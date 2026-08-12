"""
model/unet.py
==============
Tri-planar segmentation network (Sec. III-B): a shared 2D encoder E and
three view-specific decoders D_v, v in {A, C, S} (Eq. 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.anchor_head import TumorAnchorPredictionHead


_FT_CHNS = [16, 32, 64, 128, 256]
_DROPOUTS = [0.05, 0.1, 0.2, 0.3, 0.5]


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch, dropout_p),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch1, in_ch2, 2, stride=2)
        self.conv = ConvBlock(in_ch2 * 2, out_ch, 0.0)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, size=x2.shape[2:],
                                mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x2, x1], dim=1))


class Encoder(nn.Module):
    """Shared 2D encoder E (Sec. III-B)."""

    def __init__(self, in_chns):
        super().__init__()
        c, d = _FT_CHNS, _DROPOUTS
        self.in_conv = ConvBlock(in_chns, c[0], d[0])
        self.down1 = DownBlock(c[0], c[1], d[1])
        self.down2 = DownBlock(c[1], c[2], d[2])
        self.down3 = DownBlock(c[2], c[3], d[3])
        self.down4 = DownBlock(c[3], c[4], d[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    """View-specific decoder D_v (Sec. III-B)."""

    def __init__(self, seg_class_num):
        super().__init__()
        c = _FT_CHNS
        self.up1 = UpBlock(c[4], c[3], c[3])
        self.up2 = UpBlock(c[3], c[2], c[2])
        self.up3 = UpBlock(c[2], c[1], c[1])
        self.up4 = UpBlock(c[1], c[0], c[0])
        self.out = nn.Conv2d(c[0], seg_class_num, 3, padding=1)

    def forward(self, features):
        x0, x1, x2, x3, x4 = features
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return self.out(x)


def slice_at_coords(volume, coords):
    """Extracts axial/coronal/sagittal slices from a (B, C, H, W, D) volume at
    per-sample coordinates `coords` (B, 3) = (cx, cy, cz).

    Used to extract the ground-truth label slices Y^v at the same predicted
    anchor location used to extract the input slices S^v (Eq. 2, Eq. 6:
    "Y^v denotes the ground-truth segmentation mask extracted at the same
    anchor location as S^v").
    """
    B = volume.shape[0]
    ax_list, co_list, sa_list = [], [], []
    for b in range(B):
        x = int(coords[b, 0].item())
        y = int(coords[b, 1].item())
        z = int(coords[b, 2].item())
        ax_list.append(volume[b, :, :, :, z])
        co_list.append(volume[b, :, :, y, :])
        sa_list.append(volume[b, :, x, :, :])
    return (torch.stack(ax_list, dim=0),
            torch.stack(co_list, dim=0),
            torch.stack(sa_list, dim=0))


class TriplanarSSLNet(nn.Module):
    """Full TAIC model: anchor head + shared encoder + 3 view-specific decoders."""

    def __init__(self, in_chns, seg_class_num, anchor_input_size):
        super().__init__()
        self.anchor_head = TumorAnchorPredictionHead(
            in_chns=in_chns, input_size=anchor_input_size)
        self.encoder = Encoder(in_chns)
        self.dec_axial = Decoder(seg_class_num)
        self.dec_coronal = Decoder(seg_class_num)
        self.dec_sagittal = Decoder(seg_class_num)

    def encode(self, axial, coronal, sagittal):
        feat_ax = self.encoder(axial)
        feat_co = self.encoder(coronal)
        feat_sa = self.encoder(sagittal)
        return feat_ax, feat_co, feat_sa

    def predict_anchor(self, volume):
        """Eq. 1: predicted normalized anchor + voxel coordinates."""
        anchor_norm = self.anchor_head(volume)
        ax, ay, az = self.anchor_head.get_anchor_coords(volume, anchor_norm)
        return anchor_norm, (ax, ay, az)

    def extract_slices(self, volume, ax, ay, az):
        """Eq. 2: extract axial/coronal/sagittal slices at the predicted anchor."""
        B = volume.shape[0]
        H, W, D = volume.shape[2], volume.shape[3], volume.shape[4]
        sampled = []
        for b in range(B):
            x, y, z = int(ax[b].item()), int(ay[b].item()), int(az[b].item())
            sampled.append((max(0, min(x, H - 1)),
                             max(0, min(y, W - 1)),
                             max(0, min(z, D - 1))))
        sampled_coords = torch.tensor(sampled, dtype=torch.long, device=volume.device)
        slices_ax, slices_co, slices_sa = slice_at_coords(volume, sampled_coords)
        return slices_ax, slices_co, slices_sa, sampled_coords

    def forward(self, volume, axial=None, coronal=None, sagittal=None):
        anchor_norm, (pred_ax, pred_ay, pred_az) = self.predict_anchor(volume)

        if axial is not None and coronal is not None and sagittal is not None:
            # Full-volume inference (Sec. III-F): slices are supplied explicitly,
            # sweeping every axial/coronal/sagittal position rather than sampling
            # only at the predicted anchor.
            feat_ax, feat_co, feat_sa = self.encode(axial, coronal, sagittal)
            sampled_coords = None
        else:
            # Training: both labeled and unlabeled branches sample slices at the
            # predicted anchor (Sec. III-A). Ground-truth anchors supervise
            # L_anchor (Eq. 3) but are not used for slice extraction.
            axial, coronal, sagittal, sampled_coords = self.extract_slices(
                volume, pred_ax, pred_ay, pred_az)
            feat_ax, feat_co, feat_sa = self.encode(axial, coronal, sagittal)

        seg_ax = self.dec_axial(feat_ax)
        seg_co = self.dec_coronal(feat_co)
        seg_sa = self.dec_sagittal(feat_sa)

        return {
            "seg_ax": seg_ax,
            "seg_co": seg_co,
            "seg_sa": seg_sa,
            "anchor_norm": anchor_norm,
            "anchor_coords": (pred_ax, pred_ay, pred_az),
            "sampled_coords": sampled_coords,
        }
