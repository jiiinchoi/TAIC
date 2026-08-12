"""
model/anchor_head.py
=====================
3D Tumor Anchor Prediction Head (Sec. III-A, Fig. 2).

Downsamples the input volume, applies 3D convolutions, adaptive average
pooling and fully connected layers, and outputs a normalized anchor
coordinate via a sigmoid (Eq. 1).
"""

import torch.nn as nn
import torch.nn.functional as F


class TumorAnchorPredictionHead(nn.Module):
    def __init__(self, in_chns, input_size):
        super().__init__()
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Conv3d(in_chns, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
            nn.Sigmoid(),   # Eq. 1: normalized anchor c_hat in [0,1]^3
        )

    def forward(self, volume):
        x = F.interpolate(volume, size=self.input_size,
                           mode="trilinear", align_corners=False)
        x = self.encoder(x)
        x = self.pool(x)
        return self.fc(x)  # (B, 3)

    def get_anchor_coords(self, volume, anchor_norm):
        """Converts the normalized anchor to voxel coordinates in the
        original volume (used to extract S^A, S^C, S^S, Eq. 2)."""
        _, _, H, W, D = volume.shape
        ax = (anchor_norm[:, 0] * (H - 1)).long().clamp(0, H - 1)
        ay = (anchor_norm[:, 1] * (W - 1)).long().clamp(0, W - 1)
        az = (anchor_norm[:, 2] * (D - 1)).long().clamp(0, D - 1)
        return ax, ay, az
