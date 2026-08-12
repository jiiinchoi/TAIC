"""
losses/loss.py
================
L_seg      : Dice + BCE, averaged over the three views (Eq. 6)
L_anchor   : MSE between predicted and ground-truth normalized anchor (Eq. 3)
L_TAIC     : Tumor-Aware Intersection Consistency loss (Eq. 7-13)

All thresholds and weights are read from `config` — no results-affecting
constant is hardcoded in this file.
"""

import torch
import torch.nn.functional as F


# Segmentation loss (Eq. 6)

def dice_loss(pred, target, smooth):
    pred = torch.sigmoid(pred).flatten(2)
    target = target.flatten(2)
    inter = (pred * target).sum(2)
    union = pred.sum(2) + target.sum(2)
    return (1 - (2 * inter + smooth) / (union + smooth)).mean()


def seg_loss(pred, target, config):
    return dice_loss(pred, target, config.DICE_SMOOTH) + \
        F.binary_cross_entropy_with_logits(pred, target)


# Anchor loss (Eq. 3)

def anchor_loss(pred_anchor_norm, gt_anchor_norm):
    """Eq. 3: L_anchor = ||ĉ - c^gt||²₂ (squared L2 norm, summed over the
    3 coordinates and averaged over the batch) — not mean-squared-error."""
    return ((pred_anchor_norm - gt_anchor_norm) ** 2).sum(dim=1).mean()


# Tumor-Aware Intersection Consistency loss (Eq. 7-13)

def _clamp_idx(val, max_val):
    return int(torch.clamp(torch.as_tensor(val), 0, max_val - 1))


def _intersection_consistency(p_i, p_j, tau, w_high, w_low, eps):
    """
    p_i, p_j: (C, L) prediction probabilities of two views along one
    shared intersection line. Implements Eq. 8-12.
    """
    with torch.no_grad():
        s_i = p_i.max(dim=0).values          # Eq. 9 (per-view tumor likelihood)
        s_j = p_j.max(dim=0).values
        s = 0.5 * (s_i + s_j)                 # Eq. 10
        w = torch.where(s >= tau,
                         torch.full_like(s, w_high),
                         torch.full_like(s, w_low))   # Eq. 11

    d = ((p_i - p_j) ** 2).sum(dim=0)         # Eq. 8: squared L2 norm over WT/TC/ET
    return (w * d).sum() / (w.sum() + eps)    # Eq. 12


def taic_loss(seg_ax, seg_co, seg_sa, anchor, config):
    """
    seg_ax, seg_co, seg_sa: (B, 3, H, W) logits for the axial/coronal/sagittal views
    anchor: (B, 3) integer voxel coordinates (cx, cy, cz) used to sample the slices
    """
    pa = torch.sigmoid(seg_ax)
    pc = torch.sigmoid(seg_co)
    ps = torch.sigmoid(seg_sa)
    B = pa.shape[0]

    tau, w_high, w_low, eps = (
        config.TAIC_TAU, config.TAIC_W_HIGH, config.TAIC_W_LOW, config.TAIC_EPS)

    total = pa.new_zeros(())
    for b in range(B):
        cx = _clamp_idx(anchor[b, 0], pa.shape[2])
        cy = _clamp_idx(anchor[b, 1], pa.shape[3])
        cz = _clamp_idx(anchor[b, 2], pc.shape[3])

        # L^AC (Eq. 7): axial-coronal intersection line
        l_ac = _intersection_consistency(
            pa[b, :, :, cy], pc[b, :, :, cz], tau, w_high, w_low, eps)

        # L^AS: axial-sagittal intersection line
        la, ls = pa[b, :, cx, :], ps[b, :, :, cz]
        n = min(la.shape[-1], ls.shape[-1])
        l_as = _intersection_consistency(
            la[:, :n], ls[:, :n], tau, w_high, w_low, eps)

        # L^CS: coronal-sagittal intersection line
        lc, ls2 = pc[b, :, cx, :], ps[b, :, cy, :]
        n = min(lc.shape[-1], ls2.shape[-1])
        l_cs = _intersection_consistency(
            lc[:, :n], ls2[:, :n], tau, w_high, w_low, eps)

        total = total + (l_ac + l_as + l_cs) / 3.0   # Eq. 13 (average over view pairs)

    return total / B


# Combined objectives (Eq. 14-16)

def _predicted_anchor(outputs):
    """Slices are sampled using the predicted anchor for both labeled and
    unlabeled data (Sec. III-A); L_TAIC is therefore always evaluated at
    outputs['anchor_coords'], never at the ground-truth anchor."""
    ax, ay, az = outputs["anchor_coords"]
    return torch.stack([ax, ay, az], dim=1).long()


def compute_labeled_loss(lbl_slices, gt_anchor_norm, outputs, config, epoch):
    """
    lbl_slices: (Y^A, Y^C, Y^S) ground-truth label slices extracted at the
                SAME predicted-anchor location as the model's input slices
                (see model.slice_at_coords), per Eq. 6.
    gt_anchor_norm: (B, 3) ground-truth normalized anchor c^gt, supervises
                    L_anchor only (Eq. 3).
    """
    seg_ax, seg_co, seg_sa = outputs["seg_ax"], outputs["seg_co"], outputs["seg_sa"]
    y_ax, y_co, y_sa = lbl_slices

    l_seg = (seg_loss(seg_ax, y_ax, config)
             + seg_loss(seg_co, y_co, config)
             + seg_loss(seg_sa, y_sa, config)) / 3.0   # Eq. 6

    l_taic = taic_loss(seg_ax, seg_co, seg_sa, _predicted_anchor(outputs), config)
    lambda_c = config.LAMBDA_CONS * config.get_lambda_cons(epoch)

    l_anchor = anchor_loss(outputs["anchor_norm"], gt_anchor_norm)   # Eq. 3

    total = (config.LAMBDA_SEG * l_seg
             + config.LAMBDA_ANCHOR * l_anchor
             + lambda_c * l_taic)   # Eq. 14

    return {"total": total, "seg": l_seg, "taic": l_taic, "anchor": l_anchor}


def compute_unlabeled_loss(outputs, config, epoch):
    seg_ax, seg_co, seg_sa = outputs["seg_ax"], outputs["seg_co"], outputs["seg_sa"]

    l_taic = taic_loss(seg_ax, seg_co, seg_sa, _predicted_anchor(outputs), config)
    lambda_c = config.LAMBDA_CONS * config.get_lambda_cons(epoch)
    total = lambda_c * l_taic   # Eq. 15

    return {"total": total, "taic": l_taic}
