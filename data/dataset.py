import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


def load_volume(path):
    return nib.load(path).get_fdata(dtype=np.float32)


def get_voxel_spacing(path):
    """Voxel spacing (x, y, z) in mm, read from the NIfTI header."""
    return tuple(float(s) for s in nib.load(path).header.get_zooms()[:3])


def normalize(vol):
    """Z-score normalization over non-zero (brain) voxels."""
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    mean = vol[mask].mean()
    std = vol[mask].std() + 1e-8
    out = np.zeros_like(vol)
    out[mask] = (vol[mask] - mean) / std
    return out


def seg_to_multilabel(seg):
    """BraTS label convention -> 3 overlapping binary channels [WT, TC, ET]."""
    wt = (seg > 0).astype(np.float32)
    tc = ((seg == 1) | (seg == 4)).astype(np.float32)
    et = (seg == 4).astype(np.float32)
    return np.stack([wt, tc, et], axis=0)


def get_tumor_centroid(seg):
    """Ground-truth anchor c^gt: normalized center of mass of the WT mask (Sec. III-A)."""
    coords = np.argwhere(seg > 0)
    H, W, D = seg.shape
    if len(coords) == 0:
        return H // 2, W // 2, D // 2
    cx, cy, cz = coords.mean(axis=0).astype(int)
    return (int(np.clip(cx, 0, H - 1)),
            int(np.clip(cy, 0, W - 1)),
            int(np.clip(cz, 0, D - 1)))


def _parse_modality(filepath):
    base = os.path.basename(filepath).replace(".nii.gz", "").replace(".nii", "")
    return base.split("_")[-1]


def _glob_nii(cpath):
    return glob.glob(os.path.join(cpath, "*.nii")) + \
        glob.glob(os.path.join(cpath, "*.nii.gz"))


def scan_brats(root):
    """Scans a BraTS-format directory (BraTS2020, BraTS-Africa, ...) for
    cases containing t1/t1ce/t2/flair/seg volumes."""
    cases = []
    for case_dir in sorted(os.listdir(root)):
        cpath = os.path.join(root, case_dir)
        if not os.path.isdir(cpath):
            continue
        files = {_parse_modality(f): f for f in _glob_nii(cpath)}
        if not all(k in files for k in ["t1", "t1ce", "t2", "flair", "seg"]):
            continue
        cases.append({
            "case_id": case_dir,
            "t1": files["t1"], "t1ce": files["t1ce"],
            "t2": files["t2"], "flair": files["flair"], "seg": files["seg"],
        })
    return cases


class LabeledDataset(Dataset):
    """Supervised segmentation, TAIC consistency, and anchor supervision (labeled cases).

    Per Sec. III-A, the ground-truth anchor supervises L_anchor ONLY (Eq. 3).
    Tri-planar slices — for both the input volume and the GT label volume —
    are sampled using the PREDICTED anchor for labeled and unlabeled data
    alike, so no slice extraction happens here; the full volume and full
    label volume are returned and slicing happens inside the model /
    training loop once the predicted anchor is available.
    """

    def __init__(self, cases):
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        vols = [normalize(load_volume(case[m])) for m in ["t1", "t1ce", "t2", "flair"]]
        input_vol = np.stack(vols, axis=0)   # (4, H, W, D)
        seg_raw = load_volume(case["seg"])
        label_vol = seg_to_multilabel(seg_raw)   # (3, H, W, D)

        cx, cy, cz = get_tumor_centroid(seg_raw)   # c^gt — supervises L_anchor only (Eq. 3)
        H, W, D = input_vol.shape[1:]

        t = lambda x: torch.from_numpy(x.astype(np.float32))
        return {
            "volume": t(input_vol),
            "label_volume": t(label_vol),
            "anchor_norm": torch.tensor(
                [cx / (H - 1), cy / (W - 1), cz / (D - 1)], dtype=torch.float32),
            "case_id": case["case_id"],
            "is_labeled": True,
        }


class UnlabeledDataset(Dataset):
    """Consistency-only supervision (unlabeled cases): no ground-truth masks or anchors."""

    def __init__(self, cases):
        self.cases = cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        vols = [normalize(load_volume(case[m])) for m in ["t1", "t1ce", "t2", "flair"]]
        input_vol = np.stack(vols, axis=0)
        return {
            "volume": torch.from_numpy(input_vol.astype(np.float32)),
            "case_id": case["case_id"],
            "is_labeled": False,
        }
