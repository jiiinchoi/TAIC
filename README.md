# TAIC: Tumor-Aware Intersection Consistency for Semi-Supervised Brain Tumor Segmentation

Official PyTorch implementation of **Tumor-Aware Intersection Consistency (TAIC)**, a pseudo-label-free semi-supervised learning framework for tri-planar brain tumor segmentation.

TAIC enforces prediction consistency along the exact intersection lines shared by axial, coronal, and sagittal views. A tumor-aware weighting strategy emphasizes locations likely to contain tumor tissue, reducing the dominance of background regions during consistency learning.

---

## Overview

Semi-supervised medical image segmentation commonly relies on pseudo-labels or teacher-student models. However, inaccurate pseudo-labels may reinforce model errors, particularly when only a small number of labeled cases is available.

TAIC instead exploits the geometric correspondence between tri-planar MRI views. Given a 3D multi-modal MRI volume, the framework:

1. Predicts a tumor anchor using a lightweight 3D anchor prediction head.
2. Extracts axial, coronal, and sagittal slices passing through the predicted anchor.
3. Processes the three slices using a shared 2D encoder and view-specific decoders.
4. Compares predictions along the exact intersection lines shared by each pair of views.
5. Applies tumor-aware weighting to reduce background dominance.

TAIC does not generate pseudo-labels and does not require an additional teacher network.

---

## Framework

The model consists of:

- A lightweight 3D tumor anchor prediction head
- A shared 2D encoder
- Three view-specific decoders (axial, coronal, sagittal)
- A tumor-aware intersection consistency loss

For labeled volumes, the model is optimized using segmentation, anchor, and consistency losses:

```text
L_labeled = L_seg + λ_anchor * L_anchor + λ_c(t) * L_TAIC_labeled
```

For unlabeled volumes, only the consistency loss is applied (no ground truth is available):

```text
L_unlabeled = λ_c(t) * L_TAIC_unlabeled
```

The consistency weight λ_c(t) is linearly warmed up over the first `WARMUP_EPOCHS` epochs. Slices are always sampled using the predicted anchor for both labeled and unlabeled volumes; the ground-truth anchor only supervises `L_anchor`.

---

## Tumor-Aware Intersection Consistency

Each pair of tri-planar slices shares an exact intersection line:

- Axial-Coronal
- Axial-Sagittal
- Coronal-Sagittal

TAIC minimizes the prediction discrepancy between corresponding locations on these intersection lines.

For each location, tumor likelihood is estimated from the maximum predicted probability across the WT, TC, and ET channels. Locations with tumor likelihood at or above the threshold `τ` receive a higher weight (`w_h`); locations below it receive a lower weight (`w_l`).

The tumor-aware weights are computed under `torch.no_grad()` (stop-gradient), so they determine the relative importance of intersection locations but are not themselves optimized during backpropagation.

---

## Repository Structure

```text
TAIC/
├── config.py
├── train.py
├── evaluate.py
├── requirements.txt
├── data/
│   └── dataset.py
├── model/
│   ├── anchor_head.py
│   └── unet.py
├── losses/
│   └── loss.py
└── checkpoints/
```

## Dataset Preparation

Download the BraTS2020 training dataset and organize it as follows:

```text
data/
└── MICCAI_BraTS2020_TrainingData/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_t1.nii.gz
    │   ├── BraTS20_Training_001_t1ce.nii.gz
    │   ├── BraTS20_Training_001_t2.nii.gz
    │   ├── BraTS20_Training_001_flair.nii.gz
    │   └── BraTS20_Training_001_seg.nii.gz
    ├── BraTS20_Training_002/
    └── ...
```

The four MRI modalities are used as input: T1, T1ce, T2, FLAIR. The original BraTS labels are converted into three overlapping tumor regions:

| Output region         | Original BraTS labels |
|------------------------|------------------------|
| Whole Tumor (WT)       | 1, 2, and 4            |
| Tumor Core (TC)        | 1 and 4                |
| Enhancing Tumor (ET)   | 4                       |

Each MRI modality is independently normalized using the mean and standard deviation of its nonzero voxels.

Set the dataset path via `config.py` (`DATA_ROOT`) or the `--data_root` flag.

---

## Training

```bash
python train.py --label_ratio 0.05 --seed 1 --data_root /path/to/BraTS2020
```

Uses both labeled and unlabeled volumes with the full TAIC objective (L_seg + L_anchor + L_TAIC).

### Different Label Ratios

```bash
python train.py --label_ratio 0.05 --seed 1
python train.py --label_ratio 0.10 --seed 1
python train.py --label_ratio 0.20 --seed 1
```

### Multiple Seeds

```bash
python train.py --label_ratio 0.05 --seed 1
python train.py --label_ratio 0.05 --seed 2
python train.py --label_ratio 0.05 --seed 3
```

The train/validation/test partition is fixed (`SPLIT_SEED`) and identical across all runs. The labeled subset, model initialization, and data-loader shuffling vary with `--seed`.

Other CLI options: `--tau` (tumor-aware threshold), `--pred_threshold` (inference binarization threshold), `--run_name` (checkpoint directory name).

## Evaluation

```bash
python evaluate.py --label_ratio 0.05 --seed 1
```

Evaluation sweeps every axial, coronal, and sagittal slice of the full volume, reconstructs the three view-specific probability volumes, averages them voxel-wise, and binarizes using `--pred_threshold`.

Reported per subregion (WT, TC, ET):

- Dice similarity coefficient
- 95th-percentile Hausdorff distance (HD95, mm)
- Average symmetric surface distance (ASSD, mm)

HD95/ASSD are computed surface-to-surface using voxel spacing read from the NIfTI header.

```bash
python evaluate.py --label_ratio 0.05 --seed 1 --vs <comparison_run_name>
```

`--vs` runs a case-level paired Wilcoxon signed-rank test against a previously evaluated run.

Results are saved to `checkpoints/<run_name>/test_dice_3d.npy` and `test_results_3d.npy`.

---

## Reproducibility Notes

- The train/validation/test split is fixed by a separate split seed and does not change across experiment seeds or label ratios.
- The labeled subset and model initialization vary across experiment seeds.
- The ground-truth anchor is never used for slice sampling during training — only for anchor-head supervision on labeled volumes.
- Tumor-aware weights are computed with stop-gradient.
