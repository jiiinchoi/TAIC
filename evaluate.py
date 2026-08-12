"""
evaluate.py
============
3D evaluation: predicts axial/coronal/sagittal probability maps for every
slice of the full volume, tri-planar-averages them (Eq. 17), and reports
Dice / HD95 / ASSD (Sec. IV-B) plus paired Wilcoxon significance testing
against a baseline run.
"""

import argparse
import os
import json
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import wilcoxon, ttest_rel

from config import Config
from data.dataset import scan_brats, load_volume, normalize, seg_to_multilabel, get_voxel_spacing
from model.unet import TriplanarSSLNet


def compute_dice_3d(pred, gt, smooth):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    return float((2 * inter + smooth) / (union + smooth))


def compute_hd95_3d(pred, gt, spacing=None):
    """Symmetric 95th-percentile Hausdorff distance between the two surfaces.
    Distances are computed FROM each surface TO the other surface (not to the
    full foreground), matching the standard boundary-distance definition."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    try:
        pred_surf = pred & ~binary_erosion(pred)
        gt_surf = gt & ~binary_erosion(gt)
        gt_dist = distance_transform_edt(~gt_surf, sampling=spacing)
        pred_dist = distance_transform_edt(~pred_surf, sampling=spacing)
        d1 = gt_dist[pred_surf] if pred_surf.any() else np.array([0.0])
        d2 = pred_dist[gt_surf] if gt_surf.any() else np.array([0.0])
        return float(np.percentile(np.concatenate([d1, d2]), 95))
    except Exception:
        return float("nan")


def compute_assd_3d(pred, gt, spacing=None):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    pred_surf = pred & ~binary_erosion(pred)
    gt_surf = gt & ~binary_erosion(gt)
    gt_dist = distance_transform_edt(~gt_surf, sampling=spacing)
    pred_dist = distance_transform_edt(~pred_surf, sampling=spacing)
    d1 = gt_dist[pred_surf]
    d2 = pred_dist[gt_surf]
    return float((d1.sum() + d2.sum()) / (len(d1) + len(d2)))


@torch.no_grad()
def predict_3d(model, volume_np, config, device):
    """Full-volume tri-planar sweep + probability averaging (Sec. III-F, Eq. 17-18).
    No anchor-based sampling is used at inference time."""
    C, H, W, D = volume_np.shape
    vol_t = torch.from_numpy(volume_np).float().unsqueeze(0).to(device)

    prob_ax = np.zeros((3, H, W, D), dtype=np.float32)
    prob_co = np.zeros((3, H, W, D), dtype=np.float32)
    prob_sa = np.zeros((3, H, W, D), dtype=np.float32)

    for z in range(D):
        logits = model.dec_axial(model.encoder(vol_t[:, :, :, :, z]))
        prob_ax[:, :, :, z] = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    for y in range(W):
        logits = model.dec_coronal(model.encoder(vol_t[:, :, :, y, :]))
        prob_co[:, :, y, :] = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    for x in range(H):
        logits = model.dec_sagittal(model.encoder(vol_t[:, :, x, :, :]))
        prob_sa[:, x, :, :] = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    prob_avg = (prob_ax + prob_co + prob_sa) / 3.0    # Eq. 17
    return (prob_avg > config.PRED_THRESHOLD).astype(np.float32)   # Eq. 18


@torch.no_grad()
def eval_3d(cases, config, device, run_name):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, run_name, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}")
        return None

    model = TriplanarSSLNet(
        in_chns=config.IN_CHNS,
        seg_class_num=config.SEG_CLASS_NUM,
        anchor_input_size=config.ANCHOR_INPUT_SIZE,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = {k: [] for k in
               ["dice_wt", "dice_tc", "dice_et",
                "hd95_wt", "hd95_tc", "hd95_et",
                "assd_wt", "assd_tc", "assd_et"]}

    for i, case in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] {case['case_id']}", end="\r")

        vols = [normalize(load_volume(case[m])) for m in ["t1", "t1ce", "t2", "flair"]]
        volume_np = np.stack(vols, axis=0)
        label_3d = seg_to_multilabel(load_volume(case["seg"]))
        spacing = get_voxel_spacing(case["seg"])   # (x, y, z) mm

        pred_3d = predict_3d(model, volume_np, config, device)

        for j, key in enumerate(["wt", "tc", "et"]):
            p, g = pred_3d[j], label_3d[j]
            results[f"dice_{key}"].append(compute_dice_3d(p, g, config.DICE_SMOOTH))
            results[f"hd95_{key}"].append(compute_hd95_3d(p, g, spacing=spacing))
            results[f"assd_{key}"].append(compute_assd_3d(p, g, spacing=spacing))

    print()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vs", type=str, default=None,
                         help="run name to compare against (paired Wilcoxon test)")
    Config.add_cli_arguments(parser)
    args = parser.parse_args()
    Config.update_from_args(args)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    run_name = Config.get_run_name()

    print(f"\n{'='*60}\n3D Evaluating: {run_name}\nDevice: {device}\n{'='*60}")

    cases = scan_brats(Config.DATA_ROOT)
    test_json = os.path.join(Config.CHECKPOINT_DIR, run_name, "test_cases.json")
    if os.path.exists(test_json):
        with open(test_json) as f:
            test_ids = set(json.load(f))
        test_cases = [c for c in cases if c["case_id"] in test_ids]
    else:
        _, _, test_cases = Config.split_cases(cases)
    print(f"  Test cases: {len(test_cases)}")

    res = eval_3d(test_cases, Config, device, run_name)
    if res is None:
        return

    n = len(res["dice_wt"])
    print(f"\n{'='*60}\n  3D Results on test set (n={n})\n{'-'*60}")
    print(f"  {'':8} {'Dice (mean±std)':>22} {'HD95 mm (mean±std)':>22} {'ASSD mm (mean±std)':>22}")
    print(f"{'-'*60}")

    for key, label in [("wt", "WT"), ("tc", "TC"), ("et", "ET")]:
        dices = np.array(res[f"dice_{key}"])
        hd95s = np.array(res[f"hd95_{key}"])
        assds = np.array(res[f"assd_{key}"])
        hd95s_v = hd95s[~np.isnan(hd95s)]
        assds_v = assds[~np.isnan(assds)]
        mh = hd95s_v.mean() if len(hd95s_v) else float("nan")
        sh = hd95s_v.std() if len(hd95s_v) else float("nan")
        ma = assds_v.mean() if len(assds_v) else float("nan")
        sa = assds_v.std() if len(assds_v) else float("nan")
        print(f"  {label:8} {dices.mean():.4f} ± {dices.std():.4f}        "
              f"{mh:.2f} ± {sh:.2f}        {ma:.2f} ± {sa:.2f}  "
              f"(HD95 nan {np.isnan(hd95s).mean()*100:.1f}%)")

    case_dices = np.array([
        np.mean([res["dice_wt"][i], res["dice_tc"][i], res["dice_et"][i]])
        for i in range(n)
    ])
    print(f"{'-'*60}\n  {'Mean':8} {case_dices.mean():.4f} ± {case_dices.std():.4f}")

    result_dir = os.path.join(Config.CHECKPOINT_DIR, run_name)
    np.save(os.path.join(result_dir, "test_dice_3d.npy"), case_dices)
    np.save(os.path.join(result_dir, "test_results_3d.npy"),
            {k: np.array(v) for k, v in res.items()})
    print(f"\n  Saved: {result_dir}/test_dice_3d.npy")

    if args.vs is not None:
        vs_path = os.path.join(Config.CHECKPOINT_DIR, args.vs, "test_dice_3d.npy")
        if os.path.exists(vs_path):
            vs_dices = np.load(vs_path)
            n_cmp = min(len(case_dices), len(vs_dices))
            a, b = case_dices[:n_cmp], vs_dices[:n_cmp]
            _, p_w = wilcoxon(a, b)
            _, p_t = ttest_rel(a, b)
            print(f"\n  Statistical test vs [{args.vs}] (n={n_cmp}):")
            print(f"  Wilcoxon: p={p_w:.4f}  "
                  f"{'significant' if p_w < 0.05 else 'not significant'}")
            print(f"  t-test:   p={p_t:.4f}  "
                  f"{'significant' if p_t < 0.05 else 'not significant'}")
            diff = a.mean() - b.mean()
            print(f"  Mean diff: {diff:+.4f} "
                  f"({'ours better' if diff > 0 else 'baseline better'})")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
