import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from data.dataset import scan_brats, LabeledDataset, UnlabeledDataset
from model.unet import TriplanarSSLNet, slice_at_coords
from losses.loss import compute_labeled_loss, compute_unlabeled_loss


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@torch.no_grad()
def validate(model, loader, config, device):
    model.eval()
    dice_sum = torch.zeros(3, device=device)
    n = 0

    for batch in loader:
        volume = batch["volume"].to(device)
        label_volume = batch["label_volume"].to(device)

        outputs = model(volume)
        y_ax, _, _ = slice_at_coords(label_volume, outputs["sampled_coords"])

        pred = (torch.sigmoid(outputs["seg_ax"]) > config.PRED_THRESHOLD).float()
        inter = (pred * y_ax).sum(dim=(0, 2, 3))
        union = pred.sum(dim=(0, 2, 3)) + y_ax.sum(dim=(0, 2, 3))
        dice_sum += (2 * inter + config.DICE_SMOOTH) / (union + config.DICE_SMOOTH)
        n += 1

    return dice_sum / n


def train(config, device):
    ckpt_dir = config.make_dirs()

    print("Scanning dataset...")
    cases = scan_brats(config.DATA_ROOT)
    print(f"  Total: {len(cases)}")

    train_cases, val_cases, test_cases = config.split_cases(cases)
    print(f"  Train: {len(train_cases)}  Val: {len(val_cases)}  Test: {len(test_cases)}")

    with open(os.path.join(ckpt_dir, "test_cases.json"), "w") as f:
        json.dump([c["case_id"] for c in test_cases], f)

    labeled, unlabeled = config.split_labeled_unlabeled(train_cases)
    print(f"  Labeled: {len(labeled)}  Unlabeled: {len(unlabeled)}")

    g = torch.Generator()
    g.manual_seed(config.SEED)
    loader_lbl = DataLoader(
        LabeledDataset(labeled),
        batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, drop_last=True,
        worker_init_fn=seed_worker, generator=g,
    )

    loader_unlbl = None
    if len(unlabeled) > 0:
        g_u = torch.Generator()
        g_u.manual_seed(config.SEED + 1)
        loader_unlbl = DataLoader(
            UnlabeledDataset(unlabeled),
            batch_size=config.BATCH_SIZE_UNLBL, shuffle=True,
            num_workers=config.NUM_WORKERS, drop_last=True,
            worker_init_fn=seed_worker, generator=g_u,
        )

    loader_val = DataLoader(
        LabeledDataset(val_cases),
        batch_size=1, shuffle=False, num_workers=0,
    )

    model = TriplanarSSLNet(
        in_chns=config.IN_CHNS,
        seg_class_num=config.SEG_CLASS_NUM,
        anchor_input_size=config.ANCHOR_INPUT_SIZE,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.LR_SCHED_T0, eta_min=config.LR_SCHED_ETA_MIN)

    best_dice = 0.0
    no_improve = 0
    unlbl_iter = iter(loader_unlbl) if loader_unlbl else None

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        metrics = {k: 0.0 for k in ["total", "seg", "taic", "anchor", "taic_u"]}
        n = 0

        for batch_lbl in loader_lbl:
            volume = batch_lbl["volume"].to(device)
            label_volume = batch_lbl["label_volume"].to(device)
            gt_anchor_norm = batch_lbl["anchor_norm"].to(device)

            outputs_lbl = model(volume)
            lbl_slices = slice_at_coords(label_volume, outputs_lbl["sampled_coords"])

            loss_lbl = compute_labeled_loss(
                lbl_slices, gt_anchor_norm, outputs_lbl, config, epoch)
            total = loss_lbl["total"]

            if unlbl_iter is not None:
                try:
                    batch_u = next(unlbl_iter)
                except StopIteration:
                    unlbl_iter = iter(loader_unlbl)
                    batch_u = next(unlbl_iter)

                volume_u = batch_u["volume"].to(device)
                outputs_u = model(volume_u)
                loss_u = compute_unlabeled_loss(outputs_u, config, epoch)
                total = total + loss_u["total"]
                metrics["taic_u"] += loss_u["taic"].item()

            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()

            metrics["total"] += loss_lbl["total"].item()
            metrics["seg"] += loss_lbl["seg"].item()
            metrics["taic"] += loss_lbl["taic"].item()
            metrics["anchor"] += loss_lbl["anchor"].item()
            n += 1

        scheduler.step()

        print(f"  [E{epoch:03d}] seg={metrics['seg']/n:.4f}  "
              f"taic={metrics['taic']/n:.4f}  anchor={metrics['anchor']/n:.4f}")

        if epoch % config.EVAL_INTERVAL == 0:
            torch.cuda.empty_cache()
            dice = validate(model, loader_val, config, device)
            mean = dice.mean().item()
            print(f"  [Val] WT={dice[0]:.4f} TC={dice[1]:.4f} "
                  f"ET={dice[2]:.4f} Mean={mean:.4f}")

            if mean > best_dice:
                best_dice = mean
                no_improve = 0
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "best_dice": best_dice},
                           os.path.join(ckpt_dir, "best_model.pth"))
                print(f"  Saved checkpoint (dice={best_dice:.4f})")
            else:
                no_improve += 1
                if no_improve >= config.EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    return best_dice


def main():
    parser = argparse.ArgumentParser()
    Config.add_cli_arguments(parser)
    args = parser.parse_args()
    Config.update_from_args(args)

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Label ratio:   {Config.LABEL_RATIO*100:.0f}%")
    print(f"Lambda cons:   {Config.LAMBDA_CONS}  (tau={Config.TAIC_TAU}, "
          f"w_h={Config.TAIC_W_HIGH}, w_l={Config.TAIC_W_LOW})")
    print(f"Lambda anchor: {Config.LAMBDA_ANCHOR}")
    print(f"Run name:      {Config.get_run_name()}")
    print(f"Device:        {device}")
    print(f"{'='*60}\n")

    best = train(Config, device)
    print(f"\nBest val dice: {best:.4f}")


if __name__ == "__main__":
    main()
