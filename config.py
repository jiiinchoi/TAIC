"""
config.py
=========
Central configuration for TAIC. Equation numbers refer to the paper
("Anatomically Guided Tumor-Aware Intersection Consistency for
Pseudo-Label-Free Semi-Supervised Brain Tumor Segmentation").
"""

import os
import random


class Config:
    # Experiment
    LABEL_RATIO = 0.05      # fraction of training cases used as labeled data

    # Model
    IN_CHNS = 4                        # T1, T1ce, T2, FLAIR
    SEG_CLASS_NUM = 3                  # WT, TC, ET
    ANCHOR_INPUT_SIZE = (64, 64, 40)   # input size to the anchor head (Sec. III-A)

    # Loss weights (Eq. 14-16)
    LAMBDA_SEG = 1.0       # weight on L_seg
    LAMBDA_ANCHOR = 0.5    # λ_a, weight on L_anchor (Eq. 3)
    LAMBDA_CONS = 1.0      # λ_c, base weight on L_TAIC before warm-up
    WARMUP_EPOCHS = 10     # T_warm, epochs over which λ_c(t) ramps 0 → LAMBDA_CONS

    # TAIC consistency loss (Eq. 7-13)
    TAIC_TAU = 0.2      # τ, tumor-likelihood threshold (Eq. 11)
    TAIC_W_HIGH = 1.0   # w_h, weight for tumor-likely locations (Eq. 11)
    TAIC_W_LOW = 0.1    # w_l, weight for background-likely locations (Eq. 11)
    TAIC_EPS = 1e-6     # ε, numerical-stability constant (Eq. 12)

    # Segmentation loss (Eq. 6)
    DICE_SMOOTH = 1e-6

    # Inference (Eq. 17-18)
    PRED_THRESHOLD = 0.5   # binarization threshold on the tri-planar-averaged probability

    # Data split
    SPLIT_SEED = 42   # fixed train/val/test split, independent of the experiment seed
    SEED = 1          # model init, labeled/unlabeled split, DataLoader shuffling
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    # Optimization
    BATCH_SIZE = 2         # labeled batch size; 2 + 2 unlabeled = 4 per step (paper's "batch size of 4")
    BATCH_SIZE_UNLBL = 2
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 150
    GRAD_CLIP_NORM = 1.0
    LR_SCHED_T0 = 50
    LR_SCHED_ETA_MIN = 1e-6

    # Evaluation / checkpointing
    EVAL_INTERVAL = 5
    EARLY_STOP_PATIENCE = 20

    # Paths
    DATA_ROOT = "data/MICCAI_BraTS2020_TrainingData"
    CHECKPOINT_DIR = "checkpoints"
    RUN_NAME = None   # if None, derived via get_run_name()

    # Misc
    NUM_WORKERS = 4
    DEVICE = "cuda"

    @classmethod
    def get_lambda_cons(cls, epoch):
        """Linear warm-up of λ_c(t) over WARMUP_EPOCHS."""
        return min(epoch / cls.WARMUP_EPOCHS, 1.0)

    @classmethod
    def get_run_name(cls):
        if cls.RUN_NAME:
            return cls.RUN_NAME
        return f"taic_r{cls.LABEL_RATIO}_seed{cls.SEED}"

    @classmethod
    def make_dirs(cls):
        path = os.path.join(cls.CHECKPOINT_DIR, cls.get_run_name())
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def split_cases(cls, cases):
        """Fixed 8:1:1 train/val/test split. Uses SPLIT_SEED so the test set
        never changes across different experiment seeds."""
        random.seed(cls.SPLIT_SEED)
        cases = cases.copy()
        random.shuffle(cases)
        n = len(cases)
        n_test = int(n * cls.TEST_RATIO)
        n_val = int(n * cls.VAL_RATIO)
        test_cases = cases[:n_test]
        val_cases = cases[n_test:n_test + n_val]
        train_cases = cases[n_test + n_val:]
        return train_cases, val_cases, test_cases

    @classmethod
    def split_labeled_unlabeled(cls, train_cases):
        """Splits training cases into labeled/unlabeled by LABEL_RATIO, using SEED."""
        random.seed(cls.SEED)
        cases = train_cases.copy()
        random.shuffle(cases)
        n_labeled = max(1, int(len(cases) * cls.LABEL_RATIO))
        return cases[:n_labeled], cases[n_labeled:]

    @classmethod
    def add_cli_arguments(cls, parser):
        parser.add_argument("--label_ratio", type=float, default=cls.LABEL_RATIO)
        parser.add_argument("--seed", type=int, default=cls.SEED)
        parser.add_argument("--run_name", type=str, default=cls.RUN_NAME)
        parser.add_argument("--data_root", type=str, default=cls.DATA_ROOT)
        parser.add_argument("--tau", type=float, default=cls.TAIC_TAU,
                             help="tumor-likelihood threshold in the TAIC loss (Eq. 11)")
        parser.add_argument("--pred_threshold", type=float, default=cls.PRED_THRESHOLD,
                             help="binarization threshold at inference (Eq. 18)")

    @classmethod
    def update_from_args(cls, args):
        cls.LABEL_RATIO = args.label_ratio
        cls.SEED = args.seed
        if args.run_name:
            cls.RUN_NAME = args.run_name
        cls.DATA_ROOT = args.data_root
        cls.TAIC_TAU = args.tau
        cls.PRED_THRESHOLD = args.pred_threshold
