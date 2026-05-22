"""Unified results logger for week-1 baselines.

Single entrypoint: log_result(...) appends one row to baselines/baseline_table.csv.
All callers in train.py / evaluate.py / ablation scripts should use this.
"""
import csv
import hashlib
import os
from pathlib import Path
from typing import Optional

CSV_PATH = Path(__file__).parent.parent / "baselines" / "baseline_table.csv"

HEADER = [
    "dataset", "backbone", "modality", "recipe", "placement", "seed",
    "val_acc_top1", "val_acc_top5", "macro_recall",
    "train_time_min", "params_M", "gflops",
    "config_hash", "ckpt_path", "notes",
]


def config_hash(config_path: str) -> str:
    """MD5 of a YAML config file, truncated to 8 chars for readability."""
    with open(config_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def log_result(
    dataset: str,
    backbone: str,
    modality: str,
    recipe: str,
    placement: str,
    seed: int,
    val_acc_top1: float,
    val_acc_top5: float,
    macro_recall: float,
    train_time_min: float,
    params_M: float,
    gflops: float,
    config_hash: str,
    ckpt_path: str,
    notes: str = "",
    csv_path: Optional[Path] = None,
) -> None:
    """Append a single result row to baseline_table.csv.

    Creates the file with header if it doesn't exist.
    Values are formatted: accuracies as 4 decimals, time/params/flops as 2 decimals.
    """
    path = Path(csv_path) if csv_path else CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        writer.writerow([
            dataset, backbone, modality, recipe, placement, seed,
            f"{val_acc_top1:.4f}", f"{val_acc_top5:.4f}", f"{macro_recall:.4f}",
            f"{train_time_min:.2f}", f"{params_M:.2f}", f"{gflops:.2f}",
            config_hash, ckpt_path, notes,
        ])


def compute_macro_recall(y_true, y_pred) -> float:
    """Macro-averaged recall (unweighted mean of per-class recall).
    
    Required for IPN-HandS due to class imbalance (>25pp acc-vs-recall gap per paper).
    """
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, average="macro", zero_division=0)