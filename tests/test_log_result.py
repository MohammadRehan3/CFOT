"""Unit tests for utils.log_result."""
import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from utils.log_result import HEADER, log_result, compute_macro_recall


def test_log_result_creates_file_with_header(tmp_path):
    """First call creates the CSV with header row and one data row."""
    csv_path = tmp_path / "test_table.csv"
    log_result(
        dataset="ntu60_xsub", backbone="ctrgcn", modality="joint",
        recipe="published", placement="baseline", seed=0,
        val_acc_top1=0.8955, val_acc_top5=0.9876, macro_recall=0.8901,
        train_time_min=720.5, params_M=1.46, gflops=1.97,
        config_hash="abc12345",
        ckpt_path="logs/week1/ctrgcn/best.pt",
        notes="test row",
        csv_path=csv_path,
    )

    with open(csv_path) as f:
        rows = list(csv.reader(f))

    assert rows[0] == HEADER
    assert len(rows) == 2
    assert rows[1][0] == "ntu60_xsub"
    assert rows[1][6] == "0.8955"   # val_acc_top1 with 4-decimal formatting
    assert rows[1][8] == "0.8901"   # macro_recall
    assert rows[1][-1] == "test row"


def test_log_result_appends_without_duplicating_header(tmp_path):
    """Second call appends a row, doesn't re-write header."""
    csv_path = tmp_path / "test_table.csv"
    for i in range(3):
        log_result(
            dataset="ntu60_xsub", backbone="stgcn", modality="joint",
            recipe="published", placement="baseline", seed=i,
            val_acc_top1=0.81 + 0.001 * i, val_acc_top5=0.97,
            macro_recall=0.80,
            train_time_min=400.0, params_M=3.10, gflops=4.20,
            config_hash="def67890",
            ckpt_path=f"logs/week1/stgcn_s{i}/best.pt",
            notes="",
            csv_path=csv_path,
        )

    with open(csv_path) as f:
        rows = list(csv.reader(f))

    assert rows[0] == HEADER
    assert len(rows) == 4   # 1 header + 3 data rows


def test_compute_macro_recall_balanced():
    """Macro recall on balanced perfect predictions is 1.0."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    assert compute_macro_recall(y_true, y_pred) == pytest.approx(1.0)


def test_compute_macro_recall_imbalanced():
    """Macro recall penalizes ignored minority classes vs. plain accuracy.

    9 samples of class 0 (all correct) + 1 sample of class 1 (wrong):
    accuracy = 9/10 = 0.90
    macro_recall = (1.0 + 0.0) / 2 = 0.50
    The gap demonstrates why macro_recall matters for IPN-HandS imbalance.
    """
    y_true = np.array([0]*9 + [1])
    y_pred = np.array([0]*9 + [0])
    assert compute_macro_recall(y_true, y_pred) == pytest.approx(0.5)