"""IPN-Hand 2020 LSTM sanity check — week-1 pipeline shakedown.

Hyperparameters per IPN-HandS Appl. Sci. 2025 paper:
  - 3 stacked unidirectional LSTM layers
  - 256 hidden units per layer
  - dropout 0.3 (PyTorch nn.LSTM applies dropout between layers, not after the last)
  - Adam, lr=0.001
  - batch 64
  - 30 epochs
  - cross-entropy loss

Expected (IPN-HandS, NOT 2020): 91.2% acc / 78.0% macro-recall (mean of 3 runs).
Gate per Day-2 plan: if accuracy <90%, preprocessing may be wrong.

Dataset deviation note: this runs on IPN-Hand 2020 (14 classes incl. D0X, 3D xyz,
keyframe motion sampling), not IPN-HandS (14 classes excl. D0X variant, 2D + palm-bbox
normalization). Target may be unreachable due to dataset/recipe mismatch alone.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.log_result import log_result, compute_macro_recall


NUM_CLASSES = 14            # IPN-Hand 2020: 14 classes including D0X (no-gesture)
IN_FEATURES = 21 * 3        # 21 landmarks × 3 channels (xyz)


class IPNLSTM(nn.Module):
    def __init__(self, in_features, hidden=256, num_layers=3,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x shape can be either:
        #   (B, T, V, C)        — direct from IPNDataset-style preprocessing
        #   (B, C, T, V, M=1)   — NTU-style ST-GCN-family shape
        if x.dim() == 5:
            # NTU-style: (B, C, T, V, M) → (B, T, V, C)
            B, C, T, V, M = x.shape
            x = x.squeeze(-1).permute(0, 2, 3, 1)
        elif x.dim() == 4:
            # IPNDataset-style: (B, T, V, C) — use directly
            B, T, V, C = x.shape
        else:
            raise ValueError(f"Unexpected input dim {x.dim()}, expected 4 or 5")

        x = x.reshape(B, T, V * C)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def load_npz_split(data_dir: Path, split: str):
    """Loads a split directly from its dedicated standalone .npz file,
    extracting its internal x and y keys dynamically.
    """
    # Map the split type to the exact filename your preprocessor wrote
    split_map = {
        "train": "Annot_TrainList_splitted.npz",
        "val": "Annot_ValidList_splitted.npz",
        "test": "Annot_TestList.npz"
    }
    
    target_file = data_dir / split_map[split]
    
    if not target_file.exists():
        raise FileNotFoundError(
            f"Could not find the expected split file: {target_file}\n"
            f"Please verify your --data_path parameter points to the directory containing your processed .npz files."
        )
        
    d = np.load(target_file)
    
    # Dynamically find the internal key strings (e.g., 'x_Annot_TestList')
    x_key = [k for k in d.files if k.startswith("x")][0]
    y_key = [k for k in d.files if k.startswith("y")][0]
    
    x = torch.from_numpy(d[x_key]).float()
    y = torch.from_numpy(d[y_key]).long()
    return TensorDataset(x, y)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            all_y.append(y.cpu().numpy())
            all_p.append(pred.cpu().numpy())
    acc = correct / total
    return acc, np.concatenate(all_y), np.concatenate(all_p)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=Path,
                        default=Path("E:\\research\\week1\\prepare\\ipn\\data\\ipn_2020\\processed"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # --- reproducibility (single seed sanity, but pin everything we cheaply can) ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True   # speed > determinism for shakedown

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- data ---
    print(f"Loading {args.data_path}")
    train_ds = load_npz_split(args.data_path, "train")
    test_ds  = load_npz_split(args.data_path, "test")

    # Use val if present, else fall back to test
    try:
        val_ds = load_npz_split(args.data_path, "val")
        has_val = True
    except KeyError:
        val_ds = test_ds
        has_val = False
        print("No x_val/y_val in npz; using test for both validation and final eval.")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_dl   = DataLoader(val_ds,  batch_size=args.batch, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}{' (=test)' if not has_val else ''} | Test: {len(test_ds)}")

    # quick class-balance signal
    y_train_all = train_ds.tensors[1].numpy()
    print(f"Train class counts: {np.bincount(y_train_all).tolist()}")

    # --- model ---
    # model = IPNLSTM(in_features=IN_FEATURES).to(device)
    # n_params = sum(p.numel() for p in model.parameters()) / 1e6
    # print(f"Model params: {n_params:.2f} M")

    sample_x, _ = train_ds[0]
    if sample_x.dim() == 4:
        # (C, T, V, M=1) — already NTU-style, single sample
        C, T, V, M = sample_x.shape
        in_features = V * C
        print(f"Detected NTU-style shape (C,T,V,M)=({C},{T},{V},{M}); in_features={in_features}")
    elif sample_x.dim() == 3:
        # (T, V, C) — IPNDataset-style, single sample
        T, V, C = sample_x.shape
        in_features = V * C
        print(f"Detected IPNDataset-style shape (T,V,C)=({T},{V},{C}); in_features={in_features}")
    else:
        raise ValueError(f"Unexpected sample dim {sample_x.dim()}")

    model = IPNLSTM(in_features=in_features).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f} M")


    # --- optim ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # --- train ---
    best_val_acc = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n_batches = 0.0, 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        val_acc, _, _ = evaluate(model, val_dl, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        print(f"epoch {epoch:2d}/{args.epochs} | train_loss {train_loss:.4f} | "
              f"val_acc {val_acc:.4f} | best {best_val_acc:.4f}")

    train_time_min = (time.time() - t0) / 60.0

    # --- final test on best checkpoint ---
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, y_true, y_pred = evaluate(model, test_dl, device)
    macro_recall = compute_macro_recall(y_true, y_pred)

    print()
    print(f"=== Final (best-val checkpoint on test) ===")
    print(f"  test acc          : {test_acc:.4f}")
    print(f"  test macro-recall : {macro_recall:.4f}")
    print(f"  best val acc      : {best_val_acc:.4f}")
    print(f"  train time        : {train_time_min:.1f} min")
    print(f"  acc / macro-recall gap : {(test_acc - macro_recall)*100:.1f} pp "
          f"(large gap indicates class imbalance / D0X domination)")

    # --- gate ---
    gate = test_acc >= 0.90
    if gate:
        print("\nGATE PASS: test_acc >= 90% — pipeline appears healthy.")
    else:
        print("\nGATE FAIL: test_acc < 90% — per Day-2 plan, investigate preprocessing.")
        print("  Reminder: 90% target is from IPN-HandS paper, but on a different setup.")
        print("  Pre-investigation check: is test_acc > 70% with macro_recall < 60%?")
        print("  If yes, the LSTM is learning but D0X dominates; pipeline is probably fine.")

    # --- log ---
    log_result(
        dataset="ipn_2020",
        backbone="lstm",
        modality="joint",
        recipe="ipn_hands_paper_3layer_unidir_256_drop03",
        placement="N/A",
        seed=args.seed,
        val_acc_top1=test_acc,
        val_acc_top5=1.0,        # 14 classes — top-5 nearly meaningless; placeholder
        macro_recall=macro_recall,
        train_time_min=train_time_min,
        params_M=n_params,
        gflops=0.0,
        config_hash="ipn2020_lstm_sanity_v1",
        ckpt_path="N/A — sanity only, not saved",
        notes=(f"IPN-Hand 2020 (14 cls incl. D0X), 3D xyz, keyframe sampling; "
               f"target 91.2/78.0 from IPN-HandS may not apply to this setup; "
               f"best_val={best_val_acc:.4f}, gate_pass={gate}"),
    )


if __name__ == "__main__":
    main()