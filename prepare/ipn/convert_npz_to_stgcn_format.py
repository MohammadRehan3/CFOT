"""Convert IPN2020.npz to ST-GCN's legacy 4-file-per-split format.

Reads:  data/ipn_2020/IPN2020.npz
Writes:
  data/ipn_2020/stgcn_format/train_data.npy + train_label.pkl
  data/ipn_2020/stgcn_format/val_data.npy   + val_label.pkl
  data/ipn_2020/stgcn_format/test_data.npy  + test_label.pkl
"""
import pickle
from pathlib import Path
import numpy as np

SRC = Path(r"E:\research\week1\prepare\ipn\data\ipn_2020\IPN2020.npz")
OUT = Path(r"E:\research\week1\prepare\ipn\data\ipn_2020\stgcn_format")
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(SRC)
print("Source keys:", d.files)

for split in ["train", "val", "test"]:
    x_key, y_key = f"x_{split}", f"y_{split}"
    if x_key not in d.files:
        print(f"Skipping {split}: not in source npz")
        continue

    x = d[x_key]      # (N, 3, T, V, M)
    y = d[y_key]      # (N,)
    N = len(y)

    # ST-GCN expects (sample_names, labels) tuple in the pickle.
    # Sample names are usually skeleton filenames in NTU; for IPN, just synthesize them.
    sample_names = [f"ipn_{split}_{i:05d}" for i in range(N)]
    labels = y.astype(np.int64).tolist()

    data_path = OUT / f"{split}_data.npy"
    label_path = OUT / f"{split}_label.pkl"

    np.save(data_path, x.astype(np.float32))
    with open(label_path, "wb") as f:
        pickle.dump((sample_names, labels), f)

    print(f"{split}: wrote {data_path} ({x.shape}, {data_path.stat().st_size/1e6:.1f} MB)")
    print(f"         {label_path} (N={N})")