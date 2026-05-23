import numpy as np, pickle
x = np.load(r"data\ipn_2020\stgcn_format\train_data.npy")
with open(r"data\ipn_2020\stgcn_format\train_label.pkl", "rb") as f:
    names, labels = pickle.load(f)
print("data shape:", x.shape, x.dtype)
print("n labels:", len(labels), "unique:", sorted(set(labels)))
print("first 3 names:", names[:3])
print("first 3 labels:", labels[:3])