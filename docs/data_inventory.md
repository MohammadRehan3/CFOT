# NTU-60 X-Sub Data Inventory

| file | path | size (bytes) | md5 |
|---|---|---|---|
| data_joint | E:\Mohammad Code\CTR-GCN-main\CTR-GCN-main\data\ntu\NTU60_CS.npz | 10,211,198,590 | 84dc4259132b5f6d6d056b31ecc89ab8 |


## Source
CTR-GCN preprocessing pipeline, run on <21/05/2026>, from raw NTU-60 RGB+D dataset (X_Sub).



## Contents
Single .npz file containing four arrays:
- x_train (40091, 300, 150) float32
- y_train (40091, 60) float64
- x_test (16487, 300, 150) float32
- y_test (16487, 60) float64

## Notes
- Train/val split: official cross-subject (subjects 1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38 in train)

