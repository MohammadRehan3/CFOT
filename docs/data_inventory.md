# NTU-60 X-Sub Data Inventory

| file | path | size (bytes) | md5 |
|---|---|---|---|
| train_data_joint | D:/data/ntu60/xsub/train_data_joint.npy | 12,345,678,901 | abc123... |
| train_label | D:/data/ntu60/xsub/train_label.pkl | 234,567 | def456... |
| val_data_joint | D:/data/ntu60/xsub/val_data_joint.npy | 1,234,567,890 | ghi789... |
| val_label | D:/data/ntu60/xsub/val_label.pkl | 23,456 | jkl012... |

## Source
CTR-GCN preprocessing pipeline, run on <date>, from raw NTU-60 RGB+D dataset.

## Notes
- Shape of train_data_joint: (40091, 3, 300, 25, 2) — N, C, T, V, M
- Dtype: float32
- Train/val split: official cross-subject (subjects 1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38 in train)