# """IPN-Hand 2020 preprocessing — produces single-file .npz per split.

# Adapts the existing IPNDataset preprocessing logic (CFOT codebase) into a
# one-shot preprocessing script that writes train/val/test.npz files. Downstream
# loaders just np.load() — no per-sample file I/O during training.

# Key choices (carried over from IPNDataset):
#   - 14 classes including D0X (no-gesture). Labels 1..14 → 0..13 by simple shift.
#   - 4 features per landmark: (x, y, z, connectivity_degree/3).
#   - Temporal length normalization via motion-delta keyframe sampling when
#     T > max_seq_len; linear interpolation when T < max_seq_len.
#   - Missing-landmark frames filled with -1.0 sentinel (preserved from source).

# Output: data/ipn_2020/processed/{train,val,test}.npz
# Each .npz contains:
#   x: (N, T=max_seq_len, V=21, C=4) float32
#   y: (N,) int64, values in [0, 13]

# Note: this is NOT the IPN-HandS paper recipe (which uses 2D landmarks,
# sliding-window stride-32, palm-subtract + bbox-width normalize). It is the
# recipe already baked into IPNDataset, used here for week-1 pipeline shakedown
# on IPN-Hand 2020. Recipe deviation documented in baseline_table.csv notes.
# """
# import argparse
# import os
# from pathlib import Path

# import numpy as np

# # These imports come from your existing codebase
# # from utils.data_utils import top_k, interpolate_landmarks


# # MediaPipe Hands connectivity degree per landmark (how many edges touch each joint).
# # Used as the 4th feature, divided by 3 for normalization (max degree in MP hand graph is 3).
# # Derived from mediapipe.solutions.hands.HAND_CONNECTIONS — hardcoded here so the script
# # is self-contained and doesn't require mediapipe at preprocessing time.
# HAND_CONNECTIONS = [
#     (0, 1), (1, 2), (2, 3), (3, 4),                # thumb
#     (0, 5), (5, 6), (6, 7), (7, 8),                # index
#     (5, 9), (9, 10), (10, 11), (11, 12),           # middle
#     (9, 13), (13, 14), (14, 15), (15, 16),         # ring
#     (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm closure
# ]


# def build_connectivity_dict():
#     """Returns {landmark_idx: degree} for the 21-landmark MediaPipe hand graph."""
#     degree = {i: 0 for i in range(21)}
#     for i, j in HAND_CONNECTIONS:
#         degree[i] += 1
#         degree[j] += 1
#     return degree


# # Motion-delta weighting constants for keyframe sampling (carry over from IPNDataset)
# STATIC_LANDS = [0, 5, 9, 13, 17]                   # palm-side joints
# MOVING_LANDS = [i for i in range(21) if i not in STATIC_LANDS]
# LAMBDA_MOVING = 10
# LAMBDA_STATIC = 3
# LAMBDA_GLOBAL = 1

# def top_k(array, k):
#     flat = array.flatten()
#     indices = np.argpartition(flat, -k)[-k:]
#     indices = indices[np.argsort(-flat[indices])]
#     return np.sort(np.unravel_index(indices, array.shape))


# def load_landmarks(txt_file: Path, connectivity: dict) -> np.ndarray:
#     """Parse one .txt file into a (T, 21, 4) float32 array.

#     Safely skips corrupted files or sequences with fewer than 2 valid frames.
#     """
#     try:
#         # Open with utf-8, but use errors="ignore" or catch the block entirely
#         with open(txt_file, "r", encoding="utf-8") as f:
#             data = f.read()
#     except UnicodeDecodeError:
#         print(f"WARNING: File corruption detected in {txt_file}. Skipping file.")
#         return None

#     # Frames separated by blank lines
#     frames = data.split("\n\n")
#     frames = frames[:-1] if frames and frames[-1] == "" else frames

#     sequence = []
#     for frame in frames:
#         lines = frame.split("\n")
#         landmarks = []
#         joint_idx = 0 
        
#         for line in lines:
#             line = line.strip()
#             if not line:
#                 continue
                
#             if len(line) == 1:
#                 coords = [-1.0, -1.0, -1.0, -1.0]
#                 landmarks.append(coords)
#                 joint_idx += 1
#             else:
#                 parts = line.split(";")
#                 parts = [p for p in parts if len(p) > 0]
                
#                 if joint_idx < 21:
#                     try:
#                         coords = [float(x) for x in parts] + [connectivity[joint_idx] / 3.0]
#                         landmarks.append(coords)
#                     except ValueError:
#                         # Catch lines that contain corrupted non-numeric text garbage
#                         coords = [-1.0, -1.0, -1.0, -1.0]
#                         landmarks.append(coords)
#                     joint_idx += 1

#         if len(frame) == 1:
#             landmarks = np.array([[-1.0, -1.0, -1.0, -1.0]], dtype=np.float32)
#             landmarks = np.repeat(landmarks, 21, axis=0)
#         else:
#             landmarks = np.array(landmarks, dtype=np.float32)

#         if landmarks.shape[0] == 21:
#             sequence.append(landmarks)

#     if len(sequence) < 2:
#         return None

#     return np.array(sequence, dtype=np.float32)


# def interpolate_landmarks(landmarks, L):
#     l, n_landmarks, n_features = landmarks.shape # Dynamically get features (4 instead of hardcoded 6)
#     assert l > 1, "The sequence of landmarks should have at least two landmarks"

#     input_indices = np.linspace(0, l - 1, l, dtype=int)
#     output_indices = np.linspace(0, l - 1, L, dtype=float)
#     fractions = output_indices % 1
#     output_indices = np.floor(output_indices).astype(int)

#     # Fixed shape here from 6 to n_features (which will be 4)
#     interpolated_landmarks = np.zeros((L, n_landmarks, n_features), dtype=float)

#     for i in range(L):
#         if fractions[i] == 0:
#             interpolated_landmarks[i] = landmarks[input_indices[output_indices[i]]]
#         else:
#             v1 = landmarks[input_indices[output_indices[i]]]
#             v2 = landmarks[input_indices[min(output_indices[i] + 1, l - 1)]]
#             interpolated_landmarks[i] = np.mean([v1, v2], axis=0)

#     return interpolated_landmarks

# def compute_motion_delta(landmarks: np.ndarray) -> np.ndarray:
#     """Weighted motion delta per frame, used for keyframe selection."""
#     delta_moving = np.mean(
#         landmarks[1:, MOVING_LANDS, :3] - landmarks[:-1, MOVING_LANDS, :3],
#         axis=(1, 2),
#     )
#     delta_static = np.mean(
#         landmarks[1:, STATIC_LANDS, :3] - landmarks[:-1, STATIC_LANDS, :3],
#         axis=(1, 2),
#     )
#     delta_global = np.mean(
#         landmarks[1:, :, :3] - landmarks[:-1, :, :3],
#         axis=(1, 2),
#     )
#     delta = (LAMBDA_MOVING * delta_moving
#              + LAMBDA_STATIC * delta_static
#              + LAMBDA_GLOBAL * delta_global)
#     return np.concatenate(([0.0], delta))


# def normalize_sequence_length(sequence: np.ndarray, max_length: int) -> np.ndarray:
#     """If T > max_length: motion-delta keyframe sampling.
#     If T < max_length: linear interpolation upsampling.
#     If T == max_length: pass-through.
#     """
#     T = len(sequence)
#     if T > max_length:
#         delta = compute_motion_delta(sequence)
#         sampled = sequence[top_k(delta, max_length)][0]
#         return sampled
#     elif T < max_length:
#         return interpolate_landmarks(sequence, max_length)
#     return sequence


# def process_split(annotations_file: Path, data_dir: Path,
#                   max_seq_len: int, connectivity: dict):
#     """Read annotation lines, load each sequence, normalize length, collect into arrays.

#     Annotation line format (IPN-Hand 2020): comma-separated, with at least 6 fields.
#         parts[0] = folder (subject ID)
#         parts[1] = video/segment prefix
#         parts[2] = label (1-indexed, 1..14 → mapped to 0..13)
#         parts[3], parts[4], parts[5] = frame range / segment markers used in filename

#     Filename pattern: {data_dir}/{folder}/{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}.txt
#     """
#     print("annotations_file:",annotations_file)
#     print("data_dir:",data_dir)
#     print("connectivity:",connectivity)
#     all_x, all_y = [], []
#     skipped_missing_file = 0
#     skipped_too_short = 0

#     with open(annotations_file, "r") as f:
#         lines = [ln.strip() for ln in f.readlines() if ln.strip()]


#     for line in lines:
#         parts = line.split(",")
#         folder = str(parts[0])
#         label = int(parts[2]) - 1   # 1-indexed → 0-indexed, 14 classes total (0..13)

#         fname = f"{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}.txt"
#         src_path = data_dir / folder / fname

#         if not src_path.exists():
#             skipped_missing_file += 1
#             continue

#         sequence = load_landmarks(src_path, connectivity)
#         if sequence is None:
#             skipped_too_short += 1
#             continue

#         sequence = normalize_sequence_length(sequence, max_seq_len)
#         all_x.append(sequence)
#         all_y.append(label)

#     if not all_x:
#         raise RuntimeError(f"No valid sequences found in {annotations_file}")

#     x = np.stack(all_x).astype(np.float32)        # (N, T, 21, 4)
#     y = np.array(all_y, dtype=np.int64)           # (N,)
#     print(f"  -> kept {len(all_x)} sequences; "
#           f"skipped {skipped_missing_file} missing files, {skipped_too_short} too-short")
#     return x, y


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_dir", type=Path, required=True,
#                         help="Root directory with per-subject subfolders of .txt landmark files")
#     parser.add_argument("--annot_dir", type=Path, required=True,
#                         help="Directory with train.txt / val.txt / test.txt annotation files")
#     parser.add_argument("--out_dir", type=Path, default=Path("data/ipn_2020/processed"))
#     parser.add_argument("--max_seq_len", type=int, default=64,
#                         help="Temporal length all sequences are normalized to")
#     parser.add_argument("--splits", nargs="+", default=["Annot_TrainList_splitted", "Annot_ValidList_splitted", "Annot_TestList"])
#     args = parser.parse_args()

#     args.out_dir.mkdir(parents=True, exist_ok=True)
#     connectivity = build_connectivity_dict()
# # d:\Dataset\ipn (1)\annotations\Annot_TestList.txt
#     counts = {}
#     for split in args.splits:
#         annot_file = args.annot_dir / f"{split}.txt"
#         if not annot_file.exists():
#             # Some IPN-Hand distributions name them differently; try common variants
#             for alt in [f"{split}.csv", f"annot_{split}.txt", f"{split}_list.txt"]:
#                 alt_path = args.annot_dir / alt
#                 if alt_path.exists():
#                     annot_file = alt_path
#                     break
#             else:
#                 print(f"WARNING: no annotation file found for split '{split}' "
#                       f"(looked for {split}.txt, {split}.csv, annot_{split}.txt, {split}_list.txt). Skipping.")
#                 continue

#         print(f"Processing {split} from {annot_file}")
#         x, y = process_split(annot_file, args.data_dir, args.max_seq_len, connectivity)

#         unique_classes = sorted(np.unique(y).tolist())
#         class_balance = np.bincount(y).tolist()
#         print(f"  shape: {x.shape}  dtype: {x.dtype}")
#         print(f"  classes present: {unique_classes}")
#         print(f"  class counts: {class_balance}")

#         out_path = args.out_dir / f"{split}.npz"
#         np.savez_compressed(out_path, x=x, y=y)
#         print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
#         counts[split] = {"N": len(y), "shape": x.shape, "classes": unique_classes}

#     # Summary
#     print("\n=== Preprocessing summary ===")
#     for split, info in counts.items():
#         print(f"  {split}: N={info['N']}, shape={info['shape']}, "
#               f"num_classes={len(info['classes'])}")


# if __name__ == "__main__":
#     main()


"""IPN-Hand 2020 preprocessing — produces single .npz in NTU-style ST-GCN format.

Output: data/ipn_2020/IPN2020.npz
Contains:
  x_train: (N_train, 3, 64, 21, 1) float32   # N, C=xyz, T, V=21, M=1 (single hand)
  y_train: (N_train,) int64                  # labels 0..13 (14 classes incl. D0X)
  x_val:   (N_val, 3, 64, 21, 1) float32     # if val.txt provided
  y_val:   (N_val,) int64
  x_test:  (N_test, 3, 64, 21, 1) float32
  y_test:  (N_test,) int64

This matches the format your existing CTR-GCN feeder expects for NTU60_CS.npz,
so the same feeder (with a graph swap) works for IPN-Hand 2020 with minimal change.

Class setup: 14 classes including D0X (no-gesture). Labels 1..14 → 0..13.
Temporal length: motion-delta keyframe sampling (T>64) or linear interp (T<64).
Connectivity feature dropped — graph encodes it instead.
"""
import argparse
from pathlib import Path

import numpy as np

# from utils.data_utils import top_k, interpolate_landmarks


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def build_connectivity_dict():
    degree = {i: 0 for i in range(21)}
    for i, j in HAND_CONNECTIONS:
        degree[i] += 1
        degree[j] += 1
    return degree


STATIC_LANDS = [0, 5, 9, 13, 17]
MOVING_LANDS = [i for i in range(21) if i not in STATIC_LANDS]
LAMBDA_MOVING, LAMBDA_STATIC, LAMBDA_GLOBAL = 10, 3, 1


def top_k(array, k):
    flat = array.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    return np.sort(np.unravel_index(indices, array.shape))


def interpolate_landmarks(landmarks, L):
    l, n_landmarks, n_features = landmarks.shape # Dynamically get features (4 instead of hardcoded 6)
    assert l > 1, "The sequence of landmarks should have at least two landmarks"

    input_indices = np.linspace(0, l - 1, l, dtype=int)
    output_indices = np.linspace(0, l - 1, L, dtype=float)
    fractions = output_indices % 1
    output_indices = np.floor(output_indices).astype(int)

    # Fixed shape here from 6 to n_features (which will be 4)
    interpolated_landmarks = np.zeros((L, n_landmarks, n_features), dtype=float)

    for i in range(L):
        if fractions[i] == 0:
            interpolated_landmarks[i] = landmarks[input_indices[output_indices[i]]]
        else:
            v1 = landmarks[input_indices[output_indices[i]]]
            v2 = landmarks[input_indices[min(output_indices[i] + 1, l - 1)]]
            interpolated_landmarks[i] = np.mean([v1, v2], axis=0)

    return interpolated_landmarks

def load_landmarks(txt_file: Path, connectivity: dict, debug=False) -> np.ndarray:
    try:
        with open(txt_file, "r", encoding="utf-8") as f:
            data = f.read()
    except UnicodeDecodeError:
        return None

    data = data.replace("\r\n", "\n").replace("\r", "\n")
    frames = data.split("\n\n")
    frames = [f for f in frames if f.strip()]

    if debug:
        print(f"  LOAD DEBUG: {txt_file.name}")
        print(f"    file size: {len(data)} chars")
        print(f"    n raw frames after split: {len(frames)}")
        if frames:
            first_frame_lines = frames[0].split("\n")
            print(f"    first frame: {len(first_frame_lines)} lines")
            print(f"    first line of first frame: {repr(first_frame_lines[0][:80])}")

    sequence = []
    for frame in frames:
        lines = frame.split("\n")
        landmarks = []
        joint_idx = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(line) == 1:
                coords = [-1.0, -1.0, -1.0, -1.0]
                landmarks.append(coords)
                joint_idx += 1
            else:
                parts = line.split(";")
                parts = [p for p in parts if len(p) > 0]
                if joint_idx < 21:
                    try:
                        coords = [float(x) for x in parts] + [connectivity[joint_idx] / 3.0]
                        landmarks.append(coords)
                    except ValueError:
                        coords = [-1.0, -1.0, -1.0, -1.0]
                        landmarks.append(coords)
                    joint_idx += 1

        landmarks_arr = np.array(landmarks, dtype=np.float32)
        if landmarks_arr.shape[0] == 21:
            sequence.append(landmarks_arr)

    if debug:
        print(f"    valid frames assembled: {len(sequence)}")

    if len(sequence) < 2:
        return None

    return np.array(sequence, dtype=np.float32)


def compute_motion_delta(landmarks):
    delta_moving = np.mean(landmarks[1:, MOVING_LANDS, :3] - landmarks[:-1, MOVING_LANDS, :3], axis=(1, 2))
    delta_static = np.mean(landmarks[1:, STATIC_LANDS, :3] - landmarks[:-1, STATIC_LANDS, :3], axis=(1, 2))
    delta_global = np.mean(landmarks[1:, :, :3] - landmarks[:-1, :, :3], axis=(1, 2))
    delta = LAMBDA_MOVING * delta_moving + LAMBDA_STATIC * delta_static + LAMBDA_GLOBAL * delta_global
    return np.concatenate(([0.0], delta))


def normalize_sequence_length(sequence, max_length):
    T = len(sequence)
    if T > max_length:
        delta = compute_motion_delta(sequence)
        return sequence[top_k(delta, max_length)][0]
    elif T < max_length:
        return interpolate_landmarks(sequence, max_length)
    return sequence


def process_split(annotations_file: Path, data_dir: Path,
                  max_seq_len: int, connectivity: dict):
    all_x, all_y = [], []
    skipped_missing, skipped_short = 0, 0

    with open(annotations_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for i, line in enumerate(lines):
        parts = line.split(",")
        folder = str(parts[0])
        try:
            label = int(parts[2]) - 1
        except ValueError:
            skipped_missing += 1
            continue
        fname = f"{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{parts[5]}.txt"
        src_path = data_dir / folder / fname

        # DEBUG: print first 3 paths and whether they exist
        if i < 3:
            print(f"  DEBUG line {i}: parts={parts}")
            print(f"    folder={repr(folder)}")
            print(f"    fname={repr(fname)}")
            print(f"    src_path={src_path}")
            print(f"    src_path.exists()={src_path.exists()}")
            print(f"    parts[5] repr: {repr(parts[5])}")  # check for hidden chars

        if not src_path.exists():
            skipped_missing += 1
            continue

        seq = load_landmarks(src_path, connectivity)
        if seq is None:
            skipped_short += 1
            continue

        seq = normalize_sequence_length(seq, max_seq_len)   # (T, 21, 4)
        all_x.append(seq)
        all_y.append(label)

    if not all_x:
        raise RuntimeError(f"No valid sequences from {annotations_file}")

    # Stack to (N, T, 21, 4), drop connectivity feature → (N, T, 21, 3),
    # then transpose to ST-GCN-family canonical (N, C, T, V, M=1)
    x = np.stack(all_x).astype(np.float32)       # (N, T, 21, 4)
    x = x[..., :3]                                # drop connectivity column → (N, T, 21, 3)
    x = x.transpose(0, 3, 1, 2)[..., None]        # (N, 3, T, 21, 1)

    y = np.array(all_y, dtype=np.int64)

    print(f"  -> kept {len(all_y)} sequences "
          f"(skipped {skipped_missing} missing files, {skipped_short} too-short)")
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Root dir of per-subject subfolders with .txt landmark files")
    parser.add_argument("--annot_dir", type=Path, required=True,
                        help="Dir with train.txt / val.txt / test.txt")
    parser.add_argument("--out_path", type=Path,
                        default=Path("data/ipn_2020/IPN2020.npz"))
    parser.add_argument("--max_seq_len", type=int, default=64)
    args = parser.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    connectivity = build_connectivity_dict()

    split_to_annot = {
        "train": "Annot_TrainList_splitted.txt",
        "val":   "Annot_ValidList_splitted.txt",
        "test":  "Annot_TestList.txt",
    }

    arrays = {}
    summary = {}
    for split, annot_name in split_to_annot.items():
        annot = args.annot_dir / annot_name
        if not annot.exists():
            print(f"WARNING: no annotation file at {annot}. Skipping split '{split}'.")
            continue

        print(f"Processing {split} from {annot}")
        x, y = process_split(annot, args.data_dir, args.max_seq_len, connectivity)
        print(f"  shape: {x.shape}  classes present: {sorted(np.unique(y).tolist())}")
        print(f"  class counts: {np.bincount(y).tolist()}")

        arrays[f"x_{split}"] = x
        arrays[f"y_{split}"] = y
        summary[split] = (x.shape, len(np.unique(y)))
    print(f"\nSaving to {args.out_path} ...")
    np.savez_compressed(args.out_path, **arrays)
    size_mb = args.out_path.stat().st_size / 1e6
    print(f"Wrote {args.out_path}  ({size_mb:.1f} MB)")

    print("\n=== Summary ===")
    for split, (shape, nclass) in summary.items():
        print(f"  {split}: shape={shape}, num_classes={nclass}")


if __name__ == "__main__":
    main()