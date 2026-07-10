import os
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "VSL": "./dataset/VSL",
    "ASL": "./dataset/ASL",
}

display_dataset_name = {
    "ASL": "WLASL",
    "VSL": "VSL",
}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results" / "output_4_6"
EPS = 1e-6


POSE = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
}

HAND = {
    "wrist": 0,
    "thumb_cmc": 1,
    "thumb_mcp": 2,
    "thumb_ip": 3,
    "thumb_tip": 4,
    "index_mcp": 5,
    "index_pip": 6,
    "index_dip": 7,
    "index_tip": 8,
    "middle_mcp": 9,
    "middle_pip": 10,
    "middle_dip": 11,
    "middle_tip": 12,
    "ring_mcp": 13,
    "ring_pip": 14,
    "ring_dip": 15,
    "ring_tip": 16,
    "pinky_mcp": 17,
    "pinky_pip": 18,
    "pinky_dip": 19,
    "pinky_tip": 20,
}

FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def iter_npz_files():
    for dataset, folder in DATASETS.items():
        for root, _, filenames in os.walk(folder):
            for filename in filenames:
                if filename.lower().endswith(".npz"):
                    path = os.path.join(root, filename)
                    yield dataset, path


def metadata_from_path(dataset, path):
    path_obj = Path(path)
    return {
        "dataset": dataset,
        "gloss": path_obj.parent.name,
        "video": path_obj.name,
        "path": str(path_obj),
    }


def load_npz(path):
    return np.load(path, allow_pickle=True)


def _reshape_first_channels(arr, n_landmarks):
    arr = np.asarray(arr, dtype=np.float32)
    frames = arr.shape[0]
    flat = arr.reshape(frames, -1)
    needed = n_landmarks * 3
    if flat.shape[1] < needed:
        raise ValueError(f"Expected at least {needed} values, got {flat.shape[1]}")
    return flat[:, :needed].reshape(frames, n_landmarks, 3)


def extract_pose(pose):
    return _reshape_first_channels(pose, 33)


def extract_hands(hands):
    hands = np.asarray(hands, dtype=np.float32)
    frames = hands.shape[0]
    per_hand = hands.reshape(frames, 2, -1)
    needed = 21 * 3
    left = per_hand[:, 0, :needed].reshape(frames, 21, 3)
    right = per_hand[:, 1, :needed].reshape(frames, 21, 3)
    return left, right


def valid_landmark_rows(points):
    points = np.asarray(points, dtype=np.float32)
    return np.isfinite(points).all(axis=(1, 2)) & (np.abs(points).sum(axis=(1, 2)) > 0)


def valid_pose_rows(pose):
    pose = np.asarray(pose, dtype=np.float32)
    return np.isfinite(pose).all(axis=(1, 2)) & (np.abs(pose).sum(axis=(1, 2)) > 0)


def l2(a, b):
    return np.linalg.norm(a - b, axis=-1)


def safe_mean(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def safe_std(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    return float(values.std()) if values.size else np.nan


def angle_at(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + EPS
    cosang = np.sum(ba * bc, axis=-1) / denom
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def normalize_vector(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norms + EPS)


def hand_scale(hand):
    return l2(hand[:, HAND["wrist"]], hand[:, HAND["middle_mcp"]]) + EPS


def hand_centers(hand):
    return hand.mean(axis=1)


def motion_magnitude(points, valid):
    centers = hand_centers(points)
    idx = np.where(valid)[0]
    if idx.size < 2:
        return 0.0
    centers = centers[idx]
    diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return float(np.nanmean(diffs)) if diffs.size else 0.0


def write_csv(df, filename):
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


LABEL_MAPPER = {
    # Handshape
    "neutral_proxy": "Neutral",
    "closed_proxy": "Closed",
    "curved_proxy": "Curved",
    "compact_proxy": "Compact",
    "open_spread_proxy": "Open Spread",
    # Location
    "head": "Head",
    "shoulder": "Shoulder",
    "chest": "Chest",
    "waist": "Waist",
    "below_waist": "Below Waist",
    # Motion
    "two_hand": "Two-hand",
    "low_hand_activity": "Low Hand Activity",
    "right_one_hand": "Right One-hand",
    "left_one_hand": "Left One-hand",
}


def save_bar(series, title, ylabel, path):
    import matplotlib.pyplot as plt

    # Clean index labels
    series = series.copy()
    series.index = [LABEL_MAPPER.get(str(x), str(x).replace("_", " ").replace("/", " / ").title()) for x in series.index]

    plt.figure(figsize=(8, 5))
    ax = series.plot(kind="bar", color="#4C78A8", width=0.6)
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_xlabel("", fontsize=12)
    
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def read_csv(filename):
    return pd.read_csv(os.path.join(OUTPUT_DIR, filename), encoding="utf-8-sig")
