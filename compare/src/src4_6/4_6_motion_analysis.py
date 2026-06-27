import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils_4_6 import (
    OUTPUT_DIR,
    extract_hands,
    hand_centers,
    iter_npz_files,
    load_npz,
    metadata_from_path,
    motion_magnitude,
    save_bar,
    valid_landmark_rows,
    write_csv,
)


def hand_motion_features(hand, valid):
    centers = hand_centers(hand)
    idx = np.where(valid)[0]
    if idx.size < 2:
        return {
            "valid_frame_ratio": float(valid.mean()),
            "motion_magnitude": 0.0,
            "trajectory_length": 0.0,
            "motion_std": 0.0,
        }
    path_points = centers[idx]
    steps = np.linalg.norm(np.diff(path_points, axis=0), axis=1)
    return {
        "valid_frame_ratio": float(valid.mean()),
        "motion_magnitude": motion_magnitude(hand, valid),
        "trajectory_length": float(steps.sum()),
        "motion_std": float(steps.std()) if steps.size else 0.0,
    }


def activity_type(left_features, right_features, valid_threshold=0.25, motion_threshold=0.002):
    left_active = (
        left_features["valid_frame_ratio"] >= valid_threshold
        and left_features["motion_magnitude"] >= motion_threshold
    )
    right_active = (
        right_features["valid_frame_ratio"] >= valid_threshold
        and right_features["motion_magnitude"] >= motion_threshold
    )
    if left_active and right_active:
        return "two_hand"
    if left_active:
        return "left_one_hand"
    if right_active:
        return "right_one_hand"
    return "low_hand_activity"


def main():
    rows = []
    for dataset, path in tqdm(list(iter_npz_files()), desc="Motion analysis"):
        try:
            data = load_npz(path)
            left, right = extract_hands(data["hands"])
            left_features = hand_motion_features(left, valid_landmark_rows(left))
            right_features = hand_motion_features(right, valid_landmark_rows(right))
            rows.append(
                {
                    **metadata_from_path(dataset, path),
                    "left_valid_ratio": left_features["valid_frame_ratio"],
                    "right_valid_ratio": right_features["valid_frame_ratio"],
                    "left_motion_magnitude": left_features["motion_magnitude"],
                    "right_motion_magnitude": right_features["motion_magnitude"],
                    "left_trajectory_length": left_features["trajectory_length"],
                    "right_trajectory_length": right_features["trajectory_length"],
                    "left_motion_std": left_features["motion_std"],
                    "right_motion_std": right_features["motion_std"],
                    "activity_type": activity_type(left_features, right_features),
                }
            )
        except Exception as exc:
            rows.append({**metadata_from_path(dataset, path), "activity_type": "error", "error": str(exc)})

    df = pd.DataFrame(rows)
    write_csv(df, "motion_features.csv")
    summary = df.groupby(["dataset", "activity_type"]).size().reset_index(name="count")
    summary["dataset_total"] = summary.groupby("dataset")["count"].transform("sum")
    summary["percentage"] = summary["count"] / summary["dataset_total"] * 100
    write_csv(summary, "one_two_hand_statistics.csv")

    for dataset, part in summary.groupby("dataset"):
        save_bar(
            part.set_index("activity_type")["percentage"],
            f"{dataset}: one-hand vs two-hand proxy",
            "Percentage (%)",
            os.path.join(OUTPUT_DIR, f"{dataset}_one_two_hand_distribution.png"),
        )
    print("Saved motion_features.csv and one_two_hand_statistics.csv")


if __name__ == "__main__":
    main()
