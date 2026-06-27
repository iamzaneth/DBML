import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils_4_6 import (
    HAND,
    OUTPUT_DIR,
    extract_hands,
    iter_npz_files,
    load_npz,
    metadata_from_path,
    normalize_vector,
    save_bar,
    valid_landmark_rows,
    write_csv,
)


def direction_category(vectors):
    labels = []
    for vector in vectors:
        axis = int(np.argmax(np.abs(vector)))
        sign = vector[axis]
        if axis == 0:
            labels.append("right" if sign > 0 else "left")
        elif axis == 1:
            labels.append("down" if sign > 0 else "up")
        else:
            labels.append("forward" if sign > 0 else "backward")
    return labels


def analyze_hand(hand, valid):
    if valid.sum() == 0:
        return None
    hand = hand[valid]
    finger_direction = normalize_vector(hand[:, HAND["middle_tip"]] - hand[:, HAND["wrist"]])
    palm_normal = normalize_vector(
        np.cross(
            hand[:, HAND["index_mcp"]] - hand[:, HAND["wrist"]],
            hand[:, HAND["pinky_mcp"]] - hand[:, HAND["wrist"]],
        )
    )

    finger_labels = direction_category(finger_direction)
    palm_labels = direction_category(palm_normal)
    finger_main = max(set(finger_labels), key=finger_labels.count)
    palm_main = max(set(palm_labels), key=palm_labels.count)

    row = {
        "valid_frame_ratio": float(valid.mean()),
        "dominant_finger_direction": finger_main,
        "dominant_palm_orientation": palm_main,
        "finger_dir_x_mean": float(np.nanmean(finger_direction[:, 0])),
        "finger_dir_y_mean": float(np.nanmean(finger_direction[:, 1])),
        "finger_dir_z_mean": float(np.nanmean(finger_direction[:, 2])),
        "palm_normal_x_mean": float(np.nanmean(palm_normal[:, 0])),
        "palm_normal_y_mean": float(np.nanmean(palm_normal[:, 1])),
        "palm_normal_z_mean": float(np.nanmean(palm_normal[:, 2])),
    }
    return row


def main():
    rows = []
    for dataset, path in tqdm(list(iter_npz_files()), desc="Orientation analysis"):
        try:
            data = load_npz(path)
            left, right = extract_hands(data["hands"])
            base = metadata_from_path(dataset, path)
            for hand_name, hand in [("left", left), ("right", right)]:
                features = analyze_hand(hand, valid_landmark_rows(hand))
                if features:
                    rows.append({**base, "hand": hand_name, **features})
        except Exception as exc:
            rows.append({**metadata_from_path(dataset, path), "hand": "error", "error": str(exc)})

    df = pd.DataFrame(rows)
    write_csv(df, "orientation_video.csv")
    summary = (
        df[df["hand"].isin(["left", "right"])]
        .groupby(["dataset", "dominant_finger_direction", "dominant_palm_orientation"])
        .size()
        .reset_index(name="count")
    )
    summary["dataset_total"] = summary.groupby("dataset")["count"].transform("sum")
    summary["percentage"] = summary["count"] / summary["dataset_total"] * 100
    write_csv(summary, "orientation_statistics.csv")

    for dataset, part in summary.groupby("dataset"):
        top = part.head(10).copy()
        top["orientation"] = top["dominant_finger_direction"] + "/" + top["dominant_palm_orientation"]
        save_bar(
            top.set_index("orientation")["percentage"],
            f"{dataset}: orientation proxy",
            "Percentage (%)",
            os.path.join(OUTPUT_DIR, f"{dataset}_orientation_distribution.png"),
        )
    print("Saved orientation_video.csv and orientation_statistics.csv")


if __name__ == "__main__":
    main()
