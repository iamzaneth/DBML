import os

import pandas as pd
from tqdm import tqdm

from utils_4_6 import (
    HAND,
    OUTPUT_DIR,
    extract_hands,
    extract_pose,
    iter_npz_files,
    load_npz,
    metadata_from_path,
    save_bar,
    valid_landmark_rows,
    valid_pose_rows,
    write_csv,
)


def frame_location(hand_center, pose_frame):
    nose_y = pose_frame[0, 1]
    shoulder_y = (pose_frame[11, 1] + pose_frame[12, 1]) / 2
    hip_y = (pose_frame[23, 1] + pose_frame[24, 1]) / 2
    chest_y = (shoulder_y + hip_y) / 2
    y = hand_center[1]

    if y <= (nose_y + shoulder_y) / 2:
        return "head"
    if y <= shoulder_y + 0.12 * abs(hip_y - shoulder_y):
        return "shoulder"
    if y <= chest_y:
        return "chest"
    if y <= hip_y:
        return "waist"
    return "below_waist"


def analyze_video(dataset, path):
    data = load_npz(path)
    pose = extract_pose(data["pose"])
    left, right = extract_hands(data["hands"])
    pose_valid = valid_pose_rows(pose)
    base = metadata_from_path(dataset, path)
    rows = []

    for hand_name, hand in [("left", left), ("right", right)]:
        valid = pose_valid & valid_landmark_rows(hand)
        counts = {name: 0 for name in ["head", "shoulder", "chest", "waist", "below_waist"]}
        for frame_idx in valid.nonzero()[0]:
            category = frame_location(hand[frame_idx].mean(axis=0), pose[frame_idx])
            counts[category] += 1
        total = sum(counts.values())
        if total == 0:
            continue
        dominant = max(counts, key=counts.get)
        rows.append(
            {
                **base,
                "hand": hand_name,
                "valid_frame_ratio": float(valid.mean()),
                "dominant_location": dominant,
                **{f"{key}_ratio": value / total for key, value in counts.items()},
            }
        )
    return rows


def main():
    rows = []
    for dataset, path in tqdm(list(iter_npz_files()), desc="Location analysis"):
        try:
            rows.extend(analyze_video(dataset, path))
        except Exception as exc:
            rows.append({**metadata_from_path(dataset, path), "hand": "error", "error": str(exc)})

    df = pd.DataFrame(rows)
    write_csv(df, "location_video.csv")
    summary = (
        df[df["hand"].isin(["left", "right"])]
        .groupby(["dataset", "dominant_location"])
        .size()
        .reset_index(name="count")
    )
    summary["dataset_total"] = summary.groupby("dataset")["count"].transform("sum")
    summary["percentage"] = summary["count"] / summary["dataset_total"] * 100
    write_csv(summary, "location_statistics.csv")

    for dataset, part in summary.groupby("dataset"):
        save_bar(
            part.set_index("dominant_location")["percentage"],
            f"{dataset}: dominant hand location",
            "Percentage (%)",
            os.path.join(OUTPUT_DIR, f"{dataset}_location_distribution.png"),
        )
    print("Saved location_video.csv and location_statistics.csv")


if __name__ == "__main__":
    main()
