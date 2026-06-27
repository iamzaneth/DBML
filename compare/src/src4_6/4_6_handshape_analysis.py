import pandas as pd
from tqdm import tqdm

from utils_4_6 import (
    FINGERS,
    HAND,
    angle_at,
    extract_hands,
    iter_npz_files,
    l2,
    load_npz,
    metadata_from_path,
    motion_magnitude,
    safe_mean,
    safe_std,
    valid_landmark_rows,
    write_csv,
)


def handshape_features(hand, valid):
    if valid.sum() == 0:
        return None

    hand = hand[valid]
    scale = l2(hand[:, HAND["wrist"]], hand[:, HAND["middle_mcp"]]) + 1e-6
    row = {
        "valid_frame_ratio": float(valid.mean()),
        "motion_magnitude": motion_magnitude(hand, valid_landmark_rows(hand)),
    }

    fingertip_distances = []
    for finger, ids in FINGERS.items():
        if finger == "thumb":
            joints = [
                angle_at(hand[:, ids[0]], hand[:, ids[1]], hand[:, ids[2]]),
                angle_at(hand[:, ids[1]], hand[:, ids[2]], hand[:, ids[3]]),
            ]
        else:
            joints = [
                angle_at(hand[:, HAND["wrist"]], hand[:, ids[0]], hand[:, ids[1]]),
                angle_at(hand[:, ids[0]], hand[:, ids[1]], hand[:, ids[2]]),
                angle_at(hand[:, ids[1]], hand[:, ids[2]], hand[:, ids[3]]),
            ]

        for idx, values in enumerate(joints, start=1):
            row[f"{finger}_angle_{idx}_mean"] = safe_mean(values)
            row[f"{finger}_angle_{idx}_std"] = safe_std(values)

        tip_distance = l2(hand[:, ids[-1]], hand[:, HAND["wrist"]]) / scale
        fingertip_distances.append(tip_distance)
        row[f"{finger}_tip_wrist_ratio_mean"] = safe_mean(tip_distance)

    spread_pairs = [
        ("thumb_index", HAND["thumb_tip"], HAND["index_tip"]),
        ("index_middle", HAND["index_tip"], HAND["middle_tip"]),
        ("middle_ring", HAND["middle_tip"], HAND["ring_tip"]),
        ("ring_pinky", HAND["ring_tip"], HAND["pinky_tip"]),
        ("index_pinky", HAND["index_tip"], HAND["pinky_tip"]),
    ]
    for name, a, b in spread_pairs:
        ratio = l2(hand[:, a], hand[:, b]) / scale
        row[f"{name}_distance_ratio_mean"] = safe_mean(ratio)
        row[f"{name}_distance_ratio_std"] = safe_std(ratio)

    row["palm_width_ratio_mean"] = safe_mean(
        l2(hand[:, HAND["index_mcp"]], hand[:, HAND["pinky_mcp"]]) / scale
    )
    row["openness_mean"] = safe_mean(fingertip_distances)
    row["curvature_proxy_mean"] = safe_mean(
        [
            row["index_angle_2_mean"],
            row["middle_angle_2_mean"],
            row["ring_angle_2_mean"],
            row["pinky_angle_2_mean"],
        ]
    )
    return row


def analyze_video(dataset, path):
    data = load_npz(path)
    left, right = extract_hands(data["hands"])
    base = metadata_from_path(dataset, path)
    rows = []

    for hand_name, hand in [("left", left), ("right", right)]:
        valid = valid_landmark_rows(hand)
        features = handshape_features(hand, valid)
        if features is None:
            continue
        rows.append({**base, "hand": hand_name, **features})

    return rows


def main():
    all_rows = []
    files = list(iter_npz_files())
    for dataset, path in tqdm(files, desc="Handshape features"):
        try:
            all_rows.extend(analyze_video(dataset, path))
        except Exception as exc:
            all_rows.append({**metadata_from_path(dataset, path), "hand": "error", "error": str(exc)})

    df = pd.DataFrame(all_rows)
    output = write_csv(df, "handshape_features.csv")
    print(f"Saved {len(df)} hand-level rows to {output}")


if __name__ == "__main__":
    main()
