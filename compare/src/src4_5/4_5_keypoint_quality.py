import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


# ==============================
# CONFIG
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMPARE_ROOT = PROJECT_ROOT / "compare"

DATASETS = {
    "VSL": PROJECT_ROOT / "data" / "raw" / "VSL",
    "ASL": PROJECT_ROOT / "data" / "raw" / "ASL"
}

OUTPUT_DIR = COMPARE_ROOT / "results" / "output_4_5"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==============================
# LOAD NPZ
# ==============================

def load_npz(path):

    data = np.load(
        path,
        allow_pickle=True
    )

    return {
        "pose": data["pose"],
        "hands": data["hands"],
        "face": data["face"],
        "valid_mask": data["valid_mask"]
    }


# ==============================
# NORMALIZE KEYPOINTS
# ==============================

def normalize_keypoints(arr):
    if arr.ndim == 3 and arr.shape[-1] in (2, 3):
        return arr

    if arr.ndim == 2:
        if arr.shape[1] % 3 == 0:
            dim = 3
        elif arr.shape[1] % 2 == 0:
            dim = 2
        else:
            raise ValueError(
                f"Unsupported keypoint format: shape={arr.shape}"
            )
        return arr.reshape(arr.shape[0], -1, dim)

    raise ValueError(f"Unsupported array ndim={arr.ndim}")


# ==============================
# CHECK MISSING
# ==============================

def missing_ratio(arr):
    xyz = normalize_keypoints(arr)
    missing = np.all(xyz == 0, axis=2)
    return missing.mean()

def detection_rate(arr):
    xyz = normalize_keypoints(arr)
    detected = np.any(xyz != 0, axis=2)
    return detected.mean()



# ==============================
# HAND ANALYSIS
# ==============================

def analyze_hands(hands):

    frames = hands.shape[0]

    # flatten nếu cần
    hands = hands.reshape(frames, -1)

    total_values = hands.shape[1]

    # xyz
    keypoints = total_values // 3


    # chia đôi trái phải
    hands = hands.reshape(
        frames,
        2,
        keypoints // 2,
        3
    )


    left = hands[:,0]

    right = hands[:,1]


    left_missing = np.all(
        left == 0,
        axis=(1,2)
    )


    right_missing = np.all(
        right == 0,
        axis=(1,2)
    )


    both_missing = (
        left_missing &
        right_missing
    )


    return {

        "left_hand_missing_rate":
            left_missing.mean(),

        "right_hand_missing_rate":
            right_missing.mean(),

        "both_hand_missing_rate":
            both_missing.mean(),

        "hand_detection_rate":
            1 - both_missing.mean()
    }



# ==============================
# SINGLE VIDEO
# ==============================

def analyze_video(path):


    data = load_npz(path)


    pose = data["pose"]

    hands = data["hands"]

    face = data["face"]



    hand_result = analyze_hands(
        hands
    )


    result = {


        "video":
            os.path.basename(path),


        "frames":
            pose.shape[0],



        # Missing

        "pose_missing":
            missing_ratio(pose),


        "hand_missing":
            missing_ratio(hands),


        "face_missing":
            missing_ratio(face),



        # Detection

        "pose_detection":
            detection_rate(pose),


        "face_detection":
            detection_rate(face),



        **hand_result

    }



    # usable video

    result["usable"] = (
        result["hand_missing"]
        <
        0.3
    )


    return result



# ==============================
# MAIN
# ==============================

def main():


    all_results = []


    for dataset_name, folder in DATASETS.items():


        print(
            f"\nProcessing {dataset_name}"
        )


        # Search recursively inside subfolders (e.g. ASL/a/*.npz, ASL/b/*.npz)
        files = []

        for root, _, filenames in os.walk(folder):
            for f in filenames:
                if f.endswith(".npz"):
                    files.append(
                        os.path.join(root, f)
                    )

        print("Found files:", len(files))
        for file in tqdm(files):


            r = analyze_video(file)

            r["dataset"] = dataset_name


            all_results.append(r)



    df = pd.DataFrame(
        all_results
    )

    if df.empty:
        print("No NPZ files found. Check dataset path!")
        return


    # save per video

    df.to_csv(
        OUTPUT_DIR / "video_quality.csv",
        index=False
    )


    # ==============================
    # Summary
    # ==============================

    summary = (

        df
        .groupby("dataset")
        .mean(numeric_only=True)

    )


    summary.to_csv(

        OUTPUT_DIR / "keypoint_quality.csv"

    )



    print("\n===== RESULT =====")

    print(summary)



if __name__ == "__main__":

    main()
