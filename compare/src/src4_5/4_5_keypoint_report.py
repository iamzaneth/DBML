import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "compare" / "results" / "output_4_5"
INPUT = OUTPUT_DIR / "video_quality.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if not INPUT.exists() or INPUT.stat().st_size <= 2:
    raise SystemExit(
        f"Missing or empty input file: {INPUT}. Run 4_5_keypoint_quality.py after adding .npz files."
    )


df = pd.read_csv(INPUT)



# ==============================
# 1. Detection rate
# ==============================

summary = (
    df.groupby("dataset")
    .agg({

        "pose_detection":"mean",

        "face_detection":"mean",

        "hand_detection_rate":"mean"

    })
)



summary.rename(

    columns={

        "pose_detection":
            "Pose detection rate",

        "face_detection":
            "Face detection rate",

        "hand_detection_rate":
            "Hand detection rate"

    },

    inplace=True
)



# ==============================
# 2. Missing ratio
# ==============================


missing = (

    df.groupby("dataset")

    .agg({

        "pose_missing":"mean",

        "hand_missing":"mean",

        "face_missing":"mean"


    })

)



missing.rename(

    columns={


        "pose_missing":
            "Pose missing ratio",

        "hand_missing":
            "Hand missing ratio",

        "face_missing":
            "Face missing ratio"

    },

    inplace=True

)



summary = summary.join(
    missing
)



# ==============================
# 3. Left / Right hand missing
# ==============================


hand = (

    df.groupby("dataset")

    .agg({

        "left_hand_missing_rate":
            "mean",

        "right_hand_missing_rate":
            "mean",

        "both_hand_missing_rate":
            "mean"

    })

)



hand.rename(

    columns={

        "left_hand_missing_rate":
            "Left hand missing rate",

        "right_hand_missing_rate":
            "Right hand missing rate",

        "both_hand_missing_rate":
            "Both hands missing rate"

    },

    inplace=True

)



summary = summary.join(hand)



# ==============================
# 4. Usable video ratio
# ==============================


thresholds = [

    0.1,

    0.3,

    0.5,

    0.7

]


usable_result=[]



for dataset, group in df.groupby("dataset"):


    for t in thresholds:


        ratio = (

            group["hand_missing"]

            < t

        ).mean()



        usable_result.append({

            "dataset":
                dataset,

            "threshold_missing":
                t,

            "usable_video_ratio":
                ratio

        })



usable_df = pd.DataFrame(
    usable_result
)



# ==============================
# SAVE
# ==============================


summary.to_csv(

    OUTPUT_DIR / "keypoint_quality_summary.csv"

)



usable_df.to_csv(

    OUTPUT_DIR / "usable_ratio_threshold.csv",

    index=False

)



print("\n===== KEYPOINT QUALITY =====")

print(

    summary.round(4)

)



print("\n===== USABLE RATIO =====")

print(

    usable_df

)
