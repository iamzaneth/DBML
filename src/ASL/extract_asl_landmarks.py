import os
import sys
import json
import urllib.request
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Giảm log TensorFlow/MediaPipe
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Tắt logging từ Google frameworks
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ============================================================
# UTF-8 CONSOLE
# ============================================================

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
INTERIM_BASE_DIR = PROJECT_ROOT / "data" / "interim" / "ASL"
OUTPUT_BASE_DIR = PROJECT_ROOT / "data" / "processed" / "v2" / "ASL"

# ============================================================
# CONFIG
# ============================================================

# TARGET_FRAMES=None keeps all source frames; short videos are uniformly repeated
# up to MIN_TARGET_FRAMES. Set TARGET_FRAMES to an integer for fixed length.
TARGET_FRAMES = None
MIN_TARGET_FRAMES = 60
MAX_TARGET_FRAMES = None

# Use --labels for quick sample runs. None means WORKER_ID=0 runs full.
SAMPLE_LABELS = None
MAX_LABELS = None
CLI_LABELS_SELECTED = False

# Nếu file .npz đã tồn tại thì bỏ qua.
SKIP_EXISTING = True

# Optional visual check. Disabled by default because it writes extra videos.
ENABLE_PREVIEW = False
PREVIEW_BASE_DIR = PROJECT_ROOT / "data" / "processed" / "v2" / "ASL_preview"
PREVIEW_FPS = 10.0
PREVIEW_MAX_FRAMES = 240

EPSILON = 1e-6

# Trim inactive leading/trailing frames before hand interpolation. This avoids
# hallucinating hand coordinates into long idle segments where hands are missing.
ENABLE_ACTION_TRIM = True
ACTION_TRIM_MARGIN_FRAMES = 3
ACTION_TRIM_MIN_KEEP_FRAMES = 12
ACTION_TRIM_HAND_MOTION_THRESHOLD = 0.008
ACTION_TRIM_POSE_MOTION_THRESHOLD = 0.006
INTERPOLATE_EDGE_MISSING_HANDS = False
ENABLE_HAND_SIDE_STABILIZATION = True
HAND_SIDE_MINORITY_RATIO_THRESHOLD = 0.20

# Bật/tắt nhóm landmark chính.
USE_HAND = True
USE_POSE = True
USE_FACE = True

# Face được giảm nhẹ: chỉ lưu blendshapes, không lưu full 478 landmarks và matrix.
USE_FACE_BLENDSHAPES = True
USE_FACE_MATRIX = False

# Dùng float16 để giảm dung lượng. Khi train có thể cast lại float32.
STORAGE_DTYPE = np.float16

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

# ============================================================
# MULTI-WORKER CONFIG
# ============================================================
# Chế độ chạy:
# - WORKER_ID = 0: chạy toàn bộ label, không chia worker.
# - WORKER_ID = 1: xử lý label index 0-199.
# - WORKER_ID = 2: xử lý label index 200-399.
# - ...
# - WORKER_ID = 10: xử lý label index 1800-1999.

WORKER_ID = 0
LABELS_PER_WORKER = 200

# ============================================================
# MODEL URLS
# ============================================================
# Pose heavy ưu tiên chất lượng. Nếu máy yếu, có thể đổi sang full/lite.

MODEL_URLS = {
    "hand": {
        "path": MODEL_DIR / "hand_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    },
    "pose": {
        "path": MODEL_DIR / "pose_landmarker_heavy.task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    },
    "face": {
        "path": MODEL_DIR / "face_landmarker.task",
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    },
}

# ============================================================
# LANDMARK SIZE CONFIG
# ============================================================
# Dataset sau tối ưu chỉ lưu 3 nhánh:
# - pose:  33 điểm normalized x [x, y, z, visibility, presence]
#        + 33 điểm world      x [x, y, z, visibility, presence]
#        = 330 features/frame
# - hands: left/right normalized + left/right world, mỗi hand 21 x [x, y, z]
#        = 252 features/frame
# - face:  52 blendshape scores, không lưu full 478 face landmarks
#        = 52 features/frame

POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
MOUTH_LANDMARK_INDICES = [
    0, 17, 61, 291, 39, 269,
    13, 14, 78, 308, 81, 311,
]
MOUTH_LEFT_CORNER_LOCAL = MOUTH_LANDMARK_INDICES.index(61)
MOUTH_RIGHT_CORNER_LOCAL = MOUTH_LANDMARK_INDICES.index(291)
MOUTH_LANDMARK_COUNT = len(MOUTH_LANDMARK_INDICES)
FACE_BLENDSHAPE_COUNT = 52
FACE_BLENDSHAPE_DIM = FACE_BLENDSHAPE_COUNT
MOUTH_FEATURE_DIM = MOUTH_LANDMARK_COUNT * 3

POSE_NORM_DIM = POSE_LANDMARK_COUNT * 5
POSE_WORLD_DIM = POSE_LANDMARK_COUNT * 5
POSE_FEATURE_DIM = POSE_NORM_DIM + POSE_WORLD_DIM

LEFT_HAND_NORM_DIM = HAND_LANDMARK_COUNT * 3
RIGHT_HAND_NORM_DIM = HAND_LANDMARK_COUNT * 3
LEFT_HAND_WORLD_DIM = HAND_LANDMARK_COUNT * 3
RIGHT_HAND_WORLD_DIM = HAND_LANDMARK_COUNT * 3
HANDS_FEATURE_DIM = (
    LEFT_HAND_NORM_DIM
    + RIGHT_HAND_NORM_DIM
    + LEFT_HAND_WORLD_DIM
    + RIGHT_HAND_WORLD_DIM
)

FACE_FEATURE_DIM = FACE_BLENDSHAPE_COUNT if USE_FACE and USE_FACE_BLENDSHAPES else 0
TRAIN_FEATURE_DIM = POSE_FEATURE_DIM + HANDS_FEATURE_DIM + FACE_FEATURE_DIM + MOUTH_FEATURE_DIM

VALID_MASK_COLUMNS = ["pose", "left_hand", "right_hand", "face"]

# ============================================================
# UTILS
# ============================================================

def download_models() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, item in MODEL_URLS.items():
        if model_name == "hand" and not USE_HAND:
            continue
        if model_name == "pose" and not USE_POSE:
            continue
        if model_name == "face" and not USE_FACE:
            continue

        model_path = item["path"]
        model_url = item["url"]

        if model_path.exists():
            print(f"[MODEL] {model_name} đã có: {model_path}")
            continue

        print(f"[MODEL] Đang tải {model_name}...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[MODEL] Đã tải xong {model_name}: {model_path}")

def list_label_videos(label_dir: Path) -> List[Path]:
    videos: List[Path] = []

    for ext in VIDEO_EXTENSIONS:
        videos.extend(label_dir.glob(f"*{ext}"))

    videos = sorted(videos)

    return videos

class ProgressTracker:
    def __init__(self, total: int, width: int = 32):
        self.total = max(int(total), 0)
        self.width = width
        self.done = 0
        self.processed = 0
        self.skipped = 0
        self.errors = 0

    def advance(self, status: str, label: str, video_name: str) -> None:
        if self.total <= 0:
            return

        self.done += 1
        if status == "OK":
            self.processed += 1
        elif status == "SKIP":
            self.skipped += 1
        elif status == "ERROR":
            self.errors += 1

        ratio = min(self.done / self.total, 1.0)
        filled = int(round(self.width * ratio))
        bar = "█" * filled + "░" * (self.width - filled)

        # Status icon
        status_icon = "✓" if status == "OK" else ("⊘" if status == "SKIP" else ("✗" if status == "ERROR" else "•"))
        status_color = "\033[92m" if status == "OK" else ("\033[93m" if status == "SKIP" else ("\033[91m" if status == "ERROR" else "\033[0m"))
        reset_color = "\033[0m"

        # Shorten label/video_name if too long
        display_label = label if len(label) <= 15 else label[:12] + "..."
        display_video = video_name if len(video_name) <= 20 else video_name[:17] + "..."

        print(
            f"{status_color}{status_icon}{reset_color} [{bar}] {self.done:4d}/{self.total:<4d} "
            f"{ratio * 100:5.1f}% | "
            f"\033[92m✓ {self.processed}\033[0m \033[93m⊘ {self.skipped}\033[0m \033[91m✗ {self.errors}\033[0m | "
            f"{display_label:15s} / {display_video:20s}"
        )

def resolve_extraction_frames(total_frames: int) -> int:
    if total_frames <= 0:
        return 0

    extraction_frames = total_frames

    if MAX_TARGET_FRAMES is not None:
        extraction_frames = min(extraction_frames, int(MAX_TARGET_FRAMES))

    return max(1, extraction_frames)

def resolve_output_frames(sequence_frames: int) -> int:
    if sequence_frames <= 0:
        return 0

    if TARGET_FRAMES is not None:
        return max(1, int(TARGET_FRAMES))

    output_frames = sequence_frames

    if MIN_TARGET_FRAMES is not None:
        output_frames = max(output_frames, int(MIN_TARGET_FRAMES))

    if MAX_TARGET_FRAMES is not None:
        output_frames = min(output_frames, int(MAX_TARGET_FRAMES))

    return max(1, output_frames)

def get_sample_indices(total_frames: int, target_frames: int) -> List[int]:
    """
    Chuẩn hóa mỗi video thành đúng target_frames.
    Nếu video ít frame hơn target_frames, frame sẽ được lặp.
    Nếu video nhiều frame hơn target_frames, frame sẽ được sample đều theo thời gian.
    """
    if total_frames <= 0 or target_frames <= 0:
        return []

    indices = np.linspace(0, total_frames - 1, target_frames)
    indices = np.round(indices).astype(int)
    indices = np.clip(indices, 0, total_frames - 1)

    return indices.tolist()

def dtype_name(dtype) -> str:
    return np.dtype(dtype).name

def save_feature_schema(output_base_dir: Path) -> None:
    """
    Lưu một file schema duy nhất để mô tả cấu trúc tất cả file .npz.
    Không tạo schema theo từng label để tránh file dư thừa.
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    target_frames_schema = TARGET_FRAMES if TARGET_FRAMES is not None else "dynamic"

    schema = {
        "schema_version": "2.1",
        "file_format": ".npz",
        "description": "Optimized MediaPipe landmark dataset for sign-language training with NumPy keypoint preprocessing.",
        "target_frames": target_frames_schema,
        "frame_policy": {
            "target_frames": TARGET_FRAMES,
            "min_target_frames": MIN_TARGET_FRAMES,
            "max_target_frames": MAX_TARGET_FRAMES,
            "description": "Frames are extracted densely, trimmed to the action segment, then uniformly resampled after trimming. TARGET_FRAMES fixes final length; otherwise MIN_TARGET_FRAMES pads short trimmed clips.",
        },
        "action_trim": {
            "enabled": ENABLE_ACTION_TRIM,
            "margin_frames": ACTION_TRIM_MARGIN_FRAMES,
            "min_keep_frames": ACTION_TRIM_MIN_KEEP_FRAMES,
            "hand_motion_threshold": ACTION_TRIM_HAND_MOTION_THRESHOLD,
            "pose_motion_threshold": ACTION_TRIM_POSE_MOTION_THRESHOLD,
            "description": "Inactive leading/trailing frames are removed before post-trim resampling and hand interpolation.",
        },
        "hand_side_stabilization": {
            "enabled": ENABLE_HAND_SIDE_STABILIZATION,
            "minority_ratio_threshold": HAND_SIDE_MINORITY_RATIO_THRESHOLD,
            "description": "Small left/right handedness flickers are merged into the dominant hand track before interpolation.",
        },
        "storage_dtype": dtype_name(STORAGE_DTYPE),
        "train_dtype_recommendation": "Cast pose/hands/face to float32 when training.",
        "saved_keys": {
            "label": {
                "shape": [],
                "dtype": "str",
                "description": "Label folder name / gloss name.",
            },
            "video_name": {
                "shape": [],
                "dtype": "str",
                "description": "Original video filename.",
            },
            "target_frames": {
                "shape": [],
                "dtype": "int32",
                "description": "Number of sampled frames per video.",
            },
            "source_fps": {
                "shape": [],
                "dtype": "float32",
                "description": "FPS reported by OpenCV. If unavailable, fallback is 25.0.",
            },
            "source_total_frames": {
                "shape": [],
                "dtype": "int32",
                "description": "Original total frame count reported by OpenCV.",
            },
            "sample_indices": {
                "shape": [target_frames_schema],
                "dtype": "int32",
                "description": "Original frame indices sampled from the source video.",
            },
            "train_feature_dim": {
                "shape": [],
                "dtype": "int32",
                "description": "Feature dim after np.concatenate([pose, hands, face, mouth], axis=1).",
                "value": TRAIN_FEATURE_DIM,
            },
            "pose": {
                "shape": [target_frames_schema, POSE_FEATURE_DIM],
                "dtype": dtype_name(STORAGE_DTYPE),
                "contains": ["pose_normalized", "pose_world"],
                "preprocessing": "Pose xyz is translated to the neck anchor and scaled by neck-to-head distance for both normalized and world blocks.",
                "layout": {
                    "pose_normalized": {
                        "slice": [0, POSE_NORM_DIM],
                        "landmarks": POSE_LANDMARK_COUNT,
                        "values_per_landmark": ["x", "y", "z", "visibility", "presence"],
                    },
                    "pose_world": {
                        "slice": [POSE_NORM_DIM, POSE_FEATURE_DIM],
                        "landmarks": POSE_LANDMARK_COUNT,
                        "values_per_landmark": ["x", "y", "z", "visibility", "presence"],
                    },
                },
            },
            "hands": {
                "shape": [target_frames_schema, HANDS_FEATURE_DIM],
                "dtype": dtype_name(STORAGE_DTYPE),
                "contains": [
                    "left_hand_normalized",
                    "right_hand_normalized",
                    "left_hand_world",
                    "right_hand_world",
                ],
                "preprocessing": "Missing hand frames are interpolated over time, then each hand block is wrist-centered without scale normalization.",
                "layout": {
                    "left_hand_normalized": {
                        "slice": [0, LEFT_HAND_NORM_DIM],
                        "landmarks": HAND_LANDMARK_COUNT,
                        "values_per_landmark": ["x", "y", "z"],
                    },
                    "right_hand_normalized": {
                        "slice": [LEFT_HAND_NORM_DIM, LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM],
                        "landmarks": HAND_LANDMARK_COUNT,
                        "values_per_landmark": ["x", "y", "z"],
                    },
                    "left_hand_world": {
                        "slice": [LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM, LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM + LEFT_HAND_WORLD_DIM],
                        "landmarks": HAND_LANDMARK_COUNT,
                        "values_per_landmark": ["x", "y", "z"],
                    },
                    "right_hand_world": {
                        "slice": [LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM + LEFT_HAND_WORLD_DIM, HANDS_FEATURE_DIM],
                        "landmarks": HAND_LANDMARK_COUNT,
                        "values_per_landmark": ["x", "y", "z"],
                    },
                },
            },
            "face": {
                "shape": [target_frames_schema, FACE_FEATURE_DIM],
                "dtype": dtype_name(STORAGE_DTYPE),
                "contains": ["face_blendshapes"],
                "description": "52 face expression scores only. Full 478 face landmarks and 4x4 face matrix are not saved.",
            },
            "mouth": {
                "shape": [target_frames_schema, MOUTH_FEATURE_DIM],
                "dtype": dtype_name(STORAGE_DTYPE),
                "contains": ["mouth_landmarks"],
                "landmark_source": "MediaPipe face landmarks subset",
                "landmark_indices": MOUTH_LANDMARK_INDICES,
                "landmarks": MOUTH_LANDMARK_COUNT,
                "values_per_landmark": ["x", "y", "z"],
                "preprocessing": "Mouth xyz is translated to the midpoint of lip corners and scaled by lip-corner distance.",
                "description": "Compact mouth/lip landmark subset for observing mouth motion without storing full face mesh.",
            },
            "valid_mask": {
                "shape": [target_frames_schema, len(VALID_MASK_COLUMNS)],
                "dtype": "uint8",
                "columns": VALID_MASK_COLUMNS,
                "description": "Source detection mask before hand interpolation. 1 means detected by MediaPipe in that sampled frame.",
            },
        },
        "recommended_load_code": [
            "data = np.load(path)",
            "pose = data['pose'].astype(np.float32)",
            "hands = data['hands'].astype(np.float32)",
            "face = data['face'].astype(np.float32)",
            "mouth = data['mouth'].astype(np.float32)",
            "x = np.concatenate([pose, hands, face, mouth], axis=1)",
            "valid_mask = data['valid_mask']",
        ],
        "notes": [
            "No combined key is saved to avoid duplicated data.",
            "No per-label manifest CSV is created.",
            "This script extracts, preprocesses, and saves optimized .npz feature files.",
            "Hands may be non-zero after interpolation even where valid_mask marks a missed source detection.",
            "Leading/trailing idle frames can be trimmed before resampling and interpolation to avoid filling hands outside the action segment.",
            "Single-hand clips may have small handedness flickers merged before interpolation to avoid creating a second fake hand.",
        ],
    }

    schema_path = output_base_dir / "feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"[SCHEMA] Saved once: {schema_path}")

# ============================================================
# CREATE LANDMARKERS
# ============================================================

def create_hand_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_URLS["hand"]["path"])
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return vision.HandLandmarker.create_from_options(options)

def create_pose_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_URLS["pose"]["path"])
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)

def create_face_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_URLS["face"]["path"])
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
        # Chỉ lấy blendshapes để giữ thông tin biểu cảm nhưng giảm rất mạnh số chiều face.
        output_face_blendshapes=USE_FACE_BLENDSHAPES,
        # Tắt matrix để giảm output dư cho bài toán nhận diện ký hiệu.
        output_facial_transformation_matrixes=USE_FACE_MATRIX,
    )
    return vision.FaceLandmarker.create_from_options(options)

# ============================================================
# VECTOR CONVERSION
# ============================================================

def pose_to_vector(pose_landmarks) -> np.ndarray:
    """Pose: 33 x [x, y, z, visibility, presence]."""
    vector = np.zeros(POSE_LANDMARK_COUNT * 5, dtype=np.float32)

    if not pose_landmarks:
        return vector

    count = min(len(pose_landmarks), POSE_LANDMARK_COUNT)

    for i in range(count):
        lm = pose_landmarks[i]
        visibility = getattr(lm, "visibility", 0.0)
        presence = getattr(lm, "presence", 0.0)

        base = i * 5
        vector[base + 0] = lm.x
        vector[base + 1] = lm.y
        vector[base + 2] = lm.z
        vector[base + 3] = 0.0 if visibility is None else visibility
        vector[base + 4] = 0.0 if presence is None else presence

    return vector

def hand_to_vector(hand_landmarks) -> np.ndarray:
    """Hand: 21 x [x, y, z]."""
    vector = np.zeros(HAND_LANDMARK_COUNT * 3, dtype=np.float32)

    if not hand_landmarks:
        return vector

    count = min(len(hand_landmarks), HAND_LANDMARK_COUNT)

    for i in range(count):
        lm = hand_landmarks[i]
        base = i * 3
        vector[base + 0] = lm.x
        vector[base + 1] = lm.y
        vector[base + 2] = lm.z

    return vector

def face_blendshapes_to_vector(face_blendshapes) -> np.ndarray:
    """Face blendshapes: 52 expression scores."""
    vector = np.zeros(FACE_BLENDSHAPE_DIM, dtype=np.float32)

    if not face_blendshapes:
        return vector

    categories = face_blendshapes[0]
    count = min(len(categories), FACE_BLENDSHAPE_COUNT)

    for i in range(count):
        vector[i] = categories[i].score

    return vector

def mouth_landmarks_to_vector(face_landmarks) -> np.ndarray:
    """Mouth subset: selected face landmarks x [x, y, z]."""
    vector = np.zeros(MOUTH_FEATURE_DIM, dtype=np.float32)

    if not face_landmarks:
        return vector

    for out_i, face_i in enumerate(MOUTH_LANDMARK_INDICES):
        if face_i >= len(face_landmarks):
            continue

        lm = face_landmarks[face_i]
        base = out_i * 3
        vector[base + 0] = lm.x
        vector[base + 1] = lm.y
        vector[base + 2] = lm.z

    return vector

def get_handedness_label(handedness_item) -> Optional[str]:
    """MediaPipe thường trả category_name là 'Left' hoặc 'Right'."""
    try:
        if handedness_item and len(handedness_item) > 0:
            return handedness_item[0].category_name
    except Exception:
        pass

    return None

def extract_pose_features(pose_result) -> Tuple[np.ndarray, int]:
    pose_norm_vec = np.zeros(POSE_NORM_DIM, dtype=np.float32)
    pose_world_vec = np.zeros(POSE_WORLD_DIM, dtype=np.float32)
    valid_pose = 0

    if USE_POSE and pose_result is not None:
        if pose_result.pose_landmarks:
            pose_norm_vec = pose_to_vector(pose_result.pose_landmarks[0])
            valid_pose = 1

        if pose_result.pose_world_landmarks:
            pose_world_vec = pose_to_vector(pose_result.pose_world_landmarks[0])

    pose = np.concatenate([pose_norm_vec, pose_world_vec]).astype(np.float32)
    return pose, valid_pose

def extract_hands_features(hand_result) -> Tuple[np.ndarray, int, int]:
    left_hand_norm_vec = np.zeros(LEFT_HAND_NORM_DIM, dtype=np.float32)
    right_hand_norm_vec = np.zeros(RIGHT_HAND_NORM_DIM, dtype=np.float32)
    left_hand_world_vec = np.zeros(LEFT_HAND_WORLD_DIM, dtype=np.float32)
    right_hand_world_vec = np.zeros(RIGHT_HAND_WORLD_DIM, dtype=np.float32)

    valid_left_hand = 0
    valid_right_hand = 0

    if USE_HAND and hand_result is not None and hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            hand_norm_vec = hand_to_vector(hand_landmarks)

            hand_world_vec = np.zeros(HAND_LANDMARK_COUNT * 3, dtype=np.float32)
            if hand_result.hand_world_landmarks and i < len(hand_result.hand_world_landmarks):
                hand_world_vec = hand_to_vector(hand_result.hand_world_landmarks[i])

            label = None
            if hand_result.handedness and i < len(hand_result.handedness):
                label = get_handedness_label(hand_result.handedness[i])

            if label == "Left" and valid_left_hand == 0:
                left_hand_norm_vec = hand_norm_vec
                left_hand_world_vec = hand_world_vec
                valid_left_hand = 1
            elif label == "Right" and valid_right_hand == 0:
                right_hand_norm_vec = hand_norm_vec
                right_hand_world_vec = hand_world_vec
                valid_right_hand = 1
            else:
                # Fallback nếu handedness thiếu hoặc trùng nhãn.
                if valid_left_hand == 0:
                    left_hand_norm_vec = hand_norm_vec
                    left_hand_world_vec = hand_world_vec
                    valid_left_hand = 1
                elif valid_right_hand == 0:
                    right_hand_norm_vec = hand_norm_vec
                    right_hand_world_vec = hand_world_vec
                    valid_right_hand = 1

    hands = np.concatenate([
        left_hand_norm_vec,
        right_hand_norm_vec,
        left_hand_world_vec,
        right_hand_world_vec,
    ]).astype(np.float32)

    return hands, valid_left_hand, valid_right_hand

def extract_face_features(face_result) -> Tuple[np.ndarray, np.ndarray, int]:
    face_vec = np.zeros(FACE_FEATURE_DIM, dtype=np.float32)
    mouth_vec = np.zeros(MOUTH_FEATURE_DIM, dtype=np.float32)
    valid_face = 0

    if USE_FACE and face_result is not None:
        if face_result.face_landmarks:
            valid_face = 1
            mouth_vec = mouth_landmarks_to_vector(face_result.face_landmarks[0])

        if USE_FACE_BLENDSHAPES and face_result.face_blendshapes:
            face_vec = face_blendshapes_to_vector(face_result.face_blendshapes)
            valid_face = 1

    return face_vec, mouth_vec, valid_face

def extract_frame_features(hand_result, pose_result, face_result) -> Dict[str, np.ndarray]:
    pose, valid_pose = extract_pose_features(pose_result)
    hands, valid_left_hand, valid_right_hand = extract_hands_features(hand_result)
    face, mouth, valid_face = extract_face_features(face_result)

    valid_mask = np.array(
        [valid_pose, valid_left_hand, valid_right_hand, valid_face],
        dtype=np.uint8,
    )

    return {
        "pose": pose,
        "hands": hands,
        "face": face,
        "mouth": mouth,
        "valid_mask": valid_mask,
    }

def create_empty_frame_features() -> Dict[str, np.ndarray]:
    return {
        "pose": np.zeros(POSE_FEATURE_DIM, dtype=np.float32),
        "hands": np.zeros(HANDS_FEATURE_DIM, dtype=np.float32),
        "face": np.zeros(FACE_FEATURE_DIM, dtype=np.float32),
        "mouth": np.zeros(MOUTH_FEATURE_DIM, dtype=np.float32),
        "valid_mask": np.zeros(len(VALID_MASK_COLUMNS), dtype=np.uint8),
    }

# ============================================================
# ACTION TRIMMING
# ============================================================

def hand_motion_score(hand_block: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    points = hand_block.reshape(hand_block.shape[0], HAND_LANDMARK_COUNT, 3)
    motion = np.zeros(points.shape[0], dtype=np.float32)

    if points.shape[0] < 2:
        return motion

    pair_valid = valid_mask[1:].astype(bool) & valid_mask[:-1].astype(bool)
    if not np.any(pair_valid):
        return motion

    delta = np.linalg.norm(points[1:] - points[:-1], axis=2).mean(axis=1)
    delta = np.where(pair_valid, delta, 0.0).astype(np.float32)

    motion[1:] = np.maximum(motion[1:], delta)
    motion[:-1] = np.maximum(motion[:-1], delta)
    return motion

def pose_motion_score(pose_block: np.ndarray, valid_pose: np.ndarray) -> np.ndarray:
    pose_3d = pose_block.reshape(pose_block.shape[0], POSE_LANDMARK_COUNT, 5)
    upper_body = pose_3d[:, [11, 12, 13, 14, 15, 16], :3]
    motion = np.zeros(pose_3d.shape[0], dtype=np.float32)

    if pose_3d.shape[0] < 2:
        return motion

    pair_valid = valid_pose[1:].astype(bool) & valid_pose[:-1].astype(bool)
    if not np.any(pair_valid):
        return motion

    delta = np.linalg.norm(upper_body[1:] - upper_body[:-1], axis=2).mean(axis=1)
    delta = np.where(pair_valid, delta, 0.0).astype(np.float32)

    motion[1:] = np.maximum(motion[1:], delta)
    motion[:-1] = np.maximum(motion[:-1], delta)
    return motion

def expand_bounds(start: int, end: int, total: int, min_keep: int) -> Tuple[int, int]:
    start = max(0, start)
    end = min(total, end)

    if min_keep <= 0 or end - start >= min_keep or total <= end - start:
        return start, end

    missing = min_keep - (end - start)
    left_extra = missing // 2
    right_extra = missing - left_extra

    start = max(0, start - left_extra)
    end = min(total, end + right_extra)

    if end - start < min_keep:
        if start == 0:
            end = min(total, min_keep)
        elif end == total:
            start = max(0, total - min_keep)

    return start, end

def find_action_trim_bounds(stacked: Dict[str, np.ndarray]) -> Tuple[int, int, Dict[str, float]]:
    total = stacked["valid_mask"].shape[0]
    if not ENABLE_ACTION_TRIM or total == 0:
        return 0, total, {"trimmed_start": 0.0, "trimmed_end": 0.0}

    valid_mask = stacked["valid_mask"]
    hands = stacked["hands"].astype(np.float32, copy=False)
    pose = stacked["pose"].astype(np.float32, copy=False)

    left_valid = valid_mask[:, 1].astype(bool)
    right_valid = valid_mask[:, 2].astype(bool)
    hand_present = left_valid | right_valid

    left_motion = hand_motion_score(hands[:, :LEFT_HAND_NORM_DIM], left_valid)
    right_motion = hand_motion_score(
        hands[:, LEFT_HAND_NORM_DIM:LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM],
        right_valid,
    )
    pose_motion = pose_motion_score(pose[:, :POSE_NORM_DIM], valid_mask[:, 0])

    hand_active = (
        (left_motion > ACTION_TRIM_HAND_MOTION_THRESHOLD)
        | (right_motion > ACTION_TRIM_HAND_MOTION_THRESHOLD)
    )
    pose_active = pose_motion > ACTION_TRIM_POSE_MOTION_THRESHOLD

    if np.any(hand_active):
        active = hand_active
    elif np.any(hand_present):
        active = hand_present
    else:
        active = pose_active

    if not np.any(active):
        return 0, total, {
            "trimmed_start": 0.0,
            "trimmed_end": 0.0,
            "max_hand_motion": float(max(left_motion.max(), right_motion.max())),
            "max_pose_motion": float(pose_motion.max()),
        }

    active_indices = np.flatnonzero(active)
    start = int(active_indices[0]) - ACTION_TRIM_MARGIN_FRAMES
    end = int(active_indices[-1]) + ACTION_TRIM_MARGIN_FRAMES + 1

    start, end = expand_bounds(start, end, total, ACTION_TRIM_MIN_KEEP_FRAMES)

    if end <= start:
        return 0, total, {"trimmed_start": 0.0, "trimmed_end": 0.0}

    stats = {
        "trimmed_start": float(start),
        "trimmed_end": float(total - end),
        "max_hand_motion": float(max(left_motion.max(), right_motion.max())),
        "max_pose_motion": float(pose_motion.max()),
    }
    return start, end, stats

def trim_stacked_sequence(
    stacked: Dict[str, np.ndarray],
    sample_indices: List[int],
    preview_frames: List[Optional[np.ndarray]],
) -> Tuple[Dict[str, np.ndarray], List[int], List[Optional[np.ndarray]], Dict[str, float]]:
    start, end, trim_stats = find_action_trim_bounds(stacked)

    if start == 0 and end == stacked["valid_mask"].shape[0]:
        return stacked, sample_indices, preview_frames, trim_stats

    trimmed = {key: value[start:end] for key, value in stacked.items()}
    trimmed_indices = sample_indices[start:end]
    trimmed_preview = preview_frames[start:end] if ENABLE_PREVIEW else preview_frames

    return trimmed, trimmed_indices, trimmed_preview, trim_stats

def resample_stacked_sequence(
    stacked: Dict[str, np.ndarray],
    sample_indices: List[int],
    preview_frames: List[Optional[np.ndarray]],
    output_frames: int,
) -> Tuple[Dict[str, np.ndarray], List[int], List[Optional[np.ndarray]], Dict[str, int]]:
    current_frames = stacked["valid_mask"].shape[0]
    if current_frames <= 0 or output_frames <= 0:
        return stacked, sample_indices, preview_frames, {
            "frames_before_resample": current_frames,
            "frames_after_resample": current_frames,
        }

    if current_frames == output_frames:
        return stacked, sample_indices, preview_frames, {
            "frames_before_resample": current_frames,
            "frames_after_resample": current_frames,
        }

    resample_indices = get_sample_indices(current_frames, output_frames)
    resampled = {
        key: value[resample_indices]
        for key, value in stacked.items()
    }
    resampled_sample_indices = [sample_indices[i] for i in resample_indices]
    resampled_preview = (
        [preview_frames[i] for i in resample_indices]
        if ENABLE_PREVIEW
        else preview_frames
    )

    return resampled, resampled_sample_indices, resampled_preview, {
        "frames_before_resample": current_frames,
        "frames_after_resample": output_frames,
    }

# ============================================================
# NUMPY PREPROCESSING
# ============================================================

def normalize_pose_block(pose_block: np.ndarray) -> np.ndarray:
    """Anchor pose xyz at neck and scale by neck-to-head distance."""
    pose_3d = pose_block.reshape(-1, POSE_LANDMARK_COUNT, 5).astype(np.float32, copy=True)
    coords = pose_3d[:, :, :3]

    neck = (coords[:, 11, :] + coords[:, 12, :]) * 0.5
    head = (coords[:, 7, :] + coords[:, 8, :]) * 0.5
    scale = np.linalg.norm(head - neck, axis=1)

    has_pose = np.any(np.abs(coords) > EPSILON, axis=(1, 2))
    valid = has_pose & (scale > EPSILON)

    if np.any(valid):
        coords[valid] = (coords[valid] - neck[valid, None, :]) / scale[valid, None, None]
        pose_3d[:, :, :3] = coords

    return pose_3d.reshape(pose_block.shape)

def normalize_pose_sequence(pose: np.ndarray) -> np.ndarray:
    pose_out = pose.astype(np.float32, copy=True)

    pose_out[:, :POSE_NORM_DIM] = normalize_pose_block(pose_out[:, :POSE_NORM_DIM])
    pose_out[:, POSE_NORM_DIM:POSE_FEATURE_DIM] = normalize_pose_block(
        pose_out[:, POSE_NORM_DIM:POSE_FEATURE_DIM]
    )

    return pose_out

def interpolate_hand_block(hand_block: np.ndarray, valid_hint: np.ndarray) -> np.ndarray:
    """Interpolate one hand block over time using np.interp per coordinate."""
    flat = hand_block.reshape(hand_block.shape[0], -1).astype(np.float32, copy=True)
    has_coords = np.any(np.abs(flat) > EPSILON, axis=1)
    valid = has_coords & valid_hint.astype(bool)

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return flat.reshape(hand_block.shape)

    if not INTERPOLATE_EDGE_MISSING_HANDS:
        if valid_indices.size == 1:
            return flat.reshape(hand_block.shape)

        frame_axis = np.arange(flat.shape[0], dtype=np.float32)
        first_valid = int(valid_indices[0])
        last_valid = int(valid_indices[-1])
        interior_axis = frame_axis[first_valid:last_valid + 1]

        interpolated = flat.copy()
        interpolated[first_valid:last_valid + 1] = np.vstack([
            np.interp(interior_axis, valid_indices, flat[valid_indices, dim])
            for dim in range(flat.shape[1])
        ]).T

        return interpolated.astype(np.float32).reshape(hand_block.shape)

    mean_hand = flat[valid].mean(axis=0)

    if not valid[0]:
        flat[0] = mean_hand
        valid[0] = True

    if not valid[-1]:
        flat[-1] = mean_hand
        valid[-1] = True

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 1:
        flat[:] = flat[valid_indices[0]]
        return flat.reshape(hand_block.shape)

    frame_axis = np.arange(flat.shape[0], dtype=np.float32)
    interpolated = np.vstack([
        np.interp(frame_axis, valid_indices, flat[valid_indices, dim])
        for dim in range(flat.shape[1])
    ]).T

    return interpolated.astype(np.float32).reshape(hand_block.shape)

def normalize_hand_block(hand_block: np.ndarray) -> np.ndarray:
    hand_3d = hand_block.reshape(-1, HAND_LANDMARK_COUNT, 3).astype(np.float32, copy=True)
    has_hand = np.any(np.abs(hand_3d) > EPSILON, axis=(1, 2))

    if np.any(has_hand):
        wrist = hand_3d[:, 0:1, :]
        hand_3d[has_hand] = hand_3d[has_hand] - wrist[has_hand]

    return hand_3d.reshape(hand_block.shape)

def normalize_mouth_sequence(mouth: np.ndarray) -> np.ndarray:
    mouth_3d = mouth.reshape(-1, MOUTH_LANDMARK_COUNT, 3).astype(np.float32, copy=True)
    has_mouth = np.any(np.abs(mouth_3d) > EPSILON, axis=(1, 2))

    left_corner = mouth_3d[:, MOUTH_LEFT_CORNER_LOCAL, :]
    right_corner = mouth_3d[:, MOUTH_RIGHT_CORNER_LOCAL, :]
    center = (left_corner + right_corner) * 0.5
    scale = np.linalg.norm(right_corner - left_corner, axis=1)
    valid = has_mouth & (scale > EPSILON)

    if np.any(valid):
        mouth_3d[valid] = (mouth_3d[valid] - center[valid, None, :]) / scale[valid, None, None]

    return mouth_3d.reshape(mouth.shape)

def get_hand_slices(side: str) -> Tuple[slice, slice, int]:
    if side == "left":
        return (
            slice(0, LEFT_HAND_NORM_DIM),
            slice(LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM, LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM + LEFT_HAND_WORLD_DIM),
            1,
        )

    return (
        slice(LEFT_HAND_NORM_DIM, LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM),
        slice(LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM + LEFT_HAND_WORLD_DIM, HANDS_FEATURE_DIM),
        2,
    )

def stabilize_single_hand_sides(
    hands: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if not ENABLE_HAND_SIDE_STABILIZATION:
        return hands, valid_mask, {"hand_side_stabilized_frames": 0.0}

    hands_out = hands.astype(np.float32, copy=True)
    valid_out = valid_mask.astype(np.uint8, copy=True)

    left_valid = valid_out[:, 1].astype(bool)
    right_valid = valid_out[:, 2].astype(bool)
    left_count = int(left_valid.sum())
    right_count = int(right_valid.sum())

    if left_count == 0 or right_count == 0:
        return hands_out, valid_out, {
            "hand_side_stabilized_frames": 0.0,
            "left_source_frames": float(left_count),
            "right_source_frames": float(right_count),
        }

    dominant = "left" if left_count >= right_count else "right"
    minority = "right" if dominant == "left" else "left"
    dominant_count = max(left_count, right_count)
    minority_count = min(left_count, right_count)
    overlap_count = int((left_valid & right_valid).sum())

    minority_ratio = minority_count / max(dominant_count, 1)
    overlap_ratio = overlap_count / max(minority_count, 1)

    if minority_ratio > HAND_SIDE_MINORITY_RATIO_THRESHOLD:
        return hands_out, valid_out, {
            "hand_side_stabilized_frames": 0.0,
            "left_source_frames": float(left_count),
            "right_source_frames": float(right_count),
            "hand_side_minority_ratio": float(minority_ratio),
            "hand_side_overlap_ratio": float(overlap_ratio),
        }

    dom_norm, dom_world, dom_col = get_hand_slices(dominant)
    min_norm, min_world, min_col = get_hand_slices(minority)

    move_mask = valid_out[:, min_col].astype(bool) & ~valid_out[:, dom_col].astype(bool)
    clear_mask = valid_out[:, min_col].astype(bool)

    hands_out[move_mask, dom_norm] = hands_out[move_mask, min_norm]
    hands_out[move_mask, dom_world] = hands_out[move_mask, min_world]
    valid_out[move_mask, dom_col] = 1

    hands_out[clear_mask, min_norm] = 0.0
    hands_out[clear_mask, min_world] = 0.0
    valid_out[clear_mask, min_col] = 0

    return hands_out, valid_out, {
        "hand_side_stabilized_frames": float(clear_mask.sum()),
        "hand_side_dominant": 1.0 if dominant == "left" else 2.0,
        "left_source_frames": float(left_count),
        "right_source_frames": float(right_count),
        "hand_side_minority_ratio": float(minority_ratio),
        "hand_side_overlap_ratio": float(overlap_ratio),
    }

def preprocess_hands_sequence(
    hands: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    hands_out = hands.astype(np.float32, copy=True)
    preview_hands = hands.astype(np.float32, copy=True)

    hand_blocks = [
        (0, LEFT_HAND_NORM_DIM, 1),
        (LEFT_HAND_NORM_DIM, LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM, 2),
        (
            LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM,
            LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM + LEFT_HAND_WORLD_DIM,
            1,
        ),
        (
            LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM + LEFT_HAND_WORLD_DIM,
            HANDS_FEATURE_DIM,
            2,
        ),
    ]

    for start, end, valid_col in hand_blocks:
        interpolated = interpolate_hand_block(hands_out[:, start:end], valid_mask[:, valid_col])
        preview_hands[:, start:end] = interpolated
        hands_out[:, start:end] = normalize_hand_block(interpolated)

    return hands_out, preview_hands

def preprocess_stacked_landmarks(
    stacked: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    pose_float = stacked["pose"].astype(np.float32, copy=True)
    hands_float = stacked["hands"].astype(np.float32, copy=True)
    mouth_float = stacked["mouth"].astype(np.float32, copy=True)
    valid_mask = stacked["valid_mask"].astype(np.uint8, copy=True)

    preview_pose = pose_float[:, :POSE_NORM_DIM].copy()
    preview_mouth = mouth_float.copy()
    hands_float, valid_mask, hand_side_stats = stabilize_single_hand_sides(
        hands_float,
        valid_mask,
    )
    hands_processed, preview_hands = preprocess_hands_sequence(
        hands_float,
        valid_mask,
    )

    processed = {
        "pose": normalize_pose_sequence(pose_float).astype(STORAGE_DTYPE),
        "hands": hands_processed.astype(STORAGE_DTYPE),
        "face": stacked["face"].astype(STORAGE_DTYPE),
        "mouth": normalize_mouth_sequence(mouth_float).astype(STORAGE_DTYPE),
        "valid_mask": valid_mask.astype(np.uint8),
    }

    return processed, preview_pose, preview_hands, preview_mouth, hand_side_stats

# ============================================================
# PREVIEW
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (0, 7), (0, 8), (7, 11), (8, 12),
]

MOUTH_CONNECTIONS_FACE_INDEX = [
    (61, 39), (39, 0), (0, 269), (269, 291),
    (61, 17), (17, 291),
    (78, 81), (81, 13), (13, 311), (311, 308),
    (78, 14), (14, 308),
]
MOUTH_INDEX_TO_LOCAL = {
    face_index: local_index
    for local_index, face_index in enumerate(MOUTH_LANDMARK_INDICES)
}
MOUTH_CONNECTIONS = [
    (MOUTH_INDEX_TO_LOCAL[a], MOUTH_INDEX_TO_LOCAL[b])
    for a, b in MOUTH_CONNECTIONS_FACE_INDEX
]

def landmarks_to_pixels(points: np.ndarray, width: int, height: int) -> np.ndarray:
    xy = points[:, :2].astype(np.float32, copy=True)
    xy[:, 0] *= width
    xy[:, 1] *= height
    return np.round(xy).astype(np.int32)

def draw_landmark_set(
    frame: np.ndarray,
    flat_landmarks: np.ndarray,
    count: int,
    dims: int,
    connections: List[Tuple[int, int]],
    color: Tuple[int, int, int],
) -> None:
    landmarks = flat_landmarks.reshape(count, dims)
    if not np.any(np.abs(landmarks[:, :3]) > EPSILON):
        return

    height, width = frame.shape[:2]
    pixels = landmarks_to_pixels(landmarks, width, height)

    in_frame = (
        np.isfinite(landmarks[:, 0])
        & np.isfinite(landmarks[:, 1])
        & (landmarks[:, 0] >= -0.25)
        & (landmarks[:, 0] <= 1.25)
        & (landmarks[:, 1] >= -0.25)
        & (landmarks[:, 1] <= 1.25)
    )

    for a, b in connections:
        if in_frame[a] and in_frame[b]:
            cv2.line(frame, tuple(pixels[a]), tuple(pixels[b]), color, 1, cv2.LINE_AA)

    for idx, point in enumerate(pixels):
        if in_frame[idx]:
            cv2.circle(frame, tuple(point), 2, color, -1, cv2.LINE_AA)

def save_keypoint_preview(
    frames: List[Optional[np.ndarray]],
    preview_pose: np.ndarray,
    preview_hands: np.ndarray,
    preview_mouth: np.ndarray,
    target_label: str,
    video_path: Path,
) -> Optional[Path]:
    valid_frames = [frame for frame in frames if frame is not None]
    if not valid_frames:
        return None

    preview_dir = PREVIEW_BASE_DIR / target_label
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"{video_path.stem}_preview.mp4"

    first_frame = valid_frames[0]
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(preview_path), fourcc, PREVIEW_FPS, (width, height))

    max_frames = min(len(frames), PREVIEW_MAX_FRAMES)
    for i in range(max_frames):
        frame = frames[i]
        if frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = frame.copy()

        draw_landmark_set(
            frame,
            preview_pose[i],
            POSE_LANDMARK_COUNT,
            5,
            POSE_CONNECTIONS,
            (0, 255, 255),
        )
        draw_landmark_set(
            frame,
            preview_hands[i, :LEFT_HAND_NORM_DIM],
            HAND_LANDMARK_COUNT,
            3,
            HAND_CONNECTIONS,
            (0, 255, 0),
        )
        draw_landmark_set(
            frame,
            preview_hands[i, LEFT_HAND_NORM_DIM:LEFT_HAND_NORM_DIM + RIGHT_HAND_NORM_DIM],
            HAND_LANDMARK_COUNT,
            3,
            HAND_CONNECTIONS,
            (255, 0, 255),
        )
        draw_landmark_set(
            frame,
            preview_mouth[i],
            MOUTH_LANDMARK_COUNT,
            3,
            MOUTH_CONNECTIONS,
            (255, 255, 0),
        )
        writer.write(frame)

    writer.release()
    return preview_path

# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_one_video(
    video_path: Path,
    output_npz_path: Path,
    hand_landmarker,
    pose_landmarker,
    face_landmarker,
    timestamp_offset_ms: int,
    target_label: str,
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"   [ERROR] Không mở được video: {video_path}")
        return None, timestamp_offset_ms

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0:
        source_fps = 25.0

    extraction_frame_count = resolve_extraction_frames(total_frames)
    sample_indices = get_sample_indices(total_frames, extraction_frame_count)

    if len(sample_indices) == 0:
        print(f"   [ERROR] Video không có frame: {video_path}")
        cap.release()
        return None, timestamp_offset_ms

    sequence_parts = {
        "pose": [],
        "hands": [],
        "face": [],
        "mouth": [],
        "valid_mask": [],
    }
    preview_frames: List[Optional[np.ndarray]] = [] if ENABLE_PREVIEW else []

    # Timestamp giả lập 60 FPS để MediaPipe VIDEO mode nhận timestamp tăng dần.
    timestamp_step_ms = int(1000 / 60)

    for sample_i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        timestamp_ms = timestamp_offset_ms + sample_i * timestamp_step_ms

        if not ret:
            frame_features = create_empty_frame_features()
            for key in sequence_parts:
                sequence_parts[key].append(frame_features[key])
            if ENABLE_PREVIEW:
                preview_frames.append(None)
            continue

        if ENABLE_PREVIEW:
            preview_frames.append(frame.copy())

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        hand_result = None
        pose_result = None
        face_result = None

        if USE_HAND and hand_landmarker is not None:
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        if USE_POSE and pose_landmarker is not None:
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        if USE_FACE and face_landmarker is not None:
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        frame_features = extract_frame_features(
            hand_result=hand_result,
            pose_result=pose_result,
            face_result=face_result,
        )

        for key in sequence_parts:
            sequence_parts[key].append(frame_features[key])

    cap.release()

    stacked = {
        "pose": np.stack(sequence_parts["pose"], axis=0).astype(np.float32),
        "hands": np.stack(sequence_parts["hands"], axis=0).astype(np.float32),
        "face": np.stack(sequence_parts["face"], axis=0).astype(np.float32),
        "mouth": np.stack(sequence_parts["mouth"], axis=0).astype(np.float32),
        "valid_mask": np.stack(sequence_parts["valid_mask"], axis=0).astype(np.uint8),
    }

    stacked, sample_indices, preview_frames, trim_stats = trim_stacked_sequence(
        stacked=stacked,
        sample_indices=sample_indices,
        preview_frames=preview_frames,
    )

    frames_after_trim = len(sample_indices)
    target_frame_count = resolve_output_frames(frames_after_trim)
    stacked, sample_indices, preview_frames, resample_stats = resample_stacked_sequence(
        stacked=stacked,
        sample_indices=sample_indices,
        preview_frames=preview_frames,
        output_frames=target_frame_count,
    )
    target_frame_count = len(sample_indices)

    (
        stacked,
        preview_pose,
        preview_hands,
        preview_mouth,
        hand_side_stats,
    ) = preprocess_stacked_landmarks(stacked)

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)

    preview_path = None
    if ENABLE_PREVIEW:
        preview_path = save_keypoint_preview(
            frames=preview_frames,
            preview_pose=preview_pose,
            preview_hands=preview_hands,
            preview_mouth=preview_mouth,
            target_label=target_label,
            video_path=video_path,
        )

    # Không lưu combined, không lưu atomic features, không lưu manifest.
    # Khi train, dùng np.concatenate([pose, hands, face], axis=1) nếu cần full tensor.
    np.savez_compressed(
        output_npz_path,
        label=np.array(target_label),
        video_name=np.array(video_path.name),
        target_frames=np.array(target_frame_count, dtype=np.int32),
        source_fps=np.array(source_fps, dtype=np.float32),
        source_total_frames=np.array(total_frames, dtype=np.int32),
        sample_indices=np.array(sample_indices, dtype=np.int32),
        train_feature_dim=np.array(TRAIN_FEATURE_DIM, dtype=np.int32),
        pose=stacked["pose"],
        hands=stacked["hands"],
        face=stacked["face"],
        mouth=stacked["mouth"],
        valid_mask=stacked["valid_mask"],
    )

    next_timestamp_offset_ms = timestamp_offset_ms + target_frame_count * timestamp_step_ms + 1000

    valid_ratio = stacked["valid_mask"].mean(axis=0)

    result = {
        "source_fps": float(source_fps),
        "source_total_frames": int(total_frames),
        "target_frames": target_frame_count,
        "train_feature_dim": TRAIN_FEATURE_DIM,
        "pose_shape": str(stacked["pose"].shape),
        "hands_shape": str(stacked["hands"].shape),
        "face_shape": str(stacked["face"].shape),
        "mouth_shape": str(stacked["mouth"].shape),
        "preview_path": str(preview_path) if preview_path is not None else None,
        "trimmed_start_frames": int(trim_stats.get("trimmed_start", 0.0)),
        "trimmed_end_frames": int(trim_stats.get("trimmed_end", 0.0)),
        "frames_before_resample": int(resample_stats.get("frames_before_resample", target_frame_count)),
        "frames_after_resample": int(resample_stats.get("frames_after_resample", target_frame_count)),
        "hand_side_stabilized_frames": int(hand_side_stats.get("hand_side_stabilized_frames", 0.0)),
        "left_source_hand_frames": int(hand_side_stats.get("left_source_frames", 0.0)),
        "right_source_hand_frames": int(hand_side_stats.get("right_source_frames", 0.0)),
        "max_hand_motion": float(trim_stats.get("max_hand_motion", 0.0)),
        "max_pose_motion": float(trim_stats.get("max_pose_motion", 0.0)),
        "valid_pose_ratio": float(valid_ratio[0]),
        "valid_left_hand_ratio": float(valid_ratio[1]),
        "valid_right_hand_ratio": float(valid_ratio[2]),
        "valid_face_ratio": float(valid_ratio[3]),
    }

    return result, next_timestamp_offset_ms

# ============================================================
# LABEL PROCESSING
# ============================================================

def process_label(
    target_label: str,
    interim_label_dir: Path,
    output_dir: Path,
    hand_landmarker,
    pose_landmarker,
    face_landmarker,
    timestamp_offset_ms: int,
    progress: Optional[ProgressTracker] = None,
):
    print(f"\n{'=' * 80}")
    print(f"EXTRACT OPTIMIZED LANDMARKS FOR LABEL: {target_label}")
    print(f"{'=' * 80}")

    if not interim_label_dir.exists():
        print(f"[SKIP] Folder không tồn tại: {interim_label_dir}")
        return None, timestamp_offset_ms

    videos = list_label_videos(interim_label_dir)
    print(f"[1] Input folder: {interim_label_dir}")
    print(f"[2] Tìm thấy {len(videos)} video cho label '{target_label}'")

    if len(videos) == 0:
        print("[SKIP] Không có video nào trong folder này.")
        return None, timestamp_offset_ms

    processed = 0
    skipped = 0
    errors = 0

    print("\n[3] Bắt đầu xử lý video...\n")

    for idx, video_path in enumerate(videos, start=1):
        video_stem = video_path.stem
        output_npz_path = output_dir / f"{video_stem}.npz"

        print(f"[{idx}/{len(videos)}] {video_path.name}")

        if SKIP_EXISTING and output_npz_path.exists():
            print(f"   [SKIP] Đã tồn tại: {output_npz_path}")
            skipped += 1
            if progress is not None:
                progress.advance("SKIP", target_label, video_path.name)
            continue

        result, timestamp_offset_ms = process_one_video(
            video_path=video_path,
            output_npz_path=output_npz_path,
            hand_landmarker=hand_landmarker,
            pose_landmarker=pose_landmarker,
            face_landmarker=face_landmarker,
            timestamp_offset_ms=timestamp_offset_ms,
            target_label=target_label,
        )

        if result is None:
            errors += 1
            if progress is not None:
                progress.advance("ERROR", target_label, video_path.name)
            continue

        processed += 1

        print(f"   [OK] Saved: {output_npz_path}")
        print(f"   Pose shape:        {result['pose_shape']}")
        print(f"   Hands shape:       {result['hands_shape']}")
        print(f"   Face shape:        {result['face_shape']}")
        print(f"   Mouth shape:       {result['mouth_shape']}")
        print(f"   Train feature dim: {result['train_feature_dim']}")
        print(
            "   Trim action:       "
            f"start={result['trimmed_start_frames']}, "
            f"end={result['trimmed_end_frames']}, "
            f"hand_motion={result['max_hand_motion']:.4f}, "
            f"pose_motion={result['max_pose_motion']:.4f}"
        )
        print(
            "   Resample:          "
            f"{result['frames_before_resample']} -> "
            f"{result['frames_after_resample']}"
        )
        print(
            "   Hand side fix:     "
            f"merged={result['hand_side_stabilized_frames']}, "
            f"left_src={result['left_source_hand_frames']}, "
            f"right_src={result['right_source_hand_frames']}"
        )
        if result["preview_path"]:
            print(f"   Preview:           {result['preview_path']}")
        print(
            "   Valid ratio:       "
            f"pose={result['valid_pose_ratio']:.2f}, "
            f"left={result['valid_left_hand_ratio']:.2f}, "
            f"right={result['valid_right_hand_ratio']:.2f}, "
            f"face={result['valid_face_ratio']:.2f}"
        )
        if progress is not None:
            progress.advance("OK", target_label, video_path.name)

    print("\n" + "=" * 80)
    print(f"SUMMARY FOR LABEL: {target_label}")
    print("=" * 80)
    print(f"Label:             {target_label}")
    print(f"Input folder:      {interim_label_dir}")
    print(f"Processed:         {processed}")
    print(f"Skipped:           {skipped}")
    print(f"Errors:            {errors}")
    print(f"Target frames:     {TARGET_FRAMES}")
    print(f"Storage dtype:     {dtype_name(STORAGE_DTYPE)}")
    print(f"Train feature dim: {TRAIN_FEATURE_DIM}")
    print(f"Output dir:        {output_dir}")

    return {
        "label": target_label,
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
    }, timestamp_offset_ms

# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ASL MediaPipe landmarks with NumPy keypoint preprocessing."
    )

    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument(
        "--labels",
        nargs="+",
        help="Run only these label folders. Use quotes for labels with spaces.",
    )
    label_group.add_argument(
        "--full",
        action="store_true",
        help="Run all labels and all videos.",
    )

    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument(
        "--target-frames",
        type=int,
        help="Use a fixed number of uniformly sampled frames per video.",
    )
    frame_group.add_argument(
        "--all-frames",
        action="store_true",
        help="Use all source frames, with optional min/max frame bounds.",
    )

    existing_group = parser.add_mutually_exclusive_group()
    existing_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip .npz files that already exist.",
    )
    existing_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite .npz files that already exist.",
    )

    trim_group = parser.add_mutually_exclusive_group()
    trim_group.add_argument(
        "--trim-action",
        action="store_true",
        help="Trim inactive leading/trailing frames before hand interpolation.",
    )
    trim_group.add_argument(
        "--no-trim-action",
        action="store_true",
        help="Disable inactive leading/trailing frame trimming.",
    )

    hand_side_group = parser.add_mutually_exclusive_group()
    hand_side_group.add_argument(
        "--stabilize-hand-side",
        action="store_true",
        help="Merge small left/right handedness flickers before interpolation.",
    )
    hand_side_group.add_argument(
        "--no-stabilize-hand-side",
        action="store_true",
        help="Disable left/right handedness flicker stabilization.",
    )

    parser.add_argument(
        "--max-labels",
        type=int,
        help="Limit how many label folders to process after label/worker selection.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        help="Minimum output frames when using --all-frames / TARGET_FRAMES=None.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum output frames when using --all-frames / TARGET_FRAMES=None.",
    )
    parser.add_argument(
        "--trim-margin",
        type=int,
        help="Frames kept before/after detected action after trimming.",
    )
    parser.add_argument(
        "--trim-min-frames",
        type=int,
        help="Minimum frames retained after action trimming.",
    )
    parser.add_argument(
        "--hand-motion-threshold",
        type=float,
        help="Normalized hand-motion threshold used to detect action boundaries.",
    )
    parser.add_argument(
        "--pose-motion-threshold",
        type=float,
        help="Normalized upper-pose-motion threshold used as action-boundary fallback.",
    )
    parser.add_argument(
        "--hand-side-minority-ratio",
        type=float,
        help="Maximum minority/dominant hand-frame ratio to merge handedness flicker.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write preview videos with keypoints overlaid on original frames.",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        help="FPS for preview videos.",
    )
    parser.add_argument(
        "--preview-max-frames",
        type=int,
        help="Maximum frames written per preview video.",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        help="Worker id. 0 means full run; 1..10 each process exactly 200 sorted label folders.",
    )

    return parser.parse_args()

def apply_cli_args(args: argparse.Namespace) -> None:
    global TARGET_FRAMES
    global MIN_TARGET_FRAMES
    global MAX_TARGET_FRAMES
    global SAMPLE_LABELS
    global MAX_LABELS
    global CLI_LABELS_SELECTED
    global SKIP_EXISTING
    global ENABLE_PREVIEW
    global ENABLE_ACTION_TRIM
    global ACTION_TRIM_MARGIN_FRAMES
    global ACTION_TRIM_MIN_KEEP_FRAMES
    global ACTION_TRIM_HAND_MOTION_THRESHOLD
    global ACTION_TRIM_POSE_MOTION_THRESHOLD
    global ENABLE_HAND_SIDE_STABILIZATION
    global HAND_SIDE_MINORITY_RATIO_THRESHOLD
    global PREVIEW_FPS
    global PREVIEW_MAX_FRAMES
    global WORKER_ID

    if args.full:
        SAMPLE_LABELS = None
        CLI_LABELS_SELECTED = False
    elif args.labels is not None:
        SAMPLE_LABELS = args.labels
        CLI_LABELS_SELECTED = True
    elif args.worker_id is not None:
        SAMPLE_LABELS = None
        CLI_LABELS_SELECTED = False

    if args.max_labels is not None:
        MAX_LABELS = args.max_labels

    if args.target_frames is not None:
        TARGET_FRAMES = args.target_frames
    elif args.all_frames:
        TARGET_FRAMES = None

    if args.min_frames is not None:
        MIN_TARGET_FRAMES = args.min_frames

    if args.max_frames is not None:
        MAX_TARGET_FRAMES = args.max_frames

    if args.skip_existing:
        SKIP_EXISTING = True
    elif args.overwrite:
        SKIP_EXISTING = False

    if args.trim_action:
        ENABLE_ACTION_TRIM = True
    elif args.no_trim_action:
        ENABLE_ACTION_TRIM = False

    if args.trim_margin is not None:
        ACTION_TRIM_MARGIN_FRAMES = args.trim_margin

    if args.trim_min_frames is not None:
        ACTION_TRIM_MIN_KEEP_FRAMES = args.trim_min_frames

    if args.hand_motion_threshold is not None:
        ACTION_TRIM_HAND_MOTION_THRESHOLD = args.hand_motion_threshold

    if args.pose_motion_threshold is not None:
        ACTION_TRIM_POSE_MOTION_THRESHOLD = args.pose_motion_threshold

    if args.stabilize_hand_side:
        ENABLE_HAND_SIDE_STABILIZATION = True
    elif args.no_stabilize_hand_side:
        ENABLE_HAND_SIDE_STABILIZATION = False

    if args.hand_side_minority_ratio is not None:
        HAND_SIDE_MINORITY_RATIO_THRESHOLD = args.hand_side_minority_ratio

    if args.preview:
        ENABLE_PREVIEW = True

    if args.preview_fps is not None:
        PREVIEW_FPS = args.preview_fps

    if args.preview_max_frames is not None:
        PREVIEW_MAX_FRAMES = args.preview_max_frames

    if args.worker_id is not None:
        WORKER_ID = args.worker_id

def main():
    apply_cli_args(parse_args())

    print("=" * 80)
    print("EXTRACT-ONLY OPTIMIZED HAND + POSE + FACE-BLENDSHAPE LANDMARKS")
    print("=" * 80)

    if WORKER_ID == 0:
        start_idx = 0
        end_idx = None
        run_all_labels = True
        print("\n[WORKER] WORKER_ID = 0: Chạy toàn bộ labels, không chia worker")
    else:
        if WORKER_ID < 1 or WORKER_ID > 10:
            raise ValueError("WORKER_ID must be 0 or 1-10. 0 runs full; 1-10 each run exactly 200 labels.")

        run_all_labels = False
        start_idx = (WORKER_ID - 1) * LABELS_PER_WORKER

        end_idx = start_idx + LABELS_PER_WORKER
        print(f"\n[WORKER] Worker {WORKER_ID}: labels index {start_idx} to {end_idx - 1}")
    print("=" * 80)

    if not INTERIM_BASE_DIR.exists():
        print(f"[ERROR] Không tìm thấy folder input: {INTERIM_BASE_DIR}")
        print("Bạn cần có cấu trúc: data/interim/<label>/*.mp4")
        return

    download_models()
    save_feature_schema(OUTPUT_BASE_DIR)

    label_dirs = sorted([d for d in INTERIM_BASE_DIR.iterdir() if d.is_dir()])

    if not label_dirs:
        print(f"[ERROR] Không tìm thấy folder label nào trong: {INTERIM_BASE_DIR}")
        return

    print(f"\n[0] Base interim folder: {INTERIM_BASE_DIR}")
    print(f"[1] Tổng cộng tìm thấy {len(label_dirs)} label(s)")

    sample_mode = bool(SAMPLE_LABELS) and (run_all_labels or CLI_LABELS_SELECTED)
    if sample_mode:
        label_dir_by_name = {label_dir.name: label_dir for label_dir in label_dirs}
        missing_labels = [label for label in SAMPLE_LABELS if label not in label_dir_by_name]
        label_dirs = [label_dir_by_name[label] for label in SAMPLE_LABELS if label in label_dir_by_name]

        print(f"[SAMPLE] Labels: {SAMPLE_LABELS}")
        if missing_labels:
            print(f"[SAMPLE WARNING] Missing labels: {missing_labels}")

        if not label_dirs:
            print("[ERROR] SAMPLE_LABELS did not match any input label.")
            return

    if not sample_mode and not run_all_labels:
        if end_idx is None:
            label_dirs = label_dirs[start_idx:]
        else:
            label_dirs = label_dirs[start_idx:end_idx]

    if MAX_LABELS is not None:
        label_dirs = label_dirs[:MAX_LABELS]

    if sample_mode:
        print(f"[2] Sample mode: processing {len(label_dirs)} label(s):")
    elif run_all_labels:
        print(f"[2] Chế độ full: sẽ xử lý toàn bộ {len(label_dirs)} label(s):")
    else:
        print(f"[2] Worker {WORKER_ID} sẽ xử lý {len(label_dirs)} label(s):")
    for i, label_dir in enumerate(label_dirs[:5], start=1):
        print(f"    - {label_dir.name}")
    if len(label_dirs) > 5:
        print(f"    ... và {len(label_dirs) - 5} labels khác")

    total_jobs = sum(len(list_label_videos(label_dir)) for label_dir in label_dirs)
    progress = ProgressTracker(total_jobs)
    print(f"[PROGRESS] Tổng số công việc: {total_jobs} video")

    print("\n[3] Khởi tạo MediaPipe landmarkers...")

    hand_landmarker = create_hand_landmarker() if USE_HAND else None
    pose_landmarker = create_pose_landmarker() if USE_POSE else None
    face_landmarker = create_face_landmarker() if USE_FACE else None

    summary_results = []
    timestamp_offset_ms = 0

    try:
        print("\n[4] Bắt đầu xử lý labels...\n")

        for label_dir in label_dirs:
            target_label = label_dir.name
            output_label_dir = OUTPUT_BASE_DIR / target_label

            result, timestamp_offset_ms = process_label(
                target_label=target_label,
                interim_label_dir=label_dir,
                output_dir=output_label_dir,
                hand_landmarker=hand_landmarker,
                pose_landmarker=pose_landmarker,
                face_landmarker=face_landmarker,
                timestamp_offset_ms=timestamp_offset_ms,
                progress=progress,
            )

            if result:
                summary_results.append(result)

    finally:
        if hand_landmarker is not None:
            hand_landmarker.close()
        if pose_landmarker is not None:
            pose_landmarker.close()
        if face_landmarker is not None:
            face_landmarker.close()

    print("\n" + "=" * 80)
    print(f"SUMMARY FOR WORKER {WORKER_ID}")
    print("=" * 80)

    total_processed = sum(r["processed"] for r in summary_results)
    total_skipped = sum(r["skipped"] for r in summary_results)
    total_errors = sum(r["errors"] for r in summary_results)

    for result in summary_results:
        print(
            f"Label '{result['label']}': "
            f"{result['processed']} processed, "
            f"{result['skipped']} skipped, "
            f"{result['errors']} errors"
        )

    print(f"\nWorker {WORKER_ID} Summary:")
    print(f"Total processed:   {total_processed}")
    print(f"Total skipped:     {total_skipped}")
    print(f"Total errors:      {total_errors}")
    print(f"Target frames:     {TARGET_FRAMES}")
    print(f"Storage dtype:     {dtype_name(STORAGE_DTYPE)}")
    print(f"Pose dim:          {POSE_FEATURE_DIM}")
    print(f"Hands dim:         {HANDS_FEATURE_DIM}")
    print(f"Face dim:          {FACE_FEATURE_DIM}")
    print(f"Mouth dim:         {MOUTH_FEATURE_DIM}")
    print(f"Train feature dim: {TRAIN_FEATURE_DIM}")
    print(f"Output base dir:   {OUTPUT_BASE_DIR}")
    print(f"Schema:            {OUTPUT_BASE_DIR / 'feature_schema.json'}")

    print("\nDone.")

if __name__ == "__main__":
    main()
