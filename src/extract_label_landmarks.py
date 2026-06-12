import os
import sys
import csv
import json
import re
import urllib.request
from pathlib import Path

# Giảm log TensorFlow/MediaPipe
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# UTF-8 CONSOLE
# ============================================================

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
INTERIM_BASE_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_BASE_DIR = PROJECT_ROOT / "data" / "processed" / "landmarks"
PREVIEW_BASE_DIR = PROJECT_ROOT / "data" / "processed" / "landmark_preview"


# ============================================================
# CONFIG
# ============================================================

# Mỗi video sẽ được chuẩn hóa thành đúng 60 frame/time steps.
TARGET_FRAMES = 60

# Nếu muốn test ít video trước thì để số, ví dụ 5.
# Chạy toàn bộ thì để None.
MAX_VIDEOS = None

# Nếu file .npz đã tồn tại thì bỏ qua.
SKIP_EXISTING = True

# Lưu video nền đen có vẽ landmark để kiểm tra trực quan.
SAVE_BLACK_PREVIEW = True

# Bật toàn bộ để lấy chi tiết nhất có thể.
USE_HAND = True
USE_POSE = True
USE_FACE = True

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


# ============================================================
# MULTI-WORKER CONFIG (Chia việc cho nhiều người)
# ============================================================
# 
# Để chia việc cho 10 người, mỗi người chạy script với WORKER_ID khác nhau.
# - Người 1: WORKER_ID = 1 (xử lý label 0-199)
# - Người 2: WORKER_ID = 2 (xử lý label 200-399)
# - ...
# - Người 9: WORKER_ID = 9 (xử lý label 1600-1799)
# - Người 10: WORKER_ID = 10 (xử lý label 1800 trở đi)
#
# Tất cả đều lưu vào data/processed/landmarks/{label}/*.npz
# Sau đó bạn copy thủ công các folder label vào một chỗ chung.

WORKER_ID = 1  # Đặt từ 1 đến 10 tùy vào mỗi người
LABELS_PER_WORKER = 200  # Mỗi người 200 labels (trừ người cuối)


# ============================================================
# MODEL URLS
# ============================================================
# Pose dùng heavy để ưu tiên chất lượng landmark.
# Nếu máy yếu, có thể đổi pose_landmarker_heavy thành:
# - pose_landmarker_full
# - pose_landmarker_lite

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
# Lấy chi tiết:
# - Pose normalized landmarks: 33 điểm x [x, y, z, visibility, presence]
# - Pose world landmarks:      33 điểm x [x, y, z, visibility, presence]
# - Left hand normalized:      21 điểm x [x, y, z]
# - Right hand normalized:     21 điểm x [x, y, z]
# - Left hand world:           21 điểm x [x, y, z]
# - Right hand world:          21 điểm x [x, y, z]
# - Face landmarks:            478 điểm x [x, y, z]
# - Face blendshapes:          52 giá trị biểu cảm
# - Face matrix:               16 giá trị ma trận biến đổi khuôn mặt

POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
FACE_LANDMARK_COUNT = 478
FACE_BLENDSHAPE_COUNT = 52
FACE_MATRIX_DIM = 16

POSE_NORM_DIM = POSE_LANDMARK_COUNT * 5
POSE_WORLD_DIM = POSE_LANDMARK_COUNT * 5

LEFT_HAND_NORM_DIM = HAND_LANDMARK_COUNT * 3
RIGHT_HAND_NORM_DIM = HAND_LANDMARK_COUNT * 3

LEFT_HAND_WORLD_DIM = HAND_LANDMARK_COUNT * 3
RIGHT_HAND_WORLD_DIM = HAND_LANDMARK_COUNT * 3

HANDS_COMBINED_DIM = (
    LEFT_HAND_NORM_DIM
    + RIGHT_HAND_NORM_DIM
    + LEFT_HAND_WORLD_DIM
    + RIGHT_HAND_WORLD_DIM
)

FACE_LANDMARK_DIM = FACE_LANDMARK_COUNT * 3
FACE_BLENDSHAPE_DIM = FACE_BLENDSHAPE_COUNT
FACE_TRANSFORM_DIM = FACE_MATRIX_DIM

FACE_COMBINED_DIM = FACE_LANDMARK_DIM + FACE_BLENDSHAPE_DIM + FACE_TRANSFORM_DIM

POSE_COMBINED_DIM = POSE_NORM_DIM + POSE_WORLD_DIM

TOTAL_FEATURE_DIM = (
    POSE_NORM_DIM
    + POSE_WORLD_DIM
    + LEFT_HAND_NORM_DIM
    + RIGHT_HAND_NORM_DIM
    + LEFT_HAND_WORLD_DIM
    + RIGHT_HAND_WORLD_DIM
    + FACE_LANDMARK_DIM
    + FACE_BLENDSHAPE_DIM
    + FACE_TRANSFORM_DIM
)


# ============================================================
# CONNECTIONS FOR PREVIEW
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (27, 31), (28, 32)
]


# ============================================================
# UTILS
# ============================================================

def safe_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\-]+", "_", name)
    return name


def download_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, item in MODEL_URLS.items():
        model_path = item["path"]
        model_url = item["url"]

        if model_path.exists():
            print(f"[MODEL] {model_name} đã có: {model_path}")
            continue

        print(f"[MODEL] Đang tải {model_name}...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"[MODEL] Đã tải xong {model_name}: {model_path}")


def list_label_videos(label_dir: Path):
    videos = []

    for ext in VIDEO_EXTENSIONS:
        videos.extend(label_dir.glob(f"*{ext}"))

    videos = sorted(videos)

    if MAX_VIDEOS is not None:
        videos = videos[:MAX_VIDEOS]

    return videos


def get_sample_indices(total_frames: int, target_frames: int):
    """
    Chuẩn hóa mỗi video thành đúng target_frames.
    Nếu video ít frame hơn 60, frame sẽ được lặp.
    Nếu video nhiều frame hơn 60, frame sẽ được sample đều theo thời gian.
    """
    if total_frames <= 0:
        return []

    indices = np.linspace(0, total_frames - 1, target_frames)
    indices = np.round(indices).astype(int)
    indices = np.clip(indices, 0, total_frames - 1)

    return indices.tolist()


def save_feature_schema(target_label, output_dir):
    """
    Lưu schema để sau này biết từng key trong .npz có ý nghĩa gì.
    """
    schema = {
        "target_label": target_label,
        "target_frames": TARGET_FRAMES,
        "total_feature_dim": TOTAL_FEATURE_DIM,
        "file_format": ".npz",
        "keys": {
            "pose_normalized": {
                "shape": [TARGET_FRAMES, POSE_NORM_DIM],
                "landmarks": POSE_LANDMARK_COUNT,
                "values_per_landmark": ["x", "y", "z", "visibility", "presence"]
            },
            "pose_world": {
                "shape": [TARGET_FRAMES, POSE_WORLD_DIM],
                "landmarks": POSE_LANDMARK_COUNT,
                "values_per_landmark": ["x", "y", "z", "visibility", "presence"]
            },
            "pose_combined": {
                "shape": [TARGET_FRAMES, POSE_COMBINED_DIM],
                "contains": ["pose_normalized", "pose_world"]
            },
            "left_hand_normalized": {
                "shape": [TARGET_FRAMES, LEFT_HAND_NORM_DIM],
                "landmarks": HAND_LANDMARK_COUNT,
                "values_per_landmark": ["x", "y", "z"]
            },
            "right_hand_normalized": {
                "shape": [TARGET_FRAMES, RIGHT_HAND_NORM_DIM],
                "landmarks": HAND_LANDMARK_COUNT,
                "values_per_landmark": ["x", "y", "z"]
            },
            "left_hand_world": {
                "shape": [TARGET_FRAMES, LEFT_HAND_WORLD_DIM],
                "landmarks": HAND_LANDMARK_COUNT,
                "values_per_landmark": ["x", "y", "z"]
            },
            "right_hand_world": {
                "shape": [TARGET_FRAMES, RIGHT_HAND_WORLD_DIM],
                "landmarks": HAND_LANDMARK_COUNT,
                "values_per_landmark": ["x", "y", "z"]
            },
            "hands_combined": {
                "shape": [TARGET_FRAMES, HANDS_COMBINED_DIM],
                "contains": [
                    "left_hand_normalized",
                    "right_hand_normalized",
                    "left_hand_world",
                    "right_hand_world"
                ]
            },
            "face_landmarks": {
                "shape": [TARGET_FRAMES, FACE_LANDMARK_DIM],
                "landmarks": FACE_LANDMARK_COUNT,
                "values_per_landmark": ["x", "y", "z"]
            },
            "face_blendshapes": {
                "shape": [TARGET_FRAMES, FACE_BLENDSHAPE_DIM],
                "values": "expression scores"
            },
            "face_transform_matrix": {
                "shape": [TARGET_FRAMES, FACE_TRANSFORM_DIM],
                "values": "flattened 4x4 matrix"
            },
            "face_combined": {
                "shape": [TARGET_FRAMES, FACE_COMBINED_DIM],
                "contains": [
                    "face_landmarks",
                    "face_blendshapes",
                    "face_transform_matrix"
                ]
            },
            "combined": {
                "shape": [TARGET_FRAMES, TOTAL_FEATURE_DIM],
                "contains": [
                    "pose_normalized",
                    "pose_world",
                    "left_hand_normalized",
                    "right_hand_normalized",
                    "left_hand_world",
                    "right_hand_world",
                    "face_landmarks",
                    "face_blendshapes",
                    "face_transform_matrix"
                ]
            }
        }
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    schema_path = output_dir / "feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print(f"[SCHEMA] Saved: {schema_path}")


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

        # Bật để lấy thêm biểu cảm mặt và ma trận mặt
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    return vision.FaceLandmarker.create_from_options(options)


# ============================================================
# VECTOR CONVERSION
# ============================================================

def pose_to_vector(pose_landmarks):
    """
    Pose: 33 x [x, y, z, visibility, presence]
    """
    vector = np.zeros(POSE_LANDMARK_COUNT * 5, dtype=np.float32)

    if not pose_landmarks:
        return vector

    count = min(len(pose_landmarks), POSE_LANDMARK_COUNT)

    for i in range(count):
        lm = pose_landmarks[i]

        visibility = getattr(lm, "visibility", 0.0)
        presence = getattr(lm, "presence", 0.0)

        vector[i * 5 + 0] = lm.x
        vector[i * 5 + 1] = lm.y
        vector[i * 5 + 2] = lm.z
        vector[i * 5 + 3] = 0.0 if visibility is None else visibility
        vector[i * 5 + 4] = 0.0 if presence is None else presence

    return vector


def hand_to_vector(hand_landmarks):
    """
    Hand: 21 x [x, y, z]
    """
    vector = np.zeros(HAND_LANDMARK_COUNT * 3, dtype=np.float32)

    if not hand_landmarks:
        return vector

    count = min(len(hand_landmarks), HAND_LANDMARK_COUNT)

    for i in range(count):
        lm = hand_landmarks[i]

        vector[i * 3 + 0] = lm.x
        vector[i * 3 + 1] = lm.y
        vector[i * 3 + 2] = lm.z

    return vector


def face_to_vector(face_landmarks):
    """
    Face: 478 x [x, y, z]
    """
    vector = np.zeros(FACE_LANDMARK_DIM, dtype=np.float32)

    if not face_landmarks:
        return vector

    count = min(len(face_landmarks), FACE_LANDMARK_COUNT)

    for i in range(count):
        lm = face_landmarks[i]

        vector[i * 3 + 0] = lm.x
        vector[i * 3 + 1] = lm.y
        vector[i * 3 + 2] = lm.z

    return vector


def face_blendshapes_to_vector(face_blendshapes):
    """
    Face blendshape: thường 52 điểm biểu cảm.
    """
    vector = np.zeros(FACE_BLENDSHAPE_DIM, dtype=np.float32)

    if not face_blendshapes:
        return vector

    categories = face_blendshapes[0]
    count = min(len(categories), FACE_BLENDSHAPE_COUNT)

    for i in range(count):
        vector[i] = categories[i].score

    return vector


def face_matrix_to_vector(facial_transformation_matrixes):
    """
    Face transform matrix: 4x4 = 16 giá trị.
    """
    vector = np.zeros(FACE_TRANSFORM_DIM, dtype=np.float32)

    if facial_transformation_matrixes is None or len(facial_transformation_matrixes) == 0:
        return vector

    matrix = np.array(facial_transformation_matrixes[0], dtype=np.float32).reshape(-1)
    count = min(len(matrix), FACE_TRANSFORM_DIM)

    vector[:count] = matrix[:count]

    return vector


def get_handedness_label(handedness_item):
    """
    MediaPipe thường trả category_name là 'Left' hoặc 'Right'.
    """
    try:
        if handedness_item and len(handedness_item) > 0:
            return handedness_item[0].category_name
    except Exception:
        pass

    return None


def create_empty_frame_parts():
    """
    Dùng khi frame bị lỗi hoặc không đọc được.
    Vẫn giữ đúng shape để sequence không bị lệch.
    """
    pose_norm_vec = np.zeros(POSE_NORM_DIM, dtype=np.float32)
    pose_world_vec = np.zeros(POSE_WORLD_DIM, dtype=np.float32)
    pose_combined = np.zeros(POSE_COMBINED_DIM, dtype=np.float32)

    left_hand_norm_vec = np.zeros(LEFT_HAND_NORM_DIM, dtype=np.float32)
    right_hand_norm_vec = np.zeros(RIGHT_HAND_NORM_DIM, dtype=np.float32)
    left_hand_world_vec = np.zeros(LEFT_HAND_WORLD_DIM, dtype=np.float32)
    right_hand_world_vec = np.zeros(RIGHT_HAND_WORLD_DIM, dtype=np.float32)
    hands_combined = np.zeros(HANDS_COMBINED_DIM, dtype=np.float32)

    face_landmark_vec = np.zeros(FACE_LANDMARK_DIM, dtype=np.float32)
    face_blendshape_vec = np.zeros(FACE_BLENDSHAPE_DIM, dtype=np.float32)
    face_matrix_vec = np.zeros(FACE_TRANSFORM_DIM, dtype=np.float32)
    face_combined = np.zeros(FACE_COMBINED_DIM, dtype=np.float32)

    combined = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)

    return {
        "pose_normalized": pose_norm_vec,
        "pose_world": pose_world_vec,
        "pose_combined": pose_combined,

        "left_hand_normalized": left_hand_norm_vec,
        "right_hand_normalized": right_hand_norm_vec,
        "left_hand_world": left_hand_world_vec,
        "right_hand_world": right_hand_world_vec,
        "hands_combined": hands_combined,

        "face_landmarks": face_landmark_vec,
        "face_blendshapes": face_blendshape_vec,
        "face_transform_matrix": face_matrix_vec,
        "face_combined": face_combined,

        "combined": combined,
    }


def extract_frame_parts(hand_result, pose_result, face_result):
    """
    Trả về dictionary chứa từng nhóm landmark riêng,
    đồng thời có combined vector để train nhanh.

    Một frame có các nhóm:
    - pose_normalized
    - pose_world
    - pose_combined
    - left_hand_normalized
    - right_hand_normalized
    - left_hand_world
    - right_hand_world
    - hands_combined
    - face_landmarks
    - face_blendshapes
    - face_transform_matrix
    - face_combined
    - combined
    """

    pose_norm_vec = np.zeros(POSE_NORM_DIM, dtype=np.float32)
    pose_world_vec = np.zeros(POSE_WORLD_DIM, dtype=np.float32)

    left_hand_norm_vec = np.zeros(LEFT_HAND_NORM_DIM, dtype=np.float32)
    right_hand_norm_vec = np.zeros(RIGHT_HAND_NORM_DIM, dtype=np.float32)

    left_hand_world_vec = np.zeros(LEFT_HAND_WORLD_DIM, dtype=np.float32)
    right_hand_world_vec = np.zeros(RIGHT_HAND_WORLD_DIM, dtype=np.float32)

    face_landmark_vec = np.zeros(FACE_LANDMARK_DIM, dtype=np.float32)
    face_blendshape_vec = np.zeros(FACE_BLENDSHAPE_DIM, dtype=np.float32)
    face_matrix_vec = np.zeros(FACE_TRANSFORM_DIM, dtype=np.float32)

    # ----------------------------
    # POSE
    # ----------------------------

    if USE_POSE:
        if pose_result.pose_landmarks:
            pose_norm_vec = pose_to_vector(pose_result.pose_landmarks[0])

        if pose_result.pose_world_landmarks:
            pose_world_vec = pose_to_vector(pose_result.pose_world_landmarks[0])

    # ----------------------------
    # HANDS
    # ----------------------------

    if USE_HAND and hand_result.hand_landmarks:
        for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
            hand_norm_vec = hand_to_vector(hand_landmarks)

            hand_world_vec = np.zeros(HAND_LANDMARK_COUNT * 3, dtype=np.float32)
            if hand_result.hand_world_landmarks and i < len(hand_result.hand_world_landmarks):
                hand_world_vec = hand_to_vector(hand_result.hand_world_landmarks[i])

            label = None
            if hand_result.handedness and i < len(hand_result.handedness):
                label = get_handedness_label(hand_result.handedness[i])

            if label == "Left":
                left_hand_norm_vec = hand_norm_vec
                left_hand_world_vec = hand_world_vec

            elif label == "Right":
                right_hand_norm_vec = hand_norm_vec
                right_hand_world_vec = hand_world_vec

            else:
                if not np.any(left_hand_norm_vec):
                    left_hand_norm_vec = hand_norm_vec
                    left_hand_world_vec = hand_world_vec
                else:
                    right_hand_norm_vec = hand_norm_vec
                    right_hand_world_vec = hand_world_vec

    # ----------------------------
    # FACE
    # ----------------------------

    if USE_FACE:
        if face_result.face_landmarks:
            face_landmark_vec = face_to_vector(face_result.face_landmarks[0])

        if face_result.face_blendshapes:
            face_blendshape_vec = face_blendshapes_to_vector(face_result.face_blendshapes)

        if face_result.facial_transformation_matrixes:
            face_matrix_vec = face_matrix_to_vector(face_result.facial_transformation_matrixes)

    # ----------------------------
    # LOGICAL GROUPS
    # ----------------------------

    pose_combined = np.concatenate([
        pose_norm_vec,
        pose_world_vec
    ]).astype(np.float32)

    hands_combined = np.concatenate([
        left_hand_norm_vec,
        right_hand_norm_vec,
        left_hand_world_vec,
        right_hand_world_vec
    ]).astype(np.float32)

    face_combined = np.concatenate([
        face_landmark_vec,
        face_blendshape_vec,
        face_matrix_vec
    ]).astype(np.float32)

    combined = np.concatenate([
        pose_norm_vec,
        pose_world_vec,
        left_hand_norm_vec,
        right_hand_norm_vec,
        left_hand_world_vec,
        right_hand_world_vec,
        face_landmark_vec,
        face_blendshape_vec,
        face_matrix_vec
    ]).astype(np.float32)

    return {
        "pose_normalized": pose_norm_vec,
        "pose_world": pose_world_vec,
        "pose_combined": pose_combined,

        "left_hand_normalized": left_hand_norm_vec,
        "right_hand_normalized": right_hand_norm_vec,
        "left_hand_world": left_hand_world_vec,
        "right_hand_world": right_hand_world_vec,
        "hands_combined": hands_combined,

        "face_landmarks": face_landmark_vec,
        "face_blendshapes": face_blendshape_vec,
        "face_transform_matrix": face_matrix_vec,
        "face_combined": face_combined,

        "combined": combined,
    }


# ============================================================
# PREVIEW DRAWING
# ============================================================

def draw_landmarks_on_black(
    canvas,
    landmarks,
    connections=None,
    point_color=(0, 255, 0),
    line_color=(0, 255, 0),
    radius=2
):
    if not landmarks:
        return

    height, width, _ = canvas.shape
    points = []

    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))

    if connections:
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                x1, y1 = points[start_idx]
                x2, y2 = points[end_idx]

                if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                    cv2.line(canvas, (x1, y1), (x2, y2), line_color, 2)

    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), radius, point_color, -1)


def create_black_preview_frame(frame_shape, hand_result, pose_result, face_result):
    height, width, channels = frame_shape
    canvas = np.zeros((height, width, channels), dtype=np.uint8)

    # Pose: đỏ
    if USE_POSE and pose_result.pose_landmarks:
        for pose_landmarks in pose_result.pose_landmarks:
            draw_landmarks_on_black(
                canvas,
                pose_landmarks,
                connections=POSE_CONNECTIONS,
                point_color=(0, 0, 255),
                line_color=(0, 0, 255),
                radius=3
            )

    # Hands: xanh lá
    if USE_HAND and hand_result.hand_landmarks:
        for hand_landmarks in hand_result.hand_landmarks:
            draw_landmarks_on_black(
                canvas,
                hand_landmarks,
                connections=HAND_CONNECTIONS,
                point_color=(0, 255, 0),
                line_color=(0, 255, 0),
                radius=3
            )

    # Face: xanh dương, chỉ vẽ điểm cho nhẹ
    if USE_FACE and face_result.face_landmarks:
        for face_landmarks in face_result.face_landmarks:
            draw_landmarks_on_black(
                canvas,
                face_landmarks,
                connections=None,
                point_color=(255, 0, 0),
                line_color=(255, 0, 0),
                radius=1
            )

    return canvas


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_one_video(
    video_path: Path,
    output_npz_path: Path,
    preview_video_path: Path,
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

    sample_indices = get_sample_indices(total_frames, TARGET_FRAMES)

    if len(sample_indices) == 0:
        print(f"   [ERROR] Video không có frame: {video_path}")
        cap.release()
        return None, timestamp_offset_ms

    sequence_parts = {
        "pose_normalized": [],
        "pose_world": [],
        "pose_combined": [],

        "left_hand_normalized": [],
        "right_hand_normalized": [],
        "left_hand_world": [],
        "right_hand_world": [],
        "hands_combined": [],

        "face_landmarks": [],
        "face_blendshapes": [],
        "face_transform_matrix": [],
        "face_combined": [],

        "combined": [],
    }

    writer = None

    # Timestamp giả lập 60 FPS để MediaPipe VIDEO mode nhận timestamp tăng dần.
    timestamp_step_ms = int(1000 / 60)

    for sample_i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        timestamp_ms = timestamp_offset_ms + sample_i * timestamp_step_ms

        if not ret:
            frame_parts = create_empty_frame_parts()

            for key in sequence_parts:
                sequence_parts[key].append(frame_parts[key])

            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        frame_parts = extract_frame_parts(
            hand_result=hand_result,
            pose_result=pose_result,
            face_result=face_result
        )

        for key in sequence_parts:
            sequence_parts[key].append(frame_parts[key])

        if SAVE_BLACK_PREVIEW:
            preview_frame = create_black_preview_frame(
                frame_shape=frame.shape,
                hand_result=hand_result,
                pose_result=pose_result,
                face_result=face_result
            )

            if writer is None:
                preview_video_path.parent.mkdir(parents=True, exist_ok=True)

                h, w, _ = preview_frame.shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    str(preview_video_path),
                    fourcc,
                    60.0,
                    (w, h)
                )

            writer.write(preview_frame)

    cap.release()

    if writer is not None:
        writer.release()

    # Stack từng nhóm thành array shape: (60, feature_dim_của_nhóm)
    stacked = {}

    for key, values in sequence_parts.items():
        stacked[key] = np.stack(values, axis=0).astype(np.float32)

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_npz_path,

        # Metadata
        label=np.array(target_label),
        video_name=np.array(video_path.name),
        target_frames=np.array(TARGET_FRAMES, dtype=np.int32),
        source_fps=np.array(source_fps, dtype=np.float32),
        source_total_frames=np.array(total_frames, dtype=np.int32),
        total_feature_dim=np.array(TOTAL_FEATURE_DIM, dtype=np.int32),

        # Pose
        pose_normalized=stacked["pose_normalized"],
        pose_world=stacked["pose_world"],
        pose_combined=stacked["pose_combined"],

        # Hands
        left_hand_normalized=stacked["left_hand_normalized"],
        right_hand_normalized=stacked["right_hand_normalized"],
        left_hand_world=stacked["left_hand_world"],
        right_hand_world=stacked["right_hand_world"],
        hands_combined=stacked["hands_combined"],

        # Face
        face_landmarks=stacked["face_landmarks"],
        face_blendshapes=stacked["face_blendshapes"],
        face_transform_matrix=stacked["face_transform_matrix"],
        face_combined=stacked["face_combined"],

        # Main training tensor
        combined=stacked["combined"],
    )

    next_timestamp_offset_ms = timestamp_offset_ms + TARGET_FRAMES * timestamp_step_ms + 1000

    result = {
        "source_fps": float(source_fps),
        "source_total_frames": int(total_frames),
        "target_frames": TARGET_FRAMES,
        "feature_dim": TOTAL_FEATURE_DIM,

        "combined_shape": str(stacked["combined"].shape),
        "pose_shape": str(stacked["pose_combined"].shape),
        "hands_shape": str(stacked["hands_combined"].shape),
        "face_shape": str(stacked["face_combined"].shape),

        "pose_normalized_shape": str(stacked["pose_normalized"].shape),
        "pose_world_shape": str(stacked["pose_world"].shape),
        "face_landmarks_shape": str(stacked["face_landmarks"].shape),
    }

    return result, next_timestamp_offset_ms


# ============================================================
# MAIN
# ============================================================

def process_label(
    target_label: str,
    interim_label_dir: Path,
    output_dir: Path,
    preview_dir: Path,
    hand_landmarker,
    pose_landmarker,
    face_landmarker,
    timestamp_offset_ms: int,
):
    """
    Xử lý tất cả video trong một label.
    """
    print(f"\n{'=' * 80}")
    print(f"EXTRACT HAND + POSE + FACE LANDMARKS FOR LABEL: {target_label}")
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

    save_feature_schema(target_label, output_dir)

    manifest_rows = []
    processed = 0
    skipped = 0
    errors = 0
    try:
        print("\n[3] Bắt đầu xử lý video...\n")

        for idx, video_path in enumerate(videos, start=1):
            video_stem = video_path.stem

            output_npz_path = output_dir / f"{video_stem}.npz"
            preview_video_path = preview_dir / f"{video_stem}_preview.mp4"

            print(f"[{idx}/{len(videos)}] {video_path.name}")

            if SKIP_EXISTING and output_npz_path.exists():
                print(f"   [SKIP] Đã tồn tại: {output_npz_path}")
                skipped += 1
                continue

            result, timestamp_offset_ms = process_one_video(
                video_path=video_path,
                output_npz_path=output_npz_path,
                preview_video_path=preview_video_path,
                hand_landmarker=hand_landmarker,
                pose_landmarker=pose_landmarker,
                face_landmarker=face_landmarker,
                timestamp_offset_ms=timestamp_offset_ms,
                target_label=target_label,
            )

            if result is None:
                errors += 1
                continue

            processed += 1

            manifest_rows.append({
                "video_name": video_path.name,
                "video_stem": video_stem,
                "label": target_label,
                "video_path": str(video_path),
                "npz_path": str(output_npz_path),
                "preview_path": str(preview_video_path) if SAVE_BLACK_PREVIEW else "",

                "target_frames": result["target_frames"],
                "feature_dim": result["feature_dim"],

                "combined_shape": result["combined_shape"],
                "pose_shape": result["pose_shape"],
                "hands_shape": result["hands_shape"],
                "face_shape": result["face_shape"],

                "pose_normalized_shape": result["pose_normalized_shape"],
                "pose_world_shape": result["pose_world_shape"],
                "face_landmarks_shape": result["face_landmarks_shape"],

                "source_fps": result["source_fps"],
                "source_total_frames": result["source_total_frames"],
            })

            print(f"   [OK] Saved: {output_npz_path}")
            print(f"   Combined shape:        {result['combined_shape']}")
            print(f"   Pose combined shape:   {result['pose_shape']}")
            print(f"   Hands combined shape:  {result['hands_shape']}")
            print(f"   Face combined shape:   {result['face_shape']}")

    finally:
        pass

    manifest_path = output_dir / f"manifest_label_{target_label}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "video_name",
            "video_stem",
            "label",
            "video_path",
            "npz_path",
            "preview_path",

            "target_frames",
            "feature_dim",

            "combined_shape",
            "pose_shape",
            "hands_shape",
            "face_shape",

            "pose_normalized_shape",
            "pose_world_shape",
            "face_landmarks_shape",

            "source_fps",
            "source_total_frames",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 80)
    print(f"SUMMARY FOR LABEL: {target_label}")
    print("=" * 80)
    print(f"Label:             {target_label}")
    print(f"Input folder:      {interim_label_dir}")
    print(f"Processed:         {processed}")
    print(f"Skipped:           {skipped}")
    print(f"Errors:            {errors}")
    print(f"Target frames:     {TARGET_FRAMES}")
    print(f"Feature dim:       {TOTAL_FEATURE_DIM}")
    print(f"Output dir:        {output_dir}")
    print(f"Manifest:          {manifest_path}")

    if SAVE_BLACK_PREVIEW:
        print(f"Preview dir:       {preview_dir}")

    return {
        "label": target_label,
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
    }, timestamp_offset_ms


def main():
    print("=" * 80)
    print("EXTRACT HAND + POSE + FACE LANDMARKS FOR ALL LABELS")
    print("=" * 80)
    
    # ============ MULTI-WORKER INFO ============
    start_idx = (WORKER_ID - 1) * LABELS_PER_WORKER
    if WORKER_ID == 10:
        # Người thứ 10 xử lý tất cả labels từ 1800 trở đi
        end_idx = None
        print(f"\n[WORKER] Người {WORKER_ID}: Xử lý labels từ index {start_idx} trở đi (tất cả phần còn lại)")
    else:
        end_idx = start_idx + LABELS_PER_WORKER
        print(f"\n[WORKER] Người {WORKER_ID}: Xử lý labels từ index {start_idx} đến {end_idx - 1} ({LABELS_PER_WORKER} labels)")
    print("=" * 80)

    if not INTERIM_BASE_DIR.exists():
        print(f"[ERROR] Không tìm thấy folder input: {INTERIM_BASE_DIR}")
        print("Bạn cần có cấu trúc ví dụ: data/interim/label1/*.mp4, data/interim/label2/*.mp4, ...")
        return

    download_models()

    # Lấy danh sách tất cả folders (labels)
    label_dirs = sorted([d for d in INTERIM_BASE_DIR.iterdir() if d.is_dir()])

    if not label_dirs:
        print(f"[ERROR] Không tìm thấy folder nào trong: {INTERIM_BASE_DIR}")
        return

    print(f"\n[0] Base interim folder: {INTERIM_BASE_DIR}")
    print(f"[1] Tổng cộng tìm thấy {len(label_dirs)} label(s)")

    # ============ FILTER LABELS BY WORKER ============
    if end_idx is None:
        label_dirs = label_dirs[start_idx:]
    else:
        label_dirs = label_dirs[start_idx:end_idx]
    
    print(f"[2] Người {WORKER_ID} sẽ xử lý {len(label_dirs)} label(s):")
    for i, label_dir in enumerate(label_dirs[:5], start=1):
        print(f"    - {label_dir.name}")
    if len(label_dirs) > 5:
        print(f"    ... và {len(label_dirs) - 5} labels khác")

    print("\n[3] Khởi tạo MediaPipe landmarkers...")
    hand_landmarker = create_hand_landmarker()
    pose_landmarker = create_pose_landmarker()
    face_landmarker = create_face_landmarker()

    summary_results = []
    timestamp_offset_ms = 0

    try:
        print("\n[4] Bắt đầu xử lý labels...\n")

        for label_dir in label_dirs:
            target_label = label_dir.name
            output_label_dir = OUTPUT_BASE_DIR / target_label
            preview_label_dir = PREVIEW_BASE_DIR / target_label

            result, timestamp_offset_ms = process_label(
                target_label=target_label,
                interim_label_dir=label_dir,
                output_dir=output_label_dir,
                preview_dir=preview_label_dir,
                hand_landmarker=hand_landmarker,
                pose_landmarker=pose_landmarker,
                face_landmarker=face_landmarker,
                timestamp_offset_ms=timestamp_offset_ms,
            )

            if result:
                summary_results.append(result)

    finally:
        hand_landmarker.close()
        pose_landmarker.close()
        face_landmarker.close()

    print("\n" + "=" * 80)
    print(f"SUMMARY FOR WORKER {WORKER_ID}")
    print("=" * 80)

    total_processed = sum(r["processed"] for r in summary_results)
    total_skipped = sum(r["skipped"] for r in summary_results)
    total_errors = sum(r["errors"] for r in summary_results)

    for result in summary_results:
        print(f"Label '{result['label']}': {result['processed']} processed, {result['skipped']} skipped, {result['errors']} errors")

    print(f"\nWorker {WORKER_ID} Summary:")
    print(f"Total processed:   {total_processed}")
    print(f"Total skipped:     {total_skipped}")
    print(f"Total errors:      {total_errors}")
    print(f"Target frames:     {TARGET_FRAMES}")
    print(f"Feature dim:       {TOTAL_FEATURE_DIM}")
    print(f"Output base dir:   {OUTPUT_BASE_DIR}")

    if SAVE_BLACK_PREVIEW:
        print(f"Preview base dir:  {PREVIEW_BASE_DIR}")

    print("\nDone.")


if __name__ == "__main__":
    main()
