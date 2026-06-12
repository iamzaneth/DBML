import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. MODEL CONFIG
# =========================

MODEL_DIR = "models"

MODEL_URLS = {
    "hand": {
        "path": os.path.join(MODEL_DIR, "hand_landmarker.task"),
        "url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    },
    "face": {
        "path": os.path.join(MODEL_DIR, "face_landmarker.task"),
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    },
    "pose": {
        "path": os.path.join(MODEL_DIR, "pose_landmarker_heavy.task"),
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    },
}


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, item in MODEL_URLS.items():
        if not os.path.exists(item["path"]):
            print(f"Đang tải model {name}...")
            urllib.request.urlretrieve(item["url"], item["path"])
            print(f"Đã tải xong {name}")
        else:
            print(f"Model {name} đã có sẵn.")


# =========================
# 2. CONNECTIONS TỰ VẼ
# =========================

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


def get_point(landmark, width, height):
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return x, y


def draw_landmarks(frame, landmarks, connections=None, point_color=(0, 255, 0), line_color=(0, 255, 0)):
    if not landmarks:
        return

    height, width, _ = frame.shape

    # Vẽ line trước
    if connections:
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                x1, y1 = get_point(landmarks[start_idx], width, height)
                x2, y2 = get_point(landmarks[end_idx], width, height)

                if 0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height:
                    cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

    # Vẽ point
    for lm in landmarks:
        x, y = get_point(lm, width, height)

        if 0 <= x < width and 0 <= y < height:
            cv2.circle(frame, (x, y), 2, point_color, -1)


# =========================
# 3. CREATE LANDMARKERS
# =========================

def create_hand_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_URLS["hand"]["path"]
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(options)


def create_face_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_URLS["face"]["path"]
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    return vision.FaceLandmarker.create_from_options(options)


def create_pose_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_URLS["pose"]["path"]
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    return vision.PoseLandmarker.create_from_options(options)


# =========================
# 4. MAIN
# =========================

def main():
    download_models()

    hand_landmarker = create_hand_landmarker()
    face_landmarker = create_face_landmarker()
    pose_landmarker = create_pose_landmarker()

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Không mở được camera. Thử đổi cv2.VideoCapture(0) thành cv2.VideoCapture(1).")
        return

    start_time = time.time()

    print("Đã bật camera. Nhấn Q để thoát.")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Không đọc được frame từ camera.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Vẽ tay
            if hand_result.hand_landmarks:
                for hand_landmarks in hand_result.hand_landmarks:
                    draw_landmarks(
                        frame,
                        hand_landmarks,
                        HAND_CONNECTIONS,
                        point_color=(0, 255, 0),
                        line_color=(0, 255, 0)
                    )

            # Vẽ mặt
            # Face có rất nhiều điểm nên chỉ vẽ point, không vẽ line cho nhẹ
            if face_result.face_landmarks:
                for face_landmarks in face_result.face_landmarks:
                    draw_landmarks(
                        frame,
                        face_landmarks,
                        connections=None,
                        point_color=(255, 0, 0),
                        line_color=(255, 0, 0)
                    )

            # Vẽ pose
            if pose_result.pose_landmarks:
                for pose_landmarks in pose_result.pose_landmarks:
                    draw_landmarks(
                        frame,
                        pose_landmarks,
                        POSE_CONNECTIONS,
                        point_color=(0, 0, 255),
                        line_color=(0, 0, 255)
                    )

            num_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
            num_faces = len(face_result.face_landmarks) if face_result.face_landmarks else 0
            num_poses = len(pose_result.pose_landmarks) if pose_result.pose_landmarks else 0

            status = f"Hands: {num_hands} | Faces: {num_faces} | Poses: {num_poses}"

            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("MediaPipe Hand + Face + Pose", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        hand_landmarker.close()
        face_landmarker.close()
        pose_landmarker.close()


if __name__ == "__main__":
    main()