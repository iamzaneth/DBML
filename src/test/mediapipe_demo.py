from __future__ import annotations

import argparse
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class ModelResource:
    """Describe one MediaPipe task model used by the webcam demo."""

    filename: str
    url: str

    @property
    def path(self) -> Path:
        """Return the local path where the model should be stored."""

        return MODEL_DIR / self.filename

    def is_available(self) -> bool:
        """Return whether the local model file exists and is not empty."""

        return self.path.exists() and self.path.stat().st_size > 0


MODEL_RESOURCES = {
    "hand": ModelResource(
        filename="hand_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task"
        ),
    ),
    "face": ModelResource(
        filename="face_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        ),
    ),
    "pose": ModelResource(
        filename="pose_landmarker_heavy.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        ),
    ),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
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
    (27, 31), (28, 32),
]


def parse_args() -> argparse.Namespace:
    """Read webcam and frame options from the command line."""

    parser = argparse.ArgumentParser(
        description="Run a webcam demo for MediaPipe hand, face, and pose landmarks."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=640, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=480, help="Requested frame height.")
    return parser.parse_args()


def ensure_models() -> None:
    """Download missing MediaPipe model files into the project models directory."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, resource in MODEL_RESOURCES.items():
        if resource.is_available():
            print(f"{name.title()} model already exists: {resource.path}")
            continue

        print(f"Downloading {name} model...")
        urllib.request.urlretrieve(resource.url, resource.path)
        print(f"Saved {name} model to {resource.path}")


def create_hand_landmarker() -> vision.HandLandmarker:
    """Create the hand landmark detector for video frames."""

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_RESOURCES["hand"].path)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def create_face_landmarker() -> vision.FaceLandmarker:
    """Create the face landmark detector for video frames."""

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_RESOURCES["face"].path)
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


def create_pose_landmarker() -> vision.PoseLandmarker:
    """Create the pose landmark detector for video frames."""

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_RESOURCES["pose"].path)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def landmark_to_pixel(landmark, frame_width: int, frame_height: int) -> tuple[int, int]:
    """Convert a normalized MediaPipe landmark into pixel coordinates."""

    return int(landmark.x * frame_width), int(landmark.y * frame_height)


def point_is_visible(point: tuple[int, int], frame_width: int, frame_height: int) -> bool:
    """Return whether a point is inside the current video frame."""

    x, y = point
    return 0 <= x < frame_width and 0 <= y < frame_height


def draw_landmarks(
    frame,
    landmarks: Sequence,
    connections: Iterable[tuple[int, int]] | None = None,
    point_color: tuple[int, int, int] = (0, 255, 0),
    line_color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw landmark points and optional landmark connections on a frame."""

    if not landmarks:
        return

    frame_height, frame_width, _ = frame.shape

    if connections:
        for start_index, end_index in connections:
            if start_index >= len(landmarks) or end_index >= len(landmarks):
                continue

            start = landmark_to_pixel(landmarks[start_index], frame_width, frame_height)
            end = landmark_to_pixel(landmarks[end_index], frame_width, frame_height)

            if point_is_visible(start, frame_width, frame_height) and point_is_visible(
                end, frame_width, frame_height
            ):
                cv2.line(frame, start, end, line_color, 2)

    for landmark in landmarks:
        point = landmark_to_pixel(landmark, frame_width, frame_height)
        if point_is_visible(point, frame_width, frame_height):
            cv2.circle(frame, point, 2, point_color, -1)


def draw_detection_results(frame, hand_result, face_result, pose_result) -> None:
    """Render all detected landmarks and a compact status line."""

    for hand_landmarks in hand_result.hand_landmarks or []:
        draw_landmarks(
            frame,
            hand_landmarks,
            HAND_CONNECTIONS,
            point_color=(0, 255, 0),
            line_color=(0, 255, 0),
        )

    for face_landmarks in face_result.face_landmarks or []:
        draw_landmarks(frame, face_landmarks, point_color=(255, 0, 0))

    for pose_landmarks in pose_result.pose_landmarks or []:
        draw_landmarks(
            frame,
            pose_landmarks,
            POSE_CONNECTIONS,
            point_color=(0, 0, 255),
            line_color=(0, 0, 255),
        )

    status = (
        f"Hands: {len(hand_result.hand_landmarks or [])} | "
        f"Faces: {len(face_result.face_landmarks or [])} | "
        f"Poses: {len(pose_result.pose_landmarks or [])}"
    )
    cv2.putText(
        frame,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run_webcam(args: argparse.Namespace) -> None:
    """Open the webcam and process frames until the user presses Q."""

    hand_landmarker = create_hand_landmarker()
    face_landmarker = create_face_landmarker()
    pose_landmarker = create_pose_landmarker()
    camera = cv2.VideoCapture(args.camera)

    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        if not camera.isOpened():
            print(f"Could not open camera index {args.camera}.")
            return

        start_time = time.time()
        print("Camera is running. Press Q to exit.")

        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_detection_results(frame, hand_result, face_result, pose_result)
            cv2.imshow("MediaPipe Hand + Face + Pose", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        face_landmarker.close()
        pose_landmarker.close()


def main() -> None:
    """Prepare model files and run the webcam landmark demo."""

    args = parse_args()
    ensure_models()
    run_webcam(args)


if __name__ == "__main__":
    main()
