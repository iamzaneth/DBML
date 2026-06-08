"""
VSL Real-Time Inference Module.

Provides webcam-based real-time Vietnamese Sign Language recognition
using MediaPipe Holistic for landmark extraction and a trained model
for classification.

Features:
- OpenCV webcam capture with configurable resolution
- MediaPipe Holistic landmark extraction (pose, face, hands)
- 60-frame sliding window buffer
- Prediction smoothing over 3 consecutive windows
- On-screen overlay: detected sign, confidence, FPS, landmarks
- Keyboard controls: 'q' quit, 's' screenshot
"""

import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Landmark dimensions per component
# Pose: 33 landmarks × 4 (x, y, z, visibility) = 132
# Face: 468 landmarks × 3 (x, y, z) = 1404  — but we typically use a subset
# Left hand: 21 landmarks × 3 = 63
# Right hand: 21 landmarks × 3 = 63
# We'll extract exactly what matches the training input of 1605 features.
# Common split: pose(33×4=132) + face(468×3=1404) + lh(21×3=63) + rh(21×3=6) = 1662
# or: pose(33×3=99) + face(468×3=1404) + lh(21×3=63) + rh(21×3=63) = 1629
# Adjusting: We will match the training feature count of 1605 exactly.
# pose(33×3=99) + face(468×3=1404) + lh(21×3=63) + rh(21×3=63) → trim face to 126
# Alternative: Extract all and pad/truncate to 1605.

EXPECTED_FEATURES = 1605
SEQUENCE_LENGTH = 60


def _extract_landmarks_from_results(results: Any) -> np.ndarray:
    """Extract flattened landmark vector from MediaPipe Holistic results.

    Extracts pose, face, left hand, and right hand landmarks and
    concatenates them into a single feature vector. If the total
    does not match EXPECTED_FEATURES, it is padded or truncated.

    Args:
        results: MediaPipe Holistic processing results.

    Returns:
        1D numpy array of shape (EXPECTED_FEATURES,).
    """
    landmarks = []

    # Pose landmarks: 33 landmarks × (x, y, z, visibility) = 132
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        landmarks.extend([0.0] * (33 * 4))

    # Face landmarks: 468 landmarks × (x, y, z) = 1404
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (468 * 3))

    # Left hand landmarks: 21 landmarks × (x, y, z) = 63
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (21 * 3))

    # Right hand landmarks: 21 landmarks × (x, y, z) = 63
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (21 * 3))

    arr = np.array(landmarks, dtype=np.float32)

    # Pad or truncate to match expected features
    if len(arr) < EXPECTED_FEATURES:
        arr = np.pad(arr, (0, EXPECTED_FEATURES - len(arr)), mode="constant")
    elif len(arr) > EXPECTED_FEATURES:
        arr = arr[:EXPECTED_FEATURES]

    return arr


class RealtimeInference:
    """Real-time VSL sign language recognition from webcam.

    Uses MediaPipe Holistic for landmark extraction and a trained
    model for classification. Maintains a sliding window buffer
    and smooths predictions over multiple windows.

    Args:
        predictor: VSLPredictor instance for model inference.
        camera_id: OpenCV camera device ID (default 0).
        camera_width: Capture width in pixels.
        camera_height: Capture height in pixels.
        show_landmarks: Whether to draw MediaPipe landmarks on the frame.
        screenshot_dir: Directory for saving screenshots.
        smoothing_windows: Number of consecutive prediction windows to
            average for smoothing (default 3).

    Example:
        >>> from inference.predictor import VSLPredictor
        >>> predictor = VSLPredictor("outputs/best_model.h5", "labels.json")
        >>> rt = RealtimeInference(predictor)
        >>> rt.run()
    """

    def __init__(
        self,
        predictor: Any,
        camera_id: int = 0,
        camera_width: int = 1280,
        camera_height: int = 720,
        show_landmarks: bool = True,
        screenshot_dir: str = "screenshots",
        smoothing_windows: int = 3,
    ) -> None:
        self.predictor = predictor
        self.camera_id = camera_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.show_landmarks = show_landmarks
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.smoothing_windows = smoothing_windows

        # Frame buffer for sliding window
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)

        # Prediction history for smoothing
        self.prediction_history: Deque[np.ndarray] = deque(maxlen=smoothing_windows)

        # State
        self.current_prediction: str = "Waiting..."
        self.current_confidence: float = 0.0
        self.fps: float = 0.0

        logger.info(
            "RealtimeInference initialized — camera=%d, resolution=%dx%d, "
            "landmarks=%s, smoothing=%d",
            camera_id,
            camera_width,
            camera_height,
            show_landmarks,
            smoothing_windows,
        )

    def run(self) -> None:
        """Start the real-time inference loop.

        Opens the webcam, processes frames with MediaPipe, runs model
        inference, and displays results until 'q' is pressed.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        try:
            import mediapipe as mp
        except ImportError:
            raise RuntimeError(
                "mediapipe is required for real-time inference. "
                "Install it with: pip install mediapipe"
            )

        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera (id={self.camera_id}). "
                "Check that a webcam is connected and not in use."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        logger.info("Camera opened successfully. Press 'q' to quit, 's' for screenshot.")
        print("\n[VSL Real-Time Inference]")
        print("  Press 'q' to quit")
        print("  Press 's' to take a screenshot\n")

        frame_times: Deque[float] = deque(maxlen=30)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        ) as holistic:
            while cap.isOpened():
                t_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera.")
                    break

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = holistic.process(rgb_frame)
                rgb_frame.flags.writeable = True

                # Extract landmarks
                landmark_vector = _extract_landmarks_from_results(results)
                self.frame_buffer.append(landmark_vector)

                # Draw landmarks on frame if enabled
                if self.show_landmarks:
                    frame = self._draw_landmarks(
                        frame, results, mp_holistic, mp_drawing, mp_drawing_styles
                    )

                # Run inference when buffer is full
                if len(self.frame_buffer) == SEQUENCE_LENGTH:
                    self._run_prediction()

                # Calculate FPS
                t_end = time.perf_counter()
                frame_times.append(t_end - t_start)
                if len(frame_times) > 1:
                    self.fps = len(frame_times) / sum(frame_times)

                # Draw overlay
                frame = self._draw_overlay(frame)

                # Display
                cv2.imshow("VSL Real-Time Inference", frame)

                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User pressed 'q' — exiting.")
                    break
                elif key == ord("s"):
                    self._take_screenshot(frame)

        cap.release()
        cv2.destroyAllWindows()
        logger.info("Real-time inference stopped.")

    def _run_prediction(self) -> None:
        """Run model prediction on the current frame buffer."""
        sequence = np.array(list(self.frame_buffer), dtype=np.float32)

        # Get raw probabilities for smoothing
        probs = self.predictor.predict_proba(sequence)
        self.prediction_history.append(probs)

        # Smooth predictions over the last N windows
        if len(self.prediction_history) > 0:
            avg_probs = np.mean(list(self.prediction_history), axis=0)
            top_idx = int(np.argmax(avg_probs))
            self.current_prediction = self.predictor.get_label_name(top_idx)
            self.current_confidence = float(avg_probs[top_idx])

    @staticmethod
    def _draw_landmarks(
        frame: np.ndarray,
        results: Any,
        mp_holistic: Any,
        mp_drawing: Any,
        mp_drawing_styles: Any,
    ) -> np.ndarray:
        """Draw MediaPipe landmarks on the frame.

        Args:
            frame: BGR image frame.
            results: MediaPipe Holistic results.
            mp_holistic: MediaPipe holistic module.
            mp_drawing: MediaPipe drawing module.
            mp_drawing_styles: MediaPipe drawing styles module.

        Returns:
            Frame with landmarks drawn.
        """
        # Face mesh
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
            )

        # Pose
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

        # Left hand
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        # Right hand
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        return frame

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw information overlay on the frame.

        Displays detected sign (Vietnamese text), confidence score,
        FPS counter, and buffer fill status.

        Args:
            frame: BGR image frame.

        Returns:
            Frame with overlay drawn.
        """
        h, w = frame.shape[:2]

        # Semi-transparent overlay background at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Detected sign label
        label_text = f"Sign: {self.current_prediction}"
        cv2.putText(
            frame,
            label_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if self.current_confidence > 0.5 else (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Confidence bar
        conf_text = f"Confidence: {self.current_confidence:.1%}"
        cv2.putText(
            frame,
            conf_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Confidence bar visual
        bar_width = int(300 * self.current_confidence)
        bar_color = (0, 255, 0) if self.current_confidence > 0.5 else (0, 255, 255)
        cv2.rectangle(frame, (20, 85), (20 + bar_width, 100), bar_color, -1)
        cv2.rectangle(frame, (20, 85), (320, 100), (255, 255, 255), 1)

        # FPS counter
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (w - 160, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Buffer status
        buf_fill = len(self.frame_buffer)
        buf_text = f"Buffer: {buf_fill}/{SEQUENCE_LENGTH}"
        cv2.putText(
            frame,
            buf_text,
            (w - 200, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return frame

    def _take_screenshot(self, frame: np.ndarray) -> None:
        """Save the current frame as a screenshot.

        Args:
            frame: BGR image frame to save.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = str(self.screenshot_dir / filename)
        cv2.imwrite(filepath, frame)
        logger.info("Screenshot saved: %s", filepath)
        print(f"[Screenshot] Saved to {filepath}")
