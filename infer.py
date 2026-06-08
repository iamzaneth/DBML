#!/usr/bin/env python3
"""
infer.py — Main entry point for VSL sign language inference.

Usage:
    # Webcam real-time inference
    python infer.py --model outputs/best_model.h5 --labels labels.json --mode webcam

    # Video file inference
    python infer.py --model outputs/best_model.h5 --labels labels.json --mode video --input video.mp4

    # Single .npy file inference
    python infer.py --model outputs/best_model.h5 --labels labels.json --mode file --input sample.npy

Modes:
    webcam  — Real-time inference from webcam with MediaPipe
    video   — Inference on a video file (frame-by-frame with MediaPipe)
    file    — Inference on a pre-extracted .npy landmark sequence
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ────────────────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("infer")


# ────────────────────────────────────────────────────────────
# CLI argument parsing
# ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for inference.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run Vietnamese Sign Language (VSL) model inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model file (.h5 or SavedModel directory).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="labels.json",
        help="Path to labels.json for class name mapping.",
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["webcam", "video", "file"],
        default="webcam",
        help="Inference mode: 'webcam' for real-time, 'video' for video file, "
             "'file' for pre-extracted .npy sequence.",
    )

    # Input (for video/file modes)
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input path for video or .npy file (required for 'video' and 'file' modes).",
    )

    # Webcam options
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera device ID for webcam mode.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera capture width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera capture height.",
    )
    parser.add_argument(
        "--no_landmarks",
        action="store_true",
        help="Disable landmark overlay on video feed.",
    )

    # Prediction options
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top predictions to display.",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=3,
        help="Number of windows for prediction smoothing.",
    )
    parser.add_argument(
        "--screenshot_dir",
        type=str,
        default="screenshots",
        help="Directory for saving screenshots.",
    )

    # GPU
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID to use.",
    )

    return parser.parse_args()


# ────────────────────────────────────────────────────────────
# Inference on a single .npy file
# ────────────────────────────────────────────────────────────
def infer_file(
    predictor: Any,
    file_path: str,
    top_k: int = 5,
) -> None:
    """Run inference on a pre-extracted .npy landmark sequence.

    Args:
        predictor: VSLPredictor instance.
        file_path: Path to the .npy file with shape (60, 1605) or (B, 60, 1605).
        top_k: Number of top predictions to display.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    if path.suffix != ".npy":
        logger.error("Expected .npy file, got: %s", path.suffix)
        sys.exit(1)

    data = np.load(str(path))
    logger.info("Loaded %s with shape %s", path.name, data.shape)

    if data.ndim == 2:
        # Single sequence
        results = predictor.predict(data, top_k=top_k)
        print(f"\n{'='*40}")
        print(f"  Predictions for: {path.name}")
        print(f"{'='*40}")
        for rank, (label, confidence) in enumerate(results, 1):
            bar = "█" * int(confidence * 30)
            print(f"  {rank}. {label:<30s} {confidence:6.2%}  {bar}")
        print(f"{'='*40}\n")

    elif data.ndim == 3:
        # Batch of sequences
        batch_results = predictor.predict_batch(data, top_k=top_k)
        for i, results in enumerate(batch_results):
            print(f"\n--- Sample {i+1}/{len(batch_results)} ---")
            for rank, (label, confidence) in enumerate(results, 1):
                print(f"  {rank}. {label:<30s} {confidence:6.2%}")
    else:
        logger.error("Unexpected data shape: %s (expected 2D or 3D)", data.shape)
        sys.exit(1)


# ────────────────────────────────────────────────────────────
# Inference on a video file
# ────────────────────────────────────────────────────────────
def infer_video(
    predictor: Any,
    video_path: str,
    top_k: int = 5,
    show_landmarks: bool = True,
) -> None:
    """Run inference on a video file using MediaPipe for landmark extraction.

    Processes the video frame-by-frame, extracts landmarks with MediaPipe
    Holistic, collects 60-frame windows, and runs prediction.

    Args:
        predictor: VSLPredictor instance.
        video_path: Path to the video file.
        top_k: Number of top predictions to display.
        show_landmarks: Whether to draw landmarks on frames.
    """
    import cv2

    try:
        import mediapipe as mp
    except ImportError:
        logger.error("mediapipe is required for video inference. Install: pip install mediapipe")
        sys.exit(1)

    from inference.realtime_inference import (
        SEQUENCE_LENGTH,
        _extract_landmarks_from_results,
    )

    path = Path(video_path)
    if not path.exists():
        logger.error("Video file not found: %s", path)
        sys.exit(1)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Video: %s — %d frames @ %.1f FPS", path.name, total_frames, fps)

    mp_holistic = mp.solutions.holistic

    frame_buffer = []
    segment_idx = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            landmarks = _extract_landmarks_from_results(results)
            frame_buffer.append(landmarks)

            # Process when we have a full window
            if len(frame_buffer) == SEQUENCE_LENGTH:
                sequence = np.array(frame_buffer, dtype=np.float32)
                preds = predictor.predict(sequence, top_k=top_k)

                segment_idx += 1
                print(f"\n--- Segment {segment_idx} (frames {(segment_idx-1)*SEQUENCE_LENGTH + 1}–{segment_idx*SEQUENCE_LENGTH}) ---")
                for rank, (label, confidence) in enumerate(preds, 1):
                    print(f"  {rank}. {label:<30s} {confidence:6.2%}")

                # Slide window by half
                frame_buffer = frame_buffer[SEQUENCE_LENGTH // 2:]

    # Process remaining frames if buffer has enough data
    if len(frame_buffer) >= SEQUENCE_LENGTH // 2:
        # Pad to full sequence length
        while len(frame_buffer) < SEQUENCE_LENGTH:
            frame_buffer.append(np.zeros(frame_buffer[0].shape, dtype=np.float32))

        sequence = np.array(frame_buffer[:SEQUENCE_LENGTH], dtype=np.float32)
        preds = predictor.predict(sequence, top_k=top_k)

        segment_idx += 1
        print(f"\n--- Segment {segment_idx} (final, padded) ---")
        for rank, (label, confidence) in enumerate(preds, 1):
            print(f"  {rank}. {label:<30s} {confidence:6.2%}")

    cap.release()
    logger.info("Video inference complete — %d segments processed.", segment_idx)


# ────────────────────────────────────────────────────────────
# GPU configuration
# ────────────────────────────────────────────────────────────
def configure_gpu(gpu_id: str = "0") -> None:
    """Configure GPU memory growth.

    Args:
        gpu_id: GPU device ID string.
    """
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU configured: %s", [g.name for g in gpus])
        except RuntimeError as e:
            logger.warning("GPU config error: %s", e)
    else:
        logger.warning("No GPU found. Inference will run on CPU.")


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main() -> None:
    """Main inference entry point."""
    args = parse_args()

    # Configure GPU
    configure_gpu(args.gpu)

    # Validate arguments
    if args.mode in ("video", "file") and args.input is None:
        logger.error("--input is required for '%s' mode.", args.mode)
        sys.exit(1)

    # Load predictor
    from inference.predictor import VSLPredictor

    try:
        predictor = VSLPredictor(
            model_path=args.model,
            labels_path=args.labels,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load model/labels: %s", e)
        sys.exit(1)

    # Print model info
    info = predictor.get_model_info()
    print(f"\n{'='*50}")
    print("  VSL Inference")
    print(f"{'='*50}")
    print(f"  Model:      {info['model_name']}")
    print(f"  Parameters: {info['num_parameters']:,}")
    print(f"  Classes:    {info['num_classes']}")
    print(f"  Mode:       {args.mode}")
    print(f"{'='*50}\n")

    # Dispatch to mode
    if args.mode == "webcam":
        from inference.realtime_inference import RealtimeInference

        rt = RealtimeInference(
            predictor=predictor,
            camera_id=args.camera_id,
            camera_width=args.width,
            camera_height=args.height,
            show_landmarks=not args.no_landmarks,
            screenshot_dir=args.screenshot_dir,
            smoothing_windows=args.smoothing,
        )
        try:
            rt.run()
        except RuntimeError as e:
            logger.error("Real-time inference error: %s", e)
            sys.exit(1)

    elif args.mode == "video":
        infer_video(
            predictor=predictor,
            video_path=args.input,
            top_k=args.top_k,
            show_landmarks=not args.no_landmarks,
        )

    elif args.mode == "file":
        infer_file(
            predictor=predictor,
            file_path=args.input,
            top_k=args.top_k,
        )

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
