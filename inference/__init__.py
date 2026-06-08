"""
Inference package for Vietnamese Sign Language (VSL) recognition.

This package provides model prediction and real-time inference capabilities.

Modules:
    predictor: Model prediction wrapper for single/batch inference.
    realtime_inference: Webcam-based real-time sign language recognition.
"""

from inference.predictor import VSLPredictor
from inference.realtime_inference import RealtimeInference

__all__ = ["VSLPredictor", "RealtimeInference"]
