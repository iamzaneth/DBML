"""
Training package for Vietnamese Sign Language (VSL) recognition.

This package provides training and evaluation utilities for the VSL model.

Modules:
    trainer: Training loop with callbacks and mixed precision support.
    evaluate: Model evaluation with metrics, confusion matrix, and reports.
"""

from training.trainer import VSLTrainer
from training.evaluate import VSLEvaluator

__all__ = ["VSLTrainer", "VSLEvaluator"]
