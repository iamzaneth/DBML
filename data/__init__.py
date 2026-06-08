"""
Data pipeline package for VSL (Vietnamese Sign Language) Recognition.

This package provides:
    - download_dataset: Download and explore the VOYA_VSL dataset from HuggingFace
    - dataset: Load .npz files and create tf.data.Dataset pipelines
    - preprocessing: Landmark normalization and feature selection
    - augmentation: Spatial, temporal, and noise augmentations for landmark sequences

Usage:
    from data.dataset import VSLDatasetBuilder
    from data.preprocessing import LandmarkPreprocessor
    from data.augmentation import LandmarkAugmentor
"""

from data.dataset import VSLDatasetBuilder
from data.preprocessing import LandmarkPreprocessor
from data.augmentation import LandmarkAugmentor

__all__ = [
    "VSLDatasetBuilder",
    "LandmarkPreprocessor",
    "LandmarkAugmentor",
]
