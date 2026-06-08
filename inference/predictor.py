"""
VSL Predictor Module.

Provides a prediction wrapper for loading trained VSL models and
running inference on single sequences or batches.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class VSLPredictor:
    """Prediction wrapper for Vietnamese Sign Language recognition models.

    Loads a trained model and label mapping, then provides methods for
    single-sequence and batch prediction with top-k confidence scores.

    Args:
        model_path: Path to the saved Keras model (.h5 or SavedModel).
        labels_path: Path to labels.json mapping indices to label names.

    Example:
        >>> predictor = VSLPredictor("outputs/best_model.h5", "labels.json")
        >>> results = predictor.predict(sequence)  # shape (60, 1605)
        >>> for label, score in results:
        ...     print(f"{label}: {score:.4f}")
    """

    def __init__(
        self,
        model_path: str,
        labels_path: str = "labels.json",
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)

        # Load model
        self.model = self._load_model()

        # Load labels
        self.labels = self._load_labels()
        self.num_classes = len(self.labels)

        logger.info(
            "VSLPredictor ready — model: %s, classes: %d",
            self.model_path.name,
            self.num_classes,
        )

    def _load_model(self) -> keras.Model:
        """Load the Keras model from disk.

        Returns:
            Loaded Keras model.

        Raises:
            FileNotFoundError: If model file does not exist.
            ValueError: If model cannot be loaded.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            model = keras.models.load_model(str(self.model_path), compile=False)
            logger.info("Model loaded from %s", self.model_path)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model from {self.model_path}: {e}") from e

    def _load_labels(self) -> Dict[int, str]:
        """Load the label mapping from labels.json.

        Returns:
            Dictionary mapping integer indices to label names.

        Raises:
            FileNotFoundError: If labels file does not exist.
        """
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        with open(self.labels_path, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)

        # Support both {index: name} and [name1, name2, ...] formats
        if isinstance(raw_labels, list):
            labels = {i: name for i, name in enumerate(raw_labels)}
        elif isinstance(raw_labels, dict):
            labels = {int(k): v for k, v in raw_labels.items()}
        else:
            raise ValueError(f"Unsupported labels format: {type(raw_labels)}")

        logger.info("Loaded %d labels from %s", len(labels), self.labels_path)
        return labels

    def predict(
        self,
        sequence: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Predict on a single landmark sequence.

        Args:
            sequence: Input landmark sequence of shape (60, 1605) or
                (1, 60, 1605). If 2D, a batch dimension is added.
            top_k: Number of top predictions to return.

        Returns:
            List of (label_name, confidence_score) tuples sorted by confidence
            in descending order.

        Raises:
            ValueError: If input shape is invalid.
        """
        sequence = self._preprocess_input(sequence)

        # Run inference
        probs = self.model.predict(sequence, verbose=0)
        probs = probs[0]  # Remove batch dimension

        return self._decode_predictions(probs, top_k)

    def predict_batch(
        self,
        sequences: np.ndarray,
        top_k: int = 5,
    ) -> List[List[Tuple[str, float]]]:
        """Predict on a batch of landmark sequences.

        Args:
            sequences: Batch of input sequences, shape (B, 60, 1605).
            top_k: Number of top predictions per sample.

        Returns:
            List of prediction lists, one per sample in the batch.

        Raises:
            ValueError: If input shape is invalid.
        """
        if sequences.ndim != 3:
            raise ValueError(
                f"Expected 3D batch input (B, 60, 1605), got shape {sequences.shape}"
            )

        probs = self.model.predict(sequences, verbose=0)

        results = []
        for i in range(probs.shape[0]):
            results.append(self._decode_predictions(probs[i], top_k))

        return results

    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """Return raw probability distribution for a single sequence.

        Args:
            sequence: Input sequence of shape (60, 1605) or (1, 60, 1605).

        Returns:
            Probability array of shape (num_classes,).
        """
        sequence = self._preprocess_input(sequence)
        probs = self.model.predict(sequence, verbose=0)
        return probs[0]

    def _preprocess_input(self, sequence: np.ndarray) -> np.ndarray:
        """Validate and preprocess input for the model.

        Args:
            sequence: Raw input array.

        Returns:
            Preprocessed array with batch dimension, shape (1, 60, 1605).

        Raises:
            ValueError: If input dimensions are wrong.
        """
        if sequence.ndim == 2:
            # Add batch dimension: (60, 1605) -> (1, 60, 1605)
            sequence = np.expand_dims(sequence, axis=0)
        elif sequence.ndim == 3:
            pass  # Already has batch dimension
        else:
            raise ValueError(
                f"Expected 2D (60, 1605) or 3D (1, 60, 1605) input, "
                f"got {sequence.ndim}D with shape {sequence.shape}"
            )

        # Type cast
        sequence = sequence.astype(np.float32)
        return sequence

    def _decode_predictions(
        self, probs: np.ndarray, top_k: int
    ) -> List[Tuple[str, float]]:
        """Decode model output probabilities into label-score pairs.

        Args:
            probs: Probability vector of shape (num_classes,).
            top_k: Number of top predictions to return.

        Returns:
            List of (label, confidence) tuples sorted descending.
        """
        top_k = min(top_k, len(probs))
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in top_indices:
            label = self.labels.get(int(idx), f"unknown_{idx}")
            confidence = float(probs[idx])
            results.append((label, confidence))

        return results

    def get_label_name(self, index: int) -> str:
        """Get the label name for a given class index.

        Args:
            index: Class index.

        Returns:
            Label name string.
        """
        return self.labels.get(index, f"unknown_{index}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata.

        Returns:
            Dictionary with model name, parameter count, input/output shapes.
        """
        return {
            "model_name": self.model.name,
            "model_path": str(self.model_path),
            "num_parameters": self.model.count_params(),
            "num_classes": self.num_classes,
            "input_shape": str(self.model.input_shape),
            "output_shape": str(self.model.output_shape),
        }
