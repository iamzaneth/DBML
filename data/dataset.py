"""
Dataset builder for the VOYA_VSL Vietnamese Sign Language dataset.

Loads .npz files, performs stratified train/val/test splitting, applies
preprocessing and augmentation, and returns tf.data.Dataset objects
with proper batching, prefetching, and optional caching.

Usage:
    from data.dataset import VSLDatasetBuilder
    from config import get_config

    cfg = get_config()
    builder = VSLDatasetBuilder(cfg)

    train_ds, val_ds, test_ds = builder.build()
    # Each yields (sequence, label) batches as tf.Tensors

    # Or build with info
    datasets, info = builder.build_with_info()
    print(info["num_classes"], info["train_samples"])
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data.augmentation import LandmarkAugmentor, create_augmentation_fn
from data.preprocessing import LandmarkPreprocessor

logger = logging.getLogger(__name__)


class VSLDatasetBuilder:
    """Builds tf.data.Dataset pipelines for the VOYA_VSL dataset.

    Handles the full data pipeline:
        1. Load .npz files from local directory
        2. Concatenate all files into single arrays
        3. Stratified train/val/test split
        4. Apply preprocessing (normalization, feature selection)
        5. Apply augmentation (training set only)
        6. Create tf.data.Dataset with batching and prefetching

    Attributes:
        data_dir: Path to directory containing .npz files.
        num_classes: Number of sign language classes.
        seq_length: Number of frames per sequence.
        num_features: Number of features per frame.
        batch_size: Batch size for training.
        preprocessor: Landmark preprocessor instance.
        augmentor: Landmark augmentor instance (training only).
    """

    def __init__(self, config: Optional[object] = None, **kwargs) -> None:
        """Initialize the dataset builder.

        Args:
            config: Configuration DotDict object. If None, uses kwargs/defaults.
            **kwargs: Override individual settings:
                - data_dir (str): Path to .npz files
                - num_classes (int): Number of classes
                - seq_length (int): Frames per sequence
                - num_features (int): Features per frame
                - batch_size (int): Training batch size
                - train_ratio (float): Train split ratio
                - val_ratio (float): Validation split ratio
                - test_ratio (float): Test split ratio
                - split_seed (int): Random seed for splits
                - shuffle (bool): Shuffle before splitting
                - shuffle_buffer (int): tf.data shuffle buffer size
                - prefetch_buffer (int): tf.data prefetch buffer size
                - augment_train (bool): Whether to augment training data
        """
        if config is not None:
            self.data_dir = Path(getattr(config.data, "raw_dir", "data/raw"))
            self.num_classes = getattr(config.data, "num_classes", 161)
            self.seq_length = getattr(config.data, "seq_length", 60)
            self.num_features = getattr(config.data, "num_features", 1605)
            self.batch_size = getattr(config.training, "batch_size", 32)
            self.train_ratio = getattr(config.data, "train_ratio", 0.70)
            self.val_ratio = getattr(config.data, "val_ratio", 0.15)
            self.test_ratio = getattr(config.data, "test_ratio", 0.15)
            self.split_seed = getattr(config.data, "split_seed", 42)
            self.shuffle = getattr(config.data, "shuffle", True)
            self.shuffle_buffer = getattr(
                config.hardware, "shuffle_buffer", 2048
            )
            self.prefetch_buffer = getattr(
                config.hardware, "prefetch_buffer", -1
            )
            self.augment_train = getattr(
                config.augmentation, "enabled", True
            )

            # Initialize preprocessor and augmentor
            self.preprocessor = LandmarkPreprocessor(config)
            self.augmentor = LandmarkAugmentor(config) if self.augment_train else None
        else:
            self.data_dir = Path(kwargs.get("data_dir", "data/raw"))
            self.num_classes = kwargs.get("num_classes", 161)
            self.seq_length = kwargs.get("seq_length", 60)
            self.num_features = kwargs.get("num_features", 1605)
            self.batch_size = kwargs.get("batch_size", 32)
            self.train_ratio = kwargs.get("train_ratio", 0.70)
            self.val_ratio = kwargs.get("val_ratio", 0.15)
            self.test_ratio = kwargs.get("test_ratio", 0.15)
            self.split_seed = kwargs.get("split_seed", 42)
            self.shuffle = kwargs.get("shuffle", True)
            self.shuffle_buffer = kwargs.get("shuffle_buffer", 2048)
            self.prefetch_buffer = kwargs.get("prefetch_buffer", -1)
            self.augment_train = kwargs.get("augment_train", True)

            self.preprocessor = LandmarkPreprocessor(**kwargs)
            self.augmentor = (
                LandmarkAugmentor() if self.augment_train else None
            )

        # Convert prefetch_buffer -1 to tf.data.AUTOTUNE
        if self.prefetch_buffer == -1:
            self.prefetch_buffer = tf.data.AUTOTUNE

        # Validate split ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio:.4f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, "
                f"test={self.test_ratio})"
            )

        # Data holders (populated by load_data)
        self._sequences: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._is_loaded = False

        logger.info(
            f"VSLDatasetBuilder initialized: "
            f"data_dir={self.data_dir}, "
            f"batch_size={self.batch_size}, "
            f"splits=({self.train_ratio}/{self.val_ratio}/{self.test_ratio})"
        )

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load and concatenate all .npz files from the data directory.

        Scans the data directory for .npz files, loads sequences and labels
        from each, and concatenates them into single arrays.

        Returns:
            Tuple of (sequences, labels):
                - sequences: np.ndarray of shape (N, T, F)
                - labels: np.ndarray of shape (N,)

        Raises:
            FileNotFoundError: If data directory doesn't exist or has no .npz files.
            ValueError: If .npz files have inconsistent shapes.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir.resolve()}\n"
                f"Run 'python -m data.download_dataset' first."
            )

        npz_files = sorted(self.data_dir.glob("*.npz"))

        if not npz_files:
            raise FileNotFoundError(
                f"No .npz files found in {self.data_dir.resolve()}\n"
                f"Run 'python -m data.download_dataset' first."
            )

        all_sequences = []
        all_labels = []

        for npz_file in npz_files:
            logger.info(f"Loading: {npz_file.name}")

            try:
                data = np.load(npz_file)

                if "sequences" not in data or "labels" not in data:
                    logger.warning(
                        f"Skipping {npz_file.name}: missing 'sequences' or "
                        f"'labels' key. Found keys: {list(data.keys())}"
                    )
                    data.close()
                    continue

                sequences = data["sequences"].astype(np.float32)
                labels = data["labels"]
                data.close()

                # Validate shape
                if sequences.ndim != 3:
                    logger.warning(
                        f"Skipping {npz_file.name}: expected 3D sequences, "
                        f"got shape {sequences.shape}"
                    )
                    continue

                _, t, f = sequences.shape
                if t != self.seq_length:
                    logger.warning(
                        f"{npz_file.name}: seq_length={t}, expected "
                        f"{self.seq_length}. Padding/truncating."
                    )
                    sequences = self._adjust_seq_length(sequences)

                if f != self.num_features:
                    logger.warning(
                        f"{npz_file.name}: num_features={f}, expected "
                        f"{self.num_features}. Adjusting."
                    )
                    # Update num_features to match actual data
                    self.num_features = f
                    self.preprocessor.num_features = f
                    self.preprocessor._compute_landmark_boundaries()

                all_sequences.append(sequences)
                all_labels.append(labels)

                logger.info(
                    f"  Loaded {len(sequences)} samples "
                    f"(shape: {sequences.shape})"
                )

            except Exception as e:
                logger.error(f"Error loading {npz_file.name}: {e}")
                raise

        if not all_sequences:
            raise ValueError("No valid data loaded from .npz files.")

        self._sequences = np.concatenate(all_sequences, axis=0)
        self._labels = np.concatenate(all_labels, axis=0)
        self._is_loaded = True

        # Update num_classes based on actual data
        actual_classes = len(np.unique(self._labels))
        if actual_classes != self.num_classes:
            logger.info(
                f"Adjusting num_classes: {self.num_classes} -> {actual_classes}"
            )
            self.num_classes = actual_classes

        logger.info(
            f"Data loaded: {self._sequences.shape[0]} total samples, "
            f"{self.num_classes} classes, "
            f"shape: {self._sequences.shape}"
        )

        return self._sequences, self._labels

    def split_data(
        self,
        sequences: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Perform stratified train/val/test split.

        Uses sklearn's train_test_split with stratification to ensure
        each split has a proportional representation of all classes.

        Args:
            sequences: Optional sequences array. If None, uses loaded data.
            labels: Optional labels array. If None, uses loaded data.

        Returns:
            Dictionary with keys "train", "val", "test", each mapping to
            a (sequences, labels) tuple.

        Raises:
            RuntimeError: If data hasn't been loaded and no arrays provided.
        """
        if sequences is None or labels is None:
            if not self._is_loaded:
                self.load_data()
            sequences = self._sequences
            labels = self._labels

        logger.info(
            f"Splitting data: {len(sequences)} samples -> "
            f"train({self.train_ratio})/val({self.val_ratio})/"
            f"test({self.test_ratio})"
        )

        # First split: separate test set
        test_size = self.test_ratio
        val_size_adjusted = self.val_ratio / (self.train_ratio + self.val_ratio)

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            sequences,
            labels,
            test_size=test_size,
            random_state=self.split_seed,
            stratify=labels,
            shuffle=self.shuffle,
        )

        # Second split: separate val from train
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_size_adjusted,
            random_state=self.split_seed,
            stratify=y_trainval,
            shuffle=self.shuffle,
        )

        splits = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

        for name, (X, y) in splits.items():
            logger.info(
                f"  {name:>5s}: {len(X):>6d} samples, "
                f"{len(np.unique(y)):>3d} classes"
            )

        return splits

    def build(
        self,
        cache: bool = False,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Build complete tf.data.Dataset pipelines for train/val/test.

        End-to-end pipeline:
            1. Load data from .npz files
            2. Stratified split
            3. Preprocess all splits
            4. Augment training split
            5. Create tf.data.Datasets with batching and prefetching

        Args:
            cache: Whether to cache datasets in memory. Faster iteration
                but uses more RAM. Recommended for small datasets that
                fit in memory.

        Returns:
            Tuple of (train_ds, val_ds, test_ds) as tf.data.Datasets.
            Each yields batches of (sequences, labels) tensors.
        """
        datasets, _ = self.build_with_info(cache=cache)
        return datasets["train"], datasets["val"], datasets["test"]

    def build_with_info(
        self,
        cache: bool = False,
    ) -> tuple[dict[str, tf.data.Dataset], dict]:
        """Build datasets and return with metadata info.

        Args:
            cache: Whether to cache datasets in memory.

        Returns:
            Tuple of (datasets_dict, info_dict):
                - datasets_dict: {"train": ds, "val": ds, "test": ds}
                - info_dict: metadata about the dataset
        """
        # Load and split
        if not self._is_loaded:
            self.load_data()

        splits = self.split_data()

        # Preprocess all splits
        logger.info("Preprocessing data...")
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]

        # Fit preprocessor on training data only
        self.preprocessor.fit(X_train)

        # Transform all splits
        X_train = self.preprocessor.transform(X_train)
        X_val = self.preprocessor.transform(X_val)
        X_test = self.preprocessor.transform(X_test)

        logger.info(
            f"After preprocessing: "
            f"feature_dim={X_train.shape[-1]} "
            f"(subset={self.preprocessor.feature_subset})"
        )

        # Compute class weights for training
        class_weights = self._compute_class_weights(y_train)

        # Build tf.data.Datasets
        train_ds = self._create_dataset(
            X_train, y_train,
            is_training=True,
            cache=cache,
        )

        val_ds = self._create_dataset(
            X_val, y_val,
            is_training=False,
            cache=cache,
        )

        test_ds = self._create_dataset(
            X_test, y_test,
            is_training=False,
            cache=cache,
        )

        datasets = {"train": train_ds, "val": val_ds, "test": test_ds}

        info = {
            "num_classes": self.num_classes,
            "num_features": X_train.shape[-1],
            "seq_length": self.seq_length,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "total_samples": len(X_train) + len(X_val) + len(X_test),
            "class_weights": class_weights,
            "feature_info": self.preprocessor.get_feature_info(),
            "batch_size": self.batch_size,
            "train_steps": int(np.ceil(len(X_train) / self.batch_size)),
            "val_steps": int(np.ceil(len(X_val) / self.batch_size)),
            "test_steps": int(np.ceil(len(X_test) / self.batch_size)),
        }

        logger.info(
            f"Datasets built: "
            f"train={info['train_samples']} ({info['train_steps']} steps), "
            f"val={info['val_samples']} ({info['val_steps']} steps), "
            f"test={info['test_samples']} ({info['test_steps']} steps)"
        )

        return datasets, info

    def _create_dataset(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        is_training: bool = False,
        cache: bool = False,
    ) -> tf.data.Dataset:
        """Create a tf.data.Dataset from numpy arrays.

        Args:
            sequences: Shape (N, T, F).
            labels: Shape (N,).
            is_training: If True, apply shuffling and augmentation.
            cache: Whether to cache the dataset.

        Returns:
            Configured tf.data.Dataset.
        """
        # Create base dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                tf.constant(sequences, dtype=tf.float32),
                tf.constant(labels, dtype=tf.int64),
            )
        )

        # Cache before augmentation (if enabled)
        if cache:
            dataset = dataset.cache()

        # Shuffle training data
        if is_training and self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer,
                reshuffle_each_iteration=True,
            )

        # Apply augmentation to training data
        if is_training and self.augmentor is not None:
            aug_fn = create_augmentation_fn(seed=None)

            def augment_wrapper(sequence: tf.Tensor, label: tf.Tensor):
                """Apply numpy augmentation via tf.numpy_function."""
                augmented = tf.numpy_function(
                    aug_fn, [sequence], tf.float32
                )
                # Restore shape info lost by numpy_function
                augmented.set_shape(sequence.shape)
                return augmented, label

            dataset = dataset.map(
                augment_wrapper,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        # Batch
        dataset = dataset.batch(self.batch_size, drop_remainder=False)

        # Prefetch
        dataset = dataset.prefetch(self.prefetch_buffer)

        return dataset

    def _adjust_seq_length(self, sequences: np.ndarray) -> np.ndarray:
        """Adjust sequences to match the expected sequence length.

        Pads short sequences with zeros and truncates long ones.

        Args:
            sequences: Shape (N, T_actual, F).

        Returns:
            Shape (N, self.seq_length, F).
        """
        N, T_actual, F = sequences.shape
        T_target = self.seq_length

        if T_actual == T_target:
            return sequences

        result = np.zeros((N, T_target, F), dtype=np.float32)

        if T_actual > T_target:
            # Truncate (center crop)
            start = (T_actual - T_target) // 2
            result = sequences[:, start:start + T_target, :]
        else:
            # Pad with zeros (center)
            start = (T_target - T_actual) // 2
            result[:, start:start + T_actual, :] = sequences

        return result

    def _compute_class_weights(
        self,
        labels: np.ndarray,
    ) -> dict[int, float]:
        """Compute balanced class weights for handling class imbalance.

        Uses inverse frequency weighting: classes with fewer samples
        get higher weights.

        Args:
            labels: Training labels of shape (N,).

        Returns:
            Dictionary mapping class indices to weights.
        """
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        n_samples = len(labels)
        n_classes = len(unique_classes)

        # Balanced weights: n_samples / (n_classes * count_per_class)
        weights = n_samples / (n_classes * class_counts)

        class_weights = {
            int(cls): float(weight)
            for cls, weight in zip(unique_classes, weights)
        }

        # Log statistics
        weight_values = list(class_weights.values())
        logger.info(
            f"Class weights computed: "
            f"min={min(weight_values):.3f}, "
            f"max={max(weight_values):.3f}, "
            f"mean={np.mean(weight_values):.3f}"
        )

        return class_weights

    def get_label_mapping(self) -> Optional[dict[int, str]]:
        """Get a mapping from label indices to class names (if available).

        Returns:
            Dictionary mapping integer labels to string class names,
            or None if no mapping is available.
        """
        # Check if a label mapping file exists
        mapping_file = self.data_dir / "label_mapping.npy"
        if mapping_file.exists():
            try:
                mapping = np.load(mapping_file, allow_pickle=True).item()
                return mapping
            except Exception as e:
                logger.warning(f"Could not load label mapping: {e}")

        return None

    def get_sample(
        self, index: int = 0, split: str = "train"
    ) -> tuple[np.ndarray, int]:
        """Get a single sample for debugging/visualization.

        Args:
            index: Sample index within the split.
            split: Which split to get from ("train", "val", "test").

        Returns:
            Tuple of (sequence, label).

        Raises:
            RuntimeError: If data hasn't been loaded.
        """
        if not self._is_loaded:
            self.load_data()

        splits = self.split_data()
        X, y = splits[split]

        if index >= len(X):
            raise IndexError(
                f"Index {index} out of range for {split} split "
                f"({len(X)} samples)"
            )

        return X[index], int(y[index])
