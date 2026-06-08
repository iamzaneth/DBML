"""
Data augmentation for MediaPipe landmark sequences.

Provides spatial, temporal, and noise augmentations designed specifically
for sign language landmark data. All augmentations operate on numpy arrays
of shape (T, F) — a single sequence of T frames with F features per frame.

Augmentation categories:
    - **Spatial**: Scale, rotation, flip/mirror, shift
    - **Temporal**: Time warp, frame dropout, speed variation
    - **Noise**: Gaussian noise injection

Usage:
    from data.augmentation import LandmarkAugmentor
    from config import get_config

    cfg = get_config()
    augmentor = LandmarkAugmentor(cfg)

    # Augment a single sequence
    augmented = augmentor.augment(sequence)  # (60, 1605) -> (60, 1605)

    # Augment a batch
    augmented_batch = augmentor.augment_batch(sequences)  # (N, 60, 1605)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


class LandmarkAugmentor:
    """Augmentation pipeline for landmark sequences.

    Each augmentation is applied independently with its own probability.
    Only applied during training. Thread-safe for parallel data loading
    (each call uses its own RNG if seed is not fixed).

    Attributes:
        enabled: Whether augmentation is globally enabled.
        rng: NumPy random number generator.
    """

    def __init__(
        self,
        config: Optional[object] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize the augmentor with configuration.

        Args:
            config: Configuration DotDict object with augmentation settings.
            seed: Optional random seed for reproducibility.
            **kwargs: Override individual augmentation parameters.
        """
        self.rng = np.random.default_rng(seed)

        if config is not None and hasattr(config, "augmentation"):
            aug_cfg = config.augmentation
            self.enabled = getattr(aug_cfg, "enabled", True)

            # Spatial augmentations
            spatial = getattr(aug_cfg, "spatial", {})
            self.scale_cfg = _to_dict(getattr(spatial, "scale", {}))
            self.rotation_cfg = _to_dict(getattr(spatial, "rotation", {}))
            self.flip_cfg = _to_dict(getattr(spatial, "flip", {}))
            self.shift_cfg = _to_dict(getattr(spatial, "shift", {}))

            # Temporal augmentations
            temporal = getattr(aug_cfg, "temporal", {})
            self.time_warp_cfg = _to_dict(getattr(temporal, "time_warp", {}))
            self.frame_dropout_cfg = _to_dict(
                getattr(temporal, "frame_dropout", {})
            )
            self.speed_cfg = _to_dict(
                getattr(temporal, "speed_variation", {})
            )

            # Noise
            self.noise_cfg = _to_dict(getattr(aug_cfg, "noise", {}))
        else:
            self._set_defaults()

        # Apply any kwargs overrides
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Extract number of features and landmark structure
        self.coords = 3  # x, y, z

        logger.info(
            f"LandmarkAugmentor initialized: enabled={self.enabled}"
        )

    def _set_defaults(self) -> None:
        """Set default augmentation parameters."""
        self.enabled = True

        self.scale_cfg = {
            "enabled": True, "probability": 0.5,
            "min_factor": 0.85, "max_factor": 1.15,
        }
        self.rotation_cfg = {
            "enabled": True, "probability": 0.5,
            "max_angle_deg": 15.0,
        }
        self.flip_cfg = {
            "enabled": True, "probability": 0.3,
        }
        self.shift_cfg = {
            "enabled": True, "probability": 0.4,
            "max_shift": 0.05,
        }
        self.time_warp_cfg = {
            "enabled": True, "probability": 0.4,
            "sigma": 5.0, "num_knots": 4,
        }
        self.frame_dropout_cfg = {
            "enabled": True, "probability": 0.3,
            "max_dropout_ratio": 0.15,
        }
        self.speed_cfg = {
            "enabled": True, "probability": 0.3,
            "min_speed": 0.8, "max_speed": 1.2,
        }
        self.noise_cfg = {
            "enabled": True, "probability": 0.5,
            "std": 0.005,
        }

    def augment(self, sequence: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a single sequence.

        Each enabled augmentation is applied independently with its
        configured probability.

        Args:
            sequence: Landmark sequence of shape (T, F), where T is the
                number of frames and F is the number of features.

        Returns:
            Augmented sequence of shape (T, F). The original is not modified.

        Raises:
            ValueError: If sequence has wrong number of dimensions.
        """
        if not self.enabled:
            return sequence

        if sequence.ndim != 2:
            raise ValueError(
                f"Expected 2D array (T, F), got shape {sequence.shape}"
            )

        result = sequence.astype(np.float32).copy()

        # --- Spatial Augmentations ---
        if self.scale_cfg.get("enabled") and self._should_apply(
            self.scale_cfg.get("probability", 0)
        ):
            result = self._apply_scale(result)

        if self.rotation_cfg.get("enabled") and self._should_apply(
            self.rotation_cfg.get("probability", 0)
        ):
            result = self._apply_rotation(result)

        if self.flip_cfg.get("enabled") and self._should_apply(
            self.flip_cfg.get("probability", 0)
        ):
            result = self._apply_flip(result)

        if self.shift_cfg.get("enabled") and self._should_apply(
            self.shift_cfg.get("probability", 0)
        ):
            result = self._apply_shift(result)

        # --- Temporal Augmentations ---
        if self.time_warp_cfg.get("enabled") and self._should_apply(
            self.time_warp_cfg.get("probability", 0)
        ):
            result = self._apply_time_warp(result)

        if self.frame_dropout_cfg.get("enabled") and self._should_apply(
            self.frame_dropout_cfg.get("probability", 0)
        ):
            result = self._apply_frame_dropout(result)

        if self.speed_cfg.get("enabled") and self._should_apply(
            self.speed_cfg.get("probability", 0)
        ):
            result = self._apply_speed_variation(result)

        # --- Noise Injection ---
        if self.noise_cfg.get("enabled") and self._should_apply(
            self.noise_cfg.get("probability", 0)
        ):
            result = self._apply_noise(result)

        return result

    def augment_batch(self, sequences: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a batch of sequences.

        Each sequence in the batch is augmented independently (different
        random parameters for each sample).

        Args:
            sequences: Batch of sequences, shape (N, T, F).

        Returns:
            Augmented batch of shape (N, T, F).
        """
        return np.stack(
            [self.augment(seq) for seq in sequences], axis=0
        )

    def _should_apply(self, probability: float) -> bool:
        """Determine whether to apply an augmentation based on probability.

        Args:
            probability: Probability of applying [0, 1].

        Returns:
            True if the augmentation should be applied.
        """
        return self.rng.random() < probability

    # =========================================================================
    # Spatial Augmentations
    # =========================================================================

    def _apply_scale(self, sequence: np.ndarray) -> np.ndarray:
        """Apply random uniform scaling to all landmarks.

        Scales all coordinates by a random factor, simulating distance
        variation from the camera.

        Args:
            sequence: Shape (T, F).

        Returns:
            Scaled sequence.
        """
        min_f = self.scale_cfg.get("min_factor", 0.85)
        max_f = self.scale_cfg.get("max_factor", 1.15)
        factor = self.rng.uniform(min_f, max_f)

        return sequence * factor

    def _apply_rotation(self, sequence: np.ndarray) -> np.ndarray:
        """Apply random 2D rotation in the x-y plane to all landmarks.

        Rotates all (x, y) coordinates by a random angle while keeping
        z coordinates unchanged. This simulates slight camera angle
        variations.

        Args:
            sequence: Shape (T, F) where F = num_landmarks × 3.

        Returns:
            Rotated sequence.
        """
        max_angle = self.rotation_cfg.get("max_angle_deg", 15.0)
        angle_rad = np.radians(self.rng.uniform(-max_angle, max_angle))

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        result = sequence.copy()
        n_features = sequence.shape[1]
        n_landmarks = n_features // self.coords

        for lm in range(n_landmarks):
            x_idx = lm * self.coords
            y_idx = lm * self.coords + 1
            # z_idx = lm * self.coords + 2  # unchanged

            if x_idx < n_features and y_idx < n_features:
                x = sequence[:, x_idx]
                y = sequence[:, y_idx]
                result[:, x_idx] = cos_a * x - sin_a * y
                result[:, y_idx] = sin_a * x + cos_a * y

        return result

    def _apply_flip(self, sequence: np.ndarray) -> np.ndarray:
        """Apply horizontal flip (mirror) to landmarks.

        This mirrors all x-coordinates and swaps left/right hand landmarks,
        as well as corresponding left/right pose landmarks. Essential for
        sign language where some signs are handedness-specific.

        Args:
            sequence: Shape (T, F).

        Returns:
            Horizontally flipped sequence with left/right landmarks swapped.
        """
        result = sequence.copy()
        n_features = sequence.shape[1]
        n_landmarks = n_features // self.coords

        # Flip x-coordinates (negate x)
        for lm in range(n_landmarks):
            x_idx = lm * self.coords
            if x_idx < n_features:
                result[:, x_idx] = -result[:, x_idx]

        # Swap left and right hand landmarks
        # We need to know the landmark boundaries
        # Using 535 landmarks: pose(33) + face(460) + left_hand(21) + right_hand(21)
        pose_lm = 33
        face_lm = n_landmarks - pose_lm - 21 - 21  # Remaining = face
        if face_lm < 0:
            face_lm = 0

        lh_start = (pose_lm + face_lm) * self.coords
        lh_end = lh_start + 21 * self.coords
        rh_start = lh_end
        rh_end = rh_start + 21 * self.coords

        if rh_end <= n_features:
            # Swap left and right hand data
            left_hand_data = result[:, lh_start:lh_end].copy()
            right_hand_data = result[:, rh_start:rh_end].copy()
            result[:, lh_start:lh_end] = right_hand_data
            result[:, rh_start:rh_end] = left_hand_data

        # Swap bilateral pose landmarks (MediaPipe pairs)
        # Left/Right pairs in pose: (11,12), (13,14), (15,16), (17,18),
        # (19,20), (21,22), (23,24), (25,26), (27,28), (29,30), (31,32)
        pose_pairs = [
            (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
            (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
            # Also face-related: (1,4), (2,5), (3,6), (7,8), (9,10)
            (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
        ]

        for left_lm, right_lm in pose_pairs:
            if left_lm < pose_lm and right_lm < pose_lm:
                l_start = left_lm * self.coords
                r_start = right_lm * self.coords
                if l_start + self.coords <= n_features and r_start + self.coords <= n_features:
                    left_data = result[:, l_start:l_start + self.coords].copy()
                    result[:, l_start:l_start + self.coords] = result[
                        :, r_start:r_start + self.coords
                    ]
                    result[:, r_start:r_start + self.coords] = left_data

        return result

    def _apply_shift(self, sequence: np.ndarray) -> np.ndarray:
        """Apply random translation to all landmarks.

        Adds a small random offset to all x and y coordinates,
        simulating slight position variation.

        Args:
            sequence: Shape (T, F).

        Returns:
            Shifted sequence.
        """
        max_shift = self.shift_cfg.get("max_shift", 0.05)
        n_features = sequence.shape[1]
        n_landmarks = n_features // self.coords

        # Random shift for x and y (same shift for all landmarks in a frame)
        dx = self.rng.uniform(-max_shift, max_shift)
        dy = self.rng.uniform(-max_shift, max_shift)

        result = sequence.copy()

        for lm in range(n_landmarks):
            x_idx = lm * self.coords
            y_idx = lm * self.coords + 1

            if x_idx < n_features:
                result[:, x_idx] += dx
            if y_idx < n_features:
                result[:, y_idx] += dy

        return result

    # =========================================================================
    # Temporal Augmentations
    # =========================================================================

    def _apply_time_warp(self, sequence: np.ndarray) -> np.ndarray:
        """Apply smooth non-linear time warping.

        Creates a smooth, monotonically increasing mapping from original
        time indices to warped indices using cubic spline interpolation
        with random control points. This simulates natural speed variations
        in sign language performance.

        Args:
            sequence: Shape (T, F).

        Returns:
            Time-warped sequence of the same shape.
        """
        T, F = sequence.shape
        sigma = self.time_warp_cfg.get("sigma", 5.0)
        num_knots = self.time_warp_cfg.get("num_knots", 4)

        # Create random warp path
        # Start with uniformly spaced knots
        orig_knots = np.linspace(0, T - 1, num_knots + 2)

        # Add random perturbation to interior knots
        warp_knots = orig_knots.copy()
        for i in range(1, len(warp_knots) - 1):
            warp_knots[i] += self.rng.normal(0, sigma)

        # Ensure monotonicity
        warp_knots = np.sort(warp_knots)
        # Clamp to valid range
        warp_knots = np.clip(warp_knots, 0, T - 1)
        # Ensure strictly increasing
        for i in range(1, len(warp_knots)):
            if warp_knots[i] <= warp_knots[i - 1]:
                warp_knots[i] = warp_knots[i - 1] + 0.1

        # Create cubic spline mapping
        try:
            cs = CubicSpline(orig_knots, warp_knots)
            warped_indices = cs(np.arange(T))
        except ValueError:
            # Fallback: linear mapping if spline fails
            warped_indices = np.arange(T, dtype=np.float64)

        # Clamp warped indices to valid range
        warped_indices = np.clip(warped_indices, 0, T - 1)

        # Interpolate sequence at warped time points
        result = np.zeros_like(sequence)
        for f in range(F):
            result[:, f] = np.interp(
                warped_indices,
                np.arange(T),
                sequence[:, f],
            )

        return result

    def _apply_frame_dropout(self, sequence: np.ndarray) -> np.ndarray:
        """Randomly drop frames and replace with interpolated values.

        Simulates occasional tracking failures or occlusions by removing
        random frames and filling them via linear interpolation from
        neighboring frames.

        Args:
            sequence: Shape (T, F).

        Returns:
            Sequence with dropped frames interpolated.
        """
        T, F = sequence.shape
        max_ratio = self.frame_dropout_cfg.get("max_dropout_ratio", 0.15)

        num_drop = max(1, int(T * self.rng.uniform(0, max_ratio)))

        # Select random frames to drop (not first or last)
        droppable = np.arange(1, T - 1)
        if num_drop >= len(droppable):
            num_drop = max(1, len(droppable) // 2)

        drop_indices = self.rng.choice(droppable, size=num_drop, replace=False)

        result = sequence.copy()
        keep_mask = np.ones(T, dtype=bool)
        keep_mask[drop_indices] = False

        # Interpolate dropped frames
        keep_indices = np.where(keep_mask)[0]
        for f in range(F):
            result[drop_indices, f] = np.interp(
                drop_indices.astype(float),
                keep_indices.astype(float),
                sequence[keep_indices, f],
            )

        return result

    def _apply_speed_variation(self, sequence: np.ndarray) -> np.ndarray:
        """Apply uniform temporal speed variation.

        Resamples the sequence at a different temporal rate, then either
        crops or pads to maintain the original length. This simulates
        the signer performing the sign faster or slower.

        Args:
            sequence: Shape (T, F).

        Returns:
            Speed-varied sequence of the same shape.
        """
        T, F = sequence.shape
        min_speed = self.speed_cfg.get("min_speed", 0.8)
        max_speed = self.speed_cfg.get("max_speed", 1.2)

        speed_factor = self.rng.uniform(min_speed, max_speed)

        # New effective length
        new_length = int(T / speed_factor)
        new_length = max(2, new_length)  # At least 2 frames

        # Create new time indices
        new_indices = np.linspace(0, T - 1, new_length)

        # Interpolate at new time points
        resampled = np.zeros((new_length, F), dtype=np.float32)
        for f in range(F):
            resampled[:, f] = np.interp(
                new_indices, np.arange(T), sequence[:, f]
            )

        # Resample back to original length T
        if new_length != T:
            final_indices = np.linspace(0, new_length - 1, T)
            result = np.zeros_like(sequence)
            for f in range(F):
                result[:, f] = np.interp(
                    final_indices, np.arange(new_length), resampled[:, f]
                )
            return result

        return resampled

    # =========================================================================
    # Noise Augmentation
    # =========================================================================

    def _apply_noise(self, sequence: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to all coordinates.

        Simulates sensor noise and landmark detection jitter. Only adds
        noise to non-zero values (to preserve the "missing landmark"
        signal of zero-valued features).

        Args:
            sequence: Shape (T, F).

        Returns:
            Noisy sequence.
        """
        std = self.noise_cfg.get("std", 0.005)

        noise = self.rng.normal(0, std, size=sequence.shape).astype(np.float32)

        # Only add noise to non-zero features (preserve missing landmarks)
        nonzero_mask = sequence != 0
        result = sequence.copy()
        result[nonzero_mask] += noise[nonzero_mask]

        return result


def _to_dict(obj: object) -> dict:
    """Convert a DotDict or similar object to a plain dictionary.

    Args:
        obj: Object to convert (DotDict, dict, or any object with attributes).

    Returns:
        Plain dictionary.
    """
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def create_augmentation_fn(
    config: Optional[object] = None,
    seed: Optional[int] = None,
):
    """Create a numpy-compatible augmentation function for use with tf.data.

    Returns a callable that takes a sequence array and returns the
    augmented version. Useful for wrapping with tf.numpy_function.

    Args:
        config: Configuration object.
        seed: Random seed.

    Returns:
        Callable (np.ndarray) -> np.ndarray operating on single sequences.

    Example:
        >>> aug_fn = create_augmentation_fn(config)
        >>> dataset = dataset.map(
        ...     lambda x, y: (tf.numpy_function(aug_fn, [x], tf.float32), y)
        ... )
    """
    augmentor = LandmarkAugmentor(config=config, seed=seed)

    def _augment(sequence: np.ndarray) -> np.ndarray:
        return augmentor.augment(sequence)

    return _augment
