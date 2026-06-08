"""
Preprocessing utilities for MediaPipe Holistic landmark sequences.

Provides landmark normalization (root-relative, min-max, standard),
feature subset selection, missing value interpolation, and
coordinate extraction for the VOYA_VSL dataset.

The 1605-dimensional feature vector per frame consists of 535 landmarks × 3
coordinates (x, y, z) from MediaPipe Holistic:
    - Pose: 33 landmarks (indices 0–32)
    - Face: 468 landmarks (indices 33–500)  [estimated; verify with exploration]
    - Left Hand: 21 landmarks (indices 501–521) [estimated]
    - Right Hand: 21 landmarks (indices 522–534) [estimated]

Usage:
    from data.preprocessing import LandmarkPreprocessor
    from config import get_config

    cfg = get_config()
    preprocessor = LandmarkPreprocessor(cfg)
    processed = preprocessor.transform(sequences)  # (N, 60, 1605) -> (N, 60, F)
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LandmarkPreprocessor:
    """Preprocessor for MediaPipe Holistic landmark sequences.

    Handles:
        - Root-relative normalization (pose relative to shoulder center,
          hands relative to wrist)
        - Min-max normalization
        - Standard (z-score) normalization
        - Feature subset selection (full 1605 or pose+hands 225)
        - Missing/zero landmark interpolation

    Attributes:
        num_features: Number of input features per frame.
        seq_length: Expected sequence length (frames).
        normalization: Normalization strategy to use.
        feature_subset: Which features to keep ("full" or "pose_hands").
        coords_per_landmark: Number of coordinates per landmark (3 for x,y,z).
    """

    # MediaPipe Holistic landmark counts
    POSE_LANDMARKS = 33
    FACE_LANDMARKS = 468
    LEFT_HAND_LANDMARKS = 21
    RIGHT_HAND_LANDMARKS = 21
    TOTAL_LANDMARKS = 535  # Adjusted: actual may differ for 1605/3=535

    # Coordinate count
    COORDS = 3  # x, y, z

    # Important pose landmark indices (MediaPipe convention)
    LEFT_SHOULDER_IDX = 11
    RIGHT_SHOULDER_IDX = 12
    LEFT_HIP_IDX = 23
    RIGHT_HIP_IDX = 24
    NOSE_IDX = 0

    # Hand wrist is always index 0 within the hand landmark group
    WRIST_IDX = 0

    def __init__(self, config: Optional[object] = None, **kwargs) -> None:
        """Initialize the preprocessor.

        Args:
            config: Configuration DotDict object. If None, uses kwargs or defaults.
            **kwargs: Override individual settings:
                - num_features (int): Features per frame (default: 1605)
                - seq_length (int): Frames per sequence (default: 60)
                - normalization (str): "root_relative" | "min_max" | "standard"
                - feature_subset (str): "full" | "pose_hands"
                - interpolate_missing (bool): Interpolate zero landmarks
                - max_interpolation_gap (int): Max consecutive frames to interpolate
                - post_standardize (bool): Apply z-score after root-relative norm
        """
        if config is not None:
            self.num_features = getattr(config.data, "num_features", 1605)
            self.seq_length = getattr(config.data, "seq_length", 60)
            self.normalization = getattr(
                config.preprocessing, "normalization", "root_relative"
            )
            self.feature_subset = getattr(config.data, "feature_subset", "full")
            self.interpolate_missing = getattr(
                config.preprocessing, "interpolate_missing", True
            )
            self.max_interpolation_gap = getattr(
                config.preprocessing, "max_interpolation_gap", 5
            )
            self.post_standardize = getattr(
                config.preprocessing, "post_standardize", False
            )
        else:
            self.num_features = kwargs.get("num_features", 1605)
            self.seq_length = kwargs.get("seq_length", 60)
            self.normalization = kwargs.get("normalization", "root_relative")
            self.feature_subset = kwargs.get("feature_subset", "full")
            self.interpolate_missing = kwargs.get("interpolate_missing", True)
            self.max_interpolation_gap = kwargs.get("max_interpolation_gap", 5)
            self.post_standardize = kwargs.get("post_standardize", False)

        self.coords_per_landmark = self.COORDS

        # Compute landmark group boundaries (feature indices)
        self._compute_landmark_boundaries()

        # Statistics for standard normalization (fitted from training data)
        self._fit_mean: Optional[np.ndarray] = None
        self._fit_std: Optional[np.ndarray] = None

        logger.info(
            f"LandmarkPreprocessor initialized: "
            f"normalization={self.normalization}, "
            f"feature_subset={self.feature_subset}, "
            f"output_features={self.output_features}"
        )

    def _compute_landmark_boundaries(self) -> None:
        """Compute start/end feature indices for each landmark group.

        Sets attributes for pose, face, left_hand, and right_hand boundaries
        based on the assumed ordering in the feature vector.
        """
        c = self.coords_per_landmark

        # Assumed ordering: [pose, face, left_hand, right_hand]
        # Total: (33 + 468 + 21 + 21) × 3 = 543 × 3 = 1629
        # But we have 1605 = 535 × 3
        # Difference: 543 - 535 = 8 landmarks missing from somewhere
        # Most likely face has fewer: 535 - 33 - 21 - 21 = 460 face landmarks
        # Or the layout is different. We'll use 535 total and try to detect.

        self.n_landmarks = self.num_features // c
        remaining = self.num_features % c
        if remaining != 0:
            logger.warning(
                f"Feature count {self.num_features} is not divisible by "
                f"{c} coords. {remaining} features may be non-landmark data."
            )

        # Define boundaries assuming: [pose(33), face(460), lhand(21), rhand(21)]
        # This gives 535 × 3 = 1605 ✓
        self._face_landmarks = self.n_landmarks - self.POSE_LANDMARKS - \
            self.LEFT_HAND_LANDMARKS - self.RIGHT_HAND_LANDMARKS

        self.pose_start = 0
        self.pose_end = self.POSE_LANDMARKS * c  # 99

        self.face_start = self.pose_end  # 99
        self.face_end = self.face_start + self._face_landmarks * c  # 99 + 1380 = 1479

        self.left_hand_start = self.face_end  # 1479
        self.left_hand_end = self.left_hand_start + self.LEFT_HAND_LANDMARKS * c  # 1542

        self.right_hand_start = self.left_hand_end  # 1542
        self.right_hand_end = self.right_hand_start + self.RIGHT_HAND_LANDMARKS * c  # 1605

        # Compute output feature count
        if self.feature_subset == "pose_hands":
            self.output_features = (
                self.POSE_LANDMARKS + self.LEFT_HAND_LANDMARKS +
                self.RIGHT_HAND_LANDMARKS
            ) * c  # 75 × 3 = 225
        else:
            self.output_features = self.num_features

        logger.debug(
            f"Landmark boundaries: "
            f"pose=[{self.pose_start}:{self.pose_end}], "
            f"face=[{self.face_start}:{self.face_end}] ({self._face_landmarks} lm), "
            f"left_hand=[{self.left_hand_start}:{self.left_hand_end}], "
            f"right_hand=[{self.right_hand_start}:{self.right_hand_end}]"
        )

    @property
    def landmark_boundaries(self) -> dict[str, tuple[int, int]]:
        """Return landmark group boundaries as a dictionary.

        Returns:
            Dict mapping group names to (start_idx, end_idx) tuples.
        """
        return {
            "pose": (self.pose_start, self.pose_end),
            "face": (self.face_start, self.face_end),
            "left_hand": (self.left_hand_start, self.left_hand_end),
            "right_hand": (self.right_hand_start, self.right_hand_end),
        }

    def fit(self, sequences: np.ndarray) -> "LandmarkPreprocessor":
        """Fit the preprocessor on training data (compute normalization stats).

        Only needed for "standard" or "min_max" normalization. For
        "root_relative", this is a no-op but can be called safely.

        Args:
            sequences: Training sequences of shape (N, T, F).

        Returns:
            Self, for method chaining.
        """
        if self.normalization == "standard" or self.post_standardize:
            # Compute mean and std across all samples and frames
            flat = sequences.reshape(-1, sequences.shape[-1])
            self._fit_mean = np.mean(flat, axis=0).astype(np.float32)
            self._fit_std = np.std(flat, axis=0).astype(np.float32)
            # Avoid division by zero
            self._fit_std[self._fit_std < 1e-8] = 1.0
            logger.info("Fitted standard normalization statistics.")

        return self

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to sequences.

        Pipeline order:
            1. Interpolate missing landmarks (if enabled)
            2. Normalize coordinates
            3. Select feature subset
            4. Post-standardize (if enabled)

        Args:
            sequences: Input sequences of shape (N, T, F) or (T, F) for
                a single sequence. F should equal self.num_features.

        Returns:
            Preprocessed sequences. Shape depends on feature_subset:
            - "full": (N, T, 1605) or (T, 1605)
            - "pose_hands": (N, T, 225) or (T, 225)

        Raises:
            ValueError: If input shape is incompatible.
        """
        single_sample = sequences.ndim == 2
        if single_sample:
            sequences = sequences[np.newaxis, ...]

        if sequences.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features per frame, "
                f"got {sequences.shape[-1]}."
            )

        result = sequences.astype(np.float32).copy()

        # Step 1: Interpolate missing landmarks
        if self.interpolate_missing:
            result = self._interpolate_missing(result)

        # Step 2: Normalize
        if self.normalization == "root_relative":
            result = self._normalize_root_relative(result)
        elif self.normalization == "min_max":
            result = self._normalize_min_max(result)
        elif self.normalization == "standard":
            result = self._normalize_standard(result)
        else:
            logger.warning(
                f"Unknown normalization '{self.normalization}', skipping."
            )

        # Step 3: Feature subset selection
        if self.feature_subset == "pose_hands":
            result = self._select_pose_hands(result)

        # Step 4: Post-standardization
        if self.post_standardize and self._fit_mean is not None:
            # Apply to selected features
            if self.feature_subset == "pose_hands":
                # Need to select the relevant stats
                indices = self._get_pose_hands_indices()
                mean = self._fit_mean[indices]
                std = self._fit_std[indices]
            else:
                mean = self._fit_mean
                std = self._fit_std
            result = (result - mean) / std

        if single_sample:
            result = result[0]

        return result

    def fit_transform(self, sequences: np.ndarray) -> np.ndarray:
        """Fit on data and then transform it.

        Args:
            sequences: Training sequences of shape (N, T, F).

        Returns:
            Preprocessed sequences.
        """
        return self.fit(sequences).transform(sequences)

    def _interpolate_missing(self, sequences: np.ndarray) -> np.ndarray:
        """Interpolate missing (zero) landmarks via linear interpolation.

        For each landmark coordinate, if it's zero for a span of frames
        shorter than max_interpolation_gap, linearly interpolate between
        the nearest non-zero neighbors.

        Args:
            sequences: Shape (N, T, F).

        Returns:
            Sequences with interpolated missing values.
        """
        n_samples, n_frames, n_features = sequences.shape

        for i in range(n_samples):
            for f in range(n_features):
                col = sequences[i, :, f]
                zero_mask = col == 0.0

                if not np.any(zero_mask) or np.all(zero_mask):
                    continue

                # Find non-zero indices
                nonzero_idx = np.where(~zero_mask)[0]

                if len(nonzero_idx) < 2:
                    continue

                # Find zero spans and interpolate short ones
                zero_spans = _find_zero_spans(zero_mask)

                for start, end in zero_spans:
                    span_length = end - start
                    if span_length > self.max_interpolation_gap:
                        continue

                    # Find nearest non-zero neighbors
                    left_idx = start - 1 if start > 0 else None
                    right_idx = end if end < n_frames else None

                    if left_idx is not None and right_idx is not None:
                        # Linear interpolation
                        left_val = col[left_idx]
                        right_val = col[right_idx]
                        interp_values = np.linspace(
                            left_val, right_val, span_length + 2
                        )[1:-1]
                        sequences[i, start:end, f] = interp_values
                    elif left_idx is not None:
                        # Extend from left
                        sequences[i, start:end, f] = col[left_idx]
                    elif right_idx is not None:
                        # Extend from right
                        sequences[i, start:end, f] = col[right_idx]

        return sequences

    def _normalize_root_relative(self, sequences: np.ndarray) -> np.ndarray:
        """Normalize landmarks relative to root joints.

        - Pose landmarks: relative to shoulder center (midpoint of L/R shoulders)
        - Left hand landmarks: relative to left wrist
        - Right hand landmarks: relative to right wrist
        - Face landmarks: relative to nose or shoulder center

        This makes the representation translation-invariant.

        Args:
            sequences: Shape (N, T, F).

        Returns:
            Root-relative normalized sequences.
        """
        result = sequences.copy()
        c = self.coords_per_landmark

        # === Pose normalization (relative to shoulder center) ===
        # Get shoulder center coordinates
        l_shoulder_start = self.pose_start + self.LEFT_SHOULDER_IDX * c
        r_shoulder_start = self.pose_start + self.RIGHT_SHOULDER_IDX * c

        l_shoulder = sequences[:, :, l_shoulder_start:l_shoulder_start + c]
        r_shoulder = sequences[:, :, r_shoulder_start:r_shoulder_start + c]

        # Shoulder center: midpoint of left and right shoulders
        shoulder_center = (l_shoulder + r_shoulder) / 2.0  # (N, T, 3)

        # Compute shoulder width for scale normalization
        shoulder_dist = np.linalg.norm(
            l_shoulder - r_shoulder, axis=-1, keepdims=True
        )  # (N, T, 1)
        # Avoid division by zero
        shoulder_dist = np.maximum(shoulder_dist, 1e-6)

        # Normalize pose landmarks
        for lm_idx in range(self.POSE_LANDMARKS):
            feat_start = self.pose_start + lm_idx * c
            feat_end = feat_start + c
            result[:, :, feat_start:feat_end] = (
                (sequences[:, :, feat_start:feat_end] - shoulder_center)
                / shoulder_dist
            )

        # === Face normalization (relative to shoulder center, scaled) ===
        for lm_idx in range(self._face_landmarks):
            feat_start = self.face_start + lm_idx * c
            feat_end = feat_start + c
            result[:, :, feat_start:feat_end] = (
                (sequences[:, :, feat_start:feat_end] - shoulder_center)
                / shoulder_dist
            )

        # === Left hand normalization (relative to left wrist) ===
        lh_wrist_start = self.left_hand_start + self.WRIST_IDX * c
        lh_wrist = sequences[:, :, lh_wrist_start:lh_wrist_start + c]

        # Scale by hand size (distance from wrist to middle finger MCP, index 9)
        if self.LEFT_HAND_LANDMARKS > 9:
            lh_mcp_start = self.left_hand_start + 9 * c
            lh_mcp = sequences[:, :, lh_mcp_start:lh_mcp_start + c]
            lh_scale = np.linalg.norm(
                lh_mcp - lh_wrist, axis=-1, keepdims=True
            )
            lh_scale = np.maximum(lh_scale, 1e-6)
        else:
            lh_scale = shoulder_dist

        for lm_idx in range(self.LEFT_HAND_LANDMARKS):
            feat_start = self.left_hand_start + lm_idx * c
            feat_end = feat_start + c
            result[:, :, feat_start:feat_end] = (
                (sequences[:, :, feat_start:feat_end] - lh_wrist) / lh_scale
            )

        # === Right hand normalization (relative to right wrist) ===
        rh_wrist_start = self.right_hand_start + self.WRIST_IDX * c
        rh_wrist = sequences[:, :, rh_wrist_start:rh_wrist_start + c]

        if self.RIGHT_HAND_LANDMARKS > 9:
            rh_mcp_start = self.right_hand_start + 9 * c
            rh_mcp = sequences[:, :, rh_mcp_start:rh_mcp_start + c]
            rh_scale = np.linalg.norm(
                rh_mcp - rh_wrist, axis=-1, keepdims=True
            )
            rh_scale = np.maximum(rh_scale, 1e-6)
        else:
            rh_scale = shoulder_dist

        for lm_idx in range(self.RIGHT_HAND_LANDMARKS):
            feat_start = self.right_hand_start + lm_idx * c
            feat_end = feat_start + c
            result[:, :, feat_start:feat_end] = (
                (sequences[:, :, feat_start:feat_end] - rh_wrist) / rh_scale
            )

        return result

    def _normalize_min_max(self, sequences: np.ndarray) -> np.ndarray:
        """Per-sample min-max normalization to [0, 1].

        Each sample is independently normalized so that its minimum value
        maps to 0 and maximum to 1.

        Args:
            sequences: Shape (N, T, F).

        Returns:
            Min-max normalized sequences.
        """
        # Per-sample normalization
        mins = sequences.min(axis=(1, 2), keepdims=True)
        maxs = sequences.max(axis=(1, 2), keepdims=True)
        ranges = maxs - mins
        ranges[ranges < 1e-8] = 1.0

        return (sequences - mins) / ranges

    def _normalize_standard(self, sequences: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using fitted statistics.

        Uses mean and std computed during fit(). If not fitted,
        computes per-sample statistics.

        Args:
            sequences: Shape (N, T, F).

        Returns:
            Standardized sequences.
        """
        if self._fit_mean is not None and self._fit_std is not None:
            return (sequences - self._fit_mean) / self._fit_std

        # Fallback: per-sample standardization
        logger.warning(
            "Standard normalization called without fit(). "
            "Using per-sample statistics."
        )
        mean = sequences.mean(axis=(1, 2), keepdims=True)
        std = sequences.std(axis=(1, 2), keepdims=True)
        std[std < 1e-8] = 1.0
        return (sequences - mean) / std

    def _select_pose_hands(self, sequences: np.ndarray) -> np.ndarray:
        """Select only pose and hand landmarks, dropping face landmarks.

        This reduces features from 1605 to 225 (75 landmarks × 3 coords),
        focusing on the most discriminative body parts for sign language.

        Args:
            sequences: Shape (N, T, 1605).

        Returns:
            Shape (N, T, 225) with [pose(99) + left_hand(63) + right_hand(63)].
        """
        indices = self._get_pose_hands_indices()
        return sequences[:, :, indices]

    def _get_pose_hands_indices(self) -> np.ndarray:
        """Get feature indices for pose + hands subset.

        Returns:
            1-D array of integer indices into the feature dimension.
        """
        pose_indices = np.arange(self.pose_start, self.pose_end)
        lh_indices = np.arange(self.left_hand_start, self.left_hand_end)
        rh_indices = np.arange(self.right_hand_start, self.right_hand_end)
        return np.concatenate([pose_indices, lh_indices, rh_indices])

    def get_feature_info(self) -> dict:
        """Return information about the current feature configuration.

        Returns:
            Dictionary with feature counts and descriptions.
        """
        return {
            "input_features": self.num_features,
            "output_features": self.output_features,
            "feature_subset": self.feature_subset,
            "normalization": self.normalization,
            "landmark_groups": {
                "pose": {
                    "landmarks": self.POSE_LANDMARKS,
                    "features": self.pose_end - self.pose_start,
                    "range": (self.pose_start, self.pose_end),
                },
                "face": {
                    "landmarks": self._face_landmarks,
                    "features": self.face_end - self.face_start,
                    "range": (self.face_start, self.face_end),
                },
                "left_hand": {
                    "landmarks": self.LEFT_HAND_LANDMARKS,
                    "features": self.left_hand_end - self.left_hand_start,
                    "range": (self.left_hand_start, self.left_hand_end),
                },
                "right_hand": {
                    "landmarks": self.RIGHT_HAND_LANDMARKS,
                    "features": self.right_hand_end - self.right_hand_start,
                    "range": (self.right_hand_start, self.right_hand_end),
                },
            },
        }


def _find_zero_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous spans of True values in a boolean mask.

    Args:
        mask: 1-D boolean array.

    Returns:
        List of (start, end) tuples where mask[start:end] is all True.
    """
    spans = []
    in_span = False
    start = 0

    for i, val in enumerate(mask):
        if val and not in_span:
            start = i
            in_span = True
        elif not val and in_span:
            spans.append((start, i))
            in_span = False

    if in_span:
        spans.append((start, len(mask)))

    return spans


def extract_landmark_coordinates(
    frame: np.ndarray,
    landmark_group: Literal["pose", "face", "left_hand", "right_hand"],
    preprocessor: Optional[LandmarkPreprocessor] = None,
) -> np.ndarray:
    """Extract x, y, z coordinates for a specific landmark group from a frame.

    Convenience function for visualization and debugging.

    Args:
        frame: Single frame features of shape (F,) where F = num_features.
        landmark_group: Which landmark group to extract.
        preprocessor: Optional preprocessor to get boundaries from.
            If None, uses default boundaries.

    Returns:
        Array of shape (num_landmarks, 3) with x, y, z coordinates.
    """
    if preprocessor is None:
        preprocessor = LandmarkPreprocessor()

    boundaries = preprocessor.landmark_boundaries
    start, end = boundaries[landmark_group]

    group_features = frame[start:end]
    n_landmarks = len(group_features) // 3
    return group_features.reshape(n_landmarks, 3)
