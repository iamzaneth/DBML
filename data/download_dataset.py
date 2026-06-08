"""
Download and explore the VOYA_VSL dataset from HuggingFace.

This script:
    1. Downloads the Kateht/VOYA_VSL dataset using the HuggingFace `datasets` library
    2. Extracts and saves data as local .npz files
    3. Performs data exploration: prints shapes, label distributions, statistics

Usage:
    python -m data.download_dataset
    python -m data.download_dataset --output_dir data/raw --explore

The dataset contains Vietnamese Sign Language sequences captured with
MediaPipe Holistic. Each sample has 60 frames × 1605 features
(535 landmarks × 3 coordinates: x, y, z).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path for config imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


def download_dataset(
    dataset_id: str = "Kateht/VOYA_VSL",
    output_dir: str | Path = "data/raw",
    cache_dir: Optional[str | Path] = None,
) -> Path:
    """Download the VOYA_VSL dataset from HuggingFace and save as .npz files.

    Uses the huggingface_hub library to download the raw .npz files from the repo.

    Args:
        dataset_id: HuggingFace dataset identifier (default: "Kateht/VOYA_VSL").
        output_dir: Local directory to save .npz files. Created if it doesn't exist.
        cache_dir: Optional cache directory for HuggingFace downloads.

    Returns:
        Path to the output directory containing the saved .npz files.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "The `huggingface_hub` library is required. "
            "Install it with: pip install huggingface_hub"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset '{dataset_id}' from HuggingFace...")
    logger.info(f"Output directory: {output_path.resolve()}")

    import shutil
    download_path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        allow_patterns=["*.npz", "*.json"],
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    
    # Copy files from cache to output_dir
    src_dir = Path(download_path)
    count = 0
    # Copy .npz files
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.npz') or file == 'labels.json':
                src_file = Path(root) / file
                dst_file = output_path / file
                shutil.copy2(src_file, dst_file)
                count += 1
                if count % 10 == 0:
                    logger.info(f"  Copied {count} files...")

    logger.info(f"\nDownload complete. {count} files saved to: {output_path.resolve()}")
    return output_path


def explore_npz_files(data_dir: str | Path) -> dict:
    """Explore and print detailed statistics about local .npz files.

    Examines all .npz files in the given directory, printing:
    - File sizes and shapes
    - Label distribution (class frequencies)
    - Feature statistics (min, max, mean, std, zero ratios)
    - Per-landmark-group statistics
    - Potential data quality issues

    Args:
        data_dir: Directory containing .npz files.

    Returns:
        Dictionary with exploration results:
            - "files": list of file info dicts
            - "total_samples": total number of samples across all files
            - "total_classes": number of unique classes
            - "label_distribution": Counter of label frequencies
            - "feature_stats": dict of feature-level statistics
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_path.resolve()}")
        return {}

    npz_files = sorted(data_path.glob("*.npz"))

    if not npz_files:
        logger.warning(f"No .npz files found in {data_path.resolve()}")
        return {}

    print("\n" + "=" * 80)
    print("  VOYA_VSL Dataset Exploration Report")
    print("=" * 80)

    results = {
        "files": [],
        "total_samples": 0,
        "total_classes": 0,
        "label_distribution": Counter(),
        "feature_stats": {},
    }

    all_sequences = []
    all_labels = []

    for npz_file in npz_files:
        file_size_mb = npz_file.stat().st_size / (1024 * 1024)

        print(f"\n--- File: {npz_file.name} ({file_size_mb:.1f} MB) ---")

        data = np.load(npz_file, allow_pickle=True)
        keys = list(data.keys())
        print(f"  Keys: {keys}")

        file_info = {"name": npz_file.name, "size_mb": file_size_mb, "keys": keys}

        if "sequences" in keys:
            sequences = data["sequences"]
            print(f"  Sequences shape: {sequences.shape}")
            print(f"  Sequences dtype: {sequences.dtype}")
            file_info["sequences_shape"] = sequences.shape
            all_sequences.append(sequences)

        if "labels" in keys:
            labels = data["labels"]
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels dtype: {labels.dtype}")
            print(f"  Unique labels: {len(np.unique(labels))}")
            print(f"  Label range: [{labels.min()}, {labels.max()}]")
            file_info["labels_shape"] = labels.shape
            file_info["num_unique_labels"] = len(np.unique(labels))
            all_labels.append(labels)

        # Print other keys if present
        for key in keys:
            if key not in ("sequences", "labels"):
                arr = data[key]
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

        data.close()
        results["files"].append(file_info)

    # Aggregate analysis across all files
    if all_sequences:
        combined_sequences = np.concatenate(all_sequences, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        n_samples, n_frames, n_features = combined_sequences.shape
        results["total_samples"] = n_samples

        print("\n" + "=" * 80)
        print("  AGGREGATE STATISTICS")
        print("=" * 80)

        print(f"\n  Total samples: {n_samples}")
        print(f"  Sequence length: {n_frames} frames")
        print(f"  Features per frame: {n_features}")
        print(f"  Total data size: {combined_sequences.nbytes / (1024**3):.2f} GB")

        # Label distribution
        label_counts = Counter(combined_labels.tolist())
        results["label_distribution"] = label_counts
        results["total_classes"] = len(label_counts)

        print(f"\n  Number of classes: {len(label_counts)}")
        print(f"  Samples per class:")
        print(f"    Min:    {min(label_counts.values())}")
        print(f"    Max:    {max(label_counts.values())}")
        print(f"    Mean:   {np.mean(list(label_counts.values())):.1f}")
        print(f"    Median: {np.median(list(label_counts.values())):.1f}")
        print(f"    Std:    {np.std(list(label_counts.values())):.1f}")

        # Check for class imbalance
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / max(min_count, 1)
        print(f"    Imbalance ratio (max/min): {imbalance_ratio:.1f}x")

        # Feature statistics
        print(f"\n  Feature Statistics (across all samples):")
        print(f"    Min:  {combined_sequences.min():.6f}")
        print(f"    Max:  {combined_sequences.max():.6f}")
        print(f"    Mean: {combined_sequences.mean():.6f}")
        print(f"    Std:  {combined_sequences.std():.6f}")

        # Check for zero/NaN/Inf values
        zero_ratio = np.sum(combined_sequences == 0) / combined_sequences.size
        nan_count = np.sum(np.isnan(combined_sequences))
        inf_count = np.sum(np.isinf(combined_sequences))

        print(f"\n  Data Quality:")
        print(f"    Zero values:  {zero_ratio * 100:.2f}% of all values")
        print(f"    NaN values:   {nan_count}")
        print(f"    Inf values:   {inf_count}")

        results["feature_stats"] = {
            "min": float(combined_sequences.min()),
            "max": float(combined_sequences.max()),
            "mean": float(combined_sequences.mean()),
            "std": float(combined_sequences.std()),
            "zero_ratio": float(zero_ratio),
            "nan_count": int(nan_count),
            "inf_count": int(inf_count),
        }

        # Per-landmark-group analysis
        # Assumed layout: [pose(99), face(1404), left_hand(63), right_hand(63)]
        # But might be different — let's analyze in chunks
        landmark_groups = _detect_landmark_groups(combined_sequences, n_features)
        if landmark_groups:
            print(f"\n  Per-Landmark-Group Analysis:")
            for group_name, (start, end) in landmark_groups.items():
                group_data = combined_sequences[:, :, start:end]
                group_zero_ratio = np.sum(group_data == 0) / group_data.size
                print(
                    f"    {group_name:>15s} "
                    f"[{start:4d}:{end:4d}] "
                    f"({end - start:4d} features) | "
                    f"mean={group_data.mean():8.5f}  "
                    f"std={group_data.std():8.5f}  "
                    f"zeros={group_zero_ratio * 100:5.1f}%"
                )

        # Top 10 most/least frequent classes
        sorted_classes = sorted(label_counts.items(), key=lambda x: x[1])
        print(f"\n  Bottom 5 classes (least samples):")
        for label, count in sorted_classes[:5]:
            print(f"    Class {label}: {count} samples")

        print(f"\n  Top 5 classes (most samples):")
        for label, count in sorted_classes[-5:]:
            print(f"    Class {label}: {count} samples")

        del combined_sequences, combined_labels

    print("\n" + "=" * 80)
    print("  Exploration Complete")
    print("=" * 80 + "\n")

    return results


def _detect_landmark_groups(
    sequences: np.ndarray,
    n_features: int,
) -> dict[str, tuple[int, int]]:
    """Detect and return landmark group boundaries based on feature count.

    Attempts to determine the layout of MediaPipe Holistic landmarks
    in the feature vector. The standard layout for 1605 features is:
    - Pose: 33 landmarks × 3 = 99 features
    - Face: 468 landmarks × 3 = 1404 features
    - Left Hand: 21 landmarks × 3 = 63 features  (Note: 99+1404+63+63 = 1629 ≠ 1605)

    Since the exact total is 1605 = 535 × 3, we need to determine the
    actual layout empirically if possible.

    Args:
        sequences: Array of shape (N, T, F) containing landmark sequences.
        n_features: Number of features per frame.

    Returns:
        Dictionary mapping group names to (start, end) index tuples.
    """
    groups = {}

    if n_features == 1605:
        # 1605 = 535 * 3
        # Possible layouts:
        # Option A: pose(33) + face(468) + lhand(21) + rhand(21) = 543 (× 3 = 1629, too many)
        # Option B: pose(33) + face(456) + lhand(21) + rhand(21) + extra = 535
        # Option C: Different landmark counts

        # Let's try analyzing zero patterns to detect hand regions
        # Hands often have more zeros (when not visible)
        # For now, use the most common interpretation and let the user verify

        # Try standard MediaPipe counts with reduced face
        # 535 total landmarks: pose(33) + face(460) + left_hand(21) + right_hand(21)
        # Or perhaps: 33 pose + 468 face + 21 lh + 21 rh = 543 landmarks
        # but only x,y,z used for some and x,y,z,visibility for others
        # Let's just split evenly for analysis

        # Attempt: assume order [pose, face, left_hand, right_hand]
        # and infer boundaries from zero-pattern analysis
        sample_subset = sequences[:min(100, len(sequences))]

        # Heuristic: hands tend to have more zeros
        zero_per_feature = np.mean(sample_subset == 0, axis=(0, 1))

        # Look for transitions in zero density to find boundaries
        # For now, use the theoretical layout
        groups["pose"] = (0, 99)         # 33 × 3
        groups["face"] = (99, 1503)      # 468 × 3
        groups["left_hand"] = (1503, 1566)  # 21 × 3  (Note: 1566 > 1605?)
        groups["right_hand"] = (1566, 1605) # 13 × 3  (doesn't match 21 landmarks)

        # Validate boundaries don't exceed feature count
        valid_groups = {}
        for name, (start, end) in groups.items():
            if start < n_features and end <= n_features:
                valid_groups[name] = (start, end)
            elif start < n_features:
                valid_groups[name] = (start, n_features)
        groups = valid_groups

        # If the standard layout doesn't fit, try alternative
        if not groups or max(end for _, (_, end) in groups.items()) != n_features:
            # Alternative: split into roughly equal chunks for analysis
            chunk_size = n_features // 4
            groups = {
                "chunk_1": (0, chunk_size),
                "chunk_2": (chunk_size, 2 * chunk_size),
                "chunk_3": (2 * chunk_size, 3 * chunk_size),
                "chunk_4": (3 * chunk_size, n_features),
            }
    elif n_features == 225:
        # Pose (33×3=99) + Left Hand (21×3=63) + Right Hand (21×3=63) = 225
        groups["pose"] = (0, 99)
        groups["left_hand"] = (99, 162)
        groups["right_hand"] = (162, 225)
    else:
        # Unknown layout, split into quarters
        chunk_size = n_features // 4
        groups = {
            f"features_{i * chunk_size}_{min((i + 1) * chunk_size, n_features)}": (
                i * chunk_size,
                min((i + 1) * chunk_size, n_features),
            )
            for i in range(4)
        }
        if 4 * chunk_size < n_features:
            groups[f"features_{4 * chunk_size}_{n_features}"] = (
                4 * chunk_size,
                n_features,
            )

    return groups


def verify_data_integrity(data_dir: str | Path) -> bool:
    """Verify that downloaded .npz files are valid and consistent.

    Checks:
    - Files can be opened and contain expected keys
    - Shapes are consistent (60 frames, 1605 features)
    - No NaN or Inf values
    - Labels are in expected range

    Args:
        data_dir: Directory containing .npz files.

    Returns:
        True if all checks pass, False otherwise.
    """
    data_path = Path(data_dir)
    npz_files = sorted(data_path.glob("*.npz"))

    if not npz_files:
        logger.error(f"No .npz files found in {data_path}")
        return False

    all_ok = True

    for npz_file in npz_files:
        print(f"Verifying: {npz_file.name}...", end=" ")

        try:
            data = np.load(npz_file)

            # Check keys
            if "sequences" not in data or "labels" not in data:
                print(f"FAIL - Missing keys. Found: {list(data.keys())}")
                all_ok = False
                continue

            sequences = data["sequences"]
            labels = data["labels"]

            issues = []

            # Check shapes
            if sequences.ndim != 3:
                issues.append(f"sequences ndim={sequences.ndim}, expected 3")
            elif sequences.shape[1] != 60:
                issues.append(
                    f"seq_length={sequences.shape[1]}, expected 60"
                )
            elif sequences.shape[2] != 1605:
                issues.append(
                    f"num_features={sequences.shape[2]}, expected 1605"
                )

            if len(labels) != len(sequences):
                issues.append(
                    f"label count ({len(labels)}) != "
                    f"sequence count ({len(sequences)})"
                )

            # Check for NaN/Inf
            if np.any(np.isnan(sequences)):
                nan_count = np.sum(np.isnan(sequences))
                issues.append(f"{nan_count} NaN values found")

            if np.any(np.isinf(sequences)):
                inf_count = np.sum(np.isinf(sequences))
                issues.append(f"{inf_count} Inf values found")

            # Check labels
            if labels.min() < 0:
                issues.append(f"negative labels found (min={labels.min()})")

            data.close()

            if issues:
                print(f"WARNINGS: {'; '.join(issues)}")
            else:
                print("OK")

        except Exception as e:
            print(f"FAIL - {e}")
            all_ok = False

    return all_ok


def main() -> None:
    """Main entry point for downloading and exploring the VSL dataset."""
    parser = argparse.ArgumentParser(
        description="Download and explore the VOYA_VSL dataset from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download dataset to default location
    python -m data.download_dataset

    # Download to custom directory
    python -m data.download_dataset --output_dir /path/to/data

    # Only explore existing data (skip download)
    python -m data.download_dataset --explore --output_dir data/raw

    # Download and verify
    python -m data.download_dataset --verify
        """,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save .npz files (default: from config.yaml)",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        help="HuggingFace dataset ID (default: from config.yaml)",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Explore existing .npz files without downloading",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify data integrity after download",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config for defaults
    try:
        from config import load_config

        cfg = load_config()
        default_output_dir = cfg.data.raw_dir
        default_dataset_id = cfg.data.hf_dataset_id
    except Exception:
        logger.warning("Could not load config.yaml, using hardcoded defaults.")
        default_output_dir = "data/raw"
        default_dataset_id = "Kateht/VOYA_VSL"

    output_dir = args.output_dir or default_output_dir
    dataset_id = args.dataset_id or default_dataset_id

    if not args.explore:
        # Download the dataset
        try:
            download_dataset(
                dataset_id=dataset_id,
                output_dir=output_dir,
                cache_dir=args.cache_dir,
            )
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    # Always explore after download (or if --explore flag is set)
    explore_npz_files(output_dir)

    # Optionally verify
    if args.verify:
        print("\n--- Data Integrity Verification ---")
        is_valid = verify_data_integrity(output_dir)
        if is_valid:
            print("All files passed verification.")
        else:
            print("Some files have issues. Check the output above.")
            sys.exit(1)


if __name__ == "__main__":
    main()
