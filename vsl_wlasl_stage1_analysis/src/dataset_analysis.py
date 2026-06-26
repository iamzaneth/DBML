from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from video_utils import VideoMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Aggregate overview statistics for a single dataset.
    """

    dataset: str
    num_classes: int
    total_videos: int
    num_unreadable: int
    mean_samples_per_class: float
    median_samples_per_class: float
    min_samples_per_class: int
    max_samples_per_class: int
    num_signers: int = 0


def build_video_dataframe(records: list[VideoMetadata]) -> pd.DataFrame:
    """Converts a list of :class:`VideoMetadata` into a tidy DataFrame.
    """
    if not records:
        raise ValueError("Cannot build a DataFrame from an empty record list.")

    df = pd.DataFrame(
        {
            "dataset": [r.dataset for r in records],
            "class_name": [r.class_name for r in records],
            "file_path": [str(r.file_path) for r in records],
            "duration_seconds": [r.duration_seconds for r in records],
            "frame_count": [r.frame_count for r in records],
            "fps": [r.fps for r in records],
            "width": [r.width for r in records],
            "height": [r.height for r in records],
            "codec": [r.codec for r in records],
            "readable": [r.readable for r in records],
            "regional_variant": [r.regional_variant for r in records],
            "signer_id": [r.signer_id for r in records],
        }
    )
    return df


def compute_class_distribution(video_df: pd.DataFrame) -> pd.Series:
    """Computes the number of video samples per class.
    """
    if video_df.empty:
        raise ValueError("video_df is empty; cannot compute class distribution.")

    distinct_datasets = video_df["dataset"].unique()
    if distinct_datasets.size > 1:
        raise ValueError(
            "compute_class_distribution received rows from multiple datasets "
            f"({list(distinct_datasets)}); pass a single-dataset slice to avoid "
            "silently merging same-named classes across datasets."
        )

    distribution = video_df.groupby("class_name").size().sort_index()
    distribution.name = "sample_count"
    return distribution


def summarize_dataset(dataset_name: str, video_df: pd.DataFrame) -> DatasetSummary:
    """Computes overview statistics (Section 4.2 "Thong ke tong quan").
    """
    if video_df.empty:
        raise ValueError(f"No video records for dataset '{dataset_name}'.")

    distribution = compute_class_distribution(video_df)
    num_unreadable = int((~video_df["readable"]).sum())
    num_signers = int(video_df["signer_id"].dropna().nunique())

    summary = DatasetSummary(
        dataset=dataset_name,
        num_classes=int(distribution.shape[0]),
        total_videos=int(distribution.sum()),
        num_unreadable=num_unreadable,
        mean_samples_per_class=float(distribution.mean()),
        median_samples_per_class=float(distribution.median()),
        min_samples_per_class=int(distribution.min()),
        max_samples_per_class=int(distribution.max()),
        num_signers=num_signers,
    )
    logger.info(
        "Dataset '%s' summary: %d classes, %d videos, %d unreadable, %d signers",
        summary.dataset,
        summary.num_classes,
        summary.total_videos,
        summary.num_unreadable,
        summary.num_signers,
    )
    return summary


_NUMERIC_VIDEO_COLUMNS: tuple[str, ...] = ("duration_seconds", "frame_count", "fps")


def compute_video_statistics(video_df: pd.DataFrame) -> pd.DataFrame:
    """Computes descriptive statistics for video-level numeric attributes.
    """
    readable_df = video_df[video_df["readable"]]
    if readable_df.empty:
        raise ValueError("No readable video rows available to compute statistics.")

    attributes = ["duration_seconds", "frame_count", "fps", "width", "height"]
    rows = []
    for attribute in attributes:
        series = pd.to_numeric(readable_df[attribute], errors="coerce").dropna()
        if series.empty:
            logger.warning("No valid values for attribute '%s'", attribute)
            rows.append(
                {
                    "attribute": attribute,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "median": np.nan,
                    "n": 0,
                }
            )
            continue
        rows.append(
            {
                "attribute": attribute,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "n": int(series.shape[0]),
            }
        )

    stats_df = pd.DataFrame(rows).set_index("attribute")
    return stats_df


def compute_resolution_frequency(
    video_df: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Computes the frequency of (width, height) resolution pairs.
    """
    readable_df = video_df[video_df["readable"]].dropna(subset=["width", "height"])
    if readable_df.empty:
        logger.warning("No readable rows with valid resolution found.")
        return pd.DataFrame(columns=["resolution", "count", "proportion"])

    resolutions = (
        readable_df["width"].astype(int).astype(str)
        + "x"
        + readable_df["height"].astype(int).astype(str)
    )
    counts = resolutions.value_counts().head(top_n)
    total = resolutions.shape[0]

    result = pd.DataFrame(
        {
            "resolution": counts.index,
            "count": counts.values,
            "proportion": counts.values / total,
        }
    )
    return result


def compute_codec_frequency(video_df: pd.DataFrame) -> pd.DataFrame:
    """Computes the frequency of video codecs.
    """
    readable_df = video_df[video_df["readable"]]
    if readable_df.empty:
        logger.warning("No readable rows available to compute codec frequency.")
        return pd.DataFrame(columns=["codec", "count", "proportion"])

    codecs = readable_df["codec"].fillna("unknown").replace("", "unknown")
    counts = Counter(codecs)
    total = sum(counts.values())

    result = (
        pd.DataFrame(
            {
                "codec": list(counts.keys()),
                "count": list(counts.values()),
            }
        )
        .assign(proportion=lambda d: d["count"] / total)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return result


def compute_regional_variant_distribution(video_df: pd.DataFrame) -> pd.DataFrame:
    """Computes the distribution of VSL regional-dialect variants.
    """
    variants = video_df["regional_variant"].fillna("(none)")
    counts = variants.value_counts()
    total = counts.sum()

    result = pd.DataFrame(
        {
            "regional_variant": counts.index,
            "count": counts.values,
            "proportion": counts.values / total,
        }
    )
    return result


def compute_signer_distribution(video_df: pd.DataFrame) -> pd.DataFrame:
    """Computes the per-signer sample count for datasets with signer IDs.
    """
    known_signers = video_df["signer_id"].dropna()
    if known_signers.empty:
        logger.info(
            "No signer_id data available in this dataset; signer distribution is empty."
        )
        return pd.DataFrame(columns=["signer_id", "sample_count"])

    counts = known_signers.value_counts()
    result = pd.DataFrame(
        {
            "signer_id": counts.index,
            "sample_count": counts.values,
        }
    )
    return result


def summarize_signer_diversity(video_df: pd.DataFrame) -> dict:
    """Summarizes signer-count statistics for datasets with signer IDs.
    """
    signer_dist = compute_signer_distribution(video_df)
    if signer_dist.empty:
        return {
            "num_signers": 0,
            "mean_samples_per_signer": np.nan,
            "min_samples_per_signer": np.nan,
            "max_samples_per_signer": np.nan,
        }

    counts = signer_dist["sample_count"]
    return {
        "num_signers": int(signer_dist.shape[0]),
        "mean_samples_per_signer": float(counts.mean()),
        "min_samples_per_signer": int(counts.min()),
        "max_samples_per_signer": int(counts.max()),
    }


def generate_dataset_comparison_table(
    summaries: list[DatasetSummary], video_stats_by_dataset: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Builds a side-by-side comparison table across datasets.
    """
    columns: dict[str, dict[str, float]] = {}
    for summary in summaries:
        stats = video_stats_by_dataset.get(summary.dataset)
        mean_duration = (
            float(stats.loc["duration_seconds", "mean"])
            if stats is not None and "duration_seconds" in stats.index
            else np.nan
        )
        mean_fps = (
            float(stats.loc["fps", "mean"])
            if stats is not None and "fps" in stats.index
            else np.nan
        )
        columns[summary.dataset] = {
            "num_classes": summary.num_classes,
            "total_videos": summary.total_videos,
            "mean_samples_per_class": summary.mean_samples_per_class,
            "median_samples_per_class": summary.median_samples_per_class,
            "min_samples_per_class": summary.min_samples_per_class,
            "max_samples_per_class": summary.max_samples_per_class,
            "mean_duration_seconds": mean_duration,
            "mean_fps": mean_fps,
        }

    comparison_df = pd.DataFrame(columns)
    return comparison_df


def generate_academic_report(
    summaries: list[DatasetSummary], comparison_df: pd.DataFrame
) -> str:
    """Generates an automatic academic-style narrative report (Section 4.2).
    """
    if len(summaries) < 2:
        raise ValueError("At least two datasets are required for a comparison report.")

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("DATASET SCALE AND STRUCTURE ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    for summary in summaries:
        lines.append(f"[{summary.dataset}]")
        lines.append(f"  Number of classes (glosses): {summary.num_classes}")
        lines.append(f"  Total videos: {summary.total_videos}")
        lines.append(f"  Unreadable videos: {summary.num_unreadable}")
        lines.append(f"  Mean samples per class: {summary.mean_samples_per_class:.2f}")
        lines.append(
            f"  Median samples per class: {summary.median_samples_per_class:.2f}"
        )
        lines.append(f"  Min samples per class: {summary.min_samples_per_class}")
        lines.append(f"  Max samples per class: {summary.max_samples_per_class}")
        lines.append("")

    sorted_by_classes = sorted(summaries, key=lambda s: s.num_classes, reverse=True)
    larger, smaller = sorted_by_classes[0], sorted_by_classes[-1]

    lines.append("Narrative summary:")
    lines.append("-" * 80)

    for summary in summaries:
        if summary.num_unreadable > 0:
            unreadable_pct = 100.0 * summary.num_unreadable / summary.total_videos
            severity = "a non-trivial" if unreadable_pct >= 1.0 else "a small"
            lines.append(
                f"Note: {summary.num_unreadable} video(s) in {summary.dataset} "
                f"({unreadable_pct:.2f}% of the total) could not be decoded by "
                f"either OpenCV or ffprobe and were excluded from all duration, "
                f"frame-count, fps, and resolution statistics, while still being "
                f"counted toward the per-class sample size reported above. This "
                f"represents {severity} fraction of the corpus and should be "
                f"disclosed as a data-quality limitation when reporting results."
            )

    if larger.dataset != smaller.dataset and larger.num_classes > smaller.num_classes:
        class_ratio = larger.num_classes / max(smaller.num_classes, 1)
        lines.append(
            f"The {larger.dataset} dataset contains {larger.num_classes} classes "
            f"({class_ratio:.1f}x more than {smaller.dataset}'s {smaller.num_classes} "
            f"classes), indicating substantially larger lexical coverage than "
            f"{smaller.dataset}."
        )
        if larger.total_videos > smaller.total_videos:
            video_ratio = larger.total_videos / max(smaller.total_videos, 1)
            lines.append(
                f"This is accompanied by a larger total sample size "
                f"({larger.total_videos} videos, {video_ratio:.1f}x more than "
                f"{smaller.dataset}'s {smaller.total_videos}), suggesting both broader "
                f"vocabulary coverage and a more extensive data foundation."
            )
        else:
            lines.append(
                f"Despite covering more classes, {larger.dataset} has fewer total "
                f"video samples ({larger.total_videos}) than {smaller.dataset} "
                f"({smaller.total_videos}), implying a sparser per-class sampling "
                f"density in {larger.dataset}."
            )
    else:
        lines.append(
            f"{larger.dataset} and {smaller.dataset} exhibit comparable numbers of "
            f"classes ({larger.num_classes} vs {smaller.num_classes}), suggesting "
            f"similar lexical coverage between the two datasets."
        )

    mean_samples_diff = abs(
        larger.mean_samples_per_class - smaller.mean_samples_per_class
    )
    if mean_samples_diff > 0.5:
        denser = max(summaries, key=lambda s: s.mean_samples_per_class)
        sparser = min(summaries, key=lambda s: s.mean_samples_per_class)
        lines.append(
            f"On average, {denser.dataset} provides "
            f"{denser.mean_samples_per_class:.1f} samples per class, compared to "
            f"{sparser.mean_samples_per_class:.1f} for {sparser.dataset}, which may "
            f"translate into more robust per-class statistical estimates and reduced "
            f"overfitting risk during model training on {denser.dataset}."
        )

    if "mean_duration_seconds" in comparison_df.index:
        duration_row = comparison_df.loc["mean_duration_seconds"]
        longer_dataset = duration_row.idxmax()
        shorter_dataset = duration_row.idxmin()
        if longer_dataset != shorter_dataset and not duration_row.isna().any():
            lines.append(
                f"Regarding temporal characteristics, videos in {longer_dataset} have "
                f"a longer mean duration ({duration_row[longer_dataset]:.2f}s) than "
                f"those in {shorter_dataset} ({duration_row[shorter_dataset]:.2f}s), "
                f"which may reflect differences in sign articulation speed, recording "
                f"protocol, or segmentation convention between the two datasets."
            )

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)
