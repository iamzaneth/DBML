from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless rendering; must be set before pyplot import.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper")
_FIGURE_DPI = 300
_PALETTE = sns.color_palette("deep")


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Saves a Matplotlib figure to disk at publication resolution.

    Args:
        fig: The figure to save.
        output_path: Destination path. Parent directories are created if
            they do not exist.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", output_path)


def plot_class_distribution_histogram(
    distribution: pd.Series, dataset_name: str, output_path: Path
) -> None:
    """Plots a histogram of the number of samples per class.

    Args:
        distribution: Series of per-class sample counts.
        dataset_name: Name of the dataset, used in the title.
        output_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        distribution.values, bins=30, color=_PALETTE[0], ax=ax, edgecolor="white"
    )
    ax.set_xlabel("Samples per class")
    ax.set_ylabel("Number of classes")
    ax.set_title(f"{dataset_name}: Class Distribution Histogram")
    _save_figure(fig, output_path)


def plot_sorted_class_distribution(
    distribution: pd.Series, dataset_name: str, output_path: Path
) -> None:
    """Plots a descending bar chart of sample count per class.

    Args:
        distribution: Series of per-class sample counts.
        dataset_name: Name of the dataset, used in the title.
        output_path: Destination PNG path.
    """
    sorted_desc = distribution.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, sorted_desc.shape[0] * 0.08), 5))
    ax.bar(
        range(sorted_desc.shape[0]), sorted_desc.values, color=_PALETTE[1], width=1.0
    )
    ax.set_xlabel("Class rank (sorted by sample count, descending)")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"{dataset_name}: Sorted Class Distribution")
    if sorted_desc.shape[0] <= 40:
        ax.set_xticks(range(sorted_desc.shape[0]))
        ax.set_xticklabels(sorted_desc.index, rotation=90, fontsize=6)
    _save_figure(fig, output_path)


def plot_video_length_distribution(
    video_df: pd.DataFrame, dataset_name: str, output_path: Path
) -> None:
    """Plots a histogram of video durations.

    Args:
        video_df: DataFrame produced by
            ``dataset_analysis.build_video_dataframe``.
        dataset_name: Name of the dataset, used in the title.
        output_path: Destination PNG path.
    """
    durations = video_df.loc[video_df["readable"], "duration_seconds"].dropna()
    if durations.empty:
        logger.warning("No readable duration data for %s; skipping plot.", dataset_name)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(durations.values, bins=30, color=_PALETTE[2], ax=ax, edgecolor="white")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Number of videos")
    ax.set_title(f"{dataset_name}: Video Length Distribution")
    _save_figure(fig, output_path)


def plot_fps_distribution(
    video_df: pd.DataFrame, dataset_name: str, output_path: Path
) -> None:
    """Plots a histogram of frame rates (FPS).

    Args:
        video_df: DataFrame produced by
            ``dataset_analysis.build_video_dataframe``.
        dataset_name: Name of the dataset, used in the title.
        output_path: Destination PNG path.
    """
    fps_values = video_df.loc[video_df["readable"], "fps"].dropna()
    if fps_values.empty:
        logger.warning("No readable fps data for %s; skipping plot.", dataset_name)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        fps_values.values, bins=30, color=_PALETTE[3], ax=ax, edgecolor="white"
    )
    ax.set_xlabel("Frames per second (FPS)")
    ax.set_ylabel("Number of videos")
    ax.set_title(f"{dataset_name}: FPS Distribution")
    _save_figure(fig, output_path)


def plot_resolution_distribution(
    resolution_freq: pd.DataFrame, dataset_name: str, output_path: Path
) -> None:
    """Plots a bar chart of the most common video resolutions.

    Args:
        resolution_freq: Output of
            ``dataset_analysis.compute_resolution_frequency``.
        dataset_name: Name of the dataset, used in the title.
        output_path: Destination PNG path.
    """
    if resolution_freq.empty:
        logger.warning("No resolution data for %s; skipping plot.", dataset_name)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(
        resolution_freq["resolution"][::-1],
        resolution_freq["count"][::-1],
        color=_PALETTE[4],
    )
    ax.set_xlabel("Number of videos")
    ax.set_ylabel("Resolution (width x height)")
    ax.set_title(f"{dataset_name}: Top Resolution Distribution")
    _save_figure(fig, output_path)


def plot_dataset_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Plots a grouped bar chart comparing key metrics across datasets.

    Each metric is normalized to its own maximum across datasets before
    plotting, so that metrics on very different scales (e.g. number of
    classes vs mean FPS) can be shown side by side on shared axes while
    preserving relative comparison between datasets for each metric.

    Args:
        comparison_df: Output of
            ``dataset_analysis.generate_dataset_comparison_table``,
            indexed by metric name with one column per dataset.
        output_path: Destination PNG path.
    """
    metrics_to_plot = [
        "num_classes",
        "total_videos",
        "mean_samples_per_class",
        "mean_duration_seconds",
        "mean_fps",
    ]
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.index]
    if not available_metrics:
        logger.warning("No comparable metrics found; skipping comparison plot.")
        return

    subset = comparison_df.loc[available_metrics]
    normalized = subset.div(subset.max(axis=1), axis=0)

    n_metrics = len(available_metrics)
    n_datasets = subset.shape[1]
    x = np.arange(n_metrics)
    width = 0.8 / max(n_datasets, 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, dataset in enumerate(subset.columns):
        offsets = x + (i - (n_datasets - 1) / 2) * width
        bars = ax.bar(offsets, normalized[dataset].values, width=width, label=dataset)
        for bar_patch, raw_value in zip(bars, subset[dataset].values):
            if pd.notna(raw_value):
                label = f"{raw_value:.1f}" if raw_value < 1000 else f"{raw_value:.0f}"
                ax.text(
                    bar_patch.get_x() + bar_patch.get_width() / 2,
                    bar_patch.get_height() + 0.02,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(available_metrics, rotation=20, ha="right")
    ax.set_ylabel("Normalized value (relative to max across datasets)")
    ax.set_title("Dataset Comparison: VSL vs WLASL")
    ax.legend(title="Dataset")
    _save_figure(fig, output_path)


def plot_lorenz_curve(
    class_fraction: np.ndarray,
    sample_fraction: np.ndarray,
    dataset_name: str,
    output_path: Path,
) -> None:
    """Plots the Lorenz curve of class sample-count inequality.

    Args:
        class_fraction: Cumulative fraction of classes (x-axis), from
            ``imbalance_analysis.compute_lorenz_curve``.
        sample_fraction: Cumulative fraction of samples (y-axis), from
            ``imbalance_analysis.compute_lorenz_curve``.
        dataset_name: Name of the dataset, used in the title.
        output_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        class_fraction,
        sample_fraction,
        color=_PALETTE[0],
        linewidth=2,
        label="Lorenz curve",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Perfect equality",
    )
    ax.fill_between(
        class_fraction, sample_fraction, class_fraction, alpha=0.15, color=_PALETTE[0]
    )
    ax.set_xlabel("Cumulative fraction of classes")
    ax.set_ylabel("Cumulative fraction of samples")
    ax.set_title(f"{dataset_name}: Lorenz Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save_figure(fig, output_path)


def plot_cumulative_distribution(
    cumulative_df: pd.DataFrame, dataset_name: str, output_path: Path
) -> None:
    """Plots the cumulative sample fraction as classes are added by rank.

    Args:
        cumulative_df: Output of
            ``imbalance_analysis.compute_cumulative_distribution``.
        dataset_name: Name of the dataset, used in the title.
        output_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        cumulative_df["rank"],
        cumulative_df["cumulative_fraction"],
        color=_PALETTE[1],
        linewidth=2,
    )
    ax.set_xlabel("Class rank (sorted by sample count, descending)")
    ax.set_ylabel("Cumulative fraction of total samples")
    ax.set_title(f"{dataset_name}: Cumulative Distribution Curve")
    ax.set_ylim(0, 1.02)
    _save_figure(fig, output_path)


def plot_rank_frequency_loglog(
    distribution: pd.Series, dataset_name: str, output_path: Path
) -> None:
    """Plots a log-log rank-vs-frequency plot to visualize long-tail behavior.

    Args:
        distribution: Series of per-class sample counts.
        dataset_name: Name of the dataset, used in the title.
        output_path: Destination PNG path.
    """
    sorted_desc = distribution.sort_values(ascending=False)
    ranks = np.arange(1, sorted_desc.shape[0] + 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.loglog(
        ranks,
        sorted_desc.values,
        marker="o",
        linestyle="none",
        markersize=3,
        color=_PALETTE[2],
    )
    ax.set_xlabel("Class rank (log scale)")
    ax.set_ylabel("Sample count (log scale)")
    ax.set_title(f"{dataset_name}: Class Frequency Rank Plot (log-log)")
    _save_figure(fig, output_path)


def plot_boxplot_comparison(
    distributions: dict[str, pd.Series], output_path: Path
) -> None:
    """Plots a boxplot comparing per-class sample counts across datasets.

    Args:
        distributions: Mapping from dataset name to its per-class sample
            count Series.
        output_path: Destination PNG path.
    """
    plot_df = pd.concat(
        [
            pd.DataFrame({"dataset": name, "sample_count": series.values})
            for name, series in distributions.items()
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=plot_df, x="dataset", y="sample_count", hue="dataset", ax=ax, legend=False
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Samples per class")
    ax.set_title("Class Sample-Count Distribution: Boxplot Comparison")
    _save_figure(fig, output_path)


def plot_violin_comparison(
    distributions: dict[str, pd.Series], output_path: Path
) -> None:
    """Plots a violin plot comparing per-class sample count dispersion.

    Args:
        distributions: Mapping from dataset name to its per-class sample
            count Series.
        output_path: Destination PNG path.
    """
    plot_df = pd.concat(
        [
            pd.DataFrame({"dataset": name, "sample_count": series.values})
            for name, series in distributions.items()
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=plot_df, x="dataset", y="sample_count", hue="dataset", ax=ax, legend=False
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Samples per class")
    ax.set_title("Class Sample-Count Dispersion: Violin Plot Comparison")
    _save_figure(fig, output_path)
