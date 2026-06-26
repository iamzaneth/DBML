from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImbalanceMetrics:
    """Class-imbalance metrics for a single dataset.
    """

    dataset: str
    max_samples: int
    min_samples: int
    mean_samples: float
    median_samples: float
    std_samples: float
    imbalance_ratio: float
    coefficient_of_variation: float
    gini_coefficient: float
    top_10pct_class_share: float
    bottom_10pct_class_share: float


def compute_gini_coefficient(values: np.ndarray) -> float:
    """Computes the Gini coefficient of a 1-D array of non-negative values.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute Gini coefficient of an empty array.")
    if np.any(arr < 0):
        raise ValueError("Gini coefficient is undefined for negative values.")
    total = arr.sum()
    if total == 0:
        raise ValueError("Gini coefficient is undefined when all values are zero.")

    sorted_arr = np.sort(arr)
    n = sorted_arr.size
    # Sum_i (2*i - n - 1) * x_i, with i running 1..n, is algebraically
    # equivalent to the full pairwise mean-absolute-difference formulation
    # but computed in O(n log n) instead of O(n^2).
    index = np.arange(1, n + 1, dtype=np.float64)
    numerator = np.sum((2.0 * index - n - 1.0) * sorted_arr)
    gini = numerator / (n * total)
    return float(gini)


def compute_long_tail_shares(
    distribution: pd.Series, tail_fraction: float = 0.10
) -> tuple[float, float]:
    """Computes the sample share held by the top and bottom tails of classes.
    """
    if distribution.empty:
        raise ValueError("Cannot compute long-tail shares of an empty distribution.")
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError(f"tail_fraction must be in (0, 1], got {tail_fraction}.")

    n_classes = distribution.shape[0]
    tail_size = max(1, math.ceil(n_classes * tail_fraction))

    sorted_desc = distribution.sort_values(ascending=False)
    total = sorted_desc.sum()

    top_share = float(sorted_desc.iloc[:tail_size].sum() / total)
    bottom_share = float(sorted_desc.iloc[-tail_size:].sum() / total)
    return top_share, bottom_share


def compute_imbalance_metrics(
    dataset_name: str, distribution: pd.Series
) -> ImbalanceMetrics:
    """Computes the full set of class-imbalance metrics for one dataset.
    """
    if distribution.empty:
        raise ValueError(f"Class distribution for dataset '{dataset_name}' is empty.")

    values = distribution.to_numpy(dtype=np.float64)
    min_samples = int(values.min())
    max_samples = int(values.max())
    mean_samples = float(values.mean())
    median_samples = float(np.median(values))
    std_samples = float(values.std(ddof=1)) if values.size > 1 else 0.0

    if min_samples == 0:
        logger.warning(
            "Dataset '%s' has at least one class with zero samples; "
            "imbalance ratio will be infinite.",
            dataset_name,
        )
        imbalance_ratio = math.inf
    else:
        imbalance_ratio = max_samples / min_samples

    coefficient_of_variation = (
        std_samples / mean_samples if mean_samples > 0 else math.nan
    )
    gini = compute_gini_coefficient(values)
    top_share, bottom_share = compute_long_tail_shares(distribution)

    metrics = ImbalanceMetrics(
        dataset=dataset_name,
        max_samples=max_samples,
        min_samples=min_samples,
        mean_samples=mean_samples,
        median_samples=median_samples,
        std_samples=std_samples,
        imbalance_ratio=imbalance_ratio,
        coefficient_of_variation=coefficient_of_variation,
        gini_coefficient=gini,
        top_10pct_class_share=top_share,
        bottom_10pct_class_share=bottom_share,
    )
    logger.info(
        "Dataset '%s' imbalance: IR=%.2f, CV=%.3f, Gini=%.3f",
        dataset_name,
        imbalance_ratio,
        coefficient_of_variation,
        gini,
    )
    return metrics


def compute_lorenz_curve(distribution: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Computes Lorenz curve coordinates for a class distribution.
    """
    if distribution.empty:
        raise ValueError("Cannot compute Lorenz curve of an empty distribution.")

    sorted_values = np.sort(distribution.to_numpy(dtype=np.float64))
    n = sorted_values.size
    cumulative_samples = np.concatenate([[0.0], np.cumsum(sorted_values)])
    cumulative_sample_fraction = cumulative_samples / cumulative_samples[-1]
    cumulative_class_fraction = np.linspace(0.0, 1.0, n + 1)
    return cumulative_class_fraction, cumulative_sample_fraction


def compute_cumulative_distribution(distribution: pd.Series) -> pd.DataFrame:
    """Computes the cumulative distribution of samples across ranked classes.

    Args:
        distribution: Series of per-class sample counts.

    Returns:
        A DataFrame with columns ``rank`` (1-indexed, descending by
        sample count), ``class_name``, ``sample_count``, and
        ``cumulative_fraction``.

    Raises:
        ValueError: If ``distribution`` is empty.
    """
    if distribution.empty:
        raise ValueError("Cannot compute cumulative distribution of an empty Series.")

    sorted_desc = distribution.sort_values(ascending=False)
    total = sorted_desc.sum()
    cumulative_fraction = sorted_desc.cumsum() / total

    result = pd.DataFrame(
        {
            "rank": np.arange(1, sorted_desc.shape[0] + 1),
            "class_name": sorted_desc.index,
            "sample_count": sorted_desc.values,
            "cumulative_fraction": cumulative_fraction.values,
        }
    )
    return result


def generate_imbalance_report(metrics_list: list[ImbalanceMetrics]) -> str:
    """Generates an automatic academic-style class-imbalance report.
    """
    if len(metrics_list) < 2:
        raise ValueError("At least two datasets are required for a comparison report.")

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("CLASS IMBALANCE ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    for metrics in metrics_list:
        lines.append(f"[{metrics.dataset}]")
        lines.append(f"  Max samples per class: {metrics.max_samples}")
        lines.append(f"  Min samples per class: {metrics.min_samples}")
        lines.append(f"  Mean samples per class: {metrics.mean_samples:.2f}")
        lines.append(f"  Median samples per class: {metrics.median_samples:.2f}")
        lines.append(f"  Std samples per class: {metrics.std_samples:.2f}")
        lines.append(f"  Imbalance Ratio (IR): {metrics.imbalance_ratio:.2f}")
        lines.append(
            f"  Coefficient of Variation (CV): {metrics.coefficient_of_variation:.3f}"
        )
        lines.append(f"  Gini coefficient: {metrics.gini_coefficient:.3f}")
        lines.append(
            f"  Top 10% classes hold: "
            f"{metrics.top_10pct_class_share * 100:.1f}% of samples"
        )
        lines.append(
            f"  Bottom 10% classes hold: "
            f"{metrics.bottom_10pct_class_share * 100:.1f}% of samples"
        )
        lines.append("")

    lines.append("Narrative summary:")
    lines.append("-" * 80)

    finite_metrics = [m for m in metrics_list if math.isfinite(m.imbalance_ratio)]
    if len(finite_metrics) >= 2:
        most_imbalanced = max(finite_metrics, key=lambda m: m.imbalance_ratio)
        least_imbalanced = min(finite_metrics, key=lambda m: m.imbalance_ratio)
        if most_imbalanced.dataset != least_imbalanced.dataset:
            ir_ratio = most_imbalanced.imbalance_ratio / max(
                least_imbalanced.imbalance_ratio, 1e-9
            )
            lines.append(
                f"The calculated imbalance ratio of {most_imbalanced.dataset} "
                f"({most_imbalanced.imbalance_ratio:.2f}) is "
                f"{'substantially' if ir_ratio > 2 else 'moderately'} higher than "
                f"{least_imbalanced.dataset} ({least_imbalanced.imbalance_ratio:.2f}), "
                f"suggesting a stronger long-tail effect that may bias deep learning "
                f"models toward majority classes in {most_imbalanced.dataset}."
            )

    most_dispersed = max(metrics_list, key=lambda m: m.coefficient_of_variation)
    if most_dispersed.coefficient_of_variation > 0.5:
        lines.append(
            f"High coefficient of variation in {most_dispersed.dataset} "
            f"(CV = {most_dispersed.coefficient_of_variation:.2f}) indicates "
            f"substantial class frequency dispersion, meaning sample counts vary "
            f"widely across classes rather than clustering around the mean."
        )
    else:
        lines.append(
            f"Across all datasets analyzed, the coefficient of variation remains "
            f"moderate (highest observed: {most_dispersed.dataset} at "
            f"{most_dispersed.coefficient_of_variation:.2f}), suggesting relatively "
            f"consistent per-class sample counts."
        )

    most_unequal_gini = max(metrics_list, key=lambda m: m.gini_coefficient)
    least_unequal_gini = min(metrics_list, key=lambda m: m.gini_coefficient)
    if most_unequal_gini.dataset != least_unequal_gini.dataset:
        lines.append(
            f"The Gini coefficient further corroborates this pattern: "
            f"{most_unequal_gini.dataset} exhibits a higher Gini coefficient "
            f"({most_unequal_gini.gini_coefficient:.3f}) than "
            f"{least_unequal_gini.dataset} "
            f"({least_unequal_gini.gini_coefficient:.3f}), confirming a more "
            f"unequal distribution of samples across classes."
        )

    most_top_heavy = max(metrics_list, key=lambda m: m.top_10pct_class_share)
    lines.append(
        f"Long-tail analysis shows that the top 10% of classes in "
        f"{most_top_heavy.dataset} account for "
        f"{most_top_heavy.top_10pct_class_share * 100:.1f}% of all samples, while the "
        f"bottom 10% of classes contribute only "
        f"{most_top_heavy.bottom_10pct_class_share * 100:.1f}%, illustrating a "
        f"pronounced long-tail distribution that is characteristic of naturally "
        f"collected sign-language corpora."
    )

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)
