from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import dataset_analysis as da
import imbalance_analysis as ia
import visualization as viz
from video_utils import load_vsl_label_csv, load_wlasl_json, scan_dataset_from_mapping

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    """Configures root logging for the pipeline.

    Args:
        verbose: If True, sets log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed arguments with dataset paths, output directory, and
        verbosity flag.
    """
    script_dir = Path(__file__).resolve().parent
    stage1_root = script_dir.parent  # .../vsl_wlasl_stage1_analysis
    project_root = stage1_root.parent  # main project root (contains data/, src/, ...)
    data_raw_root = project_root / "data" / "raw"
    default_output = stage1_root / "outputs"

    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 analysis: dataset scale/structure (4.2) and class "
            "imbalance (4.3) comparison between VSL and WLASL."
        )
    )
    parser.add_argument(
        "--vsl-label-csv",
        type=Path,
        default=data_raw_root / "VSL" / "Dataset" / "Labels" / "label.csv",
        help="Path to the VSL label CSV file (columns: ID, VIDEO, LABEL).",
    )
    parser.add_argument(
        "--vsl-videos-dir",
        type=Path,
        default=data_raw_root / "VSL" / "Dataset" / "Videos",
        help="Path to the directory containing VSL video files.",
    )
    parser.add_argument(
        "--wlasl-json",
        type=Path,
        default=data_raw_root / "ASL" / "WLASL_v0.3.json",
        help="Path to the WLASL JSON metadata file.",
    )
    parser.add_argument(
        "--wlasl-videos-dir",
        type=Path,
        default=data_raw_root / "ASL" / "videos",
        help="Path to the directory containing WLASL video files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Root output directory; figures/, reports/, csv/ are created inside it.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG-level logging."
    )
    return parser.parse_args()


def analyze_single_dataset(dataset_name: str, records: list, output_dir: Path) -> dict:
    """Runs the full 4.2 + 4.3 analysis pipeline for one dataset.

    Args:
        dataset_name: Name of the dataset (e.g. ``"VSL"``).
        records: List of ``VideoMetadata`` produced by
            ``video_utils.scan_dataset_from_mapping``.
        output_dir: Root output directory (figures/csv/reports created inside).

    Returns:
        A dict with keys ``video_df``, ``distribution``, ``summary``,
        ``video_stats``, ``imbalance_metrics``, used later for the
        cross-dataset comparison stage.
    """
    logger.info("=== Analyzing dataset: %s ===", dataset_name)

    video_df = da.build_video_dataframe(records)
    distribution = da.compute_class_distribution(video_df)
    summary = da.summarize_dataset(dataset_name, video_df)
    video_stats = da.compute_video_statistics(video_df)
    resolution_freq = da.compute_resolution_frequency(video_df)
    codec_freq = da.compute_codec_frequency(video_df)
    imbalance_metrics = ia.compute_imbalance_metrics(dataset_name, distribution)

    figures_dir = output_dir / "figures"
    csv_dir = output_dir / "csv"

    # --- Section 4.2 figures ---
    viz.plot_class_distribution_histogram(
        distribution,
        dataset_name,
        figures_dir / f"{dataset_name}_class_distribution_histogram.png",
    )
    viz.plot_sorted_class_distribution(
        distribution,
        dataset_name,
        figures_dir / f"{dataset_name}_sorted_class_distribution.png",
    )
    viz.plot_video_length_distribution(
        video_df,
        dataset_name,
        figures_dir / f"{dataset_name}_video_length_distribution.png",
    )
    viz.plot_fps_distribution(
        video_df, dataset_name, figures_dir / f"{dataset_name}_fps_distribution.png"
    )
    viz.plot_resolution_distribution(
        resolution_freq,
        dataset_name,
        figures_dir / f"{dataset_name}_resolution_distribution.png",
    )

    # --- Section 4.3 figures ---
    class_fraction, sample_fraction = ia.compute_lorenz_curve(distribution)
    viz.plot_lorenz_curve(
        class_fraction,
        sample_fraction,
        dataset_name,
        figures_dir / f"{dataset_name}_lorenz_curve.png",
    )
    cumulative_df = ia.compute_cumulative_distribution(distribution)
    viz.plot_cumulative_distribution(
        cumulative_df,
        dataset_name,
        figures_dir / f"{dataset_name}_cumulative_distribution.png",
    )
    viz.plot_rank_frequency_loglog(
        distribution,
        dataset_name,
        figures_dir / f"{dataset_name}_rank_frequency_loglog.png",
    )

    # --- Section 4.2 CSV exports (per-dataset slices, merged later) ---
    csv_dir.mkdir(parents=True, exist_ok=True)
    codec_freq.to_csv(csv_dir / f"{dataset_name}_codec_frequency.csv", index=False)
    resolution_freq.to_csv(
        csv_dir / f"{dataset_name}_resolution_frequency.csv", index=False
    )
    cumulative_df.to_csv(
        csv_dir / f"{dataset_name}_cumulative_distribution.csv", index=False
    )

    return {
        "video_df": video_df,
        "distribution": distribution,
        "summary": summary,
        "video_stats": video_stats,
        "imbalance_metrics": imbalance_metrics,
    }


def load_dataset_records(
    dataset_name: str,
    label_csv: Path | None,
    json_path: Path | None,
    videos_dir: Path,
) -> list:
    """Loads the video-to-class mapping for a dataset and reads metadata.

    Exactly one of ``label_csv`` or ``json_path`` must be provided,
    matching whether the dataset uses a CSV (VSL) or JSON (WLASL) label
    file.

    Args:
        dataset_name: Name of the dataset (e.g. ``"VSL"``).
        label_csv: Path to a VSL-style CSV label file, or ``None``.
        json_path: Path to a WLASL-style JSON label file, or ``None``.
        videos_dir: Path to the directory containing the video files.

    Returns:
        A list of ``VideoMetadata`` records for every video resolved from
        the label file that also exists on disk.

    Raises:
        ValueError: If neither or both of ``label_csv``/``json_path`` are
            provided.
    """
    if label_csv is not None and json_path is not None:
        raise ValueError("Provide either a CSV or a JSON label file, not both.")
    if label_csv is None and json_path is None:
        raise ValueError("Provide either a CSV or a JSON label file.")

    if label_csv is not None:
        mapping = load_vsl_label_csv(label_csv, videos_dir)
    else:
        mapping = load_wlasl_json(json_path, videos_dir)

    return scan_dataset_from_mapping(mapping, dataset_name)


def main() -> None:
    """Runs the full Stage-1 VSL vs WLASL comparison pipeline."""
    args = parse_args()
    configure_logging(args.verbose)

    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    csv_dir = output_dir / "csv"
    reports_dir = output_dir / "reports"
    for directory in (figures_dir, csv_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_loaders = {
        "VSL": dict(
            label_csv=args.vsl_label_csv,
            json_path=None,
            videos_dir=args.vsl_videos_dir,
        ),
        "WLASL": dict(
            label_csv=None,
            json_path=args.wlasl_json,
            videos_dir=args.wlasl_videos_dir,
        ),
    }

    results: dict[str, dict] = {}
    for dataset_name, loader_kwargs in dataset_loaders.items():
        try:
            records = load_dataset_records(dataset_name, **loader_kwargs)
            results[dataset_name] = analyze_single_dataset(
                dataset_name, records, output_dir
            )
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            logger.error("Skipping dataset '%s': %s", dataset_name, exc)

    if len(results) < 2:
        logger.error(
            "Fewer than two datasets were successfully analyzed (%d). "
            "Cross-dataset comparison cannot proceed.",
            len(results),
        )
        sys.exit(1)

    # --- Cross-dataset CSV exports ---
    class_distribution_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset": name,
                    "class_name": result["distribution"].index,
                    "sample_count": result["distribution"].values,
                }
            )
            for name, result in results.items()
        ],
        ignore_index=True,
    )
    class_distribution_df.to_csv(csv_dir / "class_distribution.csv", index=False)

    video_statistics_df = pd.concat(
        [result["video_stats"].assign(dataset=name) for name, result in results.items()]
    ).reset_index()
    video_statistics_df.to_csv(csv_dir / "video_statistics.csv", index=False)

    summaries = [result["summary"] for result in results.values()]
    video_stats_by_dataset = {
        name: result["video_stats"] for name, result in results.items()
    }
    comparison_df = da.generate_dataset_comparison_table(
        summaries, video_stats_by_dataset
    )
    comparison_df.to_csv(csv_dir / "dataset_statistics.csv")

    imbalance_metrics_list = [
        result["imbalance_metrics"] for result in results.values()
    ]
    imbalance_df = pd.DataFrame([asdict(m) for m in imbalance_metrics_list])
    imbalance_df.to_csv(csv_dir / "imbalance_metrics.csv", index=False)

    # --- Cross-dataset figures ---
    viz.plot_dataset_comparison(comparison_df, figures_dir / "dataset_comparison.png")

    distributions = {name: result["distribution"] for name, result in results.items()}
    viz.plot_boxplot_comparison(
        distributions, figures_dir / "class_count_boxplot_comparison.png"
    )
    viz.plot_violin_comparison(
        distributions, figures_dir / "class_count_violin_comparison.png"
    )

    # --- Reports ---
    dataset_report = da.generate_academic_report(summaries, comparison_df)
    (reports_dir / "dataset_report.txt").write_text(dataset_report, encoding="utf-8")
    (output_dir / "dataset_report.txt").write_text(dataset_report, encoding="utf-8")

    imbalance_report = ia.generate_imbalance_report(imbalance_metrics_list)
    (reports_dir / "imbalance_report.txt").write_text(
        imbalance_report, encoding="utf-8"
    )

    logger.info("Pipeline completed successfully. Outputs written to: %s", output_dir)


if __name__ == "__main__":
    main()
