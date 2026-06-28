from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_DIR = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_46 = COMPARE_DIR / "results" / "output_4_6"
DEFAULT_INPUT_47 = COMPARE_DIR / "results" / "output_4_7"
DEFAULT_OUTPUT = COMPARE_DIR / "results" / "output_4_9"

HANDSHAPE_FILE = "handshape_features.csv"
LOCATION_FILE = "location_video.csv"
ORIENTATION_FILE = "orientation_video.csv"
MOTION_VIDEO_FILE = "motion_features_per_video.csv"
MOTION_LABEL_FILE = "motion_features_by_label.csv"
SEQUENCE_FILE = "sequence_length_by_label.csv"

SIGNER_OUTPUT = "signer_variation.csv"
MOTION_OUTPUT = "motion_complexity_ranking.csv"
SEQUENCE_OUTPUT = "sequence_length_variation.csv"
SUMMARY_OUTPUT = "difficulty_summary.csv"
DATASET_SUMMARY_OUTPUT = "dataset_summary.csv"
TOP20_SIGNER_OUTPUT = "top20_signer_variation.csv"
TOP20_MOTION_OUTPUT = "top20_motion_complexity.csv"
TOP20_SEQUENCE_OUTPUT = "top20_sequence_length_variation.csv"

FIGURE_DIRNAME = "figures"
LOG_DIRNAME = "logs"

RANDOM_STATE = 42
PLOT_DPI = 300


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class TableSpec:
    path: Path
    kind: str
    required_numeric: tuple[str, ...] | None = None


def configure_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline_4_9")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_dir / "pipeline_4_9.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "cp1258", "latin1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise PipelineError(f"Unable to read CSV with supported encodings: {path}") from last_error


def ensure_required_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        msg = "\n".join(f"- {path}" for path in missing)
        raise PipelineError(f"Missing required input files:\n{msg}")


def find_col(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def normalize_dataset(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_video_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\\", "/").strip()
    stem = Path(text).stem
    return stem.lower()


def select_numeric_feature_columns(
    df: pd.DataFrame,
    *,
    required_numeric: tuple[str, ...] | None = None,
) -> list[str]:
    identifier_like = {
        "dataset",
        "gloss",
        "label",
        "video",
        "video_name",
        "file",
        "path",
        "hand",
        "dominant_location",
        "dominant_finger_direction",
        "dominant_palm_orientation",
        "npz_sequence_key",
        "npz_format",
        "sample_id",
        "sample_index",
    }
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    numeric_cols = [col for col in numeric_cols if col not in identifier_like]
    if required_numeric is not None:
        missing = [col for col in required_numeric if col not in df.columns]
        if missing:
            raise PipelineError(
                f"Missing required numeric columns: {', '.join(missing)}"
            )
        numeric_cols = [col for col in required_numeric if col in numeric_cols or pd.api.types.is_numeric_dtype(df[col])]
    return numeric_cols


def source_prefix(source_name: str) -> str:
    return source_name.replace(".csv", "").lower()


def choose_key_column(df: pd.DataFrame) -> str:
    key = find_col(df.columns.tolist(), ("video", "file", "video_name", "path"))
    if key is None:
        raise PipelineError("Could not find a video/file key column.")
    return key


def standardize_feature_groups(
    df: pd.DataFrame,
    feature_groups: list[tuple[str, list[str]]],
) -> pd.DataFrame:
    scaled_groups: list[pd.DataFrame] = []
    for group_name, feature_cols in feature_groups:
        if not feature_cols:
            raise PipelineError(f"No feature columns available for standardization in group `{group_name}`.")
        matrix = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        fill_values = matrix.median(axis=0, numeric_only=True).fillna(0.0)
        matrix = matrix.fillna(fill_values)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix.to_numpy(dtype=float))
        scaled_groups.append(pd.DataFrame(scaled, columns=feature_cols, index=df.index))
    return pd.concat(scaled_groups, axis=1)


def compute_cosine_distance(matrix: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    centroid = centroid.reshape(-1)
    centroid_norm = float(np.linalg.norm(centroid))
    sample_norms = np.linalg.norm(matrix, axis=1)
    if centroid_norm == 0.0:
        return np.where(sample_norms == 0.0, 0.0, 1.0)

    denom = sample_norms * centroid_norm
    distances = np.ones(len(matrix), dtype=float)
    valid = denom > 0.0
    if np.any(valid):
        similarities = np.zeros(len(matrix), dtype=float)
        similarities[valid] = (matrix[valid] @ centroid) / denom[valid]
        distances[valid] = 1.0 - similarities[valid]
    return np.clip(distances, 0.0, 2.0)


def aggregate_video_table(
    df: pd.DataFrame,
    *,
    source_name: str,
    required_numeric: tuple[str, ...] | None = None,
    logger: logging.Logger,
) -> pd.DataFrame:
    dataset_col = find_col(df.columns.tolist(), ("dataset",))
    gloss_col = find_col(df.columns.tolist(), ("gloss", "label"))
    key_col = choose_key_column(df)
    if not dataset_col or not gloss_col:
        raise PipelineError(f"{source_name}: missing dataset/gloss or label columns.")

    numeric_cols = select_numeric_feature_columns(df, required_numeric=required_numeric)
    if not numeric_cols:
        raise PipelineError(f"{source_name}: no numeric feature columns found.")

    work = df[[dataset_col, gloss_col, key_col] + numeric_cols].copy()
    work = work.rename(columns={dataset_col: "dataset", gloss_col: "gloss", key_col: "video_key"})
    work["dataset"] = work["dataset"].map(normalize_dataset)
    work["gloss"] = work["gloss"].map(normalize_text)
    work["video_key"] = work["video_key"].map(normalize_video_key)
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["dataset", "gloss", "video_key"], how="any")
    prefix = source_name.replace(".csv", "").lower()
    rename_map = {col: f"{prefix}__{col}" for col in numeric_cols}
    work = work.rename(columns=rename_map)
    feature_cols = list(rename_map.values())

    grouped = (
        work.groupby(["dataset", "gloss", "video_key"], as_index=False, dropna=False)[feature_cols]
        .mean()
        .sort_values(["dataset", "gloss", "video_key"], kind="mergesort")
        .reset_index(drop=True)
    )

    logger.info(
        "%s: loaded %d rows, selected %d numeric columns, aggregated to %d unique videos.",
        source_name,
        len(df),
        len(feature_cols),
        len(grouped),
    )
    return grouped


def merge_video_features(tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        raise PipelineError("No feature tables were loaded.")
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on=["dataset", "gloss", "video_key"], how="outer")
    merged = merged.sort_values(["dataset", "gloss", "video_key"], kind="mergesort").reset_index(drop=True)
    return merged


def load_and_merge_signer_features(input_46: Path, input_47: Path, logger: logging.Logger) -> pd.DataFrame:
    specs = [
        TableSpec(input_46 / HANDSHAPE_FILE, "handshape"),
        TableSpec(input_46 / LOCATION_FILE, "location", required_numeric=("head_ratio", "shoulder_ratio", "chest_ratio", "waist_ratio", "below_waist_ratio")),
        TableSpec(
            input_46 / ORIENTATION_FILE,
            "orientation",
            required_numeric=(
                "finger_dir_x_mean",
                "finger_dir_y_mean",
                "finger_dir_z_mean",
                "palm_normal_x_mean",
                "palm_normal_y_mean",
                "palm_normal_z_mean",
            ),
        ),
        TableSpec(
            input_47 / MOTION_VIDEO_FILE,
            "motion",
            required_numeric=(
                "mean_motion",
                "motion_variance",
                "mean_velocity",
                "velocity_std",
                "mean_acceleration",
                "acceleration_std",
                "trajectory_length",
                "straightness_ratio",
                "direction_change_count",
                "hand_face_dist_mean",
                "hand_body_dist_mean",
                "left_right_hand_dist_mean",
            ),
        ),
    ]
    ensure_required_files([spec.path for spec in specs])

    tables: list[pd.DataFrame] = []
    for spec in tqdm(specs, desc="Loading signer feature tables", unit="table"):
        logger.info("Reading %s", spec.path)
        df = read_csv_safely(spec.path)
        grouped = aggregate_video_table(
            df,
            source_name=spec.path.name,
            required_numeric=spec.required_numeric,
            logger=logger,
        )
        tables.append(grouped)

    merged = merge_video_features(tables)
    logger.info("Merged signer feature table contains %d video-level rows.", len(merged))
    return merged


def compute_signer_variation(merged: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    handshape_prefix = f"{source_prefix(HANDSHAPE_FILE)}__"
    location_prefix = f"{source_prefix(LOCATION_FILE)}__"
    orientation_prefix = f"{source_prefix(ORIENTATION_FILE)}__"
    motion_prefix = f"{source_prefix(MOTION_VIDEO_FILE)}__"

    handshape_cols = [col for col in merged.columns if col.startswith(handshape_prefix)]
    location_cols = [f"{location_prefix}{col}" for col in ("head_ratio", "shoulder_ratio", "chest_ratio", "waist_ratio", "below_waist_ratio")]
    orientation_cols = [
        f"{orientation_prefix}{col}"
        for col in (
            "finger_dir_x_mean",
            "finger_dir_y_mean",
            "finger_dir_z_mean",
            "palm_normal_x_mean",
            "palm_normal_y_mean",
            "palm_normal_z_mean",
        )
    ]
    motion_cols = [
        f"{motion_prefix}{col}"
        for col in (
            "mean_motion",
            "motion_variance",
            "mean_velocity",
            "velocity_std",
            "mean_acceleration",
            "acceleration_std",
            "trajectory_length",
            "straightness_ratio",
            "direction_change_count",
            "hand_face_dist_mean",
            "hand_body_dist_mean",
            "left_right_hand_dist_mean",
        )
    ]

    feature_groups = [
        ("handshape", handshape_cols),
        ("location", location_cols),
        ("orientation", orientation_cols),
        ("motion", motion_cols),
    ]
    feature_cols = [col for _, cols in feature_groups for col in cols]
    missing = [col for col in feature_cols if col not in merged.columns]
    if missing:
        raise PipelineError("Missing required signer feature columns: " + ", ".join(missing))

    logger.info("Filling missing signer features using column medians and applying group-wise StandardScaler.")
    scaled_features = standardize_feature_groups(merged, feature_groups)
    scaled = pd.concat([merged[["dataset", "gloss", "video_key"]], scaled_features], axis=1)

    rows: list[dict[str, Any]] = []
    grouped = scaled.groupby(["dataset", "gloss"], sort=True, dropna=False)
    for (dataset, gloss), group in tqdm(grouped, desc="Computing signer variation", unit="gloss"):
        matrix = group[feature_cols].to_numpy(dtype=float)
        centroid = matrix.mean(axis=0, keepdims=True)
        euclidean_distances = np.linalg.norm(matrix - centroid, axis=1)
        cosine_distances = compute_cosine_distance(matrix, centroid)
        rows.append(
            {
                "dataset": dataset,
                "gloss": gloss,
                "num_samples": int(len(group)),
                "mean_euclidean_distance": float(np.mean(euclidean_distances)),
                "median_euclidean_distance": float(np.median(euclidean_distances)),
                "std_euclidean_distance": float(np.std(euclidean_distances, ddof=0)),
                "euclidean_distance_variance": float(np.var(euclidean_distances, ddof=0)),
                "max_euclidean_distance": float(np.max(euclidean_distances)),
                "mean_cosine_distance": float(np.mean(cosine_distances)),
                "median_cosine_distance": float(np.median(cosine_distances)),
                "std_cosine_distance": float(np.std(cosine_distances, ddof=0)),
                "cosine_distance_variance": float(np.var(cosine_distances, ddof=0)),
                "max_cosine_distance": float(np.max(cosine_distances)),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["mean_euclidean_distance", "dataset", "gloss"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    logger.info("Computed signer variation for %d glosses.", len(out))
    return out


def aggregate_by_gloss(
    df: pd.DataFrame,
    *,
    dataset_col: str = "dataset",
    gloss_col: str = "gloss",
    logger: logging.Logger,
) -> pd.DataFrame:
    if dataset_col not in df.columns or gloss_col not in df.columns:
        raise PipelineError(f"Expected columns `{dataset_col}` and `{gloss_col}` were not found.")

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        raise PipelineError("No numeric columns were found to aggregate.")

    if df.duplicated(subset=[dataset_col, gloss_col]).any():
        logger.info("Detected duplicate dataset+gloss rows; aggregating numeric fields by mean.")

    grouped = (
        df.assign(
            dataset=df[dataset_col].map(normalize_dataset),
            gloss=df[gloss_col].map(normalize_text),
        )
        .groupby(["dataset", "gloss"], as_index=False, dropna=False)[numeric_cols]
        .mean()
        .sort_values(["dataset", "gloss"], kind="mergesort")
        .reset_index(drop=True)
    )
    return grouped


def compute_stat_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {"mean": math.nan, "std": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    desc = stats.describe(values, ddof=0)
    return {
        "mean": float(desc.mean),
        "std": float(np.sqrt(desc.variance)),
        "median": float(np.median(values)),
        "min": float(desc.minmax[0]),
        "max": float(desc.minmax[1]),
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def create_histogram(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    *,
    color_by_dataset: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    datasets = sorted(df["dataset"].dropna().astype(str).unique())
    bins = min(30, max(10, int(np.sqrt(len(df))))) if len(df) else 10

    if color_by_dataset and len(datasets) > 1:
        colors = plt.cm.Set2(np.linspace(0.1, 0.9, len(datasets)))
        for dataset, color in zip(datasets, colors, strict=False):
            values = pd.to_numeric(df.loc[df["dataset"].astype(str) == dataset, metric], errors="coerce").dropna()
            ax.hist(values, bins=bins, alpha=0.55, label=dataset, color=color, edgecolor="white")
        ax.legend(frameon=False)
    else:
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        ax.hist(values, bins=bins, color="#4C78A8", alpha=0.8, edgecolor="white")

    ax.set_title(title, fontsize=14, pad=12, weight="bold")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Number of glosses")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def create_top20_bar_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    top = df.sort_values(metric, ascending=False, kind="mergesort").head(20).copy()
    top["label_display"] = top["dataset"].astype(str) + " | " + top["gloss"].astype(str)
    top = top.sort_values(metric, ascending=True, kind="mergesort")

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(top["label_display"], top[metric], color="#4C78A8", edgecolor="white")
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in top[metric]], padding=3, fontsize=8)
    ax.set_title(title, fontsize=14, pad=12, weight="bold")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def create_grouped_boxplot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    datasets = sorted(df["dataset"].dropna().astype(str).unique())
    data = [
        pd.to_numeric(df.loc[df["dataset"].astype(str) == dataset, metric], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
        for dataset in datasets
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(data, tick_labels=datasets, patch_artist=True, showfliers=False)
    colors = plt.cm.Set2(np.linspace(0.1, 0.9, max(len(datasets), 1)))
    for patch, color in zip(box["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor("white")
    for element in ("whiskers", "caps", "medians"):
        for artist in box[element]:
            artist.set_color("#333333")

    ax.set_title(title, fontsize=14, pad=12, weight="bold")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def build_motion_complexity_table(input_47: Path, logger: logging.Logger) -> pd.DataFrame:
    path = input_47 / MOTION_LABEL_FILE
    if not path.exists():
        raise PipelineError(f"Missing required input file: {path}")
    logger.info("Reading motion complexity summary: %s", path)
    df = read_csv_safely(path)
    dataset_col = find_col(df.columns.tolist(), ("dataset",))
    label_col = find_col(df.columns.tolist(), ("gloss", "label"))
    if not dataset_col or not label_col:
        raise PipelineError("motion_features_by_label.csv must contain dataset and label/gloss columns.")

    work = df.copy()
    work = work.rename(columns={dataset_col: "dataset", label_col: "gloss"})
    work["dataset"] = work["dataset"].map(normalize_dataset)
    work["gloss"] = work["gloss"].map(normalize_text)

    required = [
        "num_samples",
        "mean_total_motion",
        "mean_velocity",
        "motion_variance_mean",
        "acceleration_mean",
        "trajectory_length_mean",
        "direction_change_mean",
        "motion_complexity_score",
    ]
    missing = [col for col in required if col not in work.columns]
    if missing:
        raise PipelineError(
            f"motion_features_by_label.csv is missing required columns: {', '.join(missing)}"
        )

    agg = aggregate_by_gloss(work, logger=logger)
    out = agg[
        [
            "dataset",
            "gloss",
            "num_samples",
            "mean_total_motion",
            "mean_velocity",
            "motion_variance_mean",
            "acceleration_mean",
            "trajectory_length_mean",
            "direction_change_mean",
            "motion_complexity_score",
        ]
    ].copy()
    out = out.sort_values(
        ["motion_complexity_score", "dataset", "gloss"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    logger.info("Prepared motion complexity ranking for %d glosses.", len(out))
    return out


def build_sequence_length_table(input_47: Path, logger: logging.Logger) -> pd.DataFrame:
    path = input_47 / SEQUENCE_FILE
    if not path.exists():
        raise PipelineError(f"Missing required input file: {path}")
    logger.info("Reading sequence length summary: %s", path)
    df = read_csv_safely(path)
    dataset_col = find_col(df.columns.tolist(), ("dataset",))
    label_col = find_col(df.columns.tolist(), ("gloss", "label"))
    if not dataset_col or not label_col:
        raise PipelineError("sequence_length_by_label.csv must contain dataset and label/gloss columns.")

    work = df.copy()
    work = work.rename(columns={dataset_col: "dataset", label_col: "gloss"})
    work["dataset"] = work["dataset"].map(normalize_dataset)
    work["gloss"] = work["gloss"].map(normalize_text)

    required = ["num_samples", "mean_frames", "std_frames", "sequence_length_variance"]
    missing = [col for col in required if col not in work.columns]
    if missing:
        raise PipelineError(
            f"sequence_length_by_label.csv is missing required columns: {', '.join(missing)}"
        )

    agg = aggregate_by_gloss(work, logger=logger)
    out = agg[["dataset", "gloss", "num_samples", "mean_frames", "std_frames", "sequence_length_variance"]].copy()
    out = out.sort_values(
        ["sequence_length_variance", "dataset", "gloss"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    logger.info("Prepared sequence length variation ranking for %d glosses.", len(out))
    return out


def make_summary_frame(section: str, metric: str, df: pd.DataFrame) -> dict[str, Any]:
    summary = compute_stat_summary(df[metric])
    return {
        "analysis": section,
        "metric": metric,
        "num_glosses": int(len(df)),
        **summary,
    }


def build_summary_table(
    signer_df: pd.DataFrame,
    motion_df: pd.DataFrame,
    sequence_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        make_summary_frame("Signer Variation", "mean_euclidean_distance", signer_df),
        make_summary_frame("Motion Complexity", "motion_complexity_score", motion_df),
        make_summary_frame("Sequence Length Variation", "sequence_length_variance", sequence_df),
    ]
    return pd.DataFrame(rows)


def build_dataset_summary_table(
    signer_df: pd.DataFrame,
    motion_df: pd.DataFrame,
    sequence_df: pd.DataFrame,
) -> pd.DataFrame:
    datasets = sorted(
        set(signer_df["dataset"].astype(str))
        | set(motion_df["dataset"].astype(str))
        | set(sequence_df["dataset"].astype(str))
    )
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        signer_mask = signer_df["dataset"].astype(str) == dataset
        motion_mask = motion_df["dataset"].astype(str) == dataset
        sequence_mask = sequence_df["dataset"].astype(str) == dataset
        gloss_count = len(
            set(signer_df.loc[signer_mask, "gloss"].astype(str))
            | set(motion_df.loc[motion_mask, "gloss"].astype(str))
            | set(sequence_df.loc[sequence_mask, "gloss"].astype(str))
        )
        signer_summary = compute_stat_summary(signer_df.loc[signer_mask, "mean_euclidean_distance"])
        motion_summary = compute_stat_summary(motion_df.loc[motion_mask, "motion_complexity_score"])
        sequence_summary = compute_stat_summary(sequence_df.loc[sequence_mask, "sequence_length_variance"])
        rows.append(
            {
                "dataset": dataset,
                "num_glosses": int(gloss_count),
                "mean_signer_variation": signer_summary["mean"],
                "std_signer_variation": signer_summary["std"],
                "mean_motion_complexity": motion_summary["mean"],
                "std_motion_complexity": motion_summary["std"],
                "mean_sequence_length_variance": sequence_summary["mean"],
                "std_sequence_length_variance": sequence_summary["std"],
            }
        )
    return pd.DataFrame(rows)


def save_figure_bundle(
    output_dir: Path,
    figure_name_prefix: str,
    df: pd.DataFrame,
    metric: str,
    title_prefix: str,
) -> None:
    create_histogram(
        df,
        metric,
        f"{title_prefix} Histogram",
        output_dir / f"{figure_name_prefix}_histogram.png",
    )
    create_top20_bar_chart(
        df,
        metric,
        f"{title_prefix} Top-20 Glosses",
        output_dir / f"{figure_name_prefix}_top20.png",
    )


def export_top20_table(df: pd.DataFrame, metric: str, path: Path) -> None:
    write_csv(df.sort_values(metric, ascending=False, kind="mergesort").head(20).reset_index(drop=True), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Section 4.9 Dataset-level Recognition Difficulty Analysis")
    parser.add_argument("--input-46", type=Path, default=DEFAULT_INPUT_46, help="Path to output_4_6 directory")
    parser.add_argument("--input-47", type=Path, default=DEFAULT_INPUT_47, help="Path to output_4_7 directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Path to output_4_9 directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    figures_dir = output_dir / FIGURE_DIRNAME
    logs_dir = output_dir / LOG_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(logs_dir)
    logger.info("Starting Section 4.9 Dataset-level Recognition Difficulty Analysis")
    logger.info("Input 4.6 directory: %s", args.input_46)
    logger.info("Input 4.7 directory: %s", args.input_47)
    logger.info("Output directory: %s", output_dir)
    logger.info("Random state: %d", RANDOM_STATE)

    merged_signer = load_and_merge_signer_features(args.input_46, args.input_47, logger)
    signer_variation = compute_signer_variation(merged_signer, logger)
    motion_complexity = build_motion_complexity_table(args.input_47, logger)
    sequence_variation = build_sequence_length_table(args.input_47, logger)
    summary = build_summary_table(signer_variation, motion_complexity, sequence_variation)
    dataset_summary = build_dataset_summary_table(signer_variation, motion_complexity, sequence_variation)

    logger.info("Writing CSV outputs.")
    write_csv(signer_variation, output_dir / SIGNER_OUTPUT)
    write_csv(motion_complexity, output_dir / MOTION_OUTPUT)
    write_csv(sequence_variation, output_dir / SEQUENCE_OUTPUT)
    write_csv(summary, output_dir / SUMMARY_OUTPUT)
    write_csv(dataset_summary, output_dir / DATASET_SUMMARY_OUTPUT)
    export_top20_table(signer_variation, "mean_euclidean_distance", output_dir / TOP20_SIGNER_OUTPUT)
    export_top20_table(motion_complexity, "motion_complexity_score", output_dir / TOP20_MOTION_OUTPUT)
    export_top20_table(sequence_variation, "sequence_length_variance", output_dir / TOP20_SEQUENCE_OUTPUT)

    logger.info("Generating publication-quality figures at %d dpi.", PLOT_DPI)
    save_figure_bundle(figures_dir, "signer_variation", signer_variation, "mean_euclidean_distance", "Signer Variation")
    create_grouped_boxplot(
        signer_variation,
        "mean_euclidean_distance",
        "Signer Variation Boxplot by Dataset",
        figures_dir / "signer_variation_boxplot_by_dataset.png",
    )
    save_figure_bundle(
        figures_dir,
        "motion_complexity",
        motion_complexity,
        "motion_complexity_score",
        "Motion Complexity",
    )
    create_grouped_boxplot(
        motion_complexity,
        "motion_complexity_score",
        "Motion Complexity Boxplot by Dataset",
        figures_dir / "motion_complexity_boxplot_by_dataset.png",
    )
    save_figure_bundle(
        figures_dir,
        "sequence_length_variation",
        sequence_variation,
        "sequence_length_variance",
        "Sequence Length Variation",
    )
    create_grouped_boxplot(
        sequence_variation,
        "sequence_length_variance",
        "Sequence Length Variation Boxplot by Dataset",
        figures_dir / "sequence_length_variation_boxplot_by_dataset.png",
    )

    logger.info("Completed Section 4.9 successfully.")
    print(f"Section 4.9 outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
