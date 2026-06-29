from __future__ import annotations

import argparse
import logging
import math
import unicodedata
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
COMPARE_DIR = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_46 = COMPARE_DIR / "results" / "output_4_6"
DEFAULT_INPUT_47 = COMPARE_DIR / "results" / "output_4_7"
DEFAULT_OUTPUT = COMPARE_DIR / "results" / "output_4_10"

HANDSHAPE_FILE = "handshape_features.csv"
LOCATION_FILE = "location_video.csv"
ORIENTATION_FILE = "orientation_video.csv"
MOTION_FILE = "motion_features_per_video.csv"

RANDOM_STATE = 42
PLOT_DPI = 300
DATASET_ORDER = ("VSL", "ASL")


HANDSHAPE_EXCLUDE = {"dataset", "gloss", "video", "path", "hand"}
HANDSHAPE_PREFIXES = (
    "thumb_",
    "index_",
    "middle_",
    "ring_",
    "pinky_",
    "thumb_index_distance_ratio_",
    "index_middle_distance_ratio_",
    "middle_ring_distance_ratio_",
    "ring_pinky_distance_ratio_",
    "index_pinky_distance_ratio_",
)
HANDSHAPE_EXACT = (
    "valid_frame_ratio",
    "motion_magnitude",
    "palm_width_ratio_mean",
    "openness_mean",
    "curvature_proxy_mean",
)

LOCATION_FEATURES = (
    "head_ratio",
    "shoulder_ratio",
    "chest_ratio",
    "waist_ratio",
    "below_waist_ratio",
)

ORIENTATION_FEATURES = (
    "finger_dir_x_mean",
    "finger_dir_y_mean",
    "finger_dir_z_mean",
    "palm_normal_x_mean",
    "palm_normal_y_mean",
    "palm_normal_z_mean",
)

MOTION_FEATURES = (
    "total_motion",
    "mean_motion",
    "max_motion",
    "motion_variance",
    "mean_velocity",
    "max_velocity",
    "velocity_std",
    "mean_acceleration",
    "max_acceleration",
    "acceleration_std",
    "trajectory_length",
    "displacement",
    "straightness_ratio",
    "direction_change_count",
    "trajectory_bbox_area",
    "hand_face_dist_mean",
    "hand_face_dist_std",
    "hand_body_dist_mean",
    "hand_body_dist_std",
    "left_right_hand_dist_mean",
    "left_right_hand_dist_std",
)


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeatureTableSpec:
    path: Path
    name: str
    feature_columns: tuple[str, ...] | None = None
    use_handshape_rules: bool = False


def configure_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline_4_10")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_dir / "pipeline_4_10.log", encoding="utf-8")
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
    raise PipelineError(f"Cannot read CSV with supported encodings: {path}") from last_error


def ensure_required_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise PipelineError(f"Missing required input files:\n{details}")


def normalize_dataset(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if text == "WLASL":
        return "ASL"
    return text


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFC", str(value).strip())


def normalize_video_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\\", "/").strip()
    return Path(text).stem.lower()


def find_col(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def select_handshape_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        if col in HANDSHAPE_EXCLUDE:
            continue
        keep = col in HANDSHAPE_EXACT or any(col.startswith(prefix) for prefix in HANDSHAPE_PREFIXES)
        if keep:
            columns.append(col)
    missing_exact = [col for col in HANDSHAPE_EXACT if col not in df.columns]
    if missing_exact:
        raise PipelineError("handshape_features.csv missing required columns: " + ", ".join(missing_exact))
    return columns


def validate_feature_columns(df: pd.DataFrame, columns: tuple[str, ...], source_name: str) -> list[str]:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise PipelineError(f"{source_name} missing required columns: {', '.join(missing)}")
    return list(columns)


def aggregate_feature_table(spec: FeatureTableSpec, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Reading %s", spec.path)
    df = read_csv_safely(spec.path)

    dataset_col = find_col(df.columns.tolist(), ("dataset",))
    gloss_col = find_col(df.columns.tolist(), ("gloss", "label"))
    video_col = find_col(df.columns.tolist(), ("video", "file", "video_name", "path"))
    if dataset_col is None or gloss_col is None or video_col is None:
        raise PipelineError(f"{spec.path.name}: expected dataset, gloss/label, and video/file columns.")

    if spec.use_handshape_rules:
        raw_features = select_handshape_columns(df)
    elif spec.feature_columns is not None:
        raw_features = validate_feature_columns(df, spec.feature_columns, spec.path.name)
    else:
        raise PipelineError(f"{spec.path.name}: no feature selection rule was provided.")

    numeric_features: list[str] = []
    for col in raw_features:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
            numeric_features.append(col)
    if not numeric_features:
        raise PipelineError(f"{spec.path.name}: no usable numeric feature columns were found.")

    work = df[[dataset_col, gloss_col, video_col] + numeric_features].copy()
    work = work.rename(columns={dataset_col: "dataset", gloss_col: "gloss", video_col: "video"})
    work["dataset"] = work["dataset"].map(normalize_dataset)
    work["gloss"] = work["gloss"].map(normalize_text)
    work["video"] = work["video"].map(normalize_video_key)
    work = work[(work["dataset"] != "") & (work["gloss"] != "") & (work["video"] != "")]
    work = work.replace([np.inf, -np.inf], np.nan)

    rename_map = {col: f"{spec.name}__{col}" for col in numeric_features}
    work = work.rename(columns=rename_map)
    feature_cols = list(rename_map.values())

    grouped = (
        work.groupby(["dataset", "gloss", "video"], as_index=False, dropna=False)[feature_cols]
        .mean()
        .sort_values(["dataset", "gloss", "video"], kind="mergesort")
        .reset_index(drop=True)
    )
    logger.info(
        "%s: %d rows, %d numeric features, %d video-level rows after hand/video aggregation.",
        spec.path.name,
        len(df),
        len(feature_cols),
        len(grouped),
    )
    return grouped


def merge_feature_tables(tables: list[pd.DataFrame], logger: logging.Logger) -> pd.DataFrame:
    if not tables:
        raise PipelineError("No feature tables were loaded.")
    merged = tables[0]
    for table in tables[1:]:
        before = len(merged)
        merged = merged.merge(table, on=["dataset", "gloss", "video"], how="inner")
        logger.info("Inner merge reduced rows from %d to %d.", before, len(merged))
    if merged.empty:
        raise PipelineError("Merged feature table is empty. Check dataset/gloss/video keys across inputs.")
    return merged.sort_values(["dataset", "gloss", "video"], kind="mergesort").reset_index(drop=True)


def load_and_merge_features(input_46: Path, input_47: Path, logger: logging.Logger) -> pd.DataFrame:
    specs = [
        FeatureTableSpec(input_46 / HANDSHAPE_FILE, "handshape", use_handshape_rules=True),
        FeatureTableSpec(input_46 / LOCATION_FILE, "location", LOCATION_FEATURES),
        FeatureTableSpec(input_46 / ORIENTATION_FILE, "orientation", ORIENTATION_FEATURES),
        FeatureTableSpec(input_47 / MOTION_FILE, "motion", MOTION_FEATURES),
    ]
    ensure_required_files([spec.path for spec in specs])
    tables = [aggregate_feature_table(spec, logger) for spec in specs]
    merged = merge_feature_tables(tables, logger)
    logger.info("Final merged feature table has %d samples and %d columns.", len(merged), len(merged.columns))
    return merged


def write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")


def prepare_scaled_features(df: pd.DataFrame, feature_cols: list[str], logger: logging.Logger) -> np.ndarray:
    raw = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    missing_cells = int(raw.isna().sum().sum())
    logger.info("Missing numeric feature cells before imputation: %d.", missing_cells)
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(raw)
    imputed = np.nan_to_num(imputed, nan=0.0, posinf=0.0, neginf=0.0)
    return StandardScaler().fit_transform(imputed)


def color_values(labels: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    unique = sorted(labels.astype(str).unique())
    mapping = {label: idx for idx, label in enumerate(unique)}
    values = labels.astype(str).map(mapping).to_numpy(dtype=int)
    return values, mapping


def scatter_plot(embedding: np.ndarray, labels: pd.Series, title: str, path: Path) -> None:
    colors, mapping = color_values(labels)
    fig, ax = plt.subplots(figsize=(11, 8))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        cmap="tab20" if len(mapping) <= 20 else "nipy_spectral",
        s=24,
        alpha=0.78,
        linewidths=0.25,
        edgecolors="white",
    )
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if len(mapping) <= 20:
        handles, _ = scatter.legend_elements(num=len(mapping))
        ax.legend(
            handles,
            list(mapping.keys()),
            title="Gloss",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def histogram(values: pd.Series, title: str, xlabel: str, path: Path) -> None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    bins = min(40, max(10, int(np.sqrt(max(len(clean), 1)))))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(clean, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.85)
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of glosses")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def boxplot(values: pd.Series, title: str, ylabel: str, path: Path) -> None:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    box = ax.boxplot([clean], tick_labels=["Glosses"], patch_artist=True, showfliers=True)
    box["boxes"][0].set_facecolor("#72B7B2")
    box["boxes"][0].set_edgecolor("white")
    for element in ("whiskers", "caps", "medians", "fliers"):
        for artist in box[element]:
            artist.set_color("#333333")
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def heatmap(matrix: pd.DataFrame, title: str, path: Path) -> None:
    labels = matrix.index.astype(str).tolist()
    size = min(18, max(8, 0.22 * len(labels)))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    tick_step = max(1, math.ceil(len(labels) / 40))
    ticks = np.arange(0, len(labels), tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=6)
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def compute_compactness(df: pd.DataFrame, x_scaled: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for gloss, indices in df.groupby("gloss", sort=True).indices.items():
        matrix = x_scaled[np.asarray(indices)]
        centroid = matrix.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(matrix - centroid, axis=1)
        rows.append(
            {
                "gloss": gloss,
                "num_samples": int(len(indices)),
                "compactness": float(np.mean(distances)),
                "median_distance": float(np.median(distances)),
                "std_distance": float(np.std(distances, ddof=0)),
                "max_distance": float(np.max(distances)),
            }
        )
    compactness = pd.DataFrame(rows).sort_values(["compactness", "gloss"], kind="mergesort").reset_index(drop=True)
    stats = {
        "mean": float(compactness["compactness"].mean()),
        "median": float(compactness["compactness"].median()),
        "std": float(compactness["compactness"].std(ddof=0)),
    }
    return compactness, stats


def compute_centroid_distances(
    df: pd.DataFrame,
    x_scaled: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    glosses: list[str] = []
    centroids: list[np.ndarray] = []
    for gloss, indices in df.groupby("gloss", sort=True).indices.items():
        glosses.append(str(gloss))
        centroids.append(x_scaled[np.asarray(indices)].mean(axis=0))
    centroid_matrix = np.vstack(centroids)
    euclidean = pd.DataFrame(
        pairwise_distances(centroid_matrix, metric="euclidean"),
        index=glosses,
        columns=glosses,
    )
    cosine = pd.DataFrame(
        pairwise_distances(centroid_matrix, metric="cosine"),
        index=glosses,
        columns=glosses,
    )
    pair_rows: list[dict[str, Any]] = []
    for i, j in combinations(range(len(glosses)), 2):
        pair_rows.append(
            {
                "gloss_a": glosses[i],
                "gloss_b": glosses[j],
                "euclidean_distance": float(euclidean.iat[i, j]),
                "cosine_distance": float(cosine.iat[i, j]),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    stats = {}
    for name, matrix in (("euclidean", euclidean), ("cosine", cosine)):
        values = matrix.to_numpy(dtype=float)[np.triu_indices(len(glosses), k=1)]
        stats[f"{name}_mean"] = float(np.mean(values)) if values.size else math.nan
        stats[f"{name}_median"] = float(np.median(values)) if values.size else math.nan
        stats[f"{name}_std"] = float(np.std(values, ddof=0)) if values.size else math.nan
        stats[f"{name}_min"] = float(np.min(values)) if values.size else math.nan
        stats[f"{name}_max"] = float(np.max(values)) if values.size else math.nan
    return euclidean, cosine, pairs, stats


def valid_cluster_metric_shape(labels: pd.Series) -> bool:
    n_samples = len(labels)
    n_classes = labels.nunique()
    return n_samples >= 3 and 2 <= n_classes <= n_samples - 1


def compute_cluster_metrics(df: pd.DataFrame, x_scaled: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    labels = df["gloss"].astype(str)
    if not valid_cluster_metric_shape(labels):
        nan_metrics = {
            "Silhouette Score": math.nan,
            "Davies-Bouldin Index": math.nan,
            "Calinski-Harabasz Index": math.nan,
        }
        return nan_metrics, pd.DataFrame(columns=["gloss", "silhouette_mean", "silhouette_median", "num_samples"])

    silhouette = float(silhouette_score(x_scaled, labels, metric="euclidean"))
    db_index = float(davies_bouldin_score(x_scaled, labels))
    ch_index = float(calinski_harabasz_score(x_scaled, labels))
    sample_values = silhouette_samples(x_scaled, labels, metric="euclidean")
    per_gloss = (
        pd.DataFrame({"gloss": labels, "silhouette": sample_values})
        .groupby("gloss", as_index=False)
        .agg(
            silhouette_mean=("silhouette", "mean"),
            silhouette_median=("silhouette", "median"),
            num_samples=("silhouette", "size"),
        )
        .sort_values(["silhouette_mean", "gloss"], ascending=[False, True], kind="mergesort")
    )
    return {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": db_index,
        "Calinski-Harabasz Index": ch_index,
    }, per_gloss


def run_pca(x_scaled: np.ndarray, labels: pd.Series, dataset_dir: Path, dataset: str) -> tuple[float, float]:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    embedding = pca.fit_transform(x_scaled)
    scatter_plot(embedding, labels, f"{dataset} PCA Feature Space", dataset_dir / "pca.png")
    ratio = pca.explained_variance_ratio_
    pd.DataFrame(
        {
            "component": ["PC1", "PC2"],
            "explained_variance_ratio": ratio,
            "cumulative_explained_variance": np.cumsum(ratio),
        }
    ).to_csv(dataset_dir / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")
    return float(ratio[0]), float(np.sum(ratio))


def run_tsne(x_scaled: np.ndarray, labels: pd.Series, dataset_dir: Path, dataset: str) -> None:
    if len(x_scaled) < 3:
        raise PipelineError(f"{dataset}: t-SNE requires at least 3 samples.")
    perplexity = min(30, max(2, (len(x_scaled) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
    )
    embedding = tsne.fit_transform(x_scaled)
    scatter_plot(embedding, labels, f"{dataset} t-SNE Feature Space", dataset_dir / "tsne.png")


def run_umap(x_scaled: np.ndarray, labels: pd.Series, dataset_dir: Path, dataset: str) -> None:
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise PipelineError(
            "UMAP requires the optional package `umap-learn`. Install it with `pip install umap-learn`."
        ) from exc
    if len(x_scaled) < 3:
        raise PipelineError(f"{dataset}: UMAP requires at least 3 samples.")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(2, len(x_scaled) - 1)),
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    embedding = reducer.fit_transform(x_scaled)
    scatter_plot(embedding, labels, f"{dataset} UMAP Feature Space", dataset_dir / "umap.png")


def analyze_dataset(dataset_df: pd.DataFrame, dataset: str, output_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Analyzing %s with %d samples.", dataset, len(dataset_df))

    feature_cols = [col for col in dataset_df.columns if col not in {"dataset", "gloss", "video"}]
    if len(dataset_df) < 2:
        raise PipelineError(f"{dataset}: need at least 2 samples for feature-space analysis.")

    x_scaled = prepare_scaled_features(dataset_df, feature_cols, logger)
    feature_snapshot = pd.concat(
        [
            dataset_df[["dataset", "gloss", "video"]].reset_index(drop=True),
            pd.DataFrame(x_scaled, columns=feature_cols),
        ],
        axis=1,
    )
    write_csv(feature_snapshot, dataset_dir / "scaled_feature_space.csv")

    pca_pc1, pca_cumulative = run_pca(x_scaled, dataset_df["gloss"], dataset_dir, dataset)
    run_tsne(x_scaled, dataset_df["gloss"], dataset_dir, dataset)
    run_umap(x_scaled, dataset_df["gloss"], dataset_dir, dataset)

    compactness, compact_stats = compute_compactness(dataset_df, x_scaled)
    write_csv(compactness, dataset_dir / "compactness.csv")
    write_csv(compactness.head(20), dataset_dir / "top20_compact_glosses.csv")
    write_csv(compactness.tail(20).sort_values("compactness", ascending=False), dataset_dir / "top20_dispersed_glosses.csv")
    histogram(compactness["compactness"], f"{dataset} Intra-class Compactness", "Mean distance to centroid", dataset_dir / "compactness_histogram.png")
    boxplot(compactness["compactness"], f"{dataset} Compactness Distribution", "Mean distance to centroid", dataset_dir / "compactness_boxplot.png")

    euclidean, cosine, pairs, separation_stats = compute_centroid_distances(dataset_df, x_scaled)
    write_csv(euclidean.reset_index().rename(columns={"index": "gloss"}), dataset_dir / "interclass_distance_euclidean.csv")
    write_csv(cosine.reset_index().rename(columns={"index": "gloss"}), dataset_dir / "interclass_distance_cosine.csv")
    write_csv(pairs.sort_values("euclidean_distance", kind="mergesort").head(20), dataset_dir / "top20_nearest_gloss_pairs_euclidean.csv")
    write_csv(pairs.sort_values("euclidean_distance", ascending=False, kind="mergesort").head(20), dataset_dir / "top20_farthest_gloss_pairs_euclidean.csv")
    write_csv(pairs.sort_values("cosine_distance", kind="mergesort").head(20), dataset_dir / "top20_nearest_gloss_pairs_cosine.csv")
    write_csv(pairs.sort_values("cosine_distance", ascending=False, kind="mergesort").head(20), dataset_dir / "top20_farthest_gloss_pairs_cosine.csv")

    cluster_metrics, per_gloss_silhouette = compute_cluster_metrics(dataset_df, x_scaled)
    write_csv(per_gloss_silhouette, dataset_dir / "silhouette_by_gloss.csv")

    metrics = {
        "dataset": dataset,
        "num_samples": int(len(dataset_df)),
        "num_glosses": int(dataset_df["gloss"].nunique()),
        "feature_dimension": int(len(feature_cols)),
        "PCA explained variance PC1": pca_pc1,
        "PCA explained variance cumulative PC1_PC2": pca_cumulative,
        "Silhouette Score": cluster_metrics["Silhouette Score"],
        "Davies-Bouldin Index": cluster_metrics["Davies-Bouldin Index"],
        "Calinski-Harabasz Index": cluster_metrics["Calinski-Harabasz Index"],
        "Mean Compactness": compact_stats["mean"],
        "Median Compactness": compact_stats["median"],
        "Std Compactness": compact_stats["std"],
        "Mean Euclidean Separation": separation_stats["euclidean_mean"],
        "Median Euclidean Separation": separation_stats["euclidean_median"],
        "Std Euclidean Separation": separation_stats["euclidean_std"],
        "Min Euclidean Separation": separation_stats["euclidean_min"],
        "Max Euclidean Separation": separation_stats["euclidean_max"],
        "Mean Cosine Separation": separation_stats["cosine_mean"],
        "Median Cosine Separation": separation_stats["cosine_median"],
        "Std Cosine Separation": separation_stats["cosine_std"],
        "Min Cosine Separation": separation_stats["cosine_min"],
        "Max Cosine Separation": separation_stats["cosine_max"],
    }
    write_csv(pd.DataFrame([metrics]), dataset_dir / "feature_space_metrics.csv")
    return metrics


def run_pipeline(input_46: Path, input_47: Path, output_dir: Path) -> None:
    logger = configure_logging(output_dir / "logs")
    logger.info("Starting 4.10 Feature Space Analysis.")
    logger.info("Input 4.6: %s", input_46)
    logger.info("Input 4.7: %s", input_47)
    logger.info("Output: %s", output_dir)

    merged = load_and_merge_features(input_46, input_47, logger)
    write_csv(merged, output_dir / "merged_feature_space.csv")

    feature_cols = [col for col in merged.columns if col not in {"dataset", "gloss", "video"}]
    non_numeric = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(merged[col])]
    if non_numeric:
        raise PipelineError("Non-numeric columns remain after merge: " + ", ".join(non_numeric))

    metrics_rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        dataset_df = merged[merged["dataset"] == dataset].copy().reset_index(drop=True)
        if dataset_df.empty:
            logger.warning("Dataset %s has no merged rows; skipping.", dataset)
            continue
        metrics_rows.append(analyze_dataset(dataset_df, dataset, output_dir, logger))

    if not metrics_rows:
        raise PipelineError("No dataset-specific analyses were produced.")

    comparison = pd.DataFrame(metrics_rows)
    comparison["dataset"] = pd.Categorical(comparison["dataset"], categories=list(DATASET_ORDER), ordered=True)
    comparison = comparison.sort_values("dataset", kind="mergesort")
    comparison["dataset"] = comparison["dataset"].astype(str)
    write_csv(comparison, output_dir / "comparison_summary.csv")
    logger.info("Finished 4.10 Feature Space Analysis.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 4.10 Feature Space Analysis for VSL and ASL.")
    parser.add_argument("--input-4-6", type=Path, default=DEFAULT_INPUT_46, help="Directory containing output_4_6 CSV files.")
    parser.add_argument("--input-4-7", type=Path, default=DEFAULT_INPUT_47, help="Directory containing output_4_7 CSV files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory for 4.10 results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.input_4_6, args.input_4_7, args.output)


if __name__ == "__main__":
    main()
