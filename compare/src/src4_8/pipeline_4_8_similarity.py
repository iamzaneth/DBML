from __future__ import annotations

import argparse
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


QUALITY_PATH_HINTS = ("output_4_5", "quality")
REGION_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<region>[BTN])$")

HANDSHAPE_FEATURE_COLUMNS = (
    "valid_frame_ratio",
    "motion_magnitude",
    "thumb_angle_1_mean",
    "thumb_angle_1_std",
    "thumb_angle_2_mean",
    "thumb_angle_2_std",
    "thumb_tip_wrist_ratio_mean",
    "index_angle_1_mean",
    "index_angle_1_std",
    "index_angle_2_mean",
    "index_angle_2_std",
    "index_angle_3_mean",
    "index_angle_3_std",
    "index_tip_wrist_ratio_mean",
    "middle_angle_1_mean",
    "middle_angle_1_std",
    "middle_angle_2_mean",
    "middle_angle_2_std",
    "middle_angle_3_mean",
    "middle_angle_3_std",
    "middle_tip_wrist_ratio_mean",
    "ring_angle_1_mean",
    "ring_angle_1_std",
    "ring_angle_2_mean",
    "ring_angle_2_std",
    "ring_angle_3_mean",
    "ring_angle_3_std",
    "ring_tip_wrist_ratio_mean",
    "pinky_angle_1_mean",
    "pinky_angle_1_std",
    "pinky_angle_2_mean",
    "pinky_angle_2_std",
    "pinky_angle_3_mean",
    "pinky_angle_3_std",
    "pinky_tip_wrist_ratio_mean",
    "thumb_index_distance_ratio_mean",
    "thumb_index_distance_ratio_std",
    "index_middle_distance_ratio_mean",
    "index_middle_distance_ratio_std",
    "middle_ring_distance_ratio_mean",
    "middle_ring_distance_ratio_std",
    "ring_pinky_distance_ratio_mean",
    "ring_pinky_distance_ratio_std",
    "index_pinky_distance_ratio_mean",
    "index_pinky_distance_ratio_std",
    "palm_width_ratio_mean",
    "openness_mean",
    "curvature_proxy_mean",
)

LOCATION_FEATURE_COLUMNS = (
    "head_ratio",
    "shoulder_ratio",
    "chest_ratio",
    "waist_ratio",
    "below_waist_ratio",
)

ORIENTATION_FEATURE_COLUMNS = (
    "finger_dir_x_mean",
    "finger_dir_y_mean",
    "finger_dir_z_mean",
    "palm_normal_x_mean",
    "palm_normal_y_mean",
    "palm_normal_z_mean",
)

MOTION_FEATURE_COLUMNS = (
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

FEATURE_COLUMNS_BY_FILE = {
    "handshape_features.csv": HANDSHAPE_FEATURE_COLUMNS,
    "location_video.csv": LOCATION_FEATURE_COLUMNS,
    "orientation_video.csv": ORIENTATION_FEATURE_COLUMNS,
    "motion_features_per_video.csv": MOTION_FEATURE_COLUMNS,
}


class PipelineError(RuntimeError):
    pass


def read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "cp1258", "latin1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise PipelineError(f"Cannot read CSV with supported encodings: {path}") from last_error


def discover_csv_files(roots: list[Path], output_dir: Path | None = None) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    output_resolved = output_dir.resolve() if output_dir else None
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            resolved = path.resolve()
            if output_resolved and (resolved == output_resolved or output_resolved in resolved.parents):
                continue
            if resolved not in seen:
                files.append(path)
                seen.add(resolved)
    return sorted(files, key=lambda p: str(p).lower())


def profile_tables(paths: list[Path]) -> tuple[dict[Path, pd.DataFrame], list[dict[str, Any]]]:
    tables: dict[Path, pd.DataFrame] = {}
    profiles: list[dict[str, Any]] = []
    for path in paths:
        df = read_csv_safely(path)
        tables[path] = df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in df.columns if col not in numeric_cols]
        profiles.append(
            {
                "path": str(path),
                "rows": int(len(df)),
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "numeric_count": len(numeric_cols),
                "categorical_count": len(categorical_cols),
            }
        )
    return tables, profiles


def load_schema_summary(schema_path: Path) -> dict[str, list[str]]:
    if not schema_path.exists():
        raise PipelineError(f"Schema summary not found: {schema_path}")
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    current_columns: list[str] = []
    for line in schema_path.read_text(encoding="utf-8").splitlines():
        section_match = SCHEMA_SECTION_RE.match(line)
        if section_match:
            if current_section is not None:
                sections[current_section] = current_columns
            current_section = section_match.group("section")
            current_columns = []
            continue
        if current_section is None:
            continue
        column_match = SCHEMA_COLUMN_RE.match(line)
        if column_match:
            current_columns.append(column_match.group("column"))
    if current_section is not None:
        sections[current_section] = current_columns
    return sections


def normalize_video_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\\", "/").strip()
    name = Path(text).name
    for suffix in (".npz", ".mp4", ".avi", ".mov", ".mkv"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.lower()


def find_col(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_mapping_file(mapping_path: Path) -> tuple[pd.DataFrame, str, str]:
    if not mapping_path.exists():
        raise PipelineError(f"Mapping file not found: {mapping_path}")
    mapping_df = read_csv_safely(mapping_path)
    columns = mapping_df.columns.tolist()
    asl_col = find_col(columns, ("asl_label",))
    vsl_col = find_col(columns, ("vsl_label",))
    if not asl_col or not vsl_col:
        raise PipelineError(
            f"Mapping file must contain `asl_label` and `vsl_label` columns: {mapping_path}"
        )
    return mapping_df, asl_col, vsl_col


def is_quality_table(path: Path) -> bool:
    lower = str(path).lower().replace("\\", "/")
    return any(hint in lower for hint in QUALITY_PATH_HINTS)


def is_feature_candidate(path: Path, df: pd.DataFrame) -> bool:
    if is_quality_table(path):
        return False
    normalized = path.as_posix().lower()
    return any(normalized.endswith(suffix) for suffix in ALLOWED_FEATURE_SUFFIXES)


def choose_video_source(df: pd.DataFrame) -> str | None:
    columns = df.columns.tolist()
    for group in (("video",), ("file",), ("video_name",), ("path",)):
        col = find_col(columns, group)
        if col:
            return col
    return None


def prepare_feature_tables(
    tables: dict[Path, pd.DataFrame],
    schema_summary: dict[str, list[str]],
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[str]]:
    prepared: list[pd.DataFrame] = []
    merge_info: list[dict[str, Any]] = []
    warnings: list[str] = []

    for path, df in tables.items():
        whitelisted_cols = FEATURE_COLUMNS_BY_FILE.get(path.name.lower())
        if whitelisted_cols is None:
            continue
        if not is_feature_candidate(path, df):
            continue
        section = canonical_schema_suffix(path)
        if section not in schema_summary:
            warnings.append(f"Skipped {path}: schema section not found in csv_fields_summary.md.")
            continue
        schema_columns = schema_summary[section]
        dataset_col = find_col(df.columns.tolist(), ("dataset",))
        gloss_col = find_col(df.columns.tolist(), ("gloss", "label"))
        video_col = choose_video_source(df)
        missing_cols = [col for col in whitelisted_cols if col not in df.columns]
        numeric_cols = [col for col in whitelisted_cols if col in df.columns]
        if missing_cols:
            warnings.append(f"{path}: missing whitelisted feature columns: {', '.join(missing_cols)}.")
        if not (dataset_col and gloss_col and video_col):
            warnings.append(f"Skipped {path}: missing merge keys.")
            continue
        if not numeric_cols:
            warnings.append(f"Skipped {path}: no whitelisted feature columns available.")
            continue

        selected_features = [col for col in feature_candidates if col in df.columns]
        missing_features = [col for col in feature_candidates if col not in df.columns]
        ignored_metadata = [col for col in metadata_columns if col in df.columns]
        if not selected_features:
            raise PipelineError(f"No valid feature columns found for {path}.")
        if missing_features:
            warnings.append(
                f"{Path(path).name}: missing schema-defined features: {', '.join(missing_features)}"
            )

        work = df[[dataset_col, gloss_col, video_col] + selected_features].copy()
        work = work.rename(columns={dataset_col: "dataset", gloss_col: "gloss", video_col: "video_source"})
        work["dataset"] = work["dataset"].astype(str).str.strip()
        work["gloss"] = work["gloss"].astype(str).str.strip()
        work["video_key"] = work["video_source"].map(normalize_video_key)
        work = work.drop(columns=["video_source"])
        work = work.replace([np.inf, -np.inf], np.nan)
        prefix = path.stem.lower()
        rename = {col: f"{prefix}__{col}" for col in selected_features}
        work = work.rename(columns=rename)
        feature_cols = list(rename.values())
        grouped = work.groupby(["dataset", "gloss", "video_key"], as_index=False)[feature_cols].mean()
        prepared.append(grouped)
        merge_info.append(
            {
                "path": str(path),
                "key": "dataset + gloss/label + normalized video/file/video_name/path",
                "rows_before": int(len(df)),
                "rows_after_video_aggregate": int(len(grouped)),
                "feature_group": infer_feature_group(section),
                "selected_features": len(selected_features),
                "ignored_metadata": len(ignored_metadata),
                "missing_features": len(missing_features),
                "selected_feature_names": selected_features,
                "missing_feature_names": missing_features,
                "metadata_feature_names": ignored_metadata,
            }
        )

    if not prepared:
        raise PipelineError("No usable feature CSV found. Need dataset, gloss/label, video/file key, and schema-defined features.")
    return prepared, merge_info, warnings


def infer_feature_group(section: str) -> str:
    if section.endswith("handshape_features.csv"):
        return "handshape"
    if section.endswith("location_video.csv"):
        return "location"
    if section.endswith("orientation_video.csv"):
        return "orientation"
    if section.endswith("motion_features_per_video.csv"):
        return "movement"
    return "other"


def merge_video_features(feature_tables: list[pd.DataFrame]) -> pd.DataFrame:
    merged = feature_tables[0]
    for table in feature_tables[1:]:
        merged = merged.merge(table, on=["dataset", "gloss", "video_key"], how="outer")
    feature_cols = [col for col in merged.columns if col not in ("dataset", "gloss", "video_key")]
    if not feature_cols:
        raise PipelineError("No feature columns remained after merge.")
    return merged


def get_feature_columns(video_features: pd.DataFrame) -> list[str]:
    cols = [col for col in video_features.columns if col not in ("dataset", "gloss", "video_key")]
    if not cols:
        raise PipelineError("No feature column found after merge.")
    return cols


def summarize_gloss_features(video_features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    grouped = video_features.groupby(["dataset", "gloss"], dropna=False)
    mean_df = grouped[feature_cols].mean().add_suffix("__mean")
    std_df = grouped[feature_cols].std(ddof=0).fillna(0).add_suffix("__std")
    count_df = grouped.size().rename("sample_count")
    out = pd.concat([count_df, mean_df, std_df], axis=1).reset_index()
    return out


def build_scaled_gloss_matrix(gloss_summary: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [col for col in gloss_summary.columns if col.endswith("__mean")]
    matrix = gloss_summary[feature_cols].replace([np.inf, -np.inf], np.nan)
    fill_values = matrix.mean(axis=0).fillna(0)
    matrix = matrix.fillna(fill_values)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    scaled_df = pd.DataFrame(scaled, columns=feature_cols, index=gloss_summary.index)
    return scaled_df, feature_cols


def dataset_lookup(gloss_summary: pd.DataFrame, scaled_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    lookup: dict[tuple[str, str], int] = {}
    for idx, row in gloss_summary.iterrows():
        lookup[(str(row["dataset"]).upper(), str(row["gloss"]))] = idx
    return lookup


def feature_group(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("angle", "distance_ratio", "palm_width", "openness", "curvature", "handshape")):
        return "handshape"
    if any(token in lower for token in ("motion", "velocity", "acceleration", "trajectory", "straightness", "direction_change", "dtw")):
        return "movement"
    if any(token in lower for token in ("finger_dir", "palm_normal", "orientation")):
        return "orientation"
    if any(token in lower for token in ("location", "head_", "shoulder_", "chest_", "waist_", "body", "face", "left_right")):
        return "spatial relationship"
    if any(token in lower for token in ("one_two", "has_left", "has_right")):
        return "structure"
    return "other"


def contribution_summary(vec_a: np.ndarray, vec_b: np.ndarray, feature_cols: list[str], top_n: int = 3) -> str:
    diffs = np.abs(vec_a - vec_b)
    by_group: dict[str, float] = {}
    for col, diff in zip(feature_cols, diffs):
        group = feature_group(col)
        by_group[group] = by_group.get(group, 0.0) + float(diff)
    ranked = sorted(by_group.items(), key=lambda item: item[1], reverse=True)
    total = sum(value for _, value in ranked) or 1.0
    parts = [f"{name} ({value / total:.1%})" for name, value in ranked[:top_n]]
    return "; ".join(parts)


def compute_inter_language_similarity(
    mapping_df: pd.DataFrame,
    asl_col: str,
    vsl_col: str,
    gloss_summary: pd.DataFrame,
    scaled_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    lookup = dataset_lookup(gloss_summary, scaled_df)
    sample_counts = {
        (str(row["dataset"]).upper(), str(row["gloss"])): int(row["sample_count"])
        for _, row in gloss_summary.iterrows()
    }
    rows: list[dict[str, Any]] = []
    missing_vsl = 0
    missing_asl = 0
    duplicate_pairs = 0
    seen_pairs: set[tuple[str, str]] = set()

    for _, row in mapping_df.iterrows():
        vsl_gloss = str(row[vsl_col]).strip()
        asl_gloss = str(row[asl_col]).strip()
        if not vsl_gloss or not asl_gloss or vsl_gloss.lower() == "nan" or asl_gloss.lower() == "nan":
            continue
        pair = (vsl_gloss, asl_gloss)
        if pair in seen_pairs:
            duplicate_pairs += 1
            continue
        seen_pairs.add(pair)
        vsl_idx = lookup.get(("VSL", vsl_gloss))
        asl_idx = lookup.get(("ASL", asl_gloss))
        if vsl_idx is None:
            missing_vsl += 1
            continue
        if asl_idx is None:
            missing_asl += 1
            continue
        vsl_vec = scaled_df.loc[vsl_idx, feature_cols].to_numpy().reshape(1, -1)
        asl_vec = scaled_df.loc[asl_idx, feature_cols].to_numpy().reshape(1, -1)
        cosine = float(cosine_similarity(vsl_vec, asl_vec)[0, 0])
        score = (cosine + 1.0) / 2.0
        rows.append(
            {
                "VSL_gloss": vsl_gloss,
                "ASL_gloss": asl_gloss,
                "cosine_similarity": cosine,
                "similarity_score": score,
                "sample_count_VSL": sample_counts[("VSL", vsl_gloss)],
                "sample_count_ASL": sample_counts[("ASL", asl_gloss)],
                "top_difference_features": contribution_summary(vsl_vec.ravel(), asl_vec.ravel(), feature_cols),
            }
        )
    stats = {
        "mapping_rows": int(len(mapping_df)),
        "mapping_pairs_unique": len(seen_pairs),
        "mapping_pairs_matched": len(rows),
        "missing_vsl": missing_vsl,
        "missing_asl": missing_asl,
        "duplicate_pairs_removed": duplicate_pairs,
        "excluded_pairs": missing_vsl + missing_asl + duplicate_pairs,
    }
    return pd.DataFrame(rows), stats


def summary_statistics(similarity: pd.DataFrame) -> pd.DataFrame:
    if similarity.empty:
        raise PipelineError("No mapped VSL-ASL gloss pair could be matched to feature vectors.")
    scores = similarity["similarity_score"]
    return pd.DataFrame(
        [
            {"metric": "Mean Similarity", "value": float(scores.mean())},
            {"metric": "Median", "value": float(scores.median())},
            {"metric": "Standard Deviation", "value": float(scores.std(ddof=0))},
            {"metric": "Minimum", "value": float(scores.min())},
            {"metric": "Maximum", "value": float(scores.max())},
            {"metric": "Percentile 25", "value": float(scores.quantile(0.25))},
            {"metric": "Percentile 75", "value": float(scores.quantile(0.75))},
        ]
    )


def split_region(gloss: str) -> tuple[str | None, str | None]:
    match = REGION_SUFFIX_RE.match(str(gloss).strip())
    if not match:
        return None, None
    return match.group("base"), match.group("region")


def compute_regional_similarity(
    gloss_summary: pd.DataFrame,
    scaled_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vsl = gloss_summary[gloss_summary["dataset"].astype(str).str.upper() == "VSL"].copy()
    region_parts = vsl["gloss"].map(split_region)
    vsl["base_gloss"] = [item[0] for item in region_parts]
    vsl["region"] = [item[1] for item in region_parts]
    vsl = vsl.dropna(subset=["base_gloss", "region"])
    rows: list[dict[str, Any]] = []
    for base_gloss, group in vsl.groupby("base_gloss"):
        region_to_idx = {row["region"]: idx for idx, row in group.iterrows()}
        for region_a, region_b in combinations(sorted(region_to_idx), 2):
            if {region_a, region_b}.issubset({"B", "T", "N"}):
                idx_a = region_to_idx[region_a]
                idx_b = region_to_idx[region_b]
                vec_a = scaled_df.loc[idx_a, feature_cols].to_numpy().reshape(1, -1)
                vec_b = scaled_df.loc[idx_b, feature_cols].to_numpy().reshape(1, -1)
                cosine = float(cosine_similarity(vec_a, vec_b)[0, 0])
                rows.append(
                    {
                        "base_gloss": base_gloss,
                        "region_A": region_a,
                        "region_B": region_b,
                        "cosine_similarity": cosine,
                        "similarity_score": (cosine + 1.0) / 2.0,
                    }
                )
    regional = pd.DataFrame(rows)
    if regional.empty:
        regional = pd.DataFrame(columns=["base_gloss", "region_A", "region_B", "cosine_similarity", "similarity_score"])
        regional_stats = pd.DataFrame(columns=["region_pair", "mean_similarity", "pair_count"])
    else:
        regional["region_pair"] = regional.apply(lambda r: "-".join(sorted([r["region_A"], r["region_B"]])), axis=1)
        regional_stats = (
            regional.groupby("region_pair")["similarity_score"]
            .agg(mean_similarity="mean", pair_count="size")
            .reset_index()
        )
    return regional.drop(columns=["region_pair"], errors="ignore"), regional_stats


def save_barh(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = (df["VSL_gloss"].astype(str) + " - " + df["ASL_gloss"].astype(str)).tolist()
    y = np.arange(len(df))
    ax.barh(y, df["similarity_score"], color="#3b82f6")
    ax.set_yticks(y, labels=labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Similarity score")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def create_figures(
    out_dir: Path,
    similarity: pd.DataFrame,
    regional: pd.DataFrame,
    video_features: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(similarity["similarity_score"], bins=30, color="#2563eb", edgecolor="white")
    ax.set_xlabel("Similarity score")
    ax.set_ylabel("Gloss pair count")
    ax.set_title("VSL-ASL similarity distribution")
    fig.tight_layout()
    fig.savefig(figures / "01_similarity_histogram.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.boxplot(similarity["similarity_score"], vert=True)
    ax.set_ylabel("Similarity score")
    ax.set_title("VSL-ASL similarity boxplot")
    fig.tight_layout()
    fig.savefig(figures / "02_similarity_boxplot.png", dpi=160)
    plt.close(fig)

    top_heat = similarity.sort_values("similarity_score", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(9, 8))
    heat_values = top_heat[["similarity_score"]].to_numpy()
    im = ax.imshow(heat_values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(top_heat)), labels=(top_heat["VSL_gloss"] + " - " + top_heat["ASL_gloss"]).tolist(), fontsize=7)
    ax.set_xticks([0], labels=["score"])
    ax.set_title("Top mapped gloss similarity heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "03_similarity_heatmap.png", dpi=160)
    plt.close(fig)

    save_barh(similarity.sort_values("similarity_score", ascending=False).head(20), figures / "04_top20_similarity.png", "Top 20 Similar Gloss Pairs")
    save_barh(similarity.sort_values("similarity_score", ascending=True).head(20), figures / "05_bottom20_similarity.png", "Bottom 20 Similar Gloss Pairs")

    fig, ax = plt.subplots(figsize=(6, 5))
    if regional.empty:
        ax.text(0.5, 0.5, "No regional pairs detected", ha="center", va="center")
        ax.axis("off")
    else:
        pivot = regional.copy()
        pivot["pair"] = pivot["region_A"] + "-" + pivot["region_B"]
        heat = pivot.pivot_table(index="base_gloss", columns="pair", values="similarity_score", aggfunc="mean")
        heat = heat.head(40)
        im = ax.imshow(heat.fillna(np.nan).to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(heat.columns)), labels=heat.columns.tolist())
        ax.set_yticks(np.arange(len(heat.index)), labels=heat.index.tolist(), fontsize=7)
        ax.set_title("Regional similarity heatmap")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "06_regional_heatmap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if regional.empty:
        ax.text(0.5, 0.5, "No regional pairs detected", ha="center", va="center")
        ax.axis("off")
    else:
        ax.hist(regional["similarity_score"], bins=min(20, max(5, len(regional))), color="#16a34a", edgecolor="white")
        ax.set_xlabel("Regional similarity score")
        ax.set_ylabel("Pair count")
        ax.set_title("Regional similarity distribution")
    fig.tight_layout()
    fig.savefig(figures / "07_regional_similarity_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    motion_candidates = [col for col in feature_cols if "motion" in col.lower() or "trajectory" in col.lower()]
    if not motion_candidates:
        ax.text(0.5, 0.5, "No motion feature available", ha="center", va="center")
        ax.axis("off")
    else:
        # The x-axis uses 1 - similarity as a compact proxy for vector difference.
        ax.scatter(1 - similarity["similarity_score"], similarity["similarity_score"], alpha=0.65, s=18, color="#ef4444")
        ax.set_xlabel("Overall vector difference proxy")
        ax.set_ylabel("Similarity score")
        ax.set_title("Similarity vs motion/vector difference")
    fig.tight_layout()
    fig.savefig(figures / "08_similarity_vs_motion_difference.png", dpi=160)
    plt.close(fig)


def write_validation_report(
    path: Path,
    profiles: list[dict[str, Any]],
    merge_info: list[dict[str, Any]],
    video_features: pd.DataFrame | None,
    gloss_summary: pd.DataFrame | None,
    mapping_stats: dict[str, int] | None,
    warnings: list[str],
) -> None:
    lines: list[str] = ["# Data Validation Report", ""]
    lines.append("## Files Read")
    for profile in profiles:
        lines.append(f"- `{profile['path']}`")
        lines.append(f"  - rows: {profile['rows']}")
        lines.append(f"  - numeric features: {profile['numeric_count']}")
        lines.append(f"  - categorical features: {profile['categorical_count']}")
        lines.append(f"  - columns: {', '.join(profile['columns'])}")
        dtype_text = ", ".join(f"{k}: {v}" for k, v in profile["dtypes"].items())
        lines.append(f"  - dtypes: {dtype_text}")
    lines.append("")
    lines.append("## Merge Keys")
    if merge_info:
        for item in merge_info:
            lines.append(
                f"- `{item['path']}`: {item['key']}; rows {item['rows_before']} -> {item['rows_after_video_aggregate']}; "
                f"selected features {item['selected_features']}; ignored metadata {item['ignored_metadata']}; "
                f"missing features {item['missing_features']}"
            )
    else:
        lines.append("- No mergeable feature tables found.")
    lines.append("")
    lines.append("## Dataset Counts")
    if video_features is not None:
        lines.append(f"- video rows after merge: {len(video_features)}")
        lines.append(f"- unique videos: {video_features['video_key'].nunique()}")
        lines.append(f"- unique gloss: {video_features[['dataset', 'gloss']].drop_duplicates().shape[0]}")
    if gloss_summary is not None:
        lines.append(f"- gloss feature rows: {len(gloss_summary)}")
    if mapping_stats:
        lines.append(f"- mapping rows: {mapping_stats['mapping_rows']}")
        lines.append(f"- mapping unique pairs: {mapping_stats['mapping_pairs_unique']}")
        lines.append(f"- gloss pairs found in mapping and features: {mapping_stats['mapping_pairs_matched']}")
        lines.append(f"- VSL gloss not found: {mapping_stats['missing_vsl']}")
        lines.append(f"- ASL gloss not found: {mapping_stats['missing_asl']}")
        lines.append(f"- duplicate mapping pairs removed: {mapping_stats['duplicate_pairs_removed']}")
        lines.append(f"- gloss pairs excluded: {mapping_stats['excluded_pairs']}")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    video_features: pd.DataFrame,
    gloss_summary: pd.DataFrame,
    feature_cols: list[str],
    similarity: pd.DataFrame,
    stats: pd.DataFrame,
    top_similar: pd.DataFrame,
    top_different: pd.DataFrame,
    regional_stats: pd.DataFrame,
    warnings: list[str],
) -> None:
    stat_map = dict(zip(stats["metric"], stats["value"]))
    lines = [
        "# Report 4.8 - Similarity Analysis between VSL and WLASL",
        "",
        "## Pipeline",
        "The pipeline reads available CSV outputs, validates merge keys, merges per-video feature tables, aggregates video features into gloss-level vectors, standardizes all gloss vectors with one StandardScaler, then computes cosine similarity for mapped VSL-ASL gloss pairs. No deep learning model is trained and no new feature is extracted.",
        "",
        "## Dataset Summary",
        f"- video rows after merge: {len(video_features)}",
        f"- unique videos: {video_features['video_key'].nunique()}",
        f"- gloss rows: {len(gloss_summary)}",
        f"- schema-selected feature columns used before gloss aggregation: {len(feature_cols)}",
        f"- mapped VSL-ASL pairs scored: {len(similarity)}",
        "",
        "## Similarity Statistics",
        f"- Mean Similarity: {stat_map.get('Mean Similarity', math.nan):.6f}",
        f"- Median: {stat_map.get('Median', math.nan):.6f}",
        f"- Standard Deviation: {stat_map.get('Standard Deviation', math.nan):.6f}",
        f"- Minimum: {stat_map.get('Minimum', math.nan):.6f}",
        f"- Maximum: {stat_map.get('Maximum', math.nan):.6f}",
        "",
        "## Top Similar",
    ]
    for _, row in top_similar.head(10).iterrows():
        lines.append(f"- {row['VSL_gloss']} ↔ {row['ASL_gloss']}: {row['similarity_score']:.4f}")
    lines.append("")
    lines.append("## Top Different")
    for _, row in top_different.head(10).iterrows():
        lines.append(f"- {row['VSL_gloss']} ↔ {row['ASL_gloss']}: {row['similarity_score']:.4f}; main differences: {row['top_difference_features']}")
    lines.append("")
    lines.append("## Regional Similarity")
    if regional_stats.empty:
        lines.append("- No VSL gloss with `_B`, `_T`, `_N` regional suffix was detected with enough paired regions.")
    else:
        for _, row in regional_stats.iterrows():
            lines.append(f"- {row['region_pair']}: mean {row['mean_similarity']:.4f} across {int(row['pair_count'])} pairs")
    lines.append("")
    lines.append("## Automatic Notes")
    mean_score = stat_map.get("Mean Similarity", math.nan)
    std_score = stat_map.get("Standard Deviation", math.nan)
    if not math.isnan(mean_score):
        if mean_score >= 0.7:
            lines.append("- Average VSL-ASL feature similarity is high under the extracted feature space.")
        elif mean_score >= 0.5:
            lines.append("- Average VSL-ASL feature similarity is moderate; top and bottom glosses should be inspected separately.")
        else:
            lines.append("- Average VSL-ASL feature similarity is low, suggesting substantial cross-language variation in the extracted features.")
    if not math.isnan(std_score) and std_score > 0.15:
        lines.append("- Similarity scores are widely dispersed, so individual gloss-level analysis is important.")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(data_roots: list[Path], output_dir: Path, mapping_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = (Path(__file__).resolve().parents[2] / "results" / "csv_fields_summary.md").resolve()
    schema_summary = load_schema_summary(schema_path)
    csv_files = discover_csv_files(data_roots, output_dir)
    mapping_path = mapping_path.resolve()
    if mapping_path not in [path.resolve() for path in csv_files]:
        csv_files.append(mapping_path)
    if not csv_files:
        raise PipelineError(f"No CSV files found under: {', '.join(str(root) for root in data_roots)}")
    tables, profiles = profile_tables(csv_files)
    mapping_df, asl_col, vsl_col = load_mapping_file(mapping_path)
    feature_tables, merge_info, warnings = prepare_feature_tables(tables, schema_summary)
    warnings.append(f"Mapping file used exactly: {mapping_path} with columns `{vsl_col}` and `{asl_col}`.")

    for item in merge_info:
        print("Detected feature file:")
        print(Path(item["path"]).name)
        print("Selected features:")
        print(item["selected_features"])
        print("Ignored metadata:")
        print(item["ignored_metadata"])
        print("Missing features:")
        print(item["missing_features"])
        print()

    video_features = merge_video_features(feature_tables)
    feature_cols = get_feature_columns(video_features)
    group_totals = {"handshape": 0, "orientation": 0, "location": 0, "movement": 0}
    for item in merge_info:
        group = item.get("feature_group", "other")
        if group in group_totals:
            group_totals[group] += int(item["selected_features"])
    print("After merge:")
    print(f"Handshape features : {group_totals['handshape']}")
    print(f"Orientation features : {group_totals['orientation']}")
    print(f"Location features : {group_totals['location']}")
    print(f"Movement features : {group_totals['movement']}")
    print(f"Total features : {len(feature_cols)}")
    print()
    gloss_summary = summarize_gloss_features(video_features, feature_cols)
    scaled_df, gloss_feature_cols = build_scaled_gloss_matrix(gloss_summary)
    similarity, mapping_stats = compute_inter_language_similarity(
        mapping_df, asl_col, vsl_col, gloss_summary, scaled_df, gloss_feature_cols
    )
    stats = summary_statistics(similarity)
    top_similar = similarity.sort_values("similarity_score", ascending=False).head(20)
    top_different = similarity.sort_values("similarity_score", ascending=True).head(20)
    regional, regional_stats = compute_regional_similarity(gloss_summary, scaled_df, gloss_feature_cols)

    gloss_summary.to_csv(output_dir / "gloss_feature_summary.csv", index=False, encoding="utf-8-sig")
    similarity.to_csv(output_dir / "gloss_similarity.csv", index=False, encoding="utf-8-sig")
    regional.to_csv(output_dir / "regional_similarity.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(output_dir / "summary_statistics.csv", index=False, encoding="utf-8-sig")
    top_similar.to_csv(output_dir / "top_similar.csv", index=False, encoding="utf-8-sig")
    top_different.to_csv(output_dir / "top_different.csv", index=False, encoding="utf-8-sig")

    create_figures(output_dir, similarity, regional, video_features, gloss_feature_cols)
    write_validation_report(
        output_dir / "data_validation_report.md",
        profiles,
        merge_info,
        video_features,
        gloss_summary,
        mapping_stats,
        warnings,
    )
    write_report(
        output_dir / "report.md",
        video_features,
        gloss_summary,
        gloss_feature_cols,
        similarity,
        stats,
        top_similar,
        top_different,
        regional_stats,
        warnings,
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    src_dir = script_dir.parent
    compare_dir = src_dir.parent
    parser = argparse.ArgumentParser(description="Pipeline 4.8 VSL-ASL similarity analysis")
    parser.add_argument(
        "--data-root",
        action="append",
        type=Path,
        default=None,
        help="Directory to recursively scan for CSV files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=compare_dir / "results" / "output_4_8",
        help="Output directory for reports, CSVs, and figures.",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=script_dir / "mapping.csv",
        help="Exact mapping CSV to use. Default: mapping.csv next to this script.",
    )
    parser.set_defaults(
        default_data_roots=[
            src_dir,
            compare_dir / "results" / "output_4_5",
            compare_dir / "results" / "output_4_6",
            compare_dir / "results" / "output_4_7",
        ]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = args.data_root if args.data_root else args.default_data_roots
    try:
        run_pipeline([root.resolve() for root in roots], args.output_dir.resolve(), args.mapping_file.resolve())
    except PipelineError as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "data_validation_report.md").write_text(f"# Data Validation Report\n\nERROR: {exc}\n", encoding="utf-8")
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
