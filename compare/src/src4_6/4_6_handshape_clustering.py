import os

import numpy as np
import pandas as pd

from utils_4_6 import OUTPUT_DIR, read_csv, save_bar, write_csv


def kmeans_numpy(features, k=8, max_iter=100, seed=42):
    rng = np.random.default_rng(seed)
    n_samples = features.shape[0]
    k = min(k, n_samples)
    centers = features[rng.choice(n_samples, size=k, replace=False)].copy()

    for _ in range(max_iter):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        new_centers = centers.copy()
        for cluster_id in range(k):
            members = features[labels == cluster_id]
            if len(members):
                new_centers[cluster_id] = members.mean(axis=0)
        if np.allclose(centers, new_centers, atol=1e-5):
            break
        centers = new_centers

    return labels, centers


def describe_cluster(row):
    openness = row.get("openness_mean", np.nan)
    curvature = row.get("curvature_proxy_mean", np.nan)
    spread = row.get("index_pinky_distance_ratio_mean", np.nan)

    if openness >= 2.1 and spread >= 1.2:
        return "open_spread_proxy"
    if openness <= 1.35:
        return "closed_proxy"
    if np.isfinite(curvature) and curvature <= 120:
        return "curved_proxy"
    if spread <= 0.75:
        return "compact_proxy"
    return "neutral_proxy"


def main(k=8):
    df = read_csv("handshape_features.csv")
    df = df[df["hand"].isin(["left", "right"])].copy()
    numeric_cols = [
        col
        for col in df.columns
        if col
        not in {
            "dataset",
            "gloss",
            "video",
            "path",
            "hand",
            "error",
        }
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    feature_df = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    keep = feature_df.notna().mean(axis=1) >= 0.8
    df = df[keep].copy()
    feature_df = feature_df[keep].fillna(feature_df.median(numeric_only=True))
    std = feature_df.std(axis=0).replace(0, 1)
    features = ((feature_df - feature_df.mean(axis=0)) / std).to_numpy(dtype=np.float32)

    labels, _ = kmeans_numpy(features, k=k)
    df["handshape_cluster"] = labels

    cluster_profile = df.groupby("handshape_cluster")[numeric_cols].mean(numeric_only=True)
    cluster_profile["cluster_label_proxy"] = cluster_profile.apply(describe_cluster, axis=1)
    df = df.merge(
        cluster_profile[["cluster_label_proxy"]],
        left_on="handshape_cluster",
        right_index=True,
        how="left",
    )

    write_csv(df, "handshape_clusters.csv")

    summary = (
        df.groupby(["dataset", "handshape_cluster", "cluster_label_proxy"])
        .size()
        .reset_index(name="count")
    )
    summary["dataset_total"] = summary.groupby("dataset")["count"].transform("sum")
    summary["percentage"] = summary["count"] / summary["dataset_total"] * 100
    summary = summary.sort_values(["dataset", "count"], ascending=[True, False])
    write_csv(summary, "handshape_cluster_summary.csv")

    for dataset, part in summary.groupby("dataset"):
        top = part.head(8).set_index("cluster_label_proxy")["percentage"]
        save_bar(
            top,
            f"{dataset}: handshape proxy cluster distribution",
            "Percentage (%)",
            os.path.join(OUTPUT_DIR, f"{dataset}_handshape_clusters.png"),
        )

    pca_path = os.path.join(OUTPUT_DIR, "handshape_pca.png")
    centered = features - features.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for dataset in sorted(df["dataset"].unique()):
        mask = df["dataset"].to_numpy() == dataset
        plt.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.35, label=dataset)
    plt.title("Handshape feature PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pca_path, dpi=160)
    plt.close()

    print("Saved handshape_clusters.csv, handshape_cluster_summary.csv and plots")


if __name__ == "__main__":
    main()
