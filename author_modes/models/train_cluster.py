from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    import umap
except Exception:  # pragma: no cover
    umap = None

try:
    import hdbscan
except Exception:  # pragma: no cover
    hdbscan = None


@dataclass
class ModelConfig:
    scaler: str = "standard"
    embedding_method: str = "umap"
    embedding_dim: int = 16
    random_state: int = 42
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    clustering_algo: str = "hdbscan"
    hdbscan_min_cluster_size: int = 80
    hdbscan_min_samples: int = 10
    hdbscan_cluster_selection_epsilon: float = 0.0
    kmeans_k_min: int = 5
    kmeans_k_max: int = 25
    feature_drop_columns: List[str] = None
    feature_exclude: List[str] = None


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _select_feature_columns(df: pd.DataFrame, drop_columns: Optional[List[str]], exclude: Optional[List[str]]) -> List[str]:
    drop_columns = drop_columns or []
    exclude = set(exclude or [])
    candidates = [c for c in df.columns if c not in drop_columns and c not in exclude]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return numeric


def train(
    features_path: str,
    model_config_path: str,
    output_path: str,
) -> None:
    cfg_raw = _read_yaml(model_config_path)

    cfg = ModelConfig(
        scaler=cfg_raw.get("scaler", "standard"),
        embedding_method=cfg_raw.get("embedding", {}).get("method", "umap"),
        embedding_dim=cfg_raw.get("embedding", {}).get("n_components", 16),
        random_state=cfg_raw.get("embedding", {}).get("random_state", 42),
        umap_n_neighbors=cfg_raw.get("embedding", {}).get("umap", {}).get("n_neighbors", 15),
        umap_min_dist=cfg_raw.get("embedding", {}).get("umap", {}).get("min_dist", 0.1),
        clustering_algo=cfg_raw.get("clustering", {}).get("algorithm", "hdbscan"),
        hdbscan_min_cluster_size=cfg_raw.get("clustering", {}).get("hdbscan", {}).get("min_cluster_size", 80),
        hdbscan_min_samples=cfg_raw.get("clustering", {}).get("hdbscan", {}).get("min_samples", 10),
        hdbscan_cluster_selection_epsilon=cfg_raw.get("clustering", {}).get("hdbscan", {}).get("cluster_selection_epsilon", 0.0),
        kmeans_k_min=cfg_raw.get("clustering", {}).get("kmeans", {}).get("k_min", 5),
        kmeans_k_max=cfg_raw.get("clustering", {}).get("kmeans", {}).get("k_max", 25),
        feature_drop_columns=cfg_raw.get("features", {}).get("drop_columns", ["author_id"]),
        feature_exclude=cfg_raw.get("features", {}).get("exclude", []),
    )

    # Load features
    ext = os.path.splitext(features_path)[1].lower()
    if ext in [".parquet", ".pq"]:
        feat = pd.read_parquet(features_path)
    else:
        feat = pd.read_csv(features_path)

    if "author_id" not in feat.columns:
        raise ValueError("features must include author_id column")

    feature_cols = _select_feature_columns(
        df=feat,
        drop_columns=cfg.feature_drop_columns,
        exclude=cfg.feature_exclude,
    )
    X = feat[feature_cols].fillna(0).values

    # Scaling
    if cfg.scaler == "standard":
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    elif cfg.scaler == "robust":
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)
    else:
        scaler = None
        Xs = X

    # Embedding
    if cfg.embedding_method == "pca":
        reducer = PCA(n_components=cfg.embedding_dim, random_state=cfg.random_state)
        Z = reducer.fit_transform(Xs)
        embed_name = "pca"
    elif cfg.embedding_method == "umap":
        if umap is None:
            raise ImportError("umap-learn not installed; set embedding.method to 'pca' or 'none'")
        reducer = umap.UMAP(
            n_components=cfg.embedding_dim,
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            metric="euclidean",
            random_state=cfg.random_state,
        )
        Z = reducer.fit_transform(Xs)
        embed_name = "umap"
    else:
        reducer = None
        Z = Xs
        embed_name = "none"

    # Clustering
    if cfg.clustering_algo == "hdbscan":
        if hdbscan is None:
            raise ImportError("hdbscan not installed; set clustering.algorithm to 'kmeans'")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cfg.hdbscan_min_cluster_size,
            min_samples=cfg.hdbscan_min_samples,
            cluster_selection_epsilon=cfg.hdbscan_cluster_selection_epsilon,
        )
        labels = clusterer.fit_predict(Z)
        probs = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))
    else:
        best_inertia = np.inf
        best_k = None
        best_model = None
        for k in range(cfg.kmeans_k_min, cfg.kmeans_k_max + 1):
            km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=10)
            km.fit(Z)
            if km.inertia_ < best_inertia:
                best_inertia = km.inertia_
                best_k = k
                best_model = km
        clusterer = best_model
        labels = clusterer.predict(Z)
        probs = np.ones_like(labels, dtype=float)

    # Assemble outputs
    out = feat[["author_id"]].copy()
    out["cluster_label"] = labels
    out["cluster_confidence"] = probs

    # Save artifacts
    _ensure_dir(output_path)
    base, ext = os.path.splitext(output_path)
    out_path = output_path if ext in [".csv", ".parquet", ".pq"] else base + ".csv"

    if out_path.endswith((".parquet", ".pq")):
        out.to_parquet(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)

    # Save embeddings for visualization
    embed_df = pd.DataFrame(Z, columns=[f"{embed_name}_{i:02d}" for i in range(Z.shape[1])])
    embed_df.insert(0, "author_id", out["author_id"].values)
    embed_path = base + f".{embed_name}.csv"
    embed_df.to_csv(embed_path, index=False)

    print(f"Saved clusters to {out_path}; embeddings to {embed_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train clustering model for creator operating modes")
    p.add_argument("--features", required=True, help="Path to features parquet/csv")
    p.add_argument("--config", required=True, help="Path to model_config.yaml")
    p.add_argument("--output", required=True, help="Output path for cluster assignments")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(
        features_path=args.features,
        model_config_path=args.config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
