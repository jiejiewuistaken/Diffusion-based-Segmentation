from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score


def _load(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def evaluate_clusters(
    features_path: str,
    embeddings_path: Optional[str],
    assignments_path: str,
    feature_prefixes: Optional[list[str]] = None,
) -> dict:
    feat = _load(features_path)
    assign = _load(assignments_path)
    df = feat.merge(assign, on="author_id", how="inner")

    # choose representation for metrics: embeddings if provided else raw features
    if embeddings_path and os.path.exists(embeddings_path):
        emb = _load(embeddings_path)
        X = emb.drop(columns=[c for c in ["author_id", "cluster_label", "cluster_confidence"] if c in emb.columns]).values
    else:
        # use numeric feature columns
        num_cols = [c for c in df.columns if c not in ["author_id", "cluster_label", "cluster_confidence"] and pd.api.types.is_numeric_dtype(df[c])]
        X = df[num_cols].fillna(0).values

    labels = df["cluster_label"].values

    metrics = {}
    # skip metrics if only one cluster or presence of noise-only labels
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1 and not (len(unique_labels) == 1 and unique_labels[0] == -1):
        try:
            metrics["silhouette"] = float(silhouette_score(X, labels))
        except Exception:
            metrics["silhouette"] = None
        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        except Exception:
            metrics["davies_bouldin"] = None

    # cluster profiles: counts and revenue summaries if present
    grouped = df.groupby("cluster_label")
    profile = grouped.size().rename("count").to_frame()

    # revenue medians per cluster
    revenue_cols = ["video_revenue", "live_revenue", "shop_revenue", "total_revenue"]
    for col in revenue_cols:
        if col in df.columns:
            profile[col + "_median"] = grouped[col].median()

    # revenue share means per cluster (if available for channels)
    share_cols = ["video_rev_share", "live_rev_share", "shop_rev_share"]
    for col in share_cols:
        if col in df.columns:
            profile[col + "_mean"] = grouped[col].mean()

    profile = profile.reset_index()

    return {"metrics": metrics, "profile": profile}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate clustering quality and produce simple profiles")
    p.add_argument("--features", required=True)
    p.add_argument("--assignments", required=True)
    p.add_argument("--embeddings", required=False)
    p.add_argument("--output", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    res = evaluate_clusters(
        features_path=args.features,
        embeddings_path=args.embeddings,
        assignments_path=args.assignments,
    )
    metrics = res["metrics"]
    profile = res["profile"]

    base, _ = os.path.splitext(args.output)
    # write metrics json
    import json

    with open(base + ".metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    # write profile csv
    profile.to_csv(base + ".profile.csv", index=False)
    print(f"Wrote metrics to {base}.metrics.json and profile to {base}.profile.csv")


if __name__ == "__main__":
    main()
