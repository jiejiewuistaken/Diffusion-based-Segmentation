from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass
class FeatureBuildConfig:
    observation_weeks: int = 12
    timezone: str = "UTC"
    fillna_zero: bool = True


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _load_from_path(path: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    if not path or path == "REPLACE_ME":
        return pd.DataFrame()
    if not os.path.exists(path):
        # Gracefully degrade if file not present in local FS
        return pd.DataFrame()
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(path, sep=sep, parse_dates=parse_dates)
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    # Fallback: try CSV
    return pd.read_csv(path, parse_dates=parse_dates)


def _load_tables_from_data_dict(dd: dict) -> Dict[str, pd.DataFrame]:
    def path_of(name: str) -> Optional[str]:
        coll = dd.get("collections", {}).get(name, {})
        return coll.get("path")

    sv = _load_from_path(path_of("short_videos"), parse_dates=["event_ts"])
    live = _load_from_path(path_of("live_sessions"), parse_dates=["start_ts", "end_ts"])
    shop = _load_from_path(path_of("shop_window"), parse_dates=["exposure_ts"])
    orders = _load_from_path(path_of("orders"), parse_dates=["order_ts"])
    return {"short_videos": sv, "live_sessions": live, "shop_window": shop, "orders": orders}


def _infer_end_date(tables: Dict[str, pd.DataFrame]) -> Optional[pd.Timestamp]:
    candidates = []
    if not tables["short_videos"].empty:
        candidates.append(pd.to_datetime(tables["short_videos"]["event_ts"]).max())
    if not tables["live_sessions"].empty:
        candidates.append(pd.to_datetime(tables["live_sessions"]["start_ts"]).max())
    if not tables["shop_window"].empty:
        candidates.append(pd.to_datetime(tables["shop_window"]["exposure_ts"]).max())
    if not tables["orders"].empty:
        candidates.append(pd.to_datetime(tables["orders"]["order_ts"]).max())
    if not candidates:
        return None
    return max(candidates)


def _to_week(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts).dt.to_period("W-MON").dt.start_time


def _filter_last_weeks(df: pd.DataFrame, ts_col: str, end_date: pd.Timestamp, observation_weeks: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["week"] = _to_week(df[ts_col])
    end_week = _to_week(pd.Series([end_date]))[0]
    start_week = end_week - observation_weeks * pd.offsets.Week(weekday=0) + pd.offsets.Week(weekday=0)
    mask = (df["week"] >= start_week) & (df["week"] <= end_week)
    return df.loc[mask]


def _agg_short_videos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = df.copy()
    if "is_with_cart" in df.columns:
        df["is_with_cart"] = df["is_with_cart"].astype(bool)
    else:
        df["is_with_cart"] = False
    for col in ["views", "clicks", "likes", "comments", "shares", "orders", "revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0
    df_week = (
        df.groupby(["author_id", "week"], as_index=False)
        .agg(
            post_count=("content_id", "count"),
            with_cart_count=("is_with_cart", "sum"),
            non_cart_count=("is_with_cart", lambda s: (~s).sum()),
            views=("views", "sum"),
            clicks=("clicks", "sum"),
            likes=("likes", "sum"),
            comments=("comments", "sum"),
            shares=("shares", "sum"),
            orders=("orders", "sum"),
            revenue=("revenue", "sum"),
        )
    )
    df_sum = (
        df_week.groupby("author_id", as_index=False)
        .agg({
            "post_count": "sum",
            "with_cart_count": "sum",
            "non_cart_count": "sum",
            "views": "sum",
            "clicks": "sum",
            "likes": "sum",
            "comments": "sum",
            "shares": "sum",
            "orders": "sum",
            "revenue": "sum",
        })
        .rename(columns=lambda c: f"video_{c}" if c != "author_id" else c)
    )
    return df_week, df_sum


def _agg_live_sessions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = df.copy()
    for col in ["duration_sec", "views", "clicks", "orders", "revenue", "peak_concurrent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0
    df_week = (
        df.groupby(["author_id", "week"], as_index=False)
        .agg(
            sessions=("live_id", "count"),
            duration_sec=("duration_sec", "sum"),
            views=("views", "sum"),
            clicks=("clicks", "sum"),
            orders=("orders", "sum"),
            revenue=("revenue", "sum"),
            peak_concurrent=("peak_concurrent", "max"),
        )
    )
    df_sum = (
        df_week.groupby("author_id", as_index=False)
        .agg({
            "sessions": "sum",
            "duration_sec": "sum",
            "views": "sum",
            "clicks": "sum",
            "orders": "sum",
            "revenue": "sum",
            "peak_concurrent": "max",
        })
        .rename(columns=lambda c: f"live_{c}" if c != "author_id" else c)
    )
    return df_week, df_sum


def _agg_shop(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = df.copy()
    for col in ["clicks", "add_to_cart", "orders", "revenue"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0
    df_week = (
        df.groupby(["author_id", "week"], as_index=False)
        .agg(
            exposures=("exposure_ts", "count"),
            clicks=("clicks", "sum"),
            add_to_cart=("add_to_cart", "sum"),
            orders=("orders", "sum"),
            revenue=("revenue", "sum"),
        )
    )
    df_sum = (
        df_week.groupby("author_id", as_index=False)
        .agg({
            "exposures": "sum",
            "clicks": "sum",
            "add_to_cart": "sum",
            "orders": "sum",
            "revenue": "sum",
        })
        .rename(columns=lambda c: f"shop_{c}" if c != "author_id" else c)
    )
    return df_week, df_sum


def _agg_orders(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = df.copy()
    for col in ["quantity", "amount", "margin"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0
    df_week = (
        df.groupby(["author_id", "week"], as_index=False)
        .agg(
            orders=("order_ts", "count"),
            units=("quantity", "sum"),
            gmv=("amount", "sum"),
            margin=("margin", "sum"),
            refunds=("order_ts", lambda s: 0),
        )
    )
    df_sum = (
        df_week.groupby("author_id", as_index=False)
        .agg({
            "orders": "sum",
            "units": "sum",
            "gmv": "sum",
            "margin": "sum",
            "refunds": "sum",
        })
        .rename(columns=lambda c: f"order_{c}" if c != "author_id" else c)
    )
    return df_week, df_sum


def build_features(
    data_dictionary_path: str,
    config: FeatureBuildConfig,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    dd = _read_yaml(data_dictionary_path)
    tables = _load_tables_from_data_dict(dd)

    # derive end date and weekly windows
    inferred_end = _infer_end_date(tables)
    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
    else:
        end_ts = inferred_end
    if end_ts is None:
        # No data available
        return pd.DataFrame(columns=["author_id"])  # empty skeleton

    # add week columns and window filtering
    sv = _filter_last_weeks(tables["short_videos"], "event_ts", end_ts, config.observation_weeks)
    live = _filter_last_weeks(tables["live_sessions"], "start_ts", end_ts, config.observation_weeks)
    shop = _filter_last_weeks(tables["shop_window"], "exposure_ts", end_ts, config.observation_weeks)
    ords = _filter_last_weeks(tables["orders"], "order_ts", end_ts, config.observation_weeks)

    # aggregate
    sv_week, sv_sum = _agg_short_videos(sv)
    live_week, live_sum = _agg_live_sessions(live)
    shop_week, shop_sum = _agg_shop(shop)
    ord_week, ord_sum = _agg_orders(ords)

    # combine author-level summaries
    parts = []
    for df in [sv_sum, live_sum, shop_sum, ord_sum]:
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["author_id"])  # empty

    feat = parts[0]
    for df in parts[1:]:
        feat = feat.merge(df, on="author_id", how="outer")

    # derived ratios
    feat = feat.fillna(0)
    # video ratios
    if "video_clicks" in feat.columns and "video_views" in feat.columns:
        feat["video_ctr"] = np.where(feat["video_views"] > 0, feat["video_clicks"] / feat["video_views"], 0.0)
    if "video_with_cart_count" in feat.columns and "video_post_count" in feat.columns:
        feat["video_with_cart_share"] = np.where(
            feat["video_post_count"] > 0,
            feat["video_with_cart_count"] / feat["video_post_count"],
            0.0,
        )
    # revenue shares
    for col in ["video_revenue", "live_revenue", "shop_revenue"]:
        if col not in feat.columns:
            feat[col] = 0.0
    feat["total_revenue"] = feat[["video_revenue", "live_revenue", "shop_revenue"]].sum(axis=1)
    for col in ["video_revenue", "live_revenue", "shop_revenue"]:
        share_col = col.replace("_revenue", "_rev_share")
        feat[share_col] = np.where(feat["total_revenue"] > 0, feat[col] / feat["total_revenue"], 0.0)

    # unit conversions
    if "live_duration_sec" in feat.columns:
        feat["live_duration_hours"] = feat["live_duration_sec"] / 3600.0

    if config.fillna_zero:
        feat = feat.fillna(0)

    return feat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build author weekly features for operating mode analysis")
    p.add_argument("--data-dict", required=True, help="Path to data_dictionary YAML")
    p.add_argument("--config", required=False, help="Path to feature_config.yaml")
    p.add_argument("--end-date", required=False, help="YYYY-MM-DD; defaults to max date in data")
    p.add_argument("--output", required=True, help="Where to write features parquet/csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.config and os.path.exists(args.config):
        cfg_raw = _read_yaml(args.config)
        cfg = FeatureBuildConfig(
            observation_weeks=cfg_raw.get("observation_weeks", 12),
            timezone=cfg_raw.get("timezone", "UTC"),
            fillna_zero=cfg_raw.get("fillna_zero", True),
        )
    else:
        cfg = FeatureBuildConfig()

    features = build_features(
        data_dictionary_path=args.data_dict,
        config=cfg,
        end_date=args.end_date,
    )

    _ensure_dir(args.output)
    ext = os.path.splitext(args.output)[1].lower()
    if ext in [".parquet", ".pq"]:
        features.to_parquet(args.output, index=False)
    else:
        features.to_csv(args.output, index=False)
    print(f"Wrote features: {args.output} rows={len(features)} cols={len(features.columns)}")


if __name__ == "__main__":
    main()
