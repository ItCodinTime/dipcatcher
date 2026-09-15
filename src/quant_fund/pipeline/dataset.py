"""Build gold feature/label panels and numpy design matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.data.ingest import ingest
from quant_fund.data.lake import Lake
from quant_fund.features.engine import build_features
from quant_fund.labels.engine import build_labels
from quant_fund.models.ranking import available_features


def ensure_silver(config: AppConfig, *, refresh: bool = False) -> pl.DataFrame:
    lake = Lake(Path(config.data.root))
    if refresh or not lake.exists("silver/bars.parquet"):
        ingest(config)
    return lake.read_parquet("silver/bars.parquet")


def build_gold(config: AppConfig, *, refresh: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    lake = Lake(Path(config.data.root))
    bars = ensure_silver(config, refresh=refresh)
    feats = build_features(bars, config)
    labs = build_labels(bars, config)
    lake.write_parquet(feats, "gold/features.parquet")
    lake.write_parquet(labs, "gold/labels.parquet")
    return feats, labs


def panel(
    config: AppConfig, feature_names: list[str] | None = None, label: str | None = None
) -> pl.DataFrame:
    lake = Lake(Path(config.data.root))
    if not lake.exists("gold/features.parquet"):
        feats, labs = build_gold(config)
    else:
        feats = lake.read_parquet("gold/features.parquet")
        labs = lake.read_parquet("gold/labels.parquet")
    keys = ["security_id", "event_time"]
    lab_cols = [c for c in labs.columns if c.startswith("future_") or c in keys]
    out = feats.join(labs.select(lab_cols), on=keys, how="inner")
    return out.sort(["event_time", "security_id"])


def design_matrix(
    frame: pl.DataFrame,
    label: str,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    feats = available_features(frame.columns, feature_names)
    if not feats:
        # fall back to raw columns that exist
        feats = [
            c for c in ["ret_1", "mom_20", "vol_20", "reversal_1", "amihud"] if c in frame.columns
        ]
    sub = frame.select(["event_time", "security_id", label, *feats]).drop_nulls()
    x = sub.select(feats).to_numpy().astype(float)
    y = sub[label].to_numpy().astype(float)
    dates = sub["event_time"].to_numpy()
    ids = sub["security_id"].to_numpy()
    return x, y, dates, feats, ids
