"""Build gold feature/label panels and numpy design matrices."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.data.ingest import ingest
from quant_fund.data.lake import Lake
from quant_fund.data.point_in_time import validate_feature_frame
from quant_fund.features.engine import build_features
from quant_fund.features.metadata import FEATURE_SET_VERSION
from quant_fund.labels.engine import build_labels
from quant_fund.models.ranking import available_features
from quant_fund.utils.hashing import hash_file

# Process-local cache: avoid re-reading gold parquet on every asof date. The
# cache key includes content digests, not only mtimes, so an in-place artifact
# replacement cannot silently reuse stale research inputs. File metadata is a
# cheap first-level guard; a full digest is recomputed only after metadata moves.
_PANEL_CACHE: dict[tuple[str, str, str], pl.DataFrame] = {}
_FILE_DIGEST_CACHE: dict[Path, tuple[tuple[int, int, int, int, int], str]] = {}


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    metadata = os.stat(path)
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _cached_file_digest(path: Path) -> str:
    resolved = path.resolve()
    signature = _file_signature(resolved)
    cached = _FILE_DIGEST_CACHE.get(resolved)
    if cached is not None and cached[0] == signature:
        return cached[1]
    digest = hash_file(resolved)
    _FILE_DIGEST_CACHE[resolved] = (signature, digest)
    return digest


def _panel_cache_key(root: Path, feat_path: Path, lab_path: Path) -> tuple[str, str, str] | None:
    if not feat_path.is_file() or not lab_path.is_file():
        return None
    return (str(root.resolve()), _cached_file_digest(feat_path), _cached_file_digest(lab_path))


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


def clear_panel_cache() -> None:
    """Drop process-local gold panel and artifact-digest caches."""
    _PANEL_CACHE.clear()
    _FILE_DIGEST_CACHE.clear()


def panel(
    config: AppConfig, feature_names: list[str] | None = None, label: str | None = None
) -> pl.DataFrame:
    lake = Lake(Path(config.data.root))
    feat_path = Path(config.data.root) / "gold" / "features.parquet"
    lab_path = Path(config.data.root) / "gold" / "labels.parquet"

    # Fast path: reuse joined panel only when the underlying artifact bytes
    # still match the cached lineage key.
    cache_key = _panel_cache_key(Path(config.data.root), feat_path, lab_path)
    if feature_names is None and label is None and cache_key is not None:
        cached = _PANEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    if not lake.exists("gold/features.parquet"):
        feats, labs = build_gold(config)
    else:
        feats = lake.read_parquet("gold/features.parquet")
        labs = lake.read_parquet("gold/labels.parquet")

    # Cached artifacts are untrusted training inputs: enforce the same PIT
    # invariant as freshly built features instead of silently training on an
    # old or hand-edited gold file.
    if "feature_set_version" not in feats.columns:
        raise ValueError(
            "cached feature artifact is missing feature_set_version; rebuild gold data"
        )
    versions = set(feats.get_column("feature_set_version").drop_nulls().to_list())
    if versions != {FEATURE_SET_VERSION}:
        raise ValueError(
            "cached feature artifact has an incompatible feature_set_version; rebuild gold data"
        )
    if "decision_time" not in feats.columns and "event_time" not in feats.columns:
        raise ValueError(
            "cached feature artifact is missing decision timestamps; rebuild gold data"
        )
    if "decision_time" not in feats.columns and "event_time" in feats.columns:
        # Cached panels contain many as-of snapshots. Compare each row with its
        # own event-time decision clock, not the artifact's earliest timestamp.
        validate_feature_frame(
            feats.with_columns(pl.col("event_time").alias("decision_time")),
            cast(datetime, feats.get_column("event_time").min()),
        )
    else:
        validate_feature_frame(feats, cast(datetime, feats.get_column("event_time").min()))

    keys = ["security_id", "event_time"]
    lab_cols = [c for c in labs.columns if c.startswith("future_") or c in keys]
    out = feats.join(labs.select(lab_cols), on=keys, how="inner")
    if feature_names is not None:
        missing_feats = [c for c in feature_names if c not in out.columns]
        if missing_feats:
            raise ValueError(f"requested feature columns missing from gold panel: {missing_feats}")
    if label is not None and label not in out.columns:
        raise ValueError(
            f"requested label {label!r} is not present in the gold panel; rebuild gold data"
        )
    out = out.sort(["event_time", "security_id"])
    if feature_names is None and label is None:
        # Recompute after building/loading so newly materialized artifacts are
        # bound to the exact bytes that produced this joined panel.
        final_key = _panel_cache_key(Path(config.data.root), feat_path, lab_path)
        if final_key is not None:
            _PANEL_CACHE[final_key] = out
    return out


def design_matrix(
    frame: pl.DataFrame,
    label: str,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    if frame.height == 0:
        raise ValueError("design_matrix requires a non-empty frame")
    if label not in frame.columns:
        raise ValueError(f"requested label {label!r} is not present in the frame")
    feats = available_features(frame.columns, feature_names)
    if not feats:
        if feature_names is not None:
            # Explicit request with zero overlap: fail closed (do not invent columns).
            raise ValueError(
                f"none of the requested feature columns are present: {list(feature_names)}"
            )
        # fall back to raw columns that exist (default feature set empty)
        feats = [
            c for c in ["ret_1", "mom_20", "vol_20", "reversal_1", "amihud"] if c in frame.columns
        ]
    if not feats:
        raise ValueError("design_matrix found no usable feature columns")
    sub = frame.select(["event_time", "security_id", label, *feats]).drop_nulls()
    # Empty after drop_nulls is legitimate (early asof / all-null labels); return
    # zero-row arrays so callers can skip rather than inventing rows.
    x = sub.select(feats).to_numpy().astype(float)
    y = sub[label].to_numpy().astype(float)
    dates = sub["event_time"].to_numpy()
    ids = sub["security_id"].to_numpy()
    return x, y, dates, feats, ids
