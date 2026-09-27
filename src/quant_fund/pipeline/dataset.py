"""Build gold feature/label panels and numpy design matrices."""

from __future__ import annotations

import ctypes
import os
import struct
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.data.ingest import ingest
from quant_fund.data.lake import Lake
from quant_fund.data.point_in_time import validate_feature_frame
from quant_fund.data.universe import (
    require_panel_keys_in_membership,
    require_valid_membership_panel,
)
from quant_fund.features.engine import build_features
from quant_fund.features.metadata import FEATURE_SET_VERSION
from quant_fund.labels.engine import build_labels
from quant_fund.models.ranking import NEUTRAL_FILL_FEATURES, available_features
from quant_fund.schemas.errors import PointInTimeError
from quant_fund.utils.hashing import hash_file

# Process-local cache: avoid re-reading gold parquet on every asof date. The
# cache key is a content digest. Stat metadata (device, inode, size, mtime_ns,
# ctime_ns) is only a cheap hint — a same-size rewrite can keep every one of
# those fields unchanged — so a matching signature is trusted only when an
# inode watch has seen no write since the digest was stored. Without a watch,
# the digest is recomputed on every lookup.
_PANEL_CACHE: dict[tuple[str, str, str, str], pl.DataFrame] = {}
_FILE_DIGEST_CACHE: dict[Path, tuple[tuple[int, int, int, int, int], str]] = {}

# inotify(7) masks. IN_MOVE_SELF and IN_NONBLOCK share a numeric value; they
# belong to different calls and must not be mixed.
_IN_MODIFY = 0x00000002
_IN_CLOSE_WRITE = 0x00000008
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_IN_NONBLOCK = 0x00000800
_IN_CLOEXEC = 0x00080000
_WATCH_MASK = _IN_MODIFY | _IN_CLOSE_WRITE | _IN_DELETE_SELF | _IN_MOVE_SELF
_EVENT_HEADER = struct.Struct("iIII")


class _ArtifactWatch:
    """Linux inotify fast path for the artifact digest cache.

    A stat signature cannot prove bytes are unchanged. This watch reports
    in-place writes, including same-size rewrites that leave mtime, ctime,
    inode, and size untouched. When the kernel interface is unavailable, or
    a single file cannot be armed, callers must recompute the digest.
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        self._disabled = False
        self._by_path: dict[Path, int] = {}
        self._by_wd: dict[int, Path] = {}
        self._dirty: set[Path] = set()
        self._libc: ctypes.CDLL | None = None

    def arm(self, path: Path) -> bool:
        """Watch ``path``. Return whether a later ``stable`` check is meaningful."""
        if self._disabled:
            return False
        libc = self._load()
        if libc is None or self._fd is None:
            return False
        add_watch = cast(Callable[[int, bytes, int], int], libc.inotify_add_watch)
        wd = add_watch(self._fd, os.fsencode(path), _WATCH_MASK)
        if wd < 0:
            return False
        previous = self._by_path.get(path)
        if previous is not None and previous != wd:
            self._by_wd.pop(previous, None)
        self._by_path[path] = wd
        self._by_wd[wd] = path
        self._drain()
        self._dirty.discard(path)
        return True

    def stable(self, path: Path) -> bool:
        """True when ``path`` is watched and no write has been observed."""
        if path not in self._by_path:
            return False
        self._drain()
        if path in self._dirty:
            self._dirty.discard(path)
            return False
        return True

    def _load(self) -> ctypes.CDLL | None:
        if self._disabled:
            return None
        if self._libc is not None and self._fd is not None:
            return self._libc
        if sys.platform != "linux":
            self._disabled = True
            return None
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except OSError:
            self._disabled = True
            return None
        init1 = cast(Callable[[int], int], libc.inotify_init1)
        fd = init1(_IN_NONBLOCK | _IN_CLOEXEC)
        if fd < 0:
            self._disabled = True
            return None
        self._libc = libc
        self._fd = fd
        return libc

    def _drain(self) -> None:
        fd = self._fd
        if fd is None:
            return
        while True:
            try:
                raw = os.read(fd, 4096)
            except BlockingIOError:
                return
            except OSError:
                self._disable()
                return
            if not raw:
                return
            self._parse(raw)

    def _parse(self, raw: bytes) -> None:
        offset = 0
        overflow = False
        limit = len(raw)
        while offset + _EVENT_HEADER.size <= limit:
            wd, mask, _cookie, length = _EVENT_HEADER.unpack_from(raw, offset)
            offset += _EVENT_HEADER.size
            if length < 0 or offset + length > limit:
                overflow = True
                break
            offset += length
            if mask & _IN_Q_OVERFLOW:
                overflow = True
            path = self._by_wd.get(wd)
            if mask & (_IN_IGNORED | _IN_DELETE_SELF | _IN_MOVE_SELF):
                self._forget(wd)
            if path is not None:
                self._dirty.add(path)
        if overflow:
            self._dirty.update(self._by_path)

    def _forget(self, wd: int) -> None:
        path = self._by_wd.pop(wd, None)
        if path is None:
            return
        self._by_path.pop(path, None)
        self._dirty.add(path)

    def _disable(self) -> None:
        self._dirty.update(self._by_path)
        self._by_path.clear()
        self._by_wd.clear()
        fd = self._fd
        self._fd = None
        self._libc = None
        self._disabled = True
        if fd is not None:
            os.close(fd)


_ARTIFACT_WATCH = _ArtifactWatch()


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
    if cached is not None and cached[0] == signature and _ARTIFACT_WATCH.stable(resolved):
        return cached[1]
    watched = _ARTIFACT_WATCH.arm(resolved)
    digest = hash_file(resolved)
    if watched and not _ARTIFACT_WATCH.stable(resolved):
        # The file was rewritten while we hashed it. Read once more, then
        # let the next lookup observe any further write.
        digest = hash_file(resolved)
    _FILE_DIGEST_CACHE[resolved] = (_file_signature(resolved), digest)
    return digest


def _universe_path(root: Path) -> Path:
    return Path(root) / "silver" / "universe.parquet"


def _panel_cache_key(
    root: Path, feat_path: Path, lab_path: Path
) -> tuple[str, str, str, str] | None:
    univ_path = _universe_path(root)
    if not feat_path.is_file() or not lab_path.is_file() or not univ_path.is_file():
        return None
    return (
        str(root.resolve()),
        _cached_file_digest(feat_path),
        _cached_file_digest(lab_path),
        _cached_file_digest(univ_path),
    )


def ensure_silver(config: AppConfig, *, refresh: bool = False) -> pl.DataFrame:
    lake = Lake(Path(config.data.root))
    if (
        refresh
        or not lake.exists("silver/bars.parquet")
        or not lake.exists("silver/universe.parquet")
    ):
        ingest(config)
    return lake.read_parquet("silver/bars.parquet")


def read_membership_artifact(config: AppConfig) -> pl.DataFrame:
    """Read the persisted PIT universe without ingesting or rebuilding silver.

    Training ``panel()`` uses this so a missing universe cannot be repaired by
    a silent re-ingest that would disagree with already-materialized gold.
    """
    lake = Lake(Path(config.data.root))
    if not lake.exists("silver/universe.parquet"):
        raise PointInTimeError("silver/universe.parquet is required for gold and decision panels")
    membership = lake.read_parquet("silver/universe.parquet")
    require_valid_membership_panel(membership)
    if membership.is_empty():
        raise PointInTimeError(
            "universe membership is empty; refusing to materialize gold from the unfiltered panel"
        )
    return membership


def load_membership(config: AppConfig, *, refresh: bool = False) -> pl.DataFrame:
    """Load the PIT universe artifact after ensuring silver exists."""
    ensure_silver(config, refresh=refresh)
    return read_membership_artifact(config)


def build_gold(config: AppConfig, *, refresh: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    lake = Lake(Path(config.data.root))
    bars = ensure_silver(config, refresh=refresh)
    membership = load_membership(config, refresh=False)
    feats = build_features(bars, config, membership=membership)
    labs = build_labels(bars, config, membership=membership)
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

    # Wave 108 joins membership at gold materialization. Cached gold is still
    # untrusted: a later universe rebuild or hand-edit must not train names the
    # current PIT membership rejected. Do not ingest here; missing universe
    # fails closed instead of silently rebuilding silver under stale gold.
    membership = read_membership_artifact(config)
    require_panel_keys_in_membership(feats, membership)
    require_panel_keys_in_membership(labs, membership)

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


def design_frame(
    frame: pl.DataFrame,
    label: str,
    feats: list[str],
    extra_columns: list[str] | tuple[str, ...] = (),
) -> pl.DataFrame:
    """Rows usable for training: keys, label, features (+ extras), no nulls.

    Long-lookback public characteristics listed in ``NEUTRAL_FILL_FEATURES``
    are filled with their cross-sectional neutral value (0 for a robust
    z-score) instead of dropping the row, following Gu–Kelly–Xiu (2020).
    Every other null still drops the row. Callers that need row alignment
    with ``design_matrix`` must build their frame through this helper.
    """
    columns = ["event_time", "security_id", label, *feats, *extra_columns]
    sub = frame.select(columns)
    fills = [
        pl.col(name).fill_null(NEUTRAL_FILL_FEATURES[name]).alias(name)
        for name in feats
        if name in NEUTRAL_FILL_FEATURES
    ]
    if fills:
        sub = sub.with_columns(fills)
    return sub.drop_nulls()


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
    sub = design_frame(frame, label, feats)
    # Empty after drop_nulls is legitimate (early asof / all-null labels); return
    # zero-row arrays so callers can skip rather than inventing rows.
    x = sub.select(feats).to_numpy().astype(float)
    y = sub[label].to_numpy().astype(float)
    dates = sub["event_time"].to_numpy()
    ids = sub["security_id"].to_numpy()
    return x, y, dates, feats, ids
