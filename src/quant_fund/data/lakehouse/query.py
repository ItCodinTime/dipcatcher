"""DuckDB queries over lake Parquet with point-in-time constraints.

``decision_time`` is the knowledge clock. A row is visible only when both its
event time and its available time are at or before that clock. A row whose
available time precedes its event time is treated as lookahead and refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from quant_fund.config.models import UniverseConfig
from quant_fund.data.lakehouse.store import FileEntry, Snapshot, load_snapshot
from quant_fund.data.universe import membership_asof
from quant_fund.schemas.errors import DataContractError, LeakageError, PointInTimeError

_BAR_FIELDS = (
    "security_id",
    "symbol",
    "event_time",
    "available_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
)


@dataclass(frozen=True)
class BarObservation:
    """One bar visible at a decision time."""

    security_id: str
    symbol: str
    event_time: datetime
    available_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    source: str


@dataclass(frozen=True)
class UniverseMember:
    """A name that was a member on the latest knowable as-of."""

    security_id: str
    symbol: str | None
    asof: datetime


def asof_bars(
    root: Path,
    snapshot_id: str,
    decision_time: datetime,
    *,
    symbols: list[str] | tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Bars with ``event_time <= decision_time`` and ``available_time <= decision_time``."""
    decision = _require_aware(decision_time)
    snapshot = load_snapshot(root, snapshot_id)
    files = [entry for entry in snapshot.files if entry.role == "bars"]
    if not files:
        raise DataContractError("snapshot has no bars to query")
    frames = [_read_bars(root, entry, decision) for entry in files]
    frame = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    if symbols is not None:
        if "symbol" not in frame.columns:
            raise DataContractError("symbol filter requires a symbol column")
        frame = frame.filter(pl.col("symbol").is_in(list(symbols)))
    _reject_lookahead(frame, decision)
    return _select_bar_columns(frame)


def asof_observations(
    root: Path,
    snapshot_id: str,
    decision_time: datetime,
    *,
    symbols: list[str] | tuple[str, ...] | None = None,
) -> list[BarObservation]:
    """Typed rows for :func:`asof_bars`."""
    frame = asof_bars(root, snapshot_id, decision_time, symbols=symbols)
    rows: list[BarObservation] = []
    for raw in frame.iter_rows(named=True):
        volume = raw.get("volume")
        rows.append(
            BarObservation(
                security_id=str(raw.get("security_id") or raw["symbol"]),
                symbol=str(raw["symbol"]),
                event_time=_as_utc_datetime(raw["event_time"]),
                available_time=_as_utc_datetime(raw["available_time"]),
                open=float(raw["open"]),
                high=float(raw["high"]),
                low=float(raw["low"]),
                close=float(raw["close"]),
                volume=None if volume is None else float(volume),
                source=str(raw.get("source") or ""),
            )
        )
    return rows


def universe_asof(
    root: Path,
    snapshot_id: str,
    decision_time: datetime,
    *,
    config: UniverseConfig | None = None,
) -> list[UniverseMember]:
    """Survivorship-safe members at ``decision_time``.

    A membership panel (``asof``) uses the latest panel date that was already
    knowable. An interval table (``effective_from`` / ``effective_to``) keeps
    names whose listing window covers the decision and whose row was already
    available. Bars plus a security master can be filtered with
    :func:`membership_asof` when ``config`` is supplied. A bare symbol list is
    refused.
    """
    decision = _require_aware(decision_time)
    snapshot = load_snapshot(root, snapshot_id)
    universe_files = [entry for entry in snapshot.files if entry.role == "universe"]
    if universe_files:
        return _members_from_universe_files(root, universe_files, decision)
    if config is None:
        raise DataContractError(
            "snapshot has no PIT universe; refusing a point-in-time selection "
            "from an unlabeled symbol list"
        )
    bars = _load_role(root, snapshot, "bars")
    master = _load_role(root, snapshot, "security_master")
    if bars is None or master is None:
        raise DataContractError(
            "survivorship-safe universe selection needs a universe artifact "
            "or both bars and a security master"
        )
    actions = _load_role(root, snapshot, "corporate_actions")
    panel = membership_asof(bars, master, decision, config, actions=actions)
    members: list[UniverseMember] = []
    if panel.is_empty():
        return members
    for raw in panel.iter_rows(named=True):
        members.append(
            UniverseMember(
                security_id=str(raw["security_id"]),
                symbol=None if raw.get("symbol") is None else str(raw["symbol"]),
                asof=decision,
            )
        )
    return members


def _read_bars(root: Path, entry: FileEntry, decision: datetime) -> pl.DataFrame:
    path = str(Path(root) / entry.object_path)
    names = set(pl.scan_parquet(path).collect_schema().names())
    if {"event_time", "available_time"} <= names:
        sql = """
            SELECT *
            FROM read_parquet(?)
            WHERE event_time <= ? AND available_time <= ?
        """
        frame = _duckdb_frame(sql, [path, decision, decision])
        return _normalize_times(frame, ("event_time", "available_time"))
    if {"timestamp", "ticker"} <= names and entry.source == "hf_ohlcv_1m":
        # Vendor timestamp is the minute open. The bronze contract publishes
        # the bar at minute close. This is a read-time projection; the file
        # bytes stay as imported.
        sql = """
            SELECT
                upper(CAST(ticker AS VARCHAR)) AS security_id,
                upper(CAST(ticker AS VARCHAR)) AS symbol,
                timestamp + INTERVAL 1 MINUTE AS event_time,
                timestamp + INTERVAL 1 MINUTE AS available_time,
                open, high, low, close, volume,
                'hf_ohlcv_1m' AS source
            FROM read_parquet(?)
            WHERE timestamp + INTERVAL 1 MINUTE <= ?
        """
        frame = _duckdb_frame(sql, [path, decision])
        return _normalize_times(frame, ("event_time", "available_time"))
    raise DataContractError(
        f"{entry.logical_path} has no knowledge-time columns; refusing an event-time-only read"
    )


def _members_from_universe_files(
    root: Path,
    files: list[FileEntry],
    decision: datetime,
) -> list[UniverseMember]:
    frames: list[pl.DataFrame] = []
    for entry in files:
        path = str(Path(root) / entry.object_path)
        names = set(pl.scan_parquet(path).collect_schema().names())
        if "asof" in names and "security_id" in names:
            frames.append(_panel_members_frame(path, names, decision))
        elif "effective_from" in names and "security_id" in names and "available_time" in names:
            frames.append(_interval_members_frame(path, names, decision))
        else:
            raise DataContractError(
                f"{entry.logical_path} cannot support a survivorship-safe universe; "
                "need asof membership or effective_from/available_time"
            )
    frame = pl.concat(frames, how="diagonal_relaxed")
    if "asof" in frame.columns and frame.height:
        latest = frame.select(pl.col("asof").max()).item()
        frame = frame.filter(pl.col("asof") == latest)
    members: list[UniverseMember] = []
    for raw in frame.iter_rows(named=True):
        asof = raw.get("asof") or raw.get("effective_from") or decision
        symbol = raw.get("symbol")
        if symbol is None and raw.get("ticker") is not None:
            symbol = raw.get("ticker")
        members.append(
            UniverseMember(
                security_id=str(raw["security_id"]),
                symbol=None if symbol is None else str(symbol),
                asof=_as_utc_datetime(asof),
            )
        )
    return members


def _panel_members_frame(path: str, names: set[str], decision: datetime) -> pl.DataFrame:
    if "available_time" not in names:
        raise DataContractError("universe panel has no available_time; refusing it")
    # ``asof`` is reserved in DuckDB (ASOF JOIN), so the column is quoted.
    where = ['"asof" <= ?', "available_time <= ?"]
    params: list[Any] = [path, decision, decision]
    if "event_time" in names:
        where.append("event_time <= ?")
        params.append(decision)
    sql = f"""
        WITH visible AS (
            SELECT * FROM read_parquet(?)
            WHERE {" AND ".join(where)}
        )
        SELECT * FROM visible
        WHERE "asof" = (SELECT max("asof") FROM visible)
    """
    return _normalize_times(
        _duckdb_frame(sql, params),
        tuple(column for column in ("asof", "available_time", "event_time") if column in names),
    )


def _interval_members_frame(path: str, names: set[str], decision: datetime) -> pl.DataFrame:
    end_clause = ""
    if "effective_to" in names:
        end_clause = "AND (effective_to IS NULL OR effective_to > ?)"
    sql = f"""
        SELECT * FROM read_parquet(?)
        WHERE available_time <= ?
          AND effective_from <= ?
          {end_clause}
    """
    params: list[Any] = [path, decision, decision]
    if "effective_to" in names:
        params.append(decision)
    columns = ["available_time", "effective_from"]
    if "effective_to" in names:
        columns.append("effective_to")
    frame = _normalize_times(_duckdb_frame(sql, params), tuple(columns))
    if frame.is_empty():
        return frame
    return frame.with_columns(pl.lit(decision).alias("asof"))


def _load_role(root: Path, snapshot: Snapshot, role: str) -> pl.DataFrame | None:
    files = [entry for entry in snapshot.files if entry.role == role]
    if not files:
        return None
    frames = []
    for entry in files:
        path = Path(root) / entry.object_path
        if entry.kind == "lfs_pointer":
            raise DataContractError(f"{entry.logical_path} is an LFS pointer; payload is absent")
        frames.append(pl.read_parquet(path))
    return pl.concat(frames, how="diagonal_relaxed")


def _duckdb_frame(sql: str, params: list[Any]) -> pl.DataFrame:
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET TimeZone='UTC'")
        connection.execute(sql, params)
        table = connection.fetch_arrow_table()
    finally:
        connection.close()
    frame = pl.from_arrow(table)
    if isinstance(frame, pl.Series):
        frame = frame.to_frame()
    if not isinstance(frame, pl.DataFrame):
        raise DataContractError("query did not return a table")
    return frame


def _normalize_times(frame: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    for column in columns:
        if column not in frame.columns:
            continue
        dtype = frame.schema[column]
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
            frame = frame.with_columns(pl.col(column).dt.replace_time_zone("UTC"))
        elif isinstance(dtype, pl.Datetime) and dtype.time_zone != "UTC":
            frame = frame.with_columns(pl.col(column).dt.convert_time_zone("UTC"))
    return frame


def _reject_lookahead(frame: pl.DataFrame, decision: datetime) -> None:
    if frame.is_empty():
        return
    if "available_time" not in frame.columns or "event_time" not in frame.columns:
        raise PointInTimeError("as-of result is missing knowledge time or event time")
    leaked = frame.filter(
        pl.col("available_time").is_null()
        | pl.col("event_time").is_null()
        | (pl.col("available_time") < pl.col("event_time"))
        | (pl.col("available_time") > decision)
        | (pl.col("event_time") > decision)
    )
    if leaked.height:
        raise LeakageError(
            "as-of result includes a row that was not knowable at decision_time "
            "or was knowable before its event_time"
        )


def _select_bar_columns(frame: pl.DataFrame) -> pl.DataFrame:
    present = [name for name in _BAR_FIELDS if name in frame.columns]
    if not present:
        return frame
    return frame.select(present)


def _require_aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise PointInTimeError("decision_time must be timezone-aware")
    return value.astimezone(UTC)


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    raise DataContractError(f"expected a datetime, got {type(value).__name__}")
