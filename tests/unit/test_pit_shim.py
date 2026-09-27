"""Unit tests for the migration shim (DESIGN.md §4.4, §12 W1: shim mapping)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.data.lake import Lake
from quant_fund.pit import VaultUnavailableError
from quant_fund.pit.shim import guarded_read_parquet, lake_asof
from quant_fund.schemas.errors import PointInTimeError

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _legacy_frame(rows: list[tuple[str, int, int, float]]) -> pl.DataFrame:
    """Legacy PIT column set (data/point_in_time.py PIT_COLS)."""
    return pl.DataFrame(
        {
            "security_id": [r[0] for r in rows],
            "event_time": [T0 + timedelta(days=r[1]) for r in rows],
            "available_time": [T0 + timedelta(days=r[2]) for r in rows],
            "ingested_time": [T0 + timedelta(days=r[2]) for r in rows],
            "source": ["synthetic"] * len(rows),
            "revision_id": ["v1"] * len(rows),
            "close": [r[3] for r in rows],
        }
    )


def test_lake_asof_maps_available_time(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    lake.write_parquet(
        _legacy_frame([("A", 0, 0, 100.0), ("A", 0, 5, 101.5), ("B", 0, 0, 50.0)]),
        "silver/bars.parquet",
    )
    early = lake_asof(lake, "silver/bars.parquet", T0 + timedelta(days=1))
    assert early.dataset == "silver/bars.parquet"
    assert early.frame.filter(pl.col("security_id") == "A")["close"].to_list() == [100.0]
    late = lake_asof(lake, "silver/bars.parquet", T0 + timedelta(days=6))
    assert late.frame.filter(pl.col("security_id") == "A")["close"].to_list() == [101.5]
    assert late.max_known_at == T0 + timedelta(days=5)


def test_lake_asof_fail_closed_without_pit_columns(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    lake.write_parquet(pl.DataFrame({"security_id": ["A"], "close": [1.0]}), "silver/raw.parquet")
    with pytest.raises(PointInTimeError, match="event_time"):
        lake_asof(lake, "silver/raw.parquet", T0)
    lake.write_parquet(
        pl.DataFrame({"security_id": ["A"], "event_time": [T0], "close": [1.0]}),
        "silver/no_avail.parquet",
    )
    with pytest.raises(PointInTimeError, match="available_time"):
        lake_asof(lake, "silver/no_avail.parquet", T0)


def test_lake_asof_naive_timestamp_refused(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    lake.write_parquet(_legacy_frame([("A", 0, 0, 100.0)]), "silver/bars.parquet")
    with pytest.raises(PointInTimeError, match="timezone-aware"):
        lake_asof(lake, "silver/bars.parquet", datetime(2024, 1, 2))


def test_lake_asof_naive_availability_refused(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    lake.write_parquet(
        pl.DataFrame(
            {
                "security_id": ["A"],
                "event_time": [datetime(2024, 1, 1)],  # naive
                "available_time": [datetime(2024, 1, 1)],  # naive
            }
        ),
        "silver/naive.parquet",
    )
    with pytest.raises(PointInTimeError, match="timezone-aware"):
        lake_asof(lake, "silver/naive.parquet", T0)


def test_lake_asof_nothing_available(tmp_path) -> None:
    lake = Lake(tmp_path / "lake")
    lake.write_parquet(_legacy_frame([("A", 0, 5, 100.0)]), "silver/bars.parquet")
    with pytest.raises(VaultUnavailableError):
        lake_asof(lake, "silver/bars.parquet", T0 + timedelta(days=1))


def test_guarded_read_parquet(tmp_path) -> None:
    path = tmp_path / "rogue.parquet"
    _legacy_frame([("A", 0, 0, 100.0), ("A", 0, 9, 999.0)]).write_parquet(path)
    out = guarded_read_parquet(path, T0 + timedelta(days=1), dataset="rogue")
    assert out.dataset == "rogue"
    assert out.rows == 1
    assert out.frame["close"].to_list() == [100.0]


def test_guarded_read_parquet_fail_closed(tmp_path) -> None:
    path = tmp_path / "no_pit.parquet"
    pl.DataFrame({"close": [1.0]}).write_parquet(path)
    with pytest.raises(PointInTimeError):
        guarded_read_parquet(path, T0, dataset="rogue")


def test_shim_event_time_only_frame(tmp_path) -> None:
    """Frames without security_id key on event_time alone."""
    path = tmp_path / "macro.parquet"
    pl.DataFrame(
        {
            "event_time": [T0, T0],
            "available_time": [T0, T0 + timedelta(days=4)],
            "cpi": [3.1, 3.2],
        }
    ).write_parquet(path)
    early = guarded_read_parquet(path, T0 + timedelta(days=1), dataset="macro")
    assert early.frame["cpi"].to_list() == [3.1]
    late = guarded_read_parquet(path, T0 + timedelta(days=5), dataset="macro")
    assert late.frame["cpi"].to_list() == [3.2]
