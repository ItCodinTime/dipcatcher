"""Windowed parquet reads keep full-file validation and project columns."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_fund.data.adapters.parquet import ParquetMarketProvider
from quant_fund.data.lake import Lake
from quant_fund.schemas.errors import PointInTimeError


def _row(stamp: datetime, security_id: str, *, extra: str = "keep") -> dict[str, object]:
    return {
        "event_time": stamp,
        "available_time": stamp,
        "ingested_time": stamp,
        "source": "vendor-file",
        "revision_id": "v1",
        "security_id": security_id,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 1000.0,
        "unused_text": extra,
    }


def test_windowed_get_bars_projects_and_still_validates_outside_the_window(tmp_path: Path) -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    pl.DataFrame([_row(t0, "A", extra="left"), _row(t1, "B", extra="right")]).write_parquet(
        tmp_path / "bars.parquet"
    )
    provider = ParquetMarketProvider(tmp_path)
    window = provider.get_bars(end=t0, columns=["security_id", "close", "event_time"])
    assert window.columns == ["security_id", "close", "event_time"]
    assert window["security_id"].to_list() == ["A"]
    full = provider.get_bars()
    assert "unused_text" in full.columns
    assert full.height == 2

    bad = pl.DataFrame(
        [
            _row(t0, "A"),
            {
                **_row(t1, "B"),
                "close": -1.0,
            },
        ]
    )
    bad.write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="invalid OHLCV"):
        ParquetMarketProvider(tmp_path).get_bars(end=t0)


def test_lake_read_projects_columns(tmp_path: Path) -> None:
    lake = Lake(tmp_path)
    frame = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    lake.write_parquet(frame, "silver/bars.parquet")
    got = lake.read_parquet("silver/bars.parquet", columns=["b"])
    assert got.columns == ["b"]
    assert got["b"].to_list() == [3, 4]
