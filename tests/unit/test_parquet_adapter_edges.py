"""Wave 34: ParquetMarketProvider edge fixtures — missing path/cols, empty, PIT fail-closed.

Does not duplicate test_data_pit file_adapter happy/duplicate/OHLCV/impossible-order cases.
Research/infrastructure only — no live broker / vendor MD.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_fund.data.adapters.parquet import ParquetMarketProvider
from quant_fund.schemas.errors import PointInTimeError

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False

_PIT = {
    "available_time": datetime(2020, 1, 2, tzinfo=UTC),
    "ingested_time": datetime(2020, 1, 2, tzinfo=UTC),
    "source": "vendor-file",
    "revision_id": "v1",
}


def _valid_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_time": datetime(2020, 1, 2, tzinfo=UTC),
        "available_time": _PIT["available_time"],
        "ingested_time": _PIT["ingested_time"],
        "source": _PIT["source"],
        "revision_id": _PIT["revision_id"],
        "security_id": "A",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 1000.0,
    }
    row.update(overrides)
    return row


def test_parquet_missing_path_returns_empty(tmp_path: Path) -> None:
    """No bars.parquet/csv under root → empty frame (not an exception)."""
    prov = ParquetMarketProvider(tmp_path)
    bars = prov.get_bars()
    assert bars.is_empty()
    assert prov.get_corporate_actions().is_empty()
    assert prov.get_security_master().is_empty()


def test_parquet_empty_bars_frame_passthrough(tmp_path: Path) -> None:
    """Empty bars file with schema still returns empty without PIT checks."""
    pl.DataFrame(
        schema={
            "event_time": pl.Datetime("us", "UTC"),
            "security_id": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    ).write_parquet(tmp_path / "bars.parquet")
    out = ParquetMarketProvider(tmp_path).get_bars()
    assert out.is_empty()


def test_parquet_missing_required_bar_columns_fail_closed(tmp_path: Path) -> None:
    """PIT present but OHLCV incomplete → missing required columns."""
    t = datetime(2020, 1, 2, tzinfo=UTC)
    pl.DataFrame(
        {
            "event_time": [t],
            "available_time": [t],
            "ingested_time": [t],
            "source": ["vendor"],
            "revision_id": ["v1"],
            "security_id": ["A"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            # close + volume missing
        }
    ).write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="missing required columns"):
        ParquetMarketProvider(tmp_path).get_bars()


def test_parquet_blank_source_and_null_security_fail_closed(tmp_path: Path) -> None:
    pl.DataFrame([_valid_row(source="  ")]).write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="null, blank, or impossible PIT"):
        ParquetMarketProvider(tmp_path).get_bars()

    pl.DataFrame([_valid_row(security_id=None)]).write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="null, blank, or impossible PIT"):
        ParquetMarketProvider(tmp_path).get_bars()


def test_parquet_rejects_ohlc_values_outside_trading_range(tmp_path: Path) -> None:
    pl.DataFrame([_valid_row(open=12.0, high=11.0)]).write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="invalid OHLCV"):
        ParquetMarketProvider(tmp_path).get_bars()


def test_parquet_nonfinite_and_nonpositive_prices_fail_closed(tmp_path: Path) -> None:
    pl.DataFrame([_valid_row(close=float("nan"))]).write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="invalid OHLCV"):
        ParquetMarketProvider(tmp_path).get_bars()

    pl.DataFrame([_valid_row(open=0.0)]).write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="invalid OHLCV"):
        ParquetMarketProvider(tmp_path).get_bars()

    pl.DataFrame([_valid_row(close=float("-inf"))]).write_parquet(tmp_path / "bars.parquet")
    with pytest.raises(PointInTimeError, match="invalid OHLCV"):
        ParquetMarketProvider(tmp_path).get_bars()


def test_parquet_prefers_parquet_over_csv_and_end_filter(tmp_path: Path) -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    pq_rows = [_valid_row(event_time=t0, available_time=t0, ingested_time=t0, security_id="PQ")]
    csv_rows = [
        _valid_row(
            event_time=t1,
            available_time=t1,
            ingested_time=t1,
            security_id="CSV",
            open=20.0,
            high=21.0,
            low=19.0,
            close=20.5,
            volume=2000.0,
        )
    ]
    pl.DataFrame(pq_rows).write_parquet(tmp_path / "bars.parquet")
    pl.DataFrame(csv_rows).write_csv(tmp_path / "bars.csv")
    bars = ParquetMarketProvider(tmp_path).get_bars()
    assert bars.height == 1
    assert bars["security_id"].to_list() == ["PQ"]

    pl.DataFrame(
        [
            _valid_row(event_time=t0, available_time=t0, ingested_time=t0, security_id="A"),
            _valid_row(
                event_time=t1,
                available_time=t1,
                ingested_time=t1,
                security_id="B",
                open=20.0,
                high=21.0,
                low=19.0,
                close=20.5,
                volume=2000.0,
            ),
        ]
    ).write_parquet(tmp_path / "bars.parquet")
    clipped = ParquetMarketProvider(tmp_path).get_bars(end=t0)
    assert clipped.height == 1
    assert clipped["security_id"].to_list() == ["A"]


def test_parquet_corporate_actions_filters_and_empty_master(tmp_path: Path) -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 5, tzinfo=UTC)
    pl.DataFrame(
        {
            "event_time": [t0, t1],
            "security_id": ["A", "A"],
            "split_factor": [1.0, 2.0],
            "dividend": [0.0, 0.1],
        }
    ).write_parquet(tmp_path / "corporate_actions.parquet")
    prov = ParquetMarketProvider(tmp_path)
    mid = prov.get_corporate_actions(start=t0, end=t0)
    assert mid.height == 1
    assert mid["split_factor"].to_list() == [1.0]
    assert prov.get_corporate_actions(start=datetime(2099, 1, 1, tzinfo=UTC)).is_empty()
    assert prov.get_security_master().is_empty()
