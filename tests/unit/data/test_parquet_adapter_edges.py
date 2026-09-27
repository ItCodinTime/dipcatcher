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


def _ca_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_time": datetime(2020, 1, 2, tzinfo=UTC),
        "available_time": datetime(2020, 1, 2, tzinfo=UTC),
        "ingested_time": datetime(2020, 1, 2, tzinfo=UTC),
        "source": "vendor-file",
        "revision_id": "v1",
        "security_id": "A",
        "action_type": "split",
        "factor": 2.0,
        "amount": None,
    }
    row.update(overrides)
    return row


def test_parquet_corporate_actions_filters_and_empty_master(tmp_path: Path) -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 5, tzinfo=UTC)
    pl.DataFrame(
        [
            _ca_row(event_time=t0, available_time=t0, ingested_time=t0, factor=2.0),
            _ca_row(
                event_time=t1,
                available_time=t1,
                ingested_time=t1,
                action_type="cash_dividend",
                factor=None,
                amount=0.1,
            ),
        ]
    ).write_parquet(tmp_path / "corporate_actions.parquet")
    prov = ParquetMarketProvider(tmp_path)
    mid = prov.get_corporate_actions(start=t0, end=t0)
    assert mid.height == 1
    assert mid["action_type"].to_list() == ["split"]
    assert mid["factor"].to_list() == [2.0]
    assert prov.get_corporate_actions(start=datetime(2099, 1, 1, tzinfo=UTC)).is_empty()
    assert prov.get_security_master().is_empty()


def _sm_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "security_id": "A",
        "ticker": "AAA",
        "valid_from": datetime(2020, 1, 1, tzinfo=UTC),
        "valid_to": None,
        "available_time": datetime(2020, 1, 1, tzinfo=UTC),
        "ingested_time": datetime(2020, 1, 1, tzinfo=UTC),
        "source": "vendor-file",
        "revision_id": "v1",
        "sector": "tech",
        "industry": "software",
        "exchange": "XNYS",
    }
    row.update(overrides)
    return row


def test_parquet_security_master_pit_and_contract_fail_closed(tmp_path: Path) -> None:
    t0 = datetime(2020, 1, 1, tzinfo=UTC)
    pl.DataFrame([_sm_row()]).write_parquet(tmp_path / "security_master.parquet")
    got = ParquetMarketProvider(tmp_path).get_security_master()
    assert got.height == 1
    assert got["ticker"].to_list() == ["AAA"]

    pl.DataFrame([_sm_row(source="  ")]).write_parquet(tmp_path / "security_master.parquet")
    with pytest.raises(PointInTimeError, match="null, blank, or impossible PIT"):
        ParquetMarketProvider(tmp_path).get_security_master()

    pl.DataFrame([_sm_row(ticker="  ")]).write_parquet(tmp_path / "security_master.parquet")
    with pytest.raises(PointInTimeError, match="null, blank, or impossible PIT"):
        ParquetMarketProvider(tmp_path).get_security_master()

    pl.DataFrame(
        [_sm_row(available_time=datetime(2020, 1, 3, tzinfo=UTC), ingested_time=t0)]
    ).write_parquet(tmp_path / "security_master.parquet")
    with pytest.raises(PointInTimeError, match="null, blank, or impossible PIT"):
        ParquetMarketProvider(tmp_path).get_security_master()

    pl.DataFrame(
        {
            "security_id": ["A"],
            "ticker": ["AAA"],
            "valid_from": [t0],
            "sector": ["tech"],
        }
    ).write_parquet(tmp_path / "security_master.parquet")
    with pytest.raises(PointInTimeError, match="missing required columns"):
        ParquetMarketProvider(tmp_path).get_security_master()

    pl.DataFrame([_sm_row(valid_to=datetime(2019, 12, 1, tzinfo=UTC))]).write_parquet(
        tmp_path / "security_master.parquet"
    )
    with pytest.raises(PointInTimeError, match="valid_to earlier than valid_from"):
        ParquetMarketProvider(tmp_path).get_security_master()

    pl.DataFrame([_sm_row(), _sm_row(ticker="BBB")]).write_parquet(
        tmp_path / "security_master.parquet"
    )
    with pytest.raises(PointInTimeError, match="duplicate security_id/valid_from"):
        ParquetMarketProvider(tmp_path).get_security_master()

    pl.DataFrame([_sm_row(), _sm_row(security_id="B", valid_from=t0)]).write_parquet(
        tmp_path / "security_master.parquet"
    )
    with pytest.raises(PointInTimeError, match="duplicate ticker/valid_from"):
        ParquetMarketProvider(tmp_path).get_security_master()


def test_parquet_security_master_allows_late_available_restatement(tmp_path: Path) -> None:
    """valid_from may precede available_time; that is a restatement, not a PIT error."""
    t0 = datetime(2020, 1, 1, tzinfo=UTC)
    t1 = datetime(2020, 6, 1, tzinfo=UTC)
    late = datetime(2021, 1, 1, tzinfo=UTC)
    pl.DataFrame(
        [
            _sm_row(),
            _sm_row(valid_from=t1, available_time=late, ingested_time=late),
        ]
    ).write_parquet(tmp_path / "security_master.parquet")
    got = ParquetMarketProvider(tmp_path).get_security_master()
    assert got.height == 2
    assert got["available_time"].to_list()[-1] == late
    assert got["valid_from"].to_list()[0] == t0


def test_parquet_corporate_actions_pit_and_contract_fail_closed(tmp_path: Path) -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    pl.DataFrame([_ca_row(source="  ")]).write_parquet(tmp_path / "corporate_actions.parquet")
    with pytest.raises(PointInTimeError, match="null, blank, or impossible PIT"):
        ParquetMarketProvider(tmp_path).get_corporate_actions()

    pl.DataFrame([_ca_row(available_time=datetime(2019, 12, 31, tzinfo=UTC))]).write_parquet(
        tmp_path / "corporate_actions.parquet"
    )
    with pytest.raises(PointInTimeError, match="null, blank, or impossible PIT"):
        ParquetMarketProvider(tmp_path).get_corporate_actions()

    pl.DataFrame(
        {
            "event_time": [t0],
            "available_time": [t0],
            "ingested_time": [t0],
            "source": ["vendor"],
            "revision_id": ["v1"],
            "security_id": ["A"],
            "factor": [2.0],
        }
    ).write_parquet(tmp_path / "corporate_actions.parquet")
    with pytest.raises(PointInTimeError, match="missing required columns"):
        ParquetMarketProvider(tmp_path).get_corporate_actions()

    pl.DataFrame([_ca_row(action_type="merger")]).write_parquet(
        tmp_path / "corporate_actions.parquet"
    )
    with pytest.raises(PointInTimeError, match="unknown action_type"):
        ParquetMarketProvider(tmp_path).get_corporate_actions()

    pl.DataFrame([_ca_row(factor=0.0)]).write_parquet(tmp_path / "corporate_actions.parquet")
    with pytest.raises(PointInTimeError, match="non-positive or non-finite factors"):
        ParquetMarketProvider(tmp_path).get_corporate_actions()

    pl.DataFrame([_ca_row(action_type="cash_dividend", factor=None, amount=-0.25)]).write_parquet(
        tmp_path / "corporate_actions.parquet"
    )
    with pytest.raises(PointInTimeError, match="negative or non-finite amounts"):
        ParquetMarketProvider(tmp_path).get_corporate_actions()

    pl.DataFrame(
        [
            _ca_row(event_time=t1, available_time=t1, ingested_time=t1),
            _ca_row(event_time=t1, available_time=t1, ingested_time=t1, factor=1.5),
        ]
    ).write_parquet(tmp_path / "corporate_actions.parquet")
    with pytest.raises(PointInTimeError, match="duplicate security_id/event_time/action_type"):
        ParquetMarketProvider(tmp_path).get_corporate_actions()

    pl.DataFrame(
        [_ca_row(action_type="ticker_change", factor=None, new_ticker="  ")]
    ).write_parquet(tmp_path / "corporate_actions.parquet")
    with pytest.raises(PointInTimeError, match="blank new_ticker"):
        ParquetMarketProvider(tmp_path).get_corporate_actions()

    pl.DataFrame([_ca_row(action_type="ticker_change", factor=None)]).write_parquet(
        tmp_path / "corporate_actions.parquet"
    )
    with pytest.raises(PointInTimeError, match="missing new_ticker"):
        ParquetMarketProvider(tmp_path).get_corporate_actions()
