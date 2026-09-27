"""P6.3 data-integrity audit regression KATs.

Each test pins a finding from docs/AUDIT_P63_DATA.md: silent NaN/inf
swallowing, non-fail-closed paths, loop/vectorized divergence, wrong
constants, and nondeterminism.
"""

from __future__ import annotations

import http.client
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from quant_fund.config.models import AppConfig, UniverseConfig
from quant_fund.data.corporate_actions import adjust_prices
from quant_fund.data.ingest import ingest, make_provider
from quant_fund.data.lake import Lake
from quant_fund.data.point_in_time import validate_feature_frame
from quant_fund.data.sources.adapters import (
    SPOT_EARLIEST_MS,
    BinanceFundingRateSource,
    BinancePerpUniverseSource,
    BinancePublicDataSource,
    FredSource,
    SecEdgarSource,
)
from quant_fund.data.sources.base import HttpClient, SourceError, pit_frame
from quant_fund.data.universe import build_membership_panel, membership_asof
from quant_fund.schemas.errors import PointInTimeError


def _bars(sids: list[str], n: int, *, close: float = 10.0) -> pl.DataFrame:
    times = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "security_id": [s for s in sids for _ in times],
            "event_time": times * len(sids),
            "open": [close] * n * len(sids),
            "high": [close + 1] * n * len(sids),
            "low": [close - 1] * n * len(sids),
            "close": [close] * n * len(sids),
            "volume": [1e6] * n * len(sids),
        }
    )


def _master(sids: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": sids,
            "exchange": ["XNYS"] * len(sids),
            "security_type": ["common_stock"] * len(sids),
            "ticker": sids,
            "sector": ["t"] * len(sids),
            "industry": ["i"] * len(sids),
        }
    )


def _cfg(**kw: Any) -> UniverseConfig:
    base = {
        "min_price": 1.0,
        "min_adv": 0.0,
        "min_history_bars": 5,
        "top_n_adv": None,
        "exchanges": ["XNYS"],
        "security_types": ["common_stock"],
    }
    base.update(kw)
    return UniverseConfig(**base)


# --- adjust_prices: silent NaN/inf in total return -------------------------


def test_adjust_prices_rejects_zero_close() -> None:
    bars = _bars(["A"], 3).with_columns(
        pl.when(pl.arange(0, pl.len()) == 0).then(0.0).otherwise(pl.col("close")).alias("close")
    )
    with pytest.raises(PointInTimeError, match="close"):
        adjust_prices(bars, pl.DataFrame())


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -1.0])
def test_adjust_prices_rejects_nonfinite_or_negative_close(bad: float) -> None:
    bars = _bars(["A"], 3).with_columns(
        pl.when(pl.arange(0, pl.len()) == 1).then(bad).otherwise(pl.col("close")).alias("close")
    )
    with pytest.raises(PointInTimeError, match="close"):
        adjust_prices(bars, pl.DataFrame())


def test_adjust_prices_rejects_missing_and_null_close() -> None:
    with pytest.raises(PointInTimeError, match="missing close"):
        adjust_prices(_bars(["A"], 2).drop("close"), pl.DataFrame())
    bars = _bars(["A"], 2).with_columns(
        pl.when(pl.arange(0, pl.len()) == 1).then(None).otherwise(pl.col("close")).alias("close")
    )
    with pytest.raises(PointInTimeError, match="close"):
        adjust_prices(bars, pl.DataFrame())


def test_adjust_prices_total_return_golden() -> None:
    """100 -> 110 with a 5 dividend on day 2: TR index is [100, 115]."""
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [t0, t1],
            "open": [100.0, 110.0],
            "high": [100.0, 110.0],
            "low": [100.0, 110.0],
            "close": [100.0, 110.0],
            "volume": [1e6, 1e6],
        }
    )
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [t1],
            "action_type": ["cash_dividend"],
            "amount": [5.0],
        }
    )
    out = adjust_prices(bars, actions).sort("event_time")
    assert out["close_total_return"].to_list() == [100.0, pytest.approx(115.0)]


# --- universe: loop vs vectorized contract drift ---------------------------


def test_vectorized_membership_rejects_blank_new_ticker() -> None:
    bars = _bars(["A"], 25)
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 6, tzinfo=UTC)],
            "action_type": ["ticker_change"],
            "new_ticker": ["   "],
        }
    )
    asof = datetime(2020, 1, 20, tzinfo=UTC)
    with pytest.raises(PointInTimeError, match="new_ticker"):
        membership_asof(bars, _master(["A"]), asof, _cfg(), actions=actions)
    with pytest.raises(PointInTimeError, match="new_ticker"):
        build_membership_panel(bars, _master(["A"]), [asof], _cfg(), actions=actions)


def test_membership_top_n_tie_break_is_deterministic() -> None:
    """Equal ADV must pick the same name on every call and match the vector path."""
    bars = _bars(["A", "B", "C"], 25)
    master = _master(["A", "B", "C"])
    cfg = _cfg(top_n_adv=1)
    asof = datetime(2020, 1, 25, tzinfo=UTC)
    picks = {
        tuple(membership_asof(bars, master, asof, cfg)["security_id"].to_list()) for _ in range(10)
    }
    assert picks == {("A",)}
    panel = build_membership_panel(bars, master, [asof], cfg)
    assert panel["security_id"].to_list() == ["A"]


# --- sources/base: transport retry + pit_frame fail-closed ------------------


def test_http_client_retries_http_client_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def flaky(*_a: Any, **_k: Any) -> tuple[int, bytes]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise http.client.IncompleteRead(b"")
        return 200, b'{"ok": true}'

    monkeypatch.setattr("quant_fund.data.sources.base.pooled_request", flaky, raising=True)
    monkeypatch.setattr("quant_fund.data.concurrent_io._default_sleep", lambda _s: None)
    client = HttpClient(retries=3)
    assert client.get_json("https://example.invalid/x") == {"ok": True}
    assert calls["n"] == 3


def test_http_client_wraps_exhausted_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def always(*_a: Any, **_k: Any) -> tuple[int, bytes]:
        raise http.client.BadStatusLine("x")

    monkeypatch.setattr("quant_fund.data.sources.base.pooled_request", always, raising=True)
    monkeypatch.setattr("quant_fund.data.concurrent_io._default_sleep", lambda _s: None)
    with pytest.raises(SourceError):
        HttpClient(retries=1).get_json("https://example.invalid/x")


def test_pit_frame_missing_or_bad_event_time_fails_closed() -> None:
    with pytest.raises(SourceError, match="event_time"):
        pit_frame([{"value": 1}], source="s")
    with pytest.raises(SourceError):
        pit_frame([{"event_time": "not-a-date", "value": 1}], source="s")
    with pytest.raises(SourceError):
        pit_frame([{"event_time": [1, 2, 3], "value": 1}], source="s")


# --- sources/adapters --------------------------------------------------------


class _SeqClient:
    def __init__(self, payloads: list[Any]) -> None:
        self.payloads = list(payloads)
        self.urls: list[str] = []

    def get_json(self, url: str, **_: Any) -> Any:
        self.urls.append(url)
        return self.payloads.pop(0)


def test_spot_paginated_fetch_uses_spot_floor_not_perp() -> None:
    """start_time=None spot pagination must start at Binance's spot launch."""
    kline = [1_500_000_000_000, "100", "110", "90", "105", "12", 1_500_086_399_999]
    client = _SeqClient([[kline]])
    BinancePublicDataSource(client=client).fetch(
        symbol="BTCUSDT", interval="1d", max_pages=3, pause_seconds=0.0
    )
    assert f"startTime={SPOT_EARLIEST_MS}" in client.urls[0]


def test_perp_universe_missing_onboard_date_fails_closed() -> None:
    info = {
        "symbols": [
            {
                "symbol": "AAAUSDT",
                "contractType": "PERPETUAL",
                "quoteAsset": "USDT",
                "status": "TRADING",
                "baseAsset": "AAA",
            }
        ]
    }
    tickers = [{"symbol": "AAAUSDT", "quoteVolume": "10.0"}]
    with pytest.raises(SourceError, match="onboardDate"):
        BinancePerpUniverseSource(client=_SeqClient([info, tickers])).fetch()


def test_funding_rate_malformed_row_fails_closed() -> None:
    page = [[{"symbol": "BTCUSDT", "fundingRate": "0.0001"}]]  # no fundingTime
    with pytest.raises(SourceError):
        BinanceFundingRateSource(client=_SeqClient(page)).fetch(symbol="BTCUSDT", pause_seconds=0.0)


def test_sec_edgar_ragged_filing_columns_fail_closed() -> None:
    payload = {
        "filings": {
            "recent": {
                "filingDate": ["2024-01-01", "2024-02-01"],
                "form": ["10-K"],
                "accessionNumber": ["0001", "0002"],
            }
        }
    }
    with pytest.raises(SourceError, match="ragged"):
        SecEdgarSource(client=_SeqClient([payload])).fetch(cik="12345")


class _TextClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.urls: list[str] = []

    def get_text(self, url: str, **_: Any) -> str:
        self.urls.append(url)
        return self.text


def test_alfred_csv_endpoint_is_alfredgraph() -> None:
    client = _TextClient("observation_date,GDP_20260915\n2020-01-01,100.5\n")
    from quant_fund.data.sources.adapters import AlfredSource

    frame = AlfredSource(client=client).fetch(series_id="GDP")
    assert "alfred.stlouisfed.org/graph/alfredgraph.csv" in client.urls[0]
    # CSV values stay strings through normalize_observations (same as FRED).
    assert frame["value"].to_list() == ["100.5"]
    assert frame["event_time"][0] == datetime(2020, 1, 1, tzinfo=UTC)


def test_fred_csv_still_uses_bare_series_column() -> None:
    client = _TextClient("observation_date,GDP\n2020-01-01,99.5\n")
    frame = FredSource(client=client).fetch(series_id="GDP")
    assert "fredgraph.csv" in client.urls[0]
    assert frame["value"].to_list() == ["99.5"]


# --- ingest: master attrs must not depend on a "sector" column ---------------


def test_ingest_attaches_master_without_sector_column(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    pl.DataFrame(
        {
            "event_time": [t0, t1],
            "available_time": [t0, t1],
            "ingested_time": [t0, t1],
            "source": ["vendor-file", "vendor-file"],
            "revision_id": ["v1", "v1"],
            "security_id": ["A", "A"],
            "open": [10.0, 10.0],
            "high": [11.0, 11.0],
            "low": [9.0, 9.0],
            "close": [10.0, 10.0],
            "volume": [1e6, 1e6],
        }
    ).write_parquet(raw / "bars.parquet")
    pl.DataFrame(
        {
            "event_time": pl.Series([], dtype=pl.Datetime(time_zone="UTC")),
            "security_id": pl.Series([], dtype=pl.String),
            "action_type": pl.Series([], dtype=pl.String),
        }
    ).write_parquet(raw / "corporate_actions.parquet")
    # Master carries exchange/industry but no sector column.
    pl.DataFrame(
        {
            "security_id": ["A"],
            "ticker": ["AAA"],
            "exchange": ["XNYS"],
            "industry": ["software"],
            "security_type": ["common_stock"],
            "valid_from": [t0],
            "valid_to": [None],
            "available_time": [t0],
            "ingested_time": [t0],
            "source": ["vendor-file"],
            "revision_id": ["v1"],
        }
    ).write_parquet(raw / "security_master.parquet")
    config = AppConfig.model_validate(
        {"data": {"root": str(tmp_path / "lake"), "source": "file", "parquet_path": str(raw)}}
    )
    paths = ingest(config)
    silver = pl.read_parquet(paths["silver"])
    assert silver["exchange"].to_list() == ["XNYS", "XNYS"]


def test_make_provider_ccxt_is_not_a_bar_provider(tmp_path: Path) -> None:
    cfg = AppConfig.model_validate({"data": {"root": str(tmp_path), "source": "ccxt"}})
    with pytest.raises(ValueError, match="not a market-bar provider"):
        make_provider(cfg)


# --- lake: torn writes never reach the canonical path ------------------------


def test_lake_write_parquet_failure_leaves_no_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lake = Lake(tmp_path / "lake")

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("disk full mid-write")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        lake.write_parquet(pl.DataFrame({"a": [1]}), "bronze/bars.parquet")
    assert not (lake.root / "bronze" / "bars.parquet").exists()
    assert not (lake.root / "bronze" / "bars.parquet.tmp").exists()


# --- point_in_time: availability-only feature frames validate ----------------


def test_validate_feature_frame_accepts_availability_without_full_pit_stamp() -> None:
    t = datetime(2020, 1, 2, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [t, t],
            "available_time": [t, t],
            "ret_1": [0.01, 0.02],
        }
    )
    validate_feature_frame(frame, t)


def test_validate_feature_frame_still_rejects_future_availability() -> None:
    t = datetime(2020, 1, 2, tzinfo=UTC)
    future = datetime(2020, 6, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [t],
            "available_time": [future],
            "ret_1": [0.01],
        }
    )
    with pytest.raises(PointInTimeError):
        validate_feature_frame(frame, t)
