"""OHLCV resample, vendor bars, and file-adapter rejection of bad prints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant_fund.data.adapters.hf_ohlcv_1m import (
    OhlcvQualityError,
    minute_gap_report,
    normalize_vendor_frame,
    resample_ohlcv,
)
from quant_fund.data.adapters.parquet import ParquetMarketProvider
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.data.sources.base import SourceError
from quant_fund.data.sources.normalize import normalize_ohlcv
from quant_fund.schemas.errors import PointInTimeError
from tests.property._profiles import adversarial_settings

_CLOCK = datetime(2026, 1, 1, tzinfo=UTC)
_OPEN = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)  # 09:30 America/New_York


def _minute_bars(n: int, *, volume0: float = 10.0) -> pl.DataFrame:
    rows = []
    px = 100.0
    for i in range(n):
        event = _OPEN + timedelta(minutes=i + 1)
        high = px + 0.4
        low = px - 0.3
        close = px + 0.1
        rows.append(
            {
                "security_id": "A",
                "symbol": "A",
                "event_time": event,
                "available_time": event,
                "ingested_time": _CLOCK,
                "source": "fixture",
                "revision_id": "v",
                "open": px,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume0 + i,
                "currency": "USD",
                "session": "rth",
            }
        )
        px = close
    return pl.DataFrame(rows).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC")),
        pl.col("available_time").cast(pl.Datetime("us", "UTC")),
        pl.col("ingested_time").cast(pl.Datetime("us", "UTC")),
    )


@given(n=st.integers(min_value=5, max_value=25))
@adversarial_settings()
def test_resample_conserves_volume_and_ohlc_envelope(n: int) -> None:
    bars = _minute_bars(n)
    out = resample_ohlcv(bars, "5m")
    assert float(out["volume"].sum()) == pytest.approx(float(bars["volume"].sum()))
    assert int(out["n_source_minutes"].sum()) == n
    assert out.filter(
        (pl.col("high") < pl.col("low"))
        | (pl.col("open") > pl.col("high"))
        | (pl.col("open") < pl.col("low"))
        | (pl.col("close") > pl.col("high"))
        | (pl.col("close") < pl.col("low"))
    ).is_empty()
    again = resample_ohlcv(out, "5m")
    assert again.select("open", "high", "low", "close", "volume").equals(
        out.select("open", "high", "low", "close", "volume")
    )


@given(n=st.integers(min_value=1, max_value=12))
@adversarial_settings()
def test_one_minute_resample_is_idempotent(n: int) -> None:
    bars = _minute_bars(n).sort(["security_id", "event_time"])
    once = resample_ohlcv(bars, "1m")
    twice = resample_ohlcv(once, "1m")
    assert once.equals(bars)
    assert twice.equals(once)


@given(
    minutes=st.lists(st.integers(min_value=0, max_value=389), min_size=1, max_size=12, unique=True)
)
@adversarial_settings()
def test_gap_report_identity_on_unique_regular_minutes(minutes: list[int]) -> None:
    rows = []
    for minute in minutes:
        event = _OPEN + timedelta(minutes=minute + 1)
        rows.append(
            {
                "security_id": "A",
                "symbol": "A",
                "event_time": event,
                "available_time": event,
                "ingested_time": _CLOCK,
                "source": "fixture",
                "revision_id": "v",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1.0,
                "currency": "USD",
                "session": "rth",
            }
        )
    # A duplicated first print must not change the distinct-minute count.
    rows.append(dict(rows[0]))
    frame = pl.DataFrame(rows).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC")),
        pl.col("available_time").cast(pl.Datetime("us", "UTC")),
        pl.col("ingested_time").cast(pl.Datetime("us", "UTC")),
    )
    report = minute_gap_report(frame)
    assert int(report["n_rth"][0]) == len(minutes)
    assert int(report["n_rth_missing"][0]) == 390 - len(minutes)
    assert int(report["n_rth_missing"][0]) >= 0


def _vendor_row(minute: int, *, ticker: str = "AAPL") -> dict[str, object]:
    return {
        "timestamp": _OPEN + timedelta(minutes=minute),
        "open": 10.0,
        "high": 11.0,
        "low": 9.5,
        "close": 10.4,
        "volume": 100.0,
        "ticker": ticker,
    }


@given(minute=st.integers(min_value=0, max_value=20))
@adversarial_settings()
def test_vendor_normalize_rejects_duplicate_timestamps_and_nans(minute: int) -> None:
    good = pl.DataFrame([_vendor_row(minute)]).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
    )
    normalize_vendor_frame(good, clock=lambda: _CLOCK)
    dup = pl.concat([good, good])
    with pytest.raises(OhlcvQualityError, match="duplicate"):
        normalize_vendor_frame(dup, clock=lambda: _CLOCK)
    bad = good.with_columns(pl.lit(float("nan")).alias("close"))
    with pytest.raises(OhlcvQualityError, match="not repaired"):
        normalize_vendor_frame(bad, clock=lambda: _CLOCK)


@given(
    price=st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    volume=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@adversarial_settings()
def test_normalize_ohlcv_rejects_envelope_breaks_and_duplicate_keys(
    price: float,
    volume: float,
) -> None:
    event = datetime(2024, 6, 3, 14, 0, tzinfo=UTC)
    ok = {
        "security_id": "A",
        "event_time": event,
        "available_time": event,
        "open": price,
        "high": price + 1.0,
        "low": price * 0.99,
        "close": price,
        "volume": volume,
    }
    frame = normalize_ohlcv([ok], source="fixture")
    assert frame.height == 1
    assert frame["available_time"][0] >= frame["event_time"][0]
    broken = dict(ok)
    broken["high"] = broken["low"] - 1.0
    with pytest.raises(SourceError):
        normalize_ohlcv([broken], source="fixture")
    with pytest.raises(SourceError, match="duplicate"):
        normalize_ohlcv([ok, ok], source="fixture")
    nan_row = dict(ok)
    nan_row["close"] = float("nan")
    with pytest.raises(SourceError):
        normalize_ohlcv([nan_row], source="fixture")


def _pit_bar(event: datetime, *, price: float = 10.0, security_id: str = "A") -> dict[str, object]:
    return {
        "event_time": event,
        "available_time": event,
        "ingested_time": _CLOCK,
        "source": "fixture",
        "revision_id": "v",
        "security_id": security_id,
        "open": price,
        "high": price + 1.0,
        "low": price - 0.5,
        "close": price,
        "volume": 10.0,
    }


@given(price=st.floats(min_value=2.0, max_value=40.0, allow_nan=False, allow_infinity=False))
@adversarial_settings()
def test_parquet_adapter_roundtrip_and_bad_row_rejection(price: float) -> None:
    event = datetime(2024, 3, 4, tzinfo=UTC)
    frame = pl.DataFrame([_pit_bar(event, price=price)]).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC")),
        pl.col("available_time").cast(pl.Datetime("us", "UTC")),
        pl.col("ingested_time").cast(pl.Datetime("us", "UTC")),
    )
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        frame.write_parquet(root / "bars.parquet")
        loaded = ParquetMarketProvider(root).get_bars()
        assert loaded["close"].to_list() == pytest.approx([price])
        pl.concat([frame, frame]).write_parquet(root / "bars.parquet")
        with pytest.raises(PointInTimeError, match="duplicate"):
            ParquetMarketProvider(root).get_bars()
        nan = frame.with_columns(pl.lit(float("nan")).alias("low"))
        nan.write_parquet(root / "bars.parquet")
        with pytest.raises(PointInTimeError, match="invalid OHLCV"):
            ParquetMarketProvider(root).get_bars()


@given(seed=st.integers(min_value=0, max_value=5_000))
@adversarial_settings()
def test_synthetic_provider_is_seed_deterministic(seed: int) -> None:
    first = SyntheticMarketProvider(n_assets=2, n_days=16, seed=seed)
    second = SyntheticMarketProvider(n_assets=2, n_days=16, seed=seed)
    assert first.get_bars()["close"].to_list() == second.get_bars()["close"].to_list()
