"""Property tests: vendor normalizer invariants and limiter/retry math.

The core invariant under test: any well-formed vendor payload normalizes to a
frame with sorted unique event_times, finite positive prices, valid OHLC
envelopes, and a sane PIT chain — and any contract-violating payload fails
closed instead of emitting bad bars.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.data.sources.base import SourceError
from quant_fund.data.vendors.client import parse_retry_after
from quant_fund.data.vendors.errors import VendorResponseError
from quant_fund.data.vendors.ratelimit import TokenBucket
from quant_fund.data.vendors.tiingo import TiingoDailyAdapter


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.now += seconds


_price = st.floats(min_value=0.01, max_value=1e5, allow_nan=False, allow_infinity=False)


@st.composite
def tiingo_bars(draw: st.DrawFn) -> list[dict[str, object]]:
    days = draw(
        st.lists(
            st.dates(min_value=date(2000, 1, 1), max_value=date(2030, 12, 31)),
            min_size=1,
            max_size=30,
            unique=True,
        )
    )
    payload: list[dict[str, object]] = []
    for day in days:
        low = draw(_price)
        high = low + draw(st.floats(min_value=0.0, max_value=500.0))
        open_ = draw(st.floats(min_value=low, max_value=high))
        close = draw(st.floats(min_value=low, max_value=high))
        volume = draw(st.floats(min_value=0.0, max_value=1e10))
        payload.append(
            {
                "date": f"{day.isoformat()}T00:00:00.000Z",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
    return payload


@given(payload=tiingo_bars())
@settings(max_examples=50, deadline=None)
def test_normalized_bars_hold_contract_invariants(
    payload: list[dict[str, object]],
) -> None:
    frame = TiingoDailyAdapter().parse_body(payload, symbol="SPY")
    assert frame.height == len(payload)
    # Sorted, unique timestamps — the headline normalizer invariant.
    assert frame["event_time"].is_sorted()
    assert frame["event_time"].n_unique() == frame.height
    # Finite positive prices, valid envelope, non-negative volume.
    for col in ("open", "high", "low", "close"):
        assert frame[col].is_finite().all()
        assert (frame[col] > 0).all()
    assert (frame["volume"] >= 0).all()
    assert (frame["low"] <= frame["open"]).all()
    assert (frame["open"] <= frame["high"]).all()
    assert (frame["low"] <= frame["close"]).all()
    assert (frame["close"] <= frame["high"]).all()
    # PIT chain: event <= available <= ingested, all UTC.
    assert (frame["event_time"] <= frame["available_time"]).all()
    assert (frame["available_time"] <= frame["ingested_time"]).all()
    assert str(frame["event_time"].dtype.time_zone) == "UTC"
    assert frame["source"].unique().to_list() == ["tiingo"]


@given(payload=tiingo_bars())
@settings(max_examples=30, deadline=None)
def test_contract_violations_fail_closed(payload: list[dict[str, object]]) -> None:
    """A payload with even one inverted envelope must raise, never emit."""
    bad = dict(payload[0])
    bad["date"] = "2031-01-01T00:00:00.000Z"  # outside the generated range — no dup key
    bad["close"] = float(bad["high"]) + 1.0  # close above high: impossible bar
    with pytest.raises((SourceError, VendorResponseError)):
        TiingoDailyAdapter().parse_body([*payload, bad], symbol="SPY")


@given(text=st.text(max_size=80))
@settings(max_examples=100, deadline=None)
def test_parse_retry_after_never_raises(text: str) -> None:
    result = parse_retry_after(text, now=datetime(2024, 1, 1, tzinfo=UTC))
    assert result is None or result >= 0.0


@given(
    rate=st.floats(min_value=0.1, max_value=100.0),
    capacity=st.floats(min_value=1.0, max_value=10.0),
    draws=st.integers(min_value=1, max_value=25),
)
@settings(max_examples=40, deadline=None)
def test_token_bucket_respects_rate(rate: float, capacity: float, draws: int) -> None:
    clock = _FakeClock()
    bucket = TokenBucket(rate, capacity=capacity, clock=clock.monotonic, sleeper=clock.sleep)
    for _ in range(draws):
        bucket.acquire()
    # Time consumed must cover every token beyond the initial burst.
    assert clock.now >= (draws - capacity) / rate - 1e-6
