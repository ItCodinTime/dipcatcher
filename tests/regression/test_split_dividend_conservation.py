"""Counterexample: same-ex-date split plus cash dividend must conserve wealth."""

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.data.corporate_actions import adjust_prices


def _two_day_bars(closes: list[float]) -> pl.DataFrame:
    assert len(closes) == 2
    return pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [
                datetime(2020, 1, 2, tzinfo=UTC),
                datetime(2020, 1, 3, tzinfo=UTC),
            ],
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1e6, 1e6],
        }
    )


def test_same_ex_date_split_and_dividend_conserves_wealth() -> None:
    """2-for-1 and $1 per post-split share, raw close 100 then 49.

    One pre-split share becomes two shares at 49 plus $2 cash, wealth 100.
    The total-return index stays on the split-adjusted base of 50.
    Dividing the dividend by the raw previous close (100) instead of the
    split-adjusted previous close (50) reports 49.5.
    """
    bars = _two_day_bars([100.0, 49.0])
    actions = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [
                datetime(2020, 1, 3, tzinfo=UTC),
                datetime(2020, 1, 3, tzinfo=UTC),
            ],
            "action_type": ["split", "cash_dividend"],
            "factor": [2.0, None],
            "amount": [None, 1.0],
        }
    )
    out = adjust_prices(bars, actions).sort("event_time")
    assert out["close_split_adjusted"].to_list() == pytest.approx([50.0, 49.0])
    assert out["close_total_return"].to_list() == pytest.approx([50.0, 50.0])


def test_dividend_before_later_split_keeps_raw_share_yield() -> None:
    """A dividend on the pre-split basis is not scaled by a later split.

    Day-2 wealth is 110 + $1 on a $100 share (11%). The following 2-for-1
    does not change that return. Split-adjusted index: 50 * 1.11 = 55.5.
    """
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A", "A"],
            "event_time": [
                datetime(2020, 1, 2, tzinfo=UTC),
                datetime(2020, 1, 3, tzinfo=UTC),
                datetime(2020, 1, 6, tzinfo=UTC),
            ],
            "open": [100.0, 110.0, 55.0],
            "high": [101.0, 111.0, 56.0],
            "low": [99.0, 109.0, 54.0],
            "close": [100.0, 110.0, 55.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    actions = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [
                datetime(2020, 1, 3, tzinfo=UTC),
                datetime(2020, 1, 6, tzinfo=UTC),
            ],
            "action_type": ["cash_dividend", "split"],
            "factor": [None, 2.0],
            "amount": [1.0, None],
        }
    )
    out = adjust_prices(bars, actions).sort("event_time")
    assert out["close_total_return"].to_list() == pytest.approx([50.0, 55.5, 55.5])
