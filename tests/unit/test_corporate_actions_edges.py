"""Wave 37: corporate_actions adjust_prices / splits / dividends edges.

Complements test_data_pit.test_split_does_not_destroy_raw (raw preservation).
Research/infrastructure only — no live broker / vendor MD.
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.data.corporate_actions import (
    adjust_prices,
    attach_dividends,
    cumulative_split_factors,
)

RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False


def _bars_ab() -> pl.DataFrame:
    """Two sessions for A (pre/post split quote) and one for B."""
    return pl.DataFrame(
        {
            "security_id": ["A", "A", "B"],
            "event_time": [
                datetime(2020, 1, 2, tzinfo=UTC),
                datetime(2020, 1, 3, tzinfo=UTC),
                datetime(2020, 1, 2, tzinfo=UTC),
            ],
            "open": [100.0, 50.0, 20.0],
            "high": [101.0, 51.0, 21.0],
            "low": [99.0, 49.0, 19.0],
            "close": [100.0, 50.0, 20.0],
            "volume": [1e6, 1e6, 5e5],
        }
    )


def test_empty_actions_identity_factors() -> None:
    bars = _bars_ab()
    out = adjust_prices(bars, pl.DataFrame())
    assert out["split_factor"].to_list() == [1.0, 1.0, 1.0]
    assert out["dividend"].to_list() == [0.0, 0.0, 0.0]
    assert out["close"].to_list() == bars["close"].to_list()
    assert out["close_split_adjusted"].to_list() == bars["close"].to_list()


def test_actions_missing_action_type_column_noop() -> None:
    bars = _bars_ab()
    bare = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
        }
    )
    out = cumulative_split_factors(bars, bare)
    assert out["split_factor"].to_list() == [1.0, 1.0, 1.0]


def test_two_for_one_split_closed_form() -> None:
    """2-for-1 on ex-date D: pre-ex raw close / 2 == post-ex adjusted close."""
    bars = _bars_ab().filter(pl.col("security_id") == "A")
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "action_type": ["split"],
            "factor": [2.0],
            "amount": [None],
        }
    )
    out = adjust_prices(bars, actions).sort("event_time")
    # Raw preserved
    assert out["close"].to_list() == [100.0, 50.0]
    # Pre-split factor=2 → adj 50; on/after ex factor=1 → adj 50
    assert out["split_factor"].to_list() == [2.0, 1.0]
    assert out["close_split_adjusted"].to_list() == [50.0, 50.0]
    assert out["open_split_adjusted"][0] == pytest.approx(50.0)


def test_attach_dividends_cash_on_ex_date() -> None:
    bars = _bars_ab().filter(pl.col("security_id") == "A")
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "action_type": ["cash_dividend"],
            "factor": [None],
            "amount": [1.25],
        }
    )
    out = attach_dividends(bars, actions).sort("event_time")
    assert out["dividend"].to_list() == [0.0, 1.25]


def test_attach_dividends_empty_or_non_div_actions() -> None:
    bars = _bars_ab()
    assert attach_dividends(bars, pl.DataFrame())["dividend"].to_list() == [0.0, 0.0, 0.0]
    splits_only = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "action_type": ["split"],
            "factor": [2.0],
            "amount": [None],
        }
    )
    assert attach_dividends(bars, splits_only)["dividend"].to_list() == [0.0, 0.0, 0.0]


def test_missing_security_in_actions_leaves_unaffected() -> None:
    """Actions for Z must not change split factors on A/B."""
    bars = _bars_ab()
    actions = pl.DataFrame(
        {
            "security_id": ["Z"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "action_type": ["split"],
            "factor": [2.0],
            "amount": [None],
        }
    )
    out = adjust_prices(bars, actions)
    assert out["split_factor"].to_list() == [1.0, 1.0, 1.0]
    assert out["close_split_adjusted"].to_list() == out["close"].to_list()


def test_dividend_total_return_closed_form() -> None:
    """Price drop equal to cash dividend → TR flat on ex-date."""
    bars = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [
                datetime(2020, 1, 2, tzinfo=UTC),
                datetime(2020, 1, 3, tzinfo=UTC),
            ],
            "open": [100.0, 99.0],
            "high": [101.0, 100.0],
            "low": [99.0, 98.0],
            "close": [100.0, 99.0],
            "volume": [1e6, 1e6],
        }
    )
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "action_type": ["cash_dividend"],
            "factor": [None],
            "amount": [1.0],
        }
    )
    out = adjust_prices(bars, actions).sort("event_time")
    assert out["close_total_return"].to_list() == pytest.approx([100.0, 100.0])
