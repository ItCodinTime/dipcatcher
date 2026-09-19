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
    apply_listing_actions,
    attach_dividends,
    cumulative_split_factors,
)
from quant_fund.schemas.errors import PointInTimeError

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
    adjusted = adjust_prices(bars, bare)
    assert adjusted["split_factor"].to_list() == [1.0, 1.0, 1.0]
    assert adjusted["dividend"].to_list() == [0.0, 0.0, 0.0]


def test_late_split_does_not_rewrite_preavailability_history() -> None:
    bars = _bars_ab().filter(pl.col("security_id") == "A")
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "available_time": [datetime(2020, 1, 5, tzinfo=UTC)],
            "action_type": ["split"],
            "factor": [2.0],
            "amount": [None],
        }
    )
    out = adjust_prices(bars, actions).sort("event_time")
    assert out["split_factor"].to_list() == [1.0, 1.0]
    assert out["close_split_adjusted"].to_list() == [100.0, 50.0]


def test_available_split_adjusts_preex_bars() -> None:
    bars = _bars_ab().filter(pl.col("security_id") == "A")
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "available_time": [datetime(2020, 1, 2, tzinfo=UTC)],
            "action_type": ["split"],
            "factor": [2.0],
            "amount": [None],
        }
    )
    out = adjust_prices(bars, actions).sort("event_time")
    assert out["split_factor"].to_list() == [2.0, 1.0]


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


def test_late_dividend_is_not_attached() -> None:
    bars = _bars_ab().filter(pl.col("security_id") == "A")
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "available_time": [datetime(2020, 1, 4, tzinfo=UTC)],
            "action_type": ["cash_dividend"],
            "factor": [None],
            "amount": [1.25],
        }
    )
    assert attach_dividends(bars, actions)["dividend"].to_list() == [0.0, 0.0]


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


def test_same_day_cash_and_special_dividends_do_not_explode_bars() -> None:
    bars = _bars_ab().filter(pl.col("security_id") == "A")
    actions = pl.DataFrame(
        {
            "security_id": ["A", "A"],
            "event_time": [
                datetime(2020, 1, 3, tzinfo=UTC),
                datetime(2020, 1, 3, tzinfo=UTC),
            ],
            "action_type": ["cash_dividend", "special_dividend"],
            "factor": [None, None],
            "amount": [1.0, 0.5],
        }
    )
    out = attach_dividends(bars, actions).sort("event_time")
    assert out.height == bars.height
    assert out["dividend"].to_list() == pytest.approx([0.0, 1.5])


def test_nonpositive_split_factor_fail_closed() -> None:
    bars = _bars_ab().filter(pl.col("security_id") == "A")
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "action_type": ["split"],
            "factor": [0.0],
            "amount": [None],
        }
    )
    with pytest.raises(PointInTimeError, match="non-positive or non-finite factors"):
        cumulative_split_factors(bars, actions)


def test_negative_dividend_amount_fail_closed() -> None:
    bars = _bars_ab().filter(pl.col("security_id") == "A")
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "action_type": ["cash_dividend"],
            "factor": [None],
            "amount": [-1.0],
        }
    )
    with pytest.raises(PointInTimeError, match="negative or non-finite amounts"):
        attach_dividends(bars, actions)


def test_delist_drops_post_event_bars_and_keeps_last_session() -> None:
    bars = _bars_ab()
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 2, tzinfo=UTC)],
            "action_type": ["delist"],
            "factor": [None],
            "amount": [None],
        }
    )
    out = apply_listing_actions(bars, actions).sort(["security_id", "event_time"])
    assert out.filter(pl.col("security_id") == "A")["event_time"].to_list() == [
        datetime(2020, 1, 2, tzinfo=UTC)
    ]
    assert out.filter(pl.col("security_id") == "B").height == 1


def test_late_delist_does_not_drop_preavailability_bars() -> None:
    bars = _bars_ab().filter(pl.col("security_id") == "A")
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 2, tzinfo=UTC)],
            "available_time": [datetime(2020, 1, 5, tzinfo=UTC)],
            "action_type": ["delist"],
            "factor": [None],
            "amount": [None],
        }
    )
    out = apply_listing_actions(bars, actions)
    assert out.height == bars.height


def test_ticker_change_overrides_symbol_after_event() -> None:
    bars = _bars_ab().with_columns(pl.lit("AAA").alias("symbol"))
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "action_type": ["ticker_change"],
            "new_ticker": ["AA2"],
            "factor": [None],
            "amount": [None],
        }
    )
    out = apply_listing_actions(bars, actions).sort(["security_id", "event_time"])
    a = out.filter(pl.col("security_id") == "A")
    assert a["symbol"].to_list() == ["AAA", "AA2"]
    assert out.filter(pl.col("security_id") == "B")["symbol"].to_list() == ["AAA"]


def test_late_ticker_change_does_not_rewrite_preavailability_symbol() -> None:
    bars = (
        _bars_ab().filter(pl.col("security_id") == "A").with_columns(pl.lit("AAA").alias("symbol"))
    )
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 2, tzinfo=UTC)],
            "available_time": [datetime(2020, 1, 5, tzinfo=UTC)],
            "action_type": ["ticker_change"],
            "new_ticker": ["AA2"],
            "factor": [None],
            "amount": [None],
        }
    )
    out = apply_listing_actions(bars, actions).sort("event_time")
    assert out["symbol"].to_list() == ["AAA", "AAA"]


def test_blank_ticker_change_fail_closed() -> None:
    bars = _bars_ab()
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "action_type": ["ticker_change"],
            "new_ticker": ["  "],
            "factor": [None],
            "amount": [None],
        }
    )
    with pytest.raises(PointInTimeError, match="blank new_ticker"):
        apply_listing_actions(bars, actions)


def test_announced_delist_drops_immediately_when_include_delisted_false() -> None:
    bars = _bars_ab().filter(pl.col("security_id") == "A")
    actions = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [datetime(2020, 1, 3, tzinfo=UTC)],
            "available_time": [datetime(2020, 1, 2, tzinfo=UTC)],
            "action_type": ["delist"],
            "factor": [None],
            "amount": [None],
        }
    )
    kept = apply_listing_actions(bars, actions, include_delisted=True)
    assert kept.height == 2
    dropped = apply_listing_actions(bars, actions, include_delisted=False)
    assert dropped.height == 0
