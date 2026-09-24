"""Coverage for northset identity guards, NaN honesty paths, and fail-closed edges."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.northset.identities import (
    _session_ohlc,
    book_identity_frame,
    book_uncrossed_rate,
    gap_finite_rate,
    ohlc_identity_frame,
    ohlc_identity_rate,
    session_candles_from_daily,
    session_chain_rate,
    session_reconstructs_daily_rate,
    session_volume_conservation_rate,
    validate_session_book_counts,
)

_TS = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)


def _daily_frame(
    rows: list[tuple[float, float, float, float]],
    *,
    security_id: str = "A",
    volume: float = 1e6,
    start: datetime = _TS,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": [security_id] * len(rows),
            "event_time": [start + timedelta(days=i) for i in range(len(rows))],
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [volume] * len(rows),
        }
    )


def _session_frame(
    parent: datetime,
    *,
    security_id: str = "A",
    n: int = 4,
    volume: float = 8.0,
) -> pl.DataFrame:
    rows = []
    price = 10.0
    for i in range(n):
        rows.append(
            {
                "security_id": security_id,
                "parent_event_time": parent,
                "session_index": i,
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price + 0.25,
                "volume": volume / n,
            }
        )
        price += 0.25
    return pl.DataFrame(rows)


def test_ohlc_identity_frame_rejects_missing_columns() -> None:
    bars = pl.DataFrame({"open": [10.0], "high": [11.0], "low": [9.0]})
    with pytest.raises(ValueError, match="missing OHLC columns"):
        ohlc_identity_frame(bars)


def test_ohlc_identity_flags_valid_and_broken_bars() -> None:
    bars = _daily_frame(
        [
            (10.0, 11.0, 9.0, 10.5),  # valid
            (10.0, 9.5, 9.0, 10.0),  # high < close
            (float("nan"), 11.0, 9.0, 10.5),  # non-finite open
            (10.0, 11.0, 9.0, -1.0),  # non-positive close
        ]
    )
    flagged = ohlc_identity_frame(bars)
    assert flagged["ohlc_ok"].to_list() == [True, False, False, False]
    assert ohlc_identity_rate(bars) == pytest.approx(0.25)


def test_ohlc_identity_rate_nan_on_empty_frame() -> None:
    assert math.isnan(ohlc_identity_rate(pl.DataFrame()))


def test_book_identity_frame_rejects_missing_columns() -> None:
    book = pl.DataFrame({"best_bid": [10.0]})
    with pytest.raises(ValueError, match="book missing columns"):
        book_identity_frame(book)


def test_book_identity_frame_without_spread_only_checks_cross() -> None:
    book = pl.DataFrame({"best_bid": [10.0, 10.5, float("nan")], "best_ask": [10.5, 10.0, 10.0]})
    flagged = book_identity_frame(book)
    assert flagged["book_uncrossed"].to_list() == [True, False, False]
    assert book_uncrossed_rate(book) == pytest.approx(1.0 / 3.0)


def test_book_identity_frame_spread_column_gates_uncrossed() -> None:
    book = pl.DataFrame(
        {
            "best_bid": [10.0, 10.0, 10.5],
            "best_ask": [10.5, 10.5, 10.0],
            "spread": [0.5, 0.0, 0.5],
        }
    )
    flagged = book_identity_frame(book)
    # bid < ask alone is not enough: a non-positive recorded spread fails the check.
    assert flagged["book_uncrossed"].to_list() == [True, False, False]


def test_book_uncrossed_rate_nan_on_empty_frame() -> None:
    assert math.isnan(book_uncrossed_rate(pl.DataFrame()))


def test_session_ohlc_rejects_too_few_candles() -> None:
    with pytest.raises(ValueError, match="n_session_candles"):
        _session_ohlc(10.0, 11.0, 9.0, 10.5, 1, True)


def test_session_candles_rejects_missing_columns() -> None:
    bars = _daily_frame([(10.0, 11.0, 9.0, 10.5)]).drop("close")
    with pytest.raises(ValueError, match="missing required columns"):
        session_candles_from_daily(bars)


def test_session_candles_empty_frame_returns_empty_schema() -> None:
    bars = _daily_frame([]).clear()
    out = session_candles_from_daily(bars, n_candles=4)
    assert out.height == 0
    for col in ("security_id", "event_time", "parent_event_time", "session_index", "source"):
        assert col in out.columns


def test_session_candles_skips_nonfinite_and_nonpositive_rows() -> None:
    valid_ts = _TS + timedelta(days=2)
    bad_times = [_TS, _TS + timedelta(days=1)]
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 3,
            "event_time": [bad_times[0], bad_times[1], valid_ts],
            "open": [10.0, 10.0, 10.0],
            "high": [float("nan"), 11.0, 11.0],
            "low": [9.0, 9.0, 9.0],
            "close": [10.5, -1.0, 10.5],
        }
    )
    session = session_candles_from_daily(bars, n_candles=4, seed=3)
    assert session.height == 4
    assert set(session["parent_event_time"].to_list()) == {valid_ts}
    assert ohlc_identity_rate(session) == 1.0


def test_session_candles_rejects_non_datetime_event_time() -> None:
    bars = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": ["2020-01-02"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
        }
    )
    with pytest.raises(TypeError, match="event_time must be datetime"):
        session_candles_from_daily(bars, n_candles=4)


def test_session_candles_rejects_non_datetime_available_time() -> None:
    bars = pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [_TS],
            "available_time": [1_578_768_000],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
        }
    )
    with pytest.raises(TypeError, match="available_time must be datetime"):
        session_candles_from_daily(bars, n_candles=4)


def test_session_reconstructs_daily_rate_nan_on_empty_inputs() -> None:
    daily = _daily_frame([(10.0, 11.0, 9.0, 10.5)])
    session = _session_frame(_TS)
    assert math.isnan(session_reconstructs_daily_rate(daily.clear(), session))
    assert math.isnan(session_reconstructs_daily_rate(daily, session.clear()))


def test_session_reconstructs_daily_rate_nan_when_no_parents_match() -> None:
    daily = _daily_frame([(10.0, 11.0, 9.0, 10.5)])
    session = _session_frame(_TS + timedelta(days=7))
    assert math.isnan(session_reconstructs_daily_rate(daily, session))


def test_session_reconstructs_daily_rate_zero_on_broken_envelope() -> None:
    daily = _daily_frame([(10.0, 11.0, 9.0, 10.5)])
    session = _session_frame(_TS, n=4).with_columns(pl.lit(99.0).alias("high"))
    assert session_reconstructs_daily_rate(daily, session) == 0.0


def test_session_volume_conservation_rate_nan_on_empty_inputs() -> None:
    daily = _daily_frame([(10.0, 11.0, 9.0, 10.5)])
    session = _session_frame(_TS)
    assert math.isnan(session_volume_conservation_rate(daily.clear(), session))
    assert math.isnan(session_volume_conservation_rate(daily, session.clear()))


def test_session_volume_conservation_rate_nan_without_volume_columns() -> None:
    daily = _daily_frame([(10.0, 11.0, 9.0, 10.5)])
    session = _session_frame(_TS)
    assert math.isnan(session_volume_conservation_rate(daily.drop("volume"), session))
    assert math.isnan(session_volume_conservation_rate(daily, session.drop("volume")))


def test_session_volume_conservation_rate_nan_when_no_parents_match() -> None:
    daily = _daily_frame([(10.0, 11.0, 9.0, 10.5)])
    session = _session_frame(_TS + timedelta(days=7))
    assert math.isnan(session_volume_conservation_rate(daily, session))


def test_session_volume_conservation_rate_detects_drift() -> None:
    daily = _daily_frame([(10.0, 11.0, 9.0, 10.5)], volume=8.0)
    session = _session_frame(_TS, n=4, volume=8.0)
    assert session_volume_conservation_rate(daily, session) == 1.0
    drifted = session.with_columns((pl.col("volume") * 2.0).alias("volume"))
    assert session_volume_conservation_rate(daily, drifted) == 0.0


def test_session_chain_rate_nan_guards() -> None:
    session = _session_frame(_TS)
    assert math.isnan(session_chain_rate(session.clear()))
    assert math.isnan(session_chain_rate(session.drop("session_index")))


def test_session_chain_rate_nan_without_successor_pairs() -> None:
    session = _session_frame(_TS, n=4).filter(pl.col("session_index") == 0)
    assert math.isnan(session_chain_rate(session))


def test_session_chain_rate_breaks_on_gapped_opens() -> None:
    session = _session_frame(_TS, n=5)
    assert session_chain_rate(session) == 1.0
    broken = session.with_columns(
        pl.when(pl.col("session_index") == 2)
        .then(pl.col("open") + 1.0)
        .otherwise(pl.col("open"))
        .alias("open")
    )
    assert session_chain_rate(broken) == pytest.approx(0.75)


def test_gap_finite_rate_nan_guards() -> None:
    bars = _daily_frame([(10.0, 11.0, 9.0, 10.5)])
    assert math.isnan(gap_finite_rate(bars.clear()))
    assert math.isnan(gap_finite_rate(bars.drop("open")))
    assert math.isnan(gap_finite_rate(bars.drop("close")))


def test_gap_finite_rate_nan_with_single_bar_per_security() -> None:
    bars = _daily_frame([(10.0, 11.0, 9.0, 10.5)])
    assert math.isnan(gap_finite_rate(bars))


def test_gap_finite_rate_counts_finite_gaps() -> None:
    bars = _daily_frame(
        [
            (10.0, 11.0, 9.0, 10.0),
            (10.5, 11.0, 9.5, 0.0),
            (10.0, 11.0, 9.0, 10.0),
        ]
    )
    # First bar has no prior close (not eligible); third gap divides by the
    # zero previous close → non-finite.
    assert gap_finite_rate(bars) == pytest.approx(0.5)


def test_validate_session_book_counts_rejects_bad_n() -> None:
    session = _session_frame(_TS)
    with pytest.raises(ValueError, match="n_session_candles"):
        validate_session_book_counts(session, n_session_candles=1)


def test_validate_session_book_counts_rejects_missing_columns() -> None:
    session = _session_frame(_TS).drop("session_index")
    with pytest.raises(ValueError, match="missing columns"):
        validate_session_book_counts(session, n_session_candles=4)


def test_validate_session_book_counts_rejects_empty_frame() -> None:
    session = _session_frame(_TS).clear()
    with pytest.raises(ValueError, match="empty"):
        validate_session_book_counts(session, n_session_candles=4)


def test_validate_session_book_counts_enforces_snaps_per_parent() -> None:
    session = _session_frame(_TS, n=4)
    assert validate_session_book_counts(session, n_session_candles=4) is session

    short = session.filter(pl.col("session_index") != 3)
    with pytest.raises(ValueError, match="count mismatch"):
        validate_session_book_counts(short, n_session_candles=4)

    duplicated = session.with_row_index("row").with_columns(
        pl.when(pl.col("row") == 3)
        .then(pl.lit(0))
        .otherwise(pl.col("session_index"))
        .alias("session_index")
    )
    with pytest.raises(ValueError, match="count mismatch"):
        validate_session_book_counts(duplicated, n_session_candles=4)
