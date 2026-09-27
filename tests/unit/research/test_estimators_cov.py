"""Coverage tests for northset/estimators.py edge branches.

Targets the fail-closed guards (missing columns, short windows, degenerate
series) and the honest-NaN semantics that the happy-path tests skip.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

import quant_fund.northset.estimators as est
from quant_fund.northset.estimators import (
    abdi_ranaldo_spread,
    amihud_illiquidity,
    corwin_schultz_spread,
    dm_range_vs_park,
    dm_split_vs_park,
    garman_klass_vs_close_to_close,
    kyle_lambda,
    lag1_corr,
    ohlc_variance_frame,
    order_flow_imbalance,
    overnight_plus_oc_vs_close_to_close,
    overnight_share,
    parkinson_vs_close_to_close,
    queue_imbalance,
    realized_semivariance,
    rogers_satchell_vs_close_to_close,
    roll_spread,
    session_bipower_jump,
    session_realized_variance,
    session_vpin,
    true_range_frame,
    volume_over_range,
    vpin_proxy,
    yang_zhang_variance,
    yang_zhang_vs_close_to_close,
)

_T0 = datetime(2021, 3, 1, 16, 0, tzinfo=UTC)


def _ts(d: int) -> datetime:
    return _T0 + timedelta(days=d)


def _bars(
    series: dict[str, list[tuple[float, float, float, float]]],
    volume: float = 1e5,
) -> pl.DataFrame:
    """Daily bars from explicit (open, high, low, close) tuples per security."""
    rows: list[dict[str, object]] = []
    for sid, ohlc in series.items():
        for d, (opn, hi, lo, cls) in enumerate(ohlc):
            rows.append(
                {
                    "security_id": sid,
                    "event_time": _ts(d),
                    "open": opn,
                    "high": hi,
                    "low": lo,
                    "close": cls,
                    "volume": volume,
                }
            )
    return pl.DataFrame(rows)


def _random_bars(
    n_sec: int = 4,
    n_days: int = 12,
    *,
    seed: int = 1,
    volume: float = 1e5,
) -> pl.DataFrame:
    """Consistent random OHLC bars (high >= max(o, c), low <= min(o, c))."""
    rng = np.random.default_rng(seed)
    series: dict[str, list[tuple[float, float, float, float]]] = {}
    for s in range(n_sec):
        cls = 100.0 + s
        path: list[tuple[float, float, float, float]] = []
        for _ in range(n_days):
            opn = cls * (1.0 + float(rng.normal(0.0, 0.005)))
            nxt = opn * (1.0 + float(rng.normal(0.0, 0.01)))
            hi = max(opn, nxt) * (1.0 + abs(float(rng.normal(0.0, 0.004))))
            lo = min(opn, nxt) * (1.0 - abs(float(rng.normal(0.0, 0.004))))
            path.append((opn, hi, lo, nxt))
            cls = nxt
        series[f"S{s}"] = path
    return _bars(series, volume=volume)


def _sessions(
    days: list[list[tuple[float, float, float]]],
    sid: str = "S0",
) -> pl.DataFrame:
    """Session bars: per parent day a list of (open, close, volume)."""
    rows: list[dict[str, object]] = []
    for d, sessions in enumerate(days):
        for i, (opn, cls, vol) in enumerate(sessions):
            rows.append(
                {
                    "security_id": sid,
                    "parent_event_time": _ts(d),
                    "session_index": i,
                    "event_time": _ts(d) + timedelta(hours=i),
                    "open": opn,
                    "close": cls,
                    "volume": vol,
                }
            )
    return pl.DataFrame(rows)


def _book(
    snaps: dict[str, list[tuple[float, float, float, float]]],
) -> pl.DataFrame:
    """Top-of-book snapshots from (best_bid, best_ask, top_bid_size, top_ask_size)."""
    rows: list[dict[str, object]] = []
    for sid, levels in snaps.items():
        for d, (bid, ask, tb, ta) in enumerate(levels):
            rows.append(
                {
                    "security_id": sid,
                    "event_time": _ts(d),
                    "best_bid": bid,
                    "best_ask": ask,
                    "top_bid_size": tb,
                    "top_ask_size": ta,
                }
            )
    return pl.DataFrame(rows)


def _assert_inconclusive(out: dict[str, float | str]) -> None:
    assert np.isnan(out["statistic"])
    assert np.isnan(out["p_value"])
    assert out["preferred"] == "inconclusive"


# --- kyle_lambda -----------------------------------------------------------


def test_kyle_lambda_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="must align"):
        kyle_lambda(np.array([1.0, 2.0, 3.0]), np.array([1.0]))


def test_kyle_lambda_constant_flow_returns_nan() -> None:
    """Zero-variance signed volume → λ unidentified → honest NaN."""
    rng = np.random.default_rng(0)
    delta = rng.normal(0.0, 0.01, size=30)
    lam, r2 = kyle_lambda(delta, np.full(30, 5.0))
    assert np.isnan(lam) and np.isnan(r2)


def test_kyle_lambda_constant_mid_moves_gives_nan_r2() -> None:
    """Constant Δmid → ss_tot ≈ 0 → λ = 0 but R² undefined."""
    q = np.arange(30, dtype=float)
    lam, r2 = kyle_lambda(np.full(30, 0.01), q)
    assert lam == pytest.approx(0.0)
    assert np.isnan(r2)


# --- roll_spread -----------------------------------------------------------


def test_roll_spread_too_few_finite_pairs_is_nan() -> None:
    """Enough finite prices but no finite consecutive change pairs."""
    mid = np.array([1e308, -1e308] * 7)  # diffs overflow to ±inf → all masked
    with np.errstate(over="ignore"):
        assert np.isnan(roll_spread(mid))


def test_roll_spread_positive_autocov_is_nan() -> None:
    """Trending (accelerating) prices give γ₁ > 0 → Roll undefined → NaN."""
    mid = np.cumsum(np.arange(1.0, 16.0))
    assert np.isnan(roll_spread(mid))


def test_roll_spread_recovers_bounce_spread() -> None:
    """Alternating ±1 bounce → γ₁ = −1 → spread 2√1."""
    mid = np.array([10.0, 11.0] * 8)
    assert roll_spread(mid) == pytest.approx(2.0)


# --- _qlike_elements -------------------------------------------------------


def test_qlike_elements_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="must align"):
        est._qlike_elements(np.array([1.0, 2.0]), np.array([1.0]))


def test_qlike_elements_floor_and_identity() -> None:
    """Non-positive inputs clip to the floor; y = ŷ gives the 0 loss."""
    out = est._qlike_elements(np.array([0.0, -3.0, 4.0]), np.array([-1.0, 0.0, 1.0]))
    assert out[0] == pytest.approx(0.0)  # both floored at 1e-12
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(4.0 - math.log(4.0) - 1.0)


# --- ohlc_variance_frame ---------------------------------------------------


def test_ohlc_variance_frame_rejects_missing_columns() -> None:
    bars = _random_bars().drop("high")
    with pytest.raises(ValueError, match="bars missing columns"):
        ohlc_variance_frame(bars)


def test_ohlc_variance_frame_first_bar_and_invalid_rows_are_nan() -> None:
    """First bar per name has no prev close; hi < lo and non-positive prices are excluded."""
    good = (10.0, 11.0, 9.0, 10.5)
    bars = _bars(
        {
            "S0": [
                good,
                (10.0, 9.0, 11.0, -1.0),  # hi < lo and close <= 0 → invalid
                (10.0, 11.0, 9.0, 10.1),
            ]
        }
    )
    frame = ohlc_variance_frame(bars.sort("event_time"))
    assert np.isnan(frame["var_cc"][0])  # no previous close
    assert np.isnan(frame["overnight_log"][0])
    assert np.isfinite(frame["var_park"][0])  # intraday range is still defined
    assert np.isnan(frame["var_park"][1])
    assert np.isnan(frame["var_cc"][2])  # previous close belonged to an invalid bar


def test_ohlc_variance_frame_values_match_closed_forms() -> None:
    o, h, lo, c = 10.0, 11.0, 9.0, 10.5
    prev_c = 10.0
    bars = _bars({"S0": [(9.8, 10.1, 9.7, prev_c), (o, h, lo, c)]})
    frame = ohlc_variance_frame(bars.sort("event_time")).filter(pl.col("close") == c)
    hl = math.log(h / lo)
    oc = math.log(c / o)
    assert frame["var_park"][0] == pytest.approx(hl**2 / (4.0 * math.log(2.0)))
    assert frame["var_gk"][0] == pytest.approx(0.5 * hl**2 - (2.0 * math.log(2.0) - 1.0) * oc**2)
    assert frame["var_rs"][0] == pytest.approx(
        math.log(h / c) * math.log(h / o) + math.log(lo / c) * math.log(lo / o)
    )
    assert frame["var_cc"][0] == pytest.approx(math.log(c / prev_c) ** 2)


# --- *_vs_close_to_close short-window guards --------------------------------


def test_qlike_diagnostics_nan_on_short_window() -> None:
    """< 8 usable (var_cc, forecast) pairs → honest NaN for every comparator."""
    bars = _random_bars(n_sec=2, n_days=4, seed=3)  # 6 valid pairs
    assert np.isnan(parkinson_vs_close_to_close(bars))
    assert np.isnan(garman_klass_vs_close_to_close(bars))
    assert np.isnan(rogers_satchell_vs_close_to_close(bars))
    assert np.isnan(overnight_plus_oc_vs_close_to_close(bars))
    assert np.isnan(yang_zhang_variance(bars))


def test_parkinson_nan_when_all_ranges_zero() -> None:
    """Flat high == low bars → var_park = 0 everywhere → mask empties → NaN."""
    flat = [(10.0, 10.0, 10.0, 10.0)] * 12
    bars = _bars({"S0": flat, "S1": flat, "S2": flat, "S3": flat})
    assert np.isnan(parkinson_vs_close_to_close(bars))


def test_qlike_diagnostics_finite_on_adequate_bars() -> None:
    bars = _random_bars(n_sec=6, n_days=30, seed=5)
    for fn in (
        parkinson_vs_close_to_close,
        garman_klass_vs_close_to_close,
        rogers_satchell_vs_close_to_close,
        overnight_plus_oc_vs_close_to_close,
    ):
        value = fn(bars)
        assert np.isfinite(value)
        assert value >= 0.0
    assert np.isfinite(yang_zhang_variance(bars))


# --- yang_zhang_vs_close_to_close ------------------------------------------


def test_yang_zhang_oos_nan_when_few_expanding_forecasts() -> None:
    """A single 10-bar name yields only 2 expanding-window forecasts → NaN."""
    bars = _random_bars(n_sec=1, n_days=10, seed=2)
    assert np.isnan(yang_zhang_vs_close_to_close(bars))


def test_yang_zhang_oos_skips_bad_and_flat_rows() -> None:
    """Non-finite var_cc rows and zero-variance histories are skipped per name;
    other names still accumulate enough losses for a finite mean."""
    rng = np.random.default_rng(4)
    series: dict[str, list[tuple[float, float, float, float]]] = {}
    for s in range(4):
        cls = 50.0 + s
        path: list[tuple[float, float, float, float]] = []
        for _ in range(13):
            opn = cls * (1.0 + float(rng.normal(0.0, 0.01)))
            nxt = opn * (1.0 + float(rng.normal(0.0, 0.02)))
            hi = max(opn, nxt) * 1.002
            lo = min(opn, nxt) * 0.998
            path.append((opn, hi, lo, nxt))
            cls = nxt
        series[f"S{s}"] = path
    # S0: poisoned close → var_cc NaN at t and t+1, and the row stops accumulating.
    series["S0"][5] = (50.0, 51.0, 49.0, float("nan"))
    # S1: flat close repeats → var_cc = 0 → loss skipped (y > 0 required).
    for t in (9, 10):
        o, h, lo, _c = series["S1"][t]
        prev_c = series["S1"][t - 1][3]
        series["S1"][t] = (o, max(h, prev_c), min(lo, prev_c), prev_c)
    # S2: first 9 bars flat → expanding forecast ŷ = 0 → skipped even though y > 0.
    flat = [(50.0, 50.0, 50.0, 50.0)] * 9
    moving = [
        (50.0, 50.5, 49.5, 50.2),
        (50.2, 51.0, 50.0, 50.8),
        (50.8, 51.5, 50.5, 51.2),
        (51.2, 52.0, 51.0, 51.6),
    ]
    series["S2"] = flat + moving
    value = yang_zhang_vs_close_to_close(_bars(series))
    assert np.isfinite(value)
    assert value >= 0.0


# --- dm diagnostics ---------------------------------------------------------


def test_dm_range_vs_park_inconclusive_on_short_panel() -> None:
    bars = _random_bars(n_sec=2, n_days=7, seed=6)  # 14 rows < 16
    _assert_inconclusive(dm_range_vs_park(bars))
    _assert_inconclusive(dm_range_vs_park(bars, which="rs"))


def test_dm_range_vs_park_inconclusive_with_few_finite_dates() -> None:
    """≥ 16 rows but < 8 dates where every name has finite losses."""
    bars = _random_bars(n_sec=4, n_days=5, seed=7)
    _assert_inconclusive(dm_range_vs_park(bars))


def test_dm_split_vs_park_inconclusive_guards() -> None:
    _assert_inconclusive(dm_split_vs_park(_random_bars(n_sec=2, n_days=7, seed=8)))
    _assert_inconclusive(dm_split_vs_park(_random_bars(n_sec=4, n_days=5, seed=9)))


# --- corwin_schultz_spread -------------------------------------------------


def test_corwin_schultz_rejects_missing_columns() -> None:
    bars = _random_bars().drop("low")
    with pytest.raises(ValueError, match="bars missing columns"):
        corwin_schultz_spread(bars)


def test_corwin_schultz_nan_on_too_few_pairs() -> None:
    bars = _random_bars(n_sec=1, n_days=8, seed=10)  # 7 two-day pairs < 8
    assert np.isnan(corwin_schultz_spread(bars))


def test_corwin_schultz_nan_when_all_pairs_filtered() -> None:
    """Point bars (high == low) give β = 0 and negative α → spreads < 0 → all dropped."""
    days = [(10.0 + d, 10.0 + d, 10.0 + d, 10.0 + d) for d in range(10)]
    assert np.isnan(corwin_schultz_spread(_bars({"S0": days})))


def test_corwin_schultz_finite_on_adequate_bars() -> None:
    value = corwin_schultz_spread(_random_bars(n_sec=4, n_days=20, seed=11))
    assert np.isfinite(value)
    assert 0.0 < value < 1.0


# --- amihud_illiquidity -----------------------------------------------------


def test_amihud_rejects_missing_columns() -> None:
    bars = _random_bars().drop("volume")
    with pytest.raises(ValueError, match="bars missing columns"):
        amihud_illiquidity(bars)


def test_amihud_first_bar_null_and_exact_value() -> None:
    bars = _bars({"S0": [(100.0, 101.0, 99.0, 100.0), (100.0, 102.0, 99.5, 101.0)]}, volume=1000.0)
    out = amihud_illiquidity(bars.sort("event_time"))
    assert out["amihud"][0] is None  # no previous close → undefined, not zero
    assert out["amihud"][1] == pytest.approx(abs(101.0 / 100.0 - 1.0) / (101.0 * 1000.0))


# --- order_flow_imbalance ---------------------------------------------------


def test_order_flow_imbalance_rejects_missing_columns() -> None:
    book = _book({"S0": [(10.0, 11.0, 5.0, 6.0)]}).drop("best_bid")
    with pytest.raises(ValueError, match="book missing columns"):
        order_flow_imbalance(book)


def test_order_flow_imbalance_sign_conventions() -> None:
    """CKS events: bid up adds new bid size; bid down removes old; ask down
    removes new ask size; ask up adds back the old ask size."""
    book = _book(
        {
            "S0": [
                (10.0, 11.0, 5.0, 6.0),
                (10.1, 11.0, 7.0, 6.0),  # bid up, ask flat
                (10.05, 10.9, 4.0, 8.0),  # bid down, ask down
            ]
        }
    )
    out = order_flow_imbalance(book.sort("event_time"))
    ofi = out["ofi"].to_list()
    assert ofi[0] is None  # first snapshot per name has no transition
    assert ofi[1] == pytest.approx(7.0 - 0.0 - 6.0 + 6.0)
    assert ofi[2] == pytest.approx(0.0 - 7.0 - 8.0 + 0.0)


# --- session_realized_variance ---------------------------------------------


def test_session_rv_rejects_missing_columns() -> None:
    session = _sessions([[(100.0, 110.0, 1.0)]]).drop("session_index")
    with pytest.raises(ValueError, match="session missing columns"):
        session_realized_variance(session)


def test_session_rv_chains_to_previous_close_not_open() -> None:
    """session_index 0 returns log(close/open); later sessions chain close-to-close."""
    r = math.log(1.1)
    session = _sessions(
        [[(100.0, 110.0, 1.0), (115.0, 121.0, 1.0)], [(200.0, 220.0, 1.0)]],
        sid="S0",
    )
    out = session_realized_variance(session).sort("parent_event_time")
    rv = out["session_rv"].to_list()
    assert rv[0] == pytest.approx(2.0 * r * r)
    assert rv[1] == pytest.approx(r * r)


# --- overnight_share --------------------------------------------------------


def test_overnight_share_nan_on_short_window() -> None:
    assert np.isnan(overnight_share(_random_bars(n_sec=2, n_days=4, seed=12)))


def test_overnight_share_nan_on_zero_cc_variance() -> None:
    """Constant closes → mean var_cc ≈ 0 → share undefined, not inf."""
    flat_close = [(9.0 + 0.1 * d, 10.5, 9.0, 10.0) for d in range(10)]
    bars = _bars({"S0": flat_close})
    assert np.isnan(overnight_share(bars))


def test_overnight_share_one_for_pure_gap_series() -> None:
    """Every move is an overnight gap with flat open→close → share ≈ 1."""
    days: list[tuple[float, float, float, float]] = []
    cls = 100.0
    for _ in range(10):
        opn = cls * 1.01
        days.append((opn, opn * 1.001, opn * 0.999, opn))  # close == open
        cls = opn
    assert overnight_share(_bars({"S0": days})) == pytest.approx(1.0)


# --- realized_semivariance --------------------------------------------------


def test_realized_semivariance_nan_on_short_window() -> None:
    bars = _random_bars(n_sec=1, n_days=7, seed=13)
    up, down = realized_semivariance(bars)
    assert np.isnan(up) and np.isnan(down)


def test_realized_semivariance_splits_up_and_down_mass() -> None:
    """Alternating ±10% closes: 5 up returns and 4 down returns of equal size."""
    closes = [100.0 if i % 2 == 0 else 110.0 for i in range(10)]
    days = [(c * 0.998, max(c, c * 0.998) * 1.001, min(c, c * 0.998) * 0.999, c) for c in closes]
    up, down = realized_semivariance(_bars({"S0": days}))
    m2 = math.log(1.1) ** 2
    assert up == pytest.approx(5.0 * m2 / 9.0)
    assert down == pytest.approx(4.0 * m2 / 9.0)


# --- abdi_ranaldo_spread ----------------------------------------------------


def test_abdi_ranaldo_rejects_missing_columns() -> None:
    bars = _random_bars().drop("high")
    with pytest.raises(ValueError, match="bars missing columns"):
        abdi_ranaldo_spread(bars)


def test_abdi_ranaldo_nan_on_too_few_pairs() -> None:
    bars = _random_bars(n_sec=1, n_days=8, seed=14)  # 7 pairs < 8
    assert np.isnan(abdi_ranaldo_spread(bars))


def test_abdi_ranaldo_zero_when_close_equals_hl_midpoint() -> None:
    """Close always at the high-low midpoint → η = 0 → zero spread."""
    days = [(10.0, 11.0, 9.0, 10.0)] * 10
    assert abdi_ranaldo_spread(_bars({"S0": days})) == pytest.approx(0.0)


def test_abdi_ranaldo_nan_when_all_pairs_filtered() -> None:
    """Degenerate hl_mid far below close → relative spread ≥ 1 → all dropped."""
    days = [(10.0, 5.0, 5.0, 10.0)] * 10  # mid = 5, close = 10 → rel = 1
    assert np.isnan(abdi_ranaldo_spread(_bars({"S0": days})))


# --- volume_over_range ------------------------------------------------------


def test_volume_over_range_rejects_missing_columns() -> None:
    bars = _random_bars().drop("volume")
    with pytest.raises(ValueError, match="bars missing columns"):
        volume_over_range(bars)


def test_volume_over_range_floor_keeps_zero_range_finite() -> None:
    """high == low divides by the 1e-12 floor → huge but finite."""
    bars = _bars({"S0": [(10.0, 10.0, 10.0, 10.0), (10.0, 12.0, 10.0, 11.0)]}, volume=1e6)
    out = volume_over_range(bars.sort("event_time"))
    assert out["volume_over_range"][0] == pytest.approx(1e6 / 1e-12)
    assert out["volume_over_range"][1] == pytest.approx(1e6 / 2.0)


# --- true_range_frame -------------------------------------------------------


def test_true_range_rejects_missing_columns() -> None:
    bars = _random_bars().drop("close")
    with pytest.raises(ValueError, match="bars missing columns"):
        true_range_frame(bars)


def test_true_range_values_include_gap_component() -> None:
    bars = _bars(
        {
            "S0": [
                (10.0, 11.0, 9.0, 10.0),  # TR = 11 − 9 (no prev close)
                (10.0, 10.5, 9.5, 9.0),  # TR = max(1, 0.5, 0.5) = 1
                (11.0, 12.0, 11.8, 11.9),  # gap up: max(0.2, 3.0, 2.8) = 3
            ]
        }
    )
    tr = true_range_frame(bars.sort("event_time"))["true_range"].to_list()
    assert tr == pytest.approx([2.0, 1.0, 3.0])


# --- lag1_corr --------------------------------------------------------------


def test_lag1_corr_short_and_constant_series_are_nan() -> None:
    assert np.isnan(lag1_corr(np.arange(11, dtype=float)))
    assert np.isnan(lag1_corr(np.full(15, 3.0)))


def test_lag1_corr_alternating_series_is_minus_one() -> None:
    x = np.array([0.0, 1.0] * 8)
    assert lag1_corr(x) == pytest.approx(-1.0)


def test_lag1_corr_drops_non_finite_before_pairing() -> None:
    """NaN/±inf are filtered out; the survivors pair as consecutive."""
    x = np.array([1.0, np.inf, 3.0, 2.0, 5.0, 4.0, 7.0, np.nan, 6.0, 9.0, 8.0, 11.0, 10.0, 13.0])
    got = lag1_corr(x)
    finite = x[np.isfinite(x)]
    expected = float(np.corrcoef(finite[1:], finite[:-1])[0, 1])
    assert got == pytest.approx(expected)


# --- session_bipower_jump ---------------------------------------------------


def test_session_bipower_rejects_missing_columns() -> None:
    session = _sessions([[(100.0, 110.0, 1.0)]]).drop("open")
    with pytest.raises(ValueError, match="session missing columns"):
        session_bipower_jump(session)


def test_session_bipower_empty_frame_is_nan_dict() -> None:
    session = pl.DataFrame(
        schema={
            "security_id": pl.Utf8,
            "parent_event_time": pl.Datetime,
            "session_index": pl.Int64,
            "open": pl.Float64,
            "close": pl.Float64,
        }
    )
    out = session_bipower_jump(session)
    assert np.isnan(out["mean_jump_ratio"])
    assert np.isnan(out["mean_rv"])
    assert np.isnan(out["mean_bv"])


def test_session_bipower_nan_below_four_days() -> None:
    days = [[(100.0, 101.0, 1.0), (101.0, 102.0, 1.0)] for _ in range(3)]
    out = session_bipower_jump(_sessions(days))
    assert np.isnan(out["mean_jump_ratio"])


def test_session_bipower_jump_matches_closed_form() -> None:
    """Day of returns [a, a, b, a]: rv = 3a² + b², bv = (π/2)(a² + 2ab)."""
    a, b = 0.1, 0.5
    e = math.exp
    days = [
        [
            (100.0, 100.0 * e(a), 1.0),
            (0.0, 100.0 * e(2 * a), 1.0),  # open ignored past index 0
            (0.0, 100.0 * e(2 * a + b), 1.0),
            (0.0, 100.0 * e(3 * a + b), 1.0),
        ]
    ] * 4
    # Plus a zero-variance day that must be masked out (rv ≈ 0).
    days.append([(100.0, 100.0, 1.0)] * 4)
    out = session_bipower_jump(_sessions(days))
    rv = 3.0 * a * a + b * b
    bv = (math.pi / 2.0) * (a * a + 2.0 * a * b)
    jump = min(1.0, max(0.0, 1.0 - bv / rv))
    assert out["mean_rv"] == pytest.approx(rv)
    assert out["mean_bv"] == pytest.approx(bv)
    assert out["mean_jump_ratio"] == pytest.approx(jump)
    assert 0.0 <= out["mean_jump_ratio"] <= 1.0


# --- session_vpin -----------------------------------------------------------


def test_session_vpin_missing_or_empty_returns_nan() -> None:
    """Fail-closed but NaN, not a raise — unlike the raising siblings."""
    assert np.isnan(session_vpin(_sessions([[(100.0, 101.0, 1.0)]]).drop("volume")))
    empty = pl.DataFrame(
        schema={
            "security_id": pl.Utf8,
            "parent_event_time": pl.Datetime,
            "open": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )
    assert np.isnan(session_vpin(empty))


def test_session_vpin_nan_below_four_days() -> None:
    days = [[(100.0, 101.0, 1.0)] for _ in range(3)]
    assert np.isnan(session_vpin(_sessions(days)))


def test_session_vpin_signed_volume_share() -> None:
    """|Σ signed vol| / Σ vol per day: up day = 1, balanced day = 0, flat ignored."""
    days = [
        [(100.0, 101.0, 30.0), (101.0, 102.0, 10.0)],  # net 40 / 40 = 1.0
        [(100.0, 101.0, 20.0), (101.0, 100.0, 20.0)],  # net 0 / 40 = 0.0
        [(100.0, 101.0, 30.0), (101.0, 100.0, 10.0)],  # net 20 / 40 = 0.5
        [(100.0, 100.0, 50.0), (100.0, 100.0, 50.0)],  # flat bars sign 0 → 0.0
    ]
    assert session_vpin(_sessions(days)) == pytest.approx((1.0 + 0.0 + 0.5 + 0.0) / 4.0)


# --- queue_imbalance --------------------------------------------------------


def test_queue_imbalance_rejects_missing_columns() -> None:
    book = _book({"S0": [(10.0, 11.0, 5.0, 6.0)]}).drop("top_ask_size")
    with pytest.raises(ValueError, match="book missing columns"):
        queue_imbalance(book)


def test_queue_imbalance_zero_denom_is_null_not_zero() -> None:
    """Empty both sides → undefined (null), not a fake 0.0 balance."""
    book = _book({"S0": [(10.0, 11.0, 0.0, 0.0), (10.0, 11.0, 3.0, 1.0), (10.0, 11.0, 1.0, 3.0)]})
    qi = queue_imbalance(book.sort("event_time"))["queue_imbalance"].to_list()
    assert qi[0] is None
    assert qi[1] == pytest.approx(0.5)
    assert qi[2] == pytest.approx(-0.5)


# --- vpin_proxy -------------------------------------------------------------


def test_vpin_proxy_rejects_missing_columns_and_bad_params() -> None:
    book = _book({"S0": [(10.0, 11.0, 5.0, 6.0)] * 4})
    with pytest.raises(ValueError, match="book missing columns"):
        vpin_proxy(book.drop("best_ask"))
    with pytest.raises(ValueError, match="window"):
        vpin_proxy(book, window=1)
    with pytest.raises(ValueError, match="bucket_volume"):
        vpin_proxy(book, bucket_volume=0.0)


def _vpin_test_book() -> pl.DataFrame:
    return _book(
        {
            "S0": [
                (10.0, 11.0, 100.0, 100.0),
                (10.1, 11.0, 200.0, 100.0),  # buy 300 / sell 100 → tox 0.5
                (10.05, 10.9, 150.0, 300.0),  # buy 100 / sell 200 → tox 1/3
                (10.2, 11.2, 400.0, 50.0),  # buy 400 / sell 50 → tox 7/9
                (10.2, 11.3, 100.0, 60.0),  # buy 100 / sell 460 → tox 9/14
            ]
        }
    )


def test_vpin_proxy_count_window_matches_rolling_toxicity() -> None:
    out = vpin_proxy(_vpin_test_book().sort("event_time"), window=3)
    vpin = out["vpin"].to_list()
    assert vpin[0] is None and vpin[1] is None  # needs 2 non-null tox values
    tox = [0.5, 1.0 / 3.0, 7.0 / 9.0, 9.0 / 14.0]
    assert vpin[2] == pytest.approx(np.mean(tox[0:2]))
    assert vpin[3] == pytest.approx(np.mean(tox[0:3]))
    assert vpin[4] == pytest.approx(np.mean(tox[1:4]))


def test_vpin_proxy_bucket_volume_clock_carries_last_value() -> None:
    """500-vol buckets straddle rows; unfilled rows carry the previous VPIN."""
    out = vpin_proxy(_vpin_test_book().sort("event_time"), bucket_volume=500.0, window=3)
    vpin = out["vpin"].to_list()
    assert vpin[0] != vpin[0] or vpin[0] is None  # NaN before first bucket
    assert vpin[1] != vpin[1] or vpin[1] is None  # 400 < 500 accumulated
    # t2 completes bucket: (buy 300+100, sell 100+200) → |400−300|/700
    first = pytest.approx(100.0 / 700.0)
    assert vpin[2] == first
    assert vpin[3] == first  # 450 < 500 → carry
    # t4 completes bucket: (buy 400+100, sell 50+460) → |500−510|/1010
    assert vpin[4] == pytest.approx((100.0 / 700.0 + 10.0 / 1010.0) / 2.0)


def test_vpin_proxy_bucket_clock_is_per_security() -> None:
    """Each security keeps its own volume clock; buckets never bleed across."""
    snaps = {
        "S0": [(10.0, 11.0, 100.0, 100.0)] * 4,  # identical prints → tox 0 buckets
        "S1": [
            (10.0, 11.0, 300.0, 10.0),
            (10.1, 11.0, 300.0, 10.0),  # buy 310 / sell 10 → tox 15/16
            (10.0, 11.0, 300.0, 10.0),  # buy 10 / sell 310 → tox 15/16
            (10.1, 11.0, 300.0, 10.0),
        ],
    }
    out = vpin_proxy(_book(snaps).sort(["security_id", "event_time"]), bucket_volume=100.0)
    s0 = out.filter(pl.col("security_id") == "S0")["vpin"].to_list()
    s1 = out.filter(pl.col("security_id") == "S1")["vpin"].to_list()
    assert s0[0] != s0[0] or s0[0] is None
    assert all(v == pytest.approx(0.0) for v in s0[1:])
    assert s1[0] != s1[0] or s1[0] is None
    assert all(v == pytest.approx(15.0 / 16.0) for v in s1[1:])
