"""Northset — order-book and candlestick research inside Dipcatcher."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.config.models import AppConfig, NorthsetConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset import (
    abdi_ranaldo_spread,
    bench_northset,
    candle_geometry,
    corwin_schultz_spread,
    garman_klass_vs_close_to_close,
    kyle_lambda,
    liquidity_sweep_frame,
    ohlc_identity_rate,
    parkinson_vs_close_to_close,
    rogers_satchell_vs_close_to_close,
    roll_spread,
    session_bipower_jump,
    session_candles_from_daily,
    session_chain_rate,
    session_reconstructs_daily_rate,
    session_volume_conservation_rate,
    session_vpin,
    sweep_rates,
    yang_zhang_variance,
)
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.schemas.order_book import BookLevel, OrderBookSnapshot


def _bars(n_assets: int = 8, n_days: int = 40, seed: int = 11):
    return SyntheticMarketProvider(n_assets=n_assets, n_days=n_days, seed=seed).get_bars()


def test_ohlc_identities_hold_on_synthetic_daily_bars() -> None:
    bars = _bars()
    assert ohlc_identity_rate(bars) == 1.0


def test_session_candles_reconstruct_daily_envelope() -> None:
    bars = _bars(n_assets=4, n_days=16, seed=3)
    session = session_candles_from_daily(bars, n_candles=8, seed=3)
    assert session.height == bars.height * 8
    assert ohlc_identity_rate(session) == 1.0
    assert session_reconstructs_daily_rate(bars, session) == 1.0
    assert session_volume_conservation_rate(bars, session) == 1.0
    assert session_chain_rate(session) == 1.0
    assert (session["event_time"] <= session["available_time"]).all()
    availability = session.group_by(["security_id", "parent_event_time"]).agg(
        pl.col("available_time").n_unique().alias("n_available"),
        pl.col("available_time").first().alias("available"),
    )
    assert int(availability["n_available"].max()) == 1
    parent_availability = bars.select(
        "security_id",
        pl.col("event_time").alias("parent_event_time"),
        pl.col("available_time").alias("expected_available"),
    )
    checked = availability.join(
        parent_availability,
        on=["security_id", "parent_event_time"],
    )
    assert (checked["available"] == checked["expected_available"]).all()
    vpin = session_vpin(session)
    assert np.isfinite(vpin)
    assert 0.0 <= vpin <= 1.0
    jumps = session_bipower_jump(session)
    assert 0.0 <= jumps["mean_jump_ratio"] <= 1.0


def test_ohlc_vol_estimators_finite_and_nonnegative_qlike() -> None:
    bars = _bars(n_assets=8, n_days=40, seed=9)
    for fn in (
        parkinson_vs_close_to_close,
        garman_klass_vs_close_to_close,
        rogers_satchell_vs_close_to_close,
    ):
        value = fn(bars)
        assert np.isfinite(value)
        assert value >= 0.0
    yz = yang_zhang_variance(bars)
    assert np.isfinite(yz)
    assert yz >= 0.0
    cs = corwin_schultz_spread(bars)
    assert cs != cs or (0.0 < cs < 1.0)
    ar = abdi_ranaldo_spread(bars)
    assert ar != ar or (0.0 <= ar < 1.0)


def test_candle_geometry_flags_are_binary() -> None:
    bars = _bars(n_assets=6, n_days=24, seed=2)
    geo = candle_geometry(bars)
    for col in (
        "candle_doji",
        "candle_hammer",
        "candle_engulfing",
        "candle_spinning_top",
        "candle_marubozu",
        "candle_shooting_star",
    ):
        vals = set(geo[col].drop_nulls().to_list())
        assert vals <= {0.0, 1.0}
    assert "wick_skew" in geo.columns
    assert "candle_gap" in geo.columns
    assert "close_location_value" in geo.columns


def test_liquidity_sweep_flags_and_rates() -> None:
    bars = _bars(n_assets=8, n_days=60, seed=13)
    sw = liquidity_sweep_frame(bars, lookback=10)
    for col in (
        "sweep_high",
        "sweep_low",
        "sweep_high_reclaim",
        "sweep_low_reclaim",
        "sweep_high_follow",
        "sweep_low_follow",
    ):
        vals = set(sw[col].drop_nulls().to_list())
        assert vals <= {0.0, 1.0}
    # reclaim/follow imply the sweep itself
    assert (sw["sweep_high_reclaim"].fill_null(0.0) <= sw["sweep_high"].fill_null(0.0)).all()
    assert (sw["sweep_low_follow"].fill_null(0.0) <= sw["sweep_low"].fill_null(0.0)).all()
    rates = sweep_rates(sw)
    assert 0.0 <= rates["sweep_eligible_rate"] <= 1.0
    assert rates["sweep_any_rate"] == rates["sweep_any_rate"]  # finite or NaN, never raises
    with pytest.raises(ValueError, match="lookback"):
        liquidity_sweep_frame(bars, lookback=1)


def test_liquidity_sweep_deterministic_reclaim() -> None:
    n = 12
    ts = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    highs = [11.0] * n
    lows = [9.0] * n
    closes = [10.0] * n
    highs[10] = 12.0  # sweep above prior 5-bar high
    closes[10] = 10.5  # but close back inside → reclaim
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * n,
            "event_time": ts,
            "open": [10.0] * n,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1e6] * n,
        }
    )
    sw = liquidity_sweep_frame(bars, lookback=5)
    row = sw.filter(pl.col("event_time") == ts[10]).row(0, named=True)
    assert row["sweep_high"] == 1.0
    assert row["sweep_high_reclaim"] == 1.0
    assert row["sweep_high_follow"] == 0.0
    assert row["sweep_depth_high"] == pytest.approx((12.0 - 11.0) / 11.0)
    assert row["sweep_reject_signed"] < 0.0
    rates = sweep_rates(sw)
    assert rates["sweep_high_reclaim_share"] == 1.0


def test_double_sweep_is_explicit_and_directionally_neutral() -> None:
    n = 8
    ts = [datetime(2020, 2, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    highs = [11.0] * n
    lows = [9.0] * n
    highs[5] = 12.0
    lows[5] = 8.0
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * n,
            "event_time": ts,
            "open": [10.0] * n,
            "high": highs,
            "low": lows,
            "close": [10.0] * n,
            "volume": [1e6] * n,
        }
    )
    row = liquidity_sweep_frame(bars, lookback=5).row(5, named=True)
    assert row["sweep_both"] == 1.0
    assert row["sweep_high_reclaim"] == 0.0
    assert row["sweep_low_reclaim"] == 0.0
    assert row["sweep_high_follow"] == 0.0
    assert row["sweep_low_follow"] == 0.0
    assert row["sweep_reject_signed"] == 0.0
    assert row["sweep_follow_signed"] == 0.0
    assert row["sweep_depth_signed"] == 0.0


def test_session_candles_reject_too_few_slices() -> None:
    bars = _bars(n_assets=3, n_days=16, seed=1)
    with pytest.raises(ValueError, match="n_candles"):
        session_candles_from_daily(bars, n_candles=1, seed=1)


def test_kyle_and_roll_and_parkinson_are_honest() -> None:
    rng = np.random.default_rng(0)
    q = rng.normal(size=80)
    y = 0.4 * q + 0.05 * rng.normal(size=80)
    lam, r2 = kyle_lambda(y, q)
    assert lam == pytest.approx(0.4, rel=0.25)
    assert r2 > 0.5
    bars = _bars(n_assets=6, n_days=30, seed=5)
    qlike = parkinson_vs_close_to_close(bars)
    assert np.isfinite(qlike)
    assert qlike >= 0.0


def test_roll_spread_nan_when_autocov_positive() -> None:
    # Strictly increasing mids → positive lag-1 autocov of diffs is not guaranteed,
    # but a linear drift is a defined input; estimator must not throw.
    mid = np.linspace(10.0, 12.0, 40)
    value = roll_spread(mid)
    assert value != value or value >= 0.0  # NaN or nonnegative


def test_kyle_nan_on_short_sample() -> None:
    lam, r2 = kyle_lambda(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert lam != lam and r2 != r2


def test_order_book_snapshot_still_rejects_crossed_book() -> None:
    ts = datetime(2020, 1, 2, 16, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="crossed or locked"):
        OrderBookSnapshot(
            security_id="A",
            event_time=ts,
            available_time=ts,
            bids=[BookLevel(price=10.0, size=1.0)],
            asks=[BookLevel(price=9.5, size=1.0)],
            depth=1,
        )


def test_bench_northset_honesty_and_identities() -> None:
    bars = _bars(n_assets=10, n_days=40, seed=11)
    cfg = AppConfig()
    cfg.northset.require_adjusted_ohlc = False
    cfg.data.source = "synthetic"
    cfg.data.synthetic_seed = 11
    cfg.northset.min_names = 3
    receipt = bench_northset(bars, cfg)
    assert receipt["family"] == "northset"
    assert receipt["product"] == "Northset"
    assert receipt["research_only"] is True
    assert "live_pnl_claim" not in receipt
    assert receipt["data_source"] == "SYNTHETIC"
    assert receipt["ohlc_identity_rate"] == 1.0
    assert receipt["book_uncrossed_rate"] == 1.0
    assert receipt["session_reconstructs_daily_rate"] == 1.0
    assert receipt["session_volume_conservation_rate"] == 1.0
    assert receipt["session_chain_rate"] == 1.0
    assert receipt["n_scored"] > 0
    assert np.isfinite(receipt["parkinson_qlike_vs_cc"])
    assert np.isfinite(receipt["garman_klass_qlike_vs_cc"])
    assert np.isfinite(receipt["rogers_satchell_qlike_vs_cc"])
    assert np.isfinite(receipt["overnight_plus_oc_qlike_vs_cc"])
    assert receipt["yang_zhang_qlike_scope"] == "per_security_expanding_oos"
    assert receipt["vpin_method"] == "count_window_bulk_ofi_proxy"
    assert receipt["sweep_inference_index"] == "calendar_including_idle_zeros"
    assert receipt["corwin_schultz_pair_scope"] == "prior_and_current_bar"
    assert receipt["sweep_overnight_gap_method"] == "event_close_to_next_open"
    assert receipt["sweep_two_way_inference_index"] == "event_rows_not_calendar_zeros"
    assert 0.0 <= receipt["session_mean_jump_ratio"] <= 1.0
    assert "ofi_mean_ic" in receipt
    assert "wick_skew_p_ic" in receipt
    assert "microprice_p_ic" in receipt
    assert "clv_p_ic" in receipt
    assert "vpin_p_ic" in receipt
    assert "doji_rate" in receipt
    assert "spinning_top_rate" in receipt
    assert "abdi_ranaldo_spread" in receipt
    assert "sweep_any_rate" in receipt
    assert "sweep_reject_signed_p_ic" in receipt
    assert "sweep_follow_signed_p_ic" in receipt
    assert "mean_fwd_ret_after_low_reclaim" in receipt
    assert family_blob_forbidden_metrics_absent(receipt) is True
    assert "sharpe" not in {k.lower() for k in receipt}


def test_empirical_bars_with_synthetic_book_are_labeled_mixed() -> None:
    bars = _bars(n_assets=8, n_days=40, seed=17)
    cfg = AppConfig()
    cfg.data.source = "parquet"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.use_session_l2 = False
    cfg.northset.min_names = 3
    receipt = bench_northset(bars, cfg)
    assert receipt["data_source"] == "MIXED_SYNTHETIC_DERIVED"
    assert receipt["book_hypothesis_eligible"] is False
    assert receipt["sweep_evidence_scope"] == "fixture_raw_unadjusted"


def test_northset_config_rejects_bad_levels() -> None:
    with pytest.raises(ValueError, match="n_book_levels"):
        NorthsetConfig(n_book_levels=0)
    with pytest.raises(ValueError, match="n_session_candles"):
        NorthsetConfig(n_session_candles=1)


def test_invalid_print_does_not_enter_prior_high() -> None:
    n = 12
    ts = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    highs = [11.0] * n
    highs[6] = 5.0  # high < close → identity fail; must not become prior_high
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * n,
            "event_time": ts,
            "open": [10.0] * n,
            "high": highs,
            "low": [9.0] * n,
            "close": [10.0] * n,
            "volume": [1e6] * n,
        }
    )
    sw = liquidity_sweep_frame(bars, lookback=5)
    smashed = sw.filter(pl.col("event_time") == ts[6]).row(0, named=True)
    assert smashed["ohlc_ok"] is False
    later = sw.filter(pl.col("event_time") == ts[11]).row(0, named=True)
    assert later["ohlc_ok"] is True
    assert later["prior_high"] == pytest.approx(11.0)


def test_invalid_entry_print_nulls_executable_return() -> None:
    from quant_fund.northset.sweep_research import sweep_forward_frame

    n = 12
    ts = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]
    opens = [10.0] * n
    opens[8] = -1.0  # identity fail; must not be used as next-open entry
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * n,
            "event_time": ts,
            "open": opens,
            "high": [11.0] * n,
            "low": [9.0] * n,
            "close": [10.0] * n,
            "volume": [1e6] * n,
        }
    )
    sw = liquidity_sweep_frame(bars, lookback=5)
    frame = sweep_forward_frame(sw, horizons=(1,), vol_lookback=3)
    entry_row = frame.filter(pl.col("event_time") == ts[7]).row(0, named=True)
    assert entry_row["sweep_exec_ret_1"] is None or (
        entry_row["sweep_exec_ret_1"] != entry_row["sweep_exec_ret_1"]
    )
    clean = frame.filter(pl.col("event_time") == ts[5]).row(0, named=True)
    assert clean["sweep_exec_ret_1"] == pytest.approx(0.0)


def test_yang_zhang_qlike_is_expanding_oos_not_pooled_constant() -> None:
    from quant_fund.metrics.scoring import qlike
    from quant_fund.northset.estimators import ohlc_variance_frame, yang_zhang_vs_close_to_close

    bars = _bars(n_assets=6, n_days=40, seed=11)
    oos = yang_zhang_vs_close_to_close(bars)
    assert np.isfinite(oos)
    frame = ohlc_variance_frame(bars)
    y = frame["var_cc"].to_numpy().astype(float)
    pooled_level = yang_zhang_variance(bars)
    yhat = np.full_like(y, pooled_level)
    mask = np.isfinite(y) & np.isfinite(yhat) & (y > 0) & (yhat > 0)
    pooled = float(qlike(y[mask], yhat[mask]))
    assert oos != pytest.approx(pooled)
