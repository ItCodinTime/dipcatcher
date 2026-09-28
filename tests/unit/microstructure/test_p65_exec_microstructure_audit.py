"""P6.5 exec/microstructure audit regressions — KATs against cited specs.

All fixtures are synthetic and deterministic. Each test pins one claim that
was wrong, silent, or underspecified before the audit (docs/AUDIT_P65_MICRO.md).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.execution.almgren_chriss import almgren_chriss_trajectory
from quant_fund.microstructure import (
    attach_candle_book_features,
    synthesize_l2_from_bars,
)
from quant_fund.microstructure.candle_book_features import forward_close_return_labels
from quant_fund.northset import estimators as est
from quant_fund.northset.sweep_research import _attach_event_costs

_T0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _book_frame(rows: list[tuple[float, float, float, float]]) -> pl.DataFrame:
    """(best_bid, best_ask, top_bid_size, top_ask_size) snapshots, 1min apart."""
    return pl.DataFrame(
        {
            "security_id": ["A"] * len(rows),
            "event_time": [_T0 + timedelta(minutes=i) for i in range(len(rows))],
            "best_bid": [r[0] for r in rows],
            "best_ask": [r[1] for r in rows],
            "top_bid_size": [r[2] for r in rows],
            "top_ask_size": [r[3] for r in rows],
        }
    )


# --- almgren_chriss -------------------------------------------------------


def test_ac_matches_sinh_form_moderate_kappa() -> None:
    # risk_aversion=0.01, sigma=0.02, eta=1e-4 -> kappa=0.2, kappa*T=2.
    h = almgren_chriss_trajectory(100.0, 10, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=0.01)
    t = np.arange(11)
    kappa = 0.2
    expected = 100.0 * np.sinh(kappa * (10 - t)) / np.sinh(kappa * 10)
    expected[-1] = 0.0
    assert np.allclose(h, expected, atol=1e-9)


def test_ac_zero_risk_aversion_is_exact_twap() -> None:
    h = almgren_chriss_trajectory(100.0, 4, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=0.0)
    assert np.allclose(h, [100.0, 75.0, 50.0, 25.0, 0.0])


def test_ac_overflow_regime_liquidates_not_twap() -> None:
    # kappa*T ~ 3e5: old code overflowed sinh -> isfinite(denom) failed ->
    # silently returned TWAP. Correct AC limit is (near) immediate liquidation.
    h = almgren_chriss_trajectory(100.0, 10, sigma=1.0, eta=1e-3, gamma=0.0, risk_aversion=1e6)
    assert np.all(np.isfinite(h))
    assert h[1] < 1e-3 * 100.0  # far below the buggy TWAP value of 90.0
    assert np.all(np.diff(h) <= 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": float("nan")},
        {"sigma": float("inf")},
        {"sigma": 0.0},
        {"eta": float("nan")},
        {"eta": 0.0},
        {"gamma": float("nan")},
        {"risk_aversion": float("nan")},
        {"risk_aversion": -0.1},
    ],
)
def test_ac_rejects_bad_params(kwargs: dict[str, float]) -> None:
    base: dict[str, float] = {
        "sigma": 0.02,
        "eta": 1e-4,
        "gamma": 0.0,
        "risk_aversion": 0.01,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        almgren_chriss_trajectory(100.0, 10, **base)


def test_ac_rejects_bad_quantity_and_n_slices() -> None:
    with pytest.raises(ValueError):
        almgren_chriss_trajectory(
            float("nan"), 10, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=0.0
        )
    with pytest.raises(ValueError):
        almgren_chriss_trajectory(100.0, 10.5, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=0.0)
    with pytest.raises(ValueError):
        almgren_chriss_trajectory(
            100.0,
            True,
            sigma=0.02,
            eta=1e-4,
            gamma=0.0,
            risk_aversion=0.0,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        almgren_chriss_trajectory(
            100.0, 10, tau=float("nan"), sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=0.0
        )


def test_ac_integral_float_n_slices_accepted() -> None:
    h = almgren_chriss_trajectory(10.0, 4.0, sigma=0.02, eta=1e-4, gamma=0.0, risk_aversion=0.0)
    assert h.shape == (5,)


# --- order_flow_imbalance / vpin_proxy CKS legs ---------------------------

# Snapshot sequence engineered so every leg fires distinctly. CKS ties fire
# BOTH conditions on the unchanged side, so a flat ask contributes
# (-q^A_n + q^A_{n-1}) to ofi (not zero legs):
#   t1: bid up, ask flat -> ofi = q^B_1 - q^A_1 + q^A_0 = 12-10+10 = +12
#   t2: bid flat, ask DOWN -> ofi = q^B_2 - q^B_1 - q^A_2 = 8-12-6 = -10
#   t3: bid flat, ask UP   -> ofi = q^B_3 - q^B_2 + q^A_2 = 8-8+6 = +6
_CKS_BOOK = _book_frame(
    [
        (100.0, 101.0, 10.0, 10.0),  # t0: no previous row
        (100.5, 101.0, 12.0, 10.0),  # t1
        (100.5, 100.8, 8.0, 6.0),  # t2
        (100.5, 101.5, 8.0, 20.0),  # t3
    ]
)


def test_ofi_cks_known_answers() -> None:
    ofi = est.order_flow_imbalance(_CKS_BOOK)["ofi"].to_numpy()
    assert math.isnan(ofi[0])
    assert ofi[1] == pytest.approx(12.0)
    assert ofi[2] == pytest.approx(-10.0)
    assert ofi[3] == pytest.approx(6.0)


def test_vpin_buy_minus_sell_equals_ofi() -> None:
    # bucket_volume=1 gives unit-volume buckets (Easley volume clock): a
    # row of volume V completes V buckets, all at that row's toxicity, so
    # the window mean collapses to the row's own tox.
    # vpin_i = tox_i = |buy_i - sell_i| / (buy_i + sell_i).
    # CKS legs (buy = bid_up*q^B_n + ask_up*q^A_{n-1};
    #            sell = bid_dn*q^B_{n-1} + ask_dn*q^A_n):
    #   t1 buy=12+10=22 sell=0+10=10 -> tox=12/32=0.375
    #   t2 buy=8+0=8    sell=12+6=18 -> tox=10/26
    #   t3 buy=8+6=14   sell=8+0=8   -> tox=6/22
    # and buy-sell == ofi exactly on every row. The pre-fix code swapped the
    # ask legs (ask_dn*prev size into buy, ask_up*current size into sell).
    out = est.vpin_proxy(_CKS_BOOK, bucket_volume=1.0, window=2)
    vpin = out["vpin"].to_numpy()
    assert math.isnan(vpin[0])
    assert vpin[1] == pytest.approx(12.0 / 32.0)
    assert vpin[2] == pytest.approx(10.0 / 26.0)
    assert vpin[3] == pytest.approx(6.0 / 22.0)


# --- amihud_illiquidity ---------------------------------------------------


def test_amihud_zero_dollar_volume_is_null_not_1e12() -> None:
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 4,
            "event_time": [_T0 + timedelta(days=i) for i in range(4)],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1000.0, 0.0, 2000.0, -5.0],
        }
    )
    out = est.amihud_illiquidity(bars)["amihud"].to_numpy()
    assert math.isnan(out[0])  # no prev close
    assert math.isnan(out[1])  # zero dollar volume (was |r|*1e12)
    assert math.isnan(out[3])  # negative dollar volume
    assert out[2] == pytest.approx(abs(102.0 / 101.0 - 1.0) / (102.0 * 2000.0))


# --- corwin_schultz -------------------------------------------------------


def _cs_bars(ranges: list[tuple[float, float]], sec: str = "G") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "security_id": [sec] * len(ranges),
            "event_time": [_T0 + timedelta(days=i) for i in range(len(ranges))],
            "high": [r[0] for r in ranges],
            "low": [r[1] for r in ranges],
        }
    )


def test_corwin_schultz_negative_alpha_pairs_set_to_zero() -> None:
    # Overnight jumps >> daily ranges make every two-day pair's alpha < 0.
    # CS convention sets those pairs' spreads to 0 and KEEPS them in the mean;
    # the old code dropped them and returned NaN here.
    ranges = [(50.25 + 10 * i, 49.75 + 10 * i) for i in range(10)]
    assert est.corwin_schultz_spread(_cs_bars(ranges)) == 0.0


def test_corwin_schultz_zeros_stay_in_panel_mean() -> None:
    n_ranges = [(101.0, 99.0)] * 10  # constant range -> alpha = ln(101/99) > 0
    g_ranges = [(50.25 + 10 * i, 49.75 + 10 * i) for i in range(10)]
    cs_n = est.corwin_schultz_spread(_cs_bars(n_ranges, sec="N"))
    alpha = math.log(101.0 / 99.0)
    assert cs_n == pytest.approx(2.0 * (math.exp(alpha) - 1.0) / (math.exp(alpha) + 1.0))
    both = pl.concat([_cs_bars(n_ranges, sec="N"), _cs_bars(g_ranges)])
    assert est.corwin_schultz_spread(both) == pytest.approx(cs_n / 2.0)


# --- session_vpin ---------------------------------------------------------


def test_session_vpin_missing_column_raises() -> None:
    session = pl.DataFrame(
        {
            "security_id": ["A"],
            "parent_event_time": [_T0],
            "open": [100.0],
            "close": [101.0],
            # no "volume"
        }
    )
    with pytest.raises(ValueError, match="missing columns"):
        est.session_vpin(session)


def test_session_vpin_all_up_days_is_one() -> None:
    session = pl.DataFrame(
        {
            "security_id": ["A"] * 4,
            "parent_event_time": [_T0 + timedelta(days=i) for i in range(4)],
            "open": [100.0] * 4,
            "close": [101.0] * 4,
            "volume": [100.0, 200.0, 50.0, 150.0],
        }
    )
    assert est.session_vpin(session) == pytest.approx(1.0)


# --- attach_candle_book_features: real CKS ofi, not diff-size proxy -------


def _synthetic_bars() -> pl.DataFrame:
    provider = SyntheticMarketProvider(n_assets=2, n_days=16, seed=3)
    return provider.get_bars() if hasattr(provider, "get_bars") else provider._bars


def test_attach_computes_real_ofi_for_toplevel_panels() -> None:
    bars = _synthetic_bars()
    panel = synthesize_l2_from_bars(bars, depth=3, seed=5)
    if "ofi" in panel.columns:
        panel = panel.drop("ofi")
    fused = attach_candle_book_features(bars, book=panel)
    expected = est.order_flow_imbalance(panel).select(
        "security_id", pl.col("event_time").alias("book_event_time"), "ofi"
    )
    joined = fused.select("security_id", "book_event_time", "ofi").join(
        expected, on=["security_id", "book_event_time"], suffix="_exp"
    )
    got = joined["ofi"].to_numpy()
    exp = joined["ofi_exp"].to_numpy()
    mask = np.isfinite(exp)
    assert int(mask.sum()) > 0
    assert np.allclose(got[mask], exp[mask], equal_nan=False)
    assert np.isnan(got[~mask]).all()


def test_attach_respects_book_provided_ofi() -> None:
    bars = _synthetic_bars()
    panel = synthesize_l2_from_bars(bars, depth=3, seed=5)
    panel = panel.with_columns(pl.lit(7.5).alias("ofi"))
    fused = attach_candle_book_features(bars, book=panel)
    assert (fused["ofi"] == 7.5).all()


# --- forward_close_return_labels / sparse-book fused labels ---------------


def test_fwd_label_uses_next_bar_not_next_fused_row() -> None:
    # Book covers every bar EXCEPT day index 2 for A. Under the pre-fix bug the
    # fused frame's shift(-1) gave day1 the day3 return (multi-day "1-bar").
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 6,
            "event_time": [_T0 + timedelta(days=i) for i in range(6)],
            "close": [100.0, 110.0, 121.0, 110.0, 99.0, 108.9],
        }
    )
    labels = forward_close_return_labels(bars)
    # The label is always the next BAR's close return on the full panel,
    # independent of which rows any book join would keep.
    day1 = labels.filter(pl.col("event_time") == _T0 + timedelta(days=1))
    assert day1["fwd_ret_1"][0] == pytest.approx(0.10)
    last = labels.filter(pl.col("event_time") == _T0 + timedelta(days=5))
    assert last["fwd_ret_1"].is_null().all()


def test_fwd_label_value_pinned_through_join() -> None:
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 6,
            "event_time": [_T0 + timedelta(days=i) for i in range(6)],
            "close": [100.0, 110.0, 121.0, 110.0, 99.0, 108.9],
        }
    )
    labels = forward_close_return_labels(bars)
    # Simulate the post-join lookup: fused row for day0 must carry the day1
    # bar return (+10%), not the next *matched* row's.
    fused_times = pl.DataFrame(
        {
            "security_id": ["A"] * 5,
            "event_time": [_T0 + timedelta(days=i) for i in [0, 2, 3, 4, 5]],
        }
    )
    joined = fused_times.join(labels, on=["security_id", "event_time"], how="left")
    got = joined["fwd_ret_1"].to_numpy()
    # fused day0 -> next BAR day1 return (+10%), not next fused day2 (+10% by
    # coincidence of the C2/C0 two-day span under the old buggy shift).
    assert got[0] == pytest.approx(0.10)
    # fused day2 -> next BAR day3 return = 110/121 - 1 = -0.0909.
    assert got[1] == pytest.approx(110.0 / 121.0 - 1.0)


# --- synthesize_l2_from_bars empty ----------------------------------------


def test_synthesize_l2_empty_bars_raise() -> None:
    bars = _synthetic_bars().head(0)
    with pytest.raises(ValueError, match="zero snapshots|non-empty"):
        synthesize_l2_from_bars(bars, depth=3, seed=5)


# --- _attach_event_costs honors sweep_vol_lookback -------------------------


def test_attach_event_costs_uses_config_lookback() -> None:
    cfg = AppConfig()
    cfg.northset.sweep_vol_lookback = 5
    n = 10
    closes = 100.0 * np.cumprod(1.0 + 0.01 * np.array([(-1.0) ** i for i in range(n)]))
    frame = pl.DataFrame(
        {
            "security_id": ["A"] * n,
            "event_time": [_T0 + timedelta(days=i) for i in range(n)],
            "close": closes,
        }
    )
    out = _attach_event_costs(frame, cfg)
    lagged = out["sweep_lagged_vol"].to_numpy()
    # lookback=5 on log returns that start at index 1, shifted by one -> first
    # finite value at index 6. The hardcoded 20 would leave every value null.
    assert np.isnan(lagged[:6]).all()
    assert np.isfinite(lagged[6:]).all()


# --- northset __all__ exports resolvable -----------------------------------


def test_northset_all_exports_resolve() -> None:
    import quant_fund.northset as ns

    assert len(ns.__all__) == len(set(ns.__all__))  # no duplicates
    for name in ns.__all__:
        assert callable(getattr(ns, name)) or isinstance(
            getattr(ns, name), (int, float, str, tuple, list, dict)
        ), name
