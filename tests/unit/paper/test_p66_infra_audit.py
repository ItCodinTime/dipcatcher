"""P6.6 infra audit KATs — small deterministic checks on fixed findings.

Each test names the bug it regresses:
  * sim_live funding window was forward-looking (look-ahead into fund_cut)
  * sim_live _equity_stats annualized a negative NAV ratio into a complex number
  * recon.reconcile_broker_states never compared last_marks *values*
  * recon.reconcile_equity last_nav_delta was join-order, not chronological
  * ledger.promotion_dry_run let max_l1=NaN emit would_promote_paper=True
  * loop borrow accrual used /252 per bar regardless of bar spacing
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.paper.ledger import promotion_dry_run, validate_promotion_dry_run_receipt
from quant_fund.paper.loop import run_paper_loop
from quant_fund.paper.recon import reconcile_broker_states, reconcile_equity
from quant_fund.paper.sim_live import _bars_per_year, _equity_stats, _trailing_window_sum

T0 = datetime(2024, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# sim_live funding window — was forward-looking (roll[i] summed i..i+n-1)
# ---------------------------------------------------------------------------


def test_trailing_window_sum_is_causal_kat() -> None:
    per_bar = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    roll = _trailing_window_sum(per_bar, 3)
    # Trailing: out[i] = sum(per_bar[max(0,i-2)..i]).
    np.testing.assert_allclose(roll, [1.0, 3.0, 6.0, 9.0, 12.0])


def test_trailing_window_sum_never_reads_future() -> None:
    rng = np.random.default_rng(0)
    per_bar = rng.normal(size=50)
    roll = _trailing_window_sum(per_bar, 7)
    # Zero the future tail; outputs at earlier indices must be untouched.
    clipped = per_bar.copy()
    clipped[10:] = 0.0
    roll_clipped = _trailing_window_sum(clipped, 7)
    np.testing.assert_allclose(roll[:4], roll_clipped[:4])
    assert roll[9] == pytest.approx(float(per_bar[3:10].sum()))
    # roll_n > len is legal (shrinking window), roll_n < 1 is not.
    np.testing.assert_allclose(_trailing_window_sum(per_bar[:3], 99), np.cumsum(per_bar[:3]))
    with pytest.raises(ValueError):
        _trailing_window_sum(per_bar, 0)


def test_funding_roll_n_derived_from_interval() -> None:
    # 7 calendar days in bar units: 1d->7, 4h->42, 1h->168.
    for interval, expected in (("1d", 7), ("4h", 42), ("1h", 168)):
        bars = _bars_per_year(interval)
        assert max(1, int(round(7.0 * bars / 365.25))) == expected


def test_equity_stats_negative_nav_ratio_is_nan_not_complex() -> None:
    eq = pl.DataFrame({"nav": [100.0, -50.0, -25.0]})
    stats = _equity_stats(eq, 365.25)
    assert stats["status"] == "ok"
    assert math.isnan(stats["ann_return"])
    assert stats["total_return"] == pytest.approx(-1.25)


# ---------------------------------------------------------------------------
# recon — last_marks values compared, equity ordering/null-nav semantics
# ---------------------------------------------------------------------------


def _broker_state(**over):
    state = {
        "cash": 1000.0,
        "shares": {"A": 10.0},
        "last_marks": {"A": 100.0},
        "reject_count": 1,
        "halt_count": 0,
        "risk_gate_reject_count": 0,
        "n_orders": 5,
        "n_fills": 2,
        "kill_state": "ENABLED",
        "slot": "champion",
    }
    state.update(over)
    return state


def test_broker_state_mark_value_drift_is_caught() -> None:
    exp = _broker_state()
    drifted = _broker_state(last_marks={"A": 101.0})
    rep = reconcile_broker_states(exp, drifted)
    assert rep["match"] is False
    assert any("last_marks[A]" in m for m in rep["mismatches"])
    # Same key set, equal values still matches.
    assert reconcile_broker_states(exp, dict(exp))["match"] is True


def test_broker_state_mark_nan_is_flagged() -> None:
    exp = _broker_state()
    act = _broker_state(last_marks={"A": float("nan")})
    rep = reconcile_broker_states(exp, act)
    assert rep["match"] is False
    assert any("last_marks[A]" in m for m in rep["mismatches"])


def test_reconcile_equity_last_delta_is_chronological() -> None:
    times = [T0 + timedelta(days=d) for d in range(4)]
    expected = pl.DataFrame({"event_time": times, "nav": [100.0, 101.0, 102.0, 103.0]})
    # Same timestamps, shuffled row order; delta at t_i is expected+ (2i+1):
    # deltas = [1, 3, 5, 7] — the chronological last delta is 7, which an
    # unordered join would not reliably emit last.
    order = [2, 0, 3, 1]
    actual = pl.DataFrame(
        {
            "event_time": [times[i] for i in order],
            "nav": [(100.0 + i) + (2 * i + 1) for i in order],
        }
    )
    rep = reconcile_equity(expected, actual)
    assert rep["n_matched"] == 4
    assert rep["last_nav_delta"] == pytest.approx(7.0)
    assert rep["max_abs_nav_delta"] == pytest.approx(7.0)


def test_reconcile_equity_null_nav_not_silently_unmatched() -> None:
    times = [T0 + timedelta(days=d) for d in range(3)]
    expected = pl.DataFrame({"event_time": times, "nav": [1.0, 2.0, 3.0]})
    actual = pl.DataFrame({"event_time": times, "nav": [1.0, None, 3.0]})
    rep = reconcile_equity(expected, actual)
    assert rep["match"] is False
    assert rep["matched_null_nav"] == 1
    assert rep["unmatched_expected"] == 0 and rep["unmatched_actual"] == 0


# ---------------------------------------------------------------------------
# ledger.promotion_dry_run — max_l1 NaN must gate would_promote_paper
# ---------------------------------------------------------------------------


def _promo_kwargs(**over):
    kw = {
        "mean_l1": 0.05,
        "max_l1": 0.10,
        "n_steps": 100,
        "champion_nav": 1_000_000.0,
        "shadow_gross": 0.5,
        "max_mean_l1": 0.25,
        "min_steps": 10,
        "data_source": "REAL",
        "allow_missing_divergence": True,
    }
    kw.update(over)
    return kw


def test_promotion_dry_run_max_l1_nan_gates_promotion() -> None:
    receipt = promotion_dry_run(**_promo_kwargs(max_l1=float("nan")))
    assert receipt["would_promote_paper"] is False
    assert "missing_divergence" in receipt["reasons"]
    # The emitted receipt must pass its own validator (self-consistency).
    assert validate_promotion_dry_run_receipt(receipt) == []


def test_promotion_dry_run_mean_l1_nan_still_gates() -> None:
    receipt = promotion_dry_run(**_promo_kwargs(mean_l1=float("nan")))
    assert receipt["would_promote_paper"] is False
    assert "missing_divergence" in receipt["reasons"]


def test_promotion_dry_run_nan_rejected_without_allowance() -> None:
    with pytest.raises(ValueError):
        promotion_dry_run(**_promo_kwargs(max_l1=float("nan"), allow_missing_divergence=False))


# ---------------------------------------------------------------------------
# loop borrow accrual — elapsed wall-time, not /252 per bar
# ---------------------------------------------------------------------------


def _borrow_cfg(tmp_path: Path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 4.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.frictionless = False
    cfg.costs.commission_bps = 0.0
    cfg.costs.half_spread_bps = 0.0
    cfg.costs.impact_y = 0.0
    cfg.costs.borrow_bps_per_year = 10000.0  # 100%/yr so accrual is easy to read
    return cfg


def _intraday_bars(hours_per_bar: int, n: int, price: float = 100.0) -> pl.DataFrame:
    rows = []
    for d in range(n):
        t = T0 + timedelta(hours=hours_per_bar * d)
        rows.append(
            {
                "security_id": "A",
                "event_time": t,
                "open": price,
                "close": price,
                "close_total_return": price,
                "volume": 1e9,
                "adv": 1e12,
                "vol_20": 0.01,
                "source": "synthetic",
            }
        )
    return pl.DataFrame(rows)


def _short_weights(n: int, w: float = -0.5) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [T0 + timedelta(hours=4 * d) for d in range(n)],
            "security_id": "A",
            "target_weight": w,
        }
    )


def test_borrow_accrues_by_elapsed_time_not_fixed_daily(tmp_path: Path) -> None:
    """4h-spaced bars: borrow for a held -0.5 short must scale with bar gap.

    Old code charged /252 per bar (~1 day of borrow every 4h). With elapsed
    accrual each 4h bar charges 4h of borrow:  notional * apr * (4h/yr).
    """
    n = 6
    cfg = _borrow_cfg(tmp_path)
    result = run_paper_loop(
        _intraday_bars(4, n),
        cfg,
        champion_weights=_short_weights(n),
        initial_nav=1_000_000.0,
        run_id="borrow-4h",
        prefer_latest=False,
    )
    cash_events = pl.read_parquet(result.paths["cash_ledger"])
    borrows = cash_events.filter(pl.col("side") == "borrow")
    assert borrows.height >= 1
    expected_per_bar = 0.5 * 1_000_000.0 * (1.0) * (4 * 3600.0 / (365.0 * 86400.0))
    for fee in borrows["fees"].to_list():
        assert fee == pytest.approx(expected_per_bar, rel=1e-3)


def test_borrow_zero_on_first_bar_and_gaps_scale(tmp_path: Path) -> None:
    """A single-step run has no prior exec bar — zero elapsed, zero borrow."""
    cfg = _borrow_cfg(tmp_path)
    result = run_paper_loop(
        _intraday_bars(4, 3),
        cfg,
        champion_weights=_short_weights(3),
        initial_nav=1_000_000.0,
        run_id="borrow-first",
        max_steps=1,
        prefer_latest=False,
    )
    cash_events = pl.read_parquet(result.paths["cash_ledger"])
    assert cash_events.filter(pl.col("side") == "borrow").height == 0
