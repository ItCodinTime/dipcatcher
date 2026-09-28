"""Sizing NAV at bar t must use only marks knowable at t.

A held name with no execution print on bar t must be valued at the mark
from bars ≤ t−1 (the pre-bar mark). Using that bar's close/CTR is look-ahead:
it sizes orders off a print that has not happened yet. Mutating the same-bar
close of such a name must not change fill quantities on any path
(event loop, numba kernel, interpreted fallback).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_fund.backtest import fast_replay
from quant_fund.backtest.engine import _run_backtest_event_loop, run_backtest
from quant_fund.backtest.fast_replay import run_backtest_fast
from quant_fund.config.loader import load_config

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _cfg(tmp_path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.risk_gate.max_name = 10.0
    cfg.risk_gate.max_net = 10.0
    cfg.risk_gate.max_gross = 10.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_predicted_vol = 1e9
    cfg.risk_gate.stale_price_bars = 3
    return cfg


def _panel(b_exec_close: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """A flat; B flat then no open on the last bar with ``b_exec_close`` close.

    Rebalance on bar 3 targets more A. Sizing that order must not see B's
    same-bar close — only the prior mark of 100.
    """
    bars = pl.DataFrame(
        {
            "security_id": ["A"] * 5 + ["B"] * 5,
            "event_time": [T0 + timedelta(days=i) for i in range(5)] * 2,
            "open": [100.0] * 5 + [100.0, 100.0, 100.0, 100.0, None],
            "close": [100.0] * 5 + [100.0, 100.0, 100.0, 100.0, b_exec_close],
            "close_total_return": [100.0] * 5 + [100.0, 100.0, 100.0, 100.0, b_exec_close],
            "volume": [1_000_000.0] * 10,
            "source": ["synthetic"] * 10,
        }
    )
    weights = pl.DataFrame(
        {
            "event_time": [
                T0,
                T0,
                T0 + timedelta(days=3),
            ],
            "security_id": ["A", "B", "A"],
            "target_weight": [0.5, 0.2, 0.6],
        }
    )
    return bars, weights


def _a_fill_qty(result) -> float:
    fills = result.fills.filter(pl.col("security_id") == "A")
    assert fills.height >= 1
    return float(fills.row(-1, named=True)["quantity"])


@pytest.mark.parametrize("b_close", [100.0, 200.0, 1_000_000.0])
def test_sizing_nav_invariant_to_held_name_same_bar_close(tmp_path, b_close: float) -> None:
    """Causal delta is +2000; a leaked B close of C would size +20*C."""
    cfg = _cfg(tmp_path)
    bars, weights = _panel(b_close)
    # Causal NAV 2e6 → desired A 12000 → delta +2000 from the prior 10000.
    assert _a_fill_qty(run_backtest(bars, weights, cfg, initial_nav=2e6, fast=True)) == pytest.approx(
        2000.0
    )
    assert _a_fill_qty(
        run_backtest(bars, weights, cfg, initial_nav=2e6, fast=False)
    ) == pytest.approx(2000.0)


def test_fast_replay_paths_agree_and_ignore_post_mark_close(tmp_path) -> None:
    """Numba, interpreted fallback, and the event loop all size off pre-bar marks."""
    cfg = _cfg(tmp_path)
    bars_mild, weights = _panel(200.0)
    bars_wild, _ = _panel(9_999_999.0)

    mild_loop = _run_backtest_event_loop(bars_mild, weights, cfg, initial_nav=2e6)
    mild_fast = run_backtest_fast(bars_mild, weights, cfg, initial_nav=2e6)
    wild_fast = run_backtest_fast(bars_wild, weights, cfg, initial_nav=2e6)

    original = fast_replay.HAVE_NUMBA
    fast_replay.HAVE_NUMBA = False
    try:
        mild_interp = run_backtest_fast(bars_mild, weights, cfg, initial_nav=2e6)
        wild_interp = run_backtest_fast(bars_wild, weights, cfg, initial_nav=2e6)
    finally:
        fast_replay.HAVE_NUMBA = original

    expected = 2000.0
    for result in (mild_loop, mild_fast, mild_interp, wild_fast, wild_interp):
        assert _a_fill_qty(result) == pytest.approx(expected)

    # Same fills byte-for-byte across the two close panels: same-bar close
    # of the unprinted held name is not an input to sizing.
    assert mild_fast.fills.equals(wild_fast.fills)
    assert mild_interp.fills.equals(wild_interp.fills)
