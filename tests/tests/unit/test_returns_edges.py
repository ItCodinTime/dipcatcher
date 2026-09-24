"""Wave 23/41: returns metrics empty/NaN/short/constant edges — honest NaN.

Sharpe / Sortino / Calmar here are research-diagnostic only.
They are NOT live P&L claims and must never be treated as promotion edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.returns import (
    annualized_vol,
    cagr,
    calmar_ratio,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
    wealth_index,
)


def test_wealth_index_empty() -> None:
    out = wealth_index(np.array([]))
    assert out.size == 0
    assert out.dtype == float


def test_wealth_index_nan_propagates() -> None:
    out = wealth_index(np.array([0.01, np.nan, 0.02]))
    assert np.isclose(out[0], 1.01)
    assert np.isnan(out[1])
    assert np.isnan(out[2])


def test_drawdown_empty_and_nan() -> None:
    assert drawdown_series(np.array([])).size == 0
    assert max_drawdown(np.array([])) == 0.0
    dd = drawdown_series(np.array([np.nan, np.nan]))
    assert dd.size == 2
    assert np.isnan(max_drawdown(np.array([np.nan, np.nan])))
    # A later invalid observation invalidates the whole path; do not report a
    # truncated maximum drawdown from only the finite prefix.
    assert np.isnan(max_drawdown(np.array([0.1, -0.5, np.nan])))


def test_drawdown_includes_initial_capital_before_first_loss() -> None:
    returns = np.array([-0.2, 0.1, 0.0])
    assert np.allclose(drawdown_series(returns), [-0.2, -0.12, -0.12])
    assert max_drawdown(returns) == pytest.approx(-0.2)
    assert max_drawdown(np.array([-0.2])) == pytest.approx(-0.2)


def test_cagr_empty_nan_nonfinite() -> None:
    assert np.isnan(cagr(np.array([])))
    assert np.isnan(cagr(np.array([np.nan])))
    # All losses wipe wealth ≤ 0 → NaN
    assert np.isnan(cagr(np.array([-1.0])))


def test_annualized_vol_empty_short_nan() -> None:
    assert np.isnan(annualized_vol(np.array([])))
    assert np.isnan(annualized_vol(np.array([0.01])))
    assert np.isnan(annualized_vol(np.array([0.01, np.nan, 0.02])))
    finite = annualized_vol(np.array([0.01, -0.01, 0.02, -0.02]))
    assert np.isfinite(finite) and finite > 0.0


def test_turnover_empty_mismatch_nan() -> None:
    assert turnover(np.array([]), np.array([])) == 0.0
    with pytest.raises(ValueError, match="mismatch"):
        turnover(np.array([0.1, 0.2]), np.array([0.1]))
    assert np.isnan(turnover(np.array([0.1, np.nan]), np.array([0.0, 0.0])))
    assert np.isclose(turnover(np.array([0.5, -0.2]), np.array([0.0, 0.0])), 0.7)


# --- Wave 41: ratio edges (research diagnostic only; not live Sharpe) ---


def test_sharpe_ratio_empty_constant_nan() -> None:
    """Empty / constant / non-finite → NaN; diagnostic flag stays False."""
    empty = sharpe_ratio(np.array([]))
    assert empty["n"] == 0
    assert np.isnan(empty["sharpe"])
    assert empty["flag_high_sharpe"] is False

    short = sharpe_ratio(np.array([0.01]))
    assert short["n"] == 1
    assert np.isnan(short["sharpe"])

    constant = sharpe_ratio(np.array([0.01, 0.01, 0.01, 0.01]))
    assert np.isnan(constant["sharpe"])
    assert constant["flag_high_sharpe"] is False

    with_nan = sharpe_ratio(np.array([0.01, np.nan, -0.01]))
    assert np.isnan(with_nan["sharpe"])
    assert with_nan["flag_high_sharpe"] is False

    # Finite varying path yields a finite diagnostic number (lab only).
    # Noisy path keeps |SR| ≤ 5 so flag_high_sharpe stays False (lab sanity).
    rng = np.random.default_rng(42)
    noisy = 0.001 + 0.02 * rng.normal(size=60)
    ok = sharpe_ratio(noisy)
    assert np.isfinite(ok["sharpe"])
    assert ok["flag_high_sharpe"] is False


def test_sortino_ratio_empty_constant_nan() -> None:
    """Empty / zero-downside / non-finite → NaN (research diagnostic only)."""
    assert np.isnan(sortino_ratio(np.array([])))
    assert np.isnan(sortino_ratio(np.array([0.01])))
    # All returns ≥ MAR → zero downside deviation → NaN
    assert np.isnan(sortino_ratio(np.array([0.01, 0.02, 0.0, 0.03]), mar=0.0))
    assert np.isnan(sortino_ratio(np.array([0.01, np.nan, -0.01])))
    mixed = sortino_ratio(np.array([0.02, -0.01, 0.03, -0.02]), mar=0.0)
    assert np.isfinite(mixed)


def test_calmar_ratio_empty_constant_nan() -> None:
    """Empty / zero-drawdown / non-finite → NaN (research diagnostic only)."""
    assert np.isnan(calmar_ratio(np.array([])))
    # Monotone wealth → mdd == 0 → NaN (no invented Calmar)
    assert np.isnan(calmar_ratio(np.array([0.01, 0.02, 0.0, 0.03])))
    assert np.isnan(calmar_ratio(np.array([0.01, np.nan, -0.02])))
    # Path with a drawdown yields a finite diagnostic Calmar
    with_dd = calmar_ratio(np.array([0.1, -0.5, 0.2, 0.1]), periods_per_year=4.0)
    assert np.isfinite(with_dd)
