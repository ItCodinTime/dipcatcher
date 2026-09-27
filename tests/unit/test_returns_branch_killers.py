"""Example tests aimed at surviving mutants in ``metrics/returns.py``.

Sharpe, Sortino, and Calmar values here are research diagnostics. They are
not live P&L claims.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from quant_fund.metrics.returns import (
    annualized_vol,
    cagr,
    calmar_ratio,
    downside_deviation,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    turnover,
    wealth_index,
)

_PERIODS = "periods_per_year must be finite and positive"


def _exact(message: str) -> str:
    return rf"^{re.escape(message)}$"


def test_wealth_and_drawdown_match_the_compound_path() -> None:
    returns = np.array([0.1, -0.5, 0.2])
    wealth = wealth_index(returns)
    assert wealth.tolist() == pytest.approx([1.1, 0.55, 0.66])
    # Peak includes the unit of starting capital, so the first loss is a drawdown.
    assert drawdown_series(returns).tolist() == pytest.approx(
        [0.0, 0.55 / 1.1 - 1.0, 0.66 / 1.1 - 1.0]
    )
    assert max_drawdown(returns) == pytest.approx(0.55 / 1.1 - 1.0)
    assert wealth_index(np.array([])).size == 0
    assert drawdown_series(np.array([])).size == 0
    assert max_drawdown(np.array([])) == 0.0
    assert math.isnan(max_drawdown(np.array([0.1, math.nan])))


def test_cagr_closed_form_and_invalid_periods() -> None:
    flat = np.array([0.0, 0.0])
    assert cagr(flat, periods_per_year=2.0) == 0.0
    one = np.array([0.1])
    assert cagr(one, periods_per_year=1.0) == pytest.approx(0.1)
    assert math.isnan(cagr(np.array([])))
    assert math.isnan(cagr(np.array([-1.0])))
    assert math.isnan(cagr(np.array([math.nan])))
    with pytest.raises(ValueError, match=_exact(_PERIODS)):
        cagr(np.array([0.01, 0.02]), periods_per_year=0.0)
    with pytest.raises(ValueError, match=_exact(_PERIODS)):
        cagr(np.array([0.01]), periods_per_year=float("nan"))
    # Two years, so multiplying by years is not the same as dividing by years.
    assert cagr(np.array([0.0, 0.21]), periods_per_year=1.0) == pytest.approx(0.1)


def test_annualized_vol_uses_sample_std() -> None:
    returns = np.array([0.01, 0.03])
    expected = float(np.std(returns, ddof=1) * math.sqrt(4.0))
    assert annualized_vol(returns, periods_per_year=4.0) == expected
    assert math.isnan(annualized_vol(np.array([0.01])))
    assert math.isnan(annualized_vol(np.array([0.01, math.nan])))
    with pytest.raises(ValueError, match=_exact(_PERIODS)):
        annualized_vol(np.array([0.01, 0.02]), periods_per_year=-1.0)


def test_downside_deviation_ignores_gains_above_mar() -> None:
    returns = np.array([0.05, -0.02, 0.01, -0.04])
    mar = 0.01
    shortfall = np.minimum(returns - mar, 0.0)
    expected = float(math.sqrt(np.mean(shortfall**2)) * math.sqrt(4.0))
    assert downside_deviation(returns, mar=mar, periods_per_year=4.0) == expected
    assert math.isnan(downside_deviation(np.array([0.01])))
    assert downside_deviation(np.array([0.02, 0.03]), mar=0.0, periods_per_year=1.0) == 0.0
    with pytest.raises(ValueError, match=_exact("mar must be finite")):
        downside_deviation(np.array([0.01, -0.01]), mar=float("nan"))


def test_sharpe_closed_form_boundary_and_flag() -> None:
    returns = np.array([0.02, -0.01, 0.03, 0.0])
    rf = 0.01
    excess = returns - rf
    periodic = float(np.mean(excess) / np.std(excess, ddof=1))
    out = sharpe_ratio(returns, rf_per_period=rf, periods_per_year=16.0)
    assert out["n"] == 4
    assert out["periods_per_year"] == 16.0
    assert out["annualized"] is True
    assert out["sharpe"] == periodic * math.sqrt(16.0)
    irregular = sharpe_ratio(returns, rf_per_period=rf, periods_per_year=16.0, irregular=True)
    assert irregular["annualized"] is False
    assert irregular["sharpe"] == periodic
    # Strictly above 5 flags; the boundary at 5 does not.
    gap = 2.0
    at_five = 5.0 * math.sqrt(2.0) - gap / 2.0
    above = 6.0 * math.sqrt(2.0) - gap / 2.0
    below = 4.0 * math.sqrt(2.0) - gap / 2.0
    assert (
        sharpe_ratio(np.array([at_five, at_five + gap]), periods_per_year=1.0)["flag_high_sharpe"]
        is False
    )
    assert (
        sharpe_ratio(np.array([above, above + gap]), periods_per_year=1.0)["flag_high_sharpe"]
        is True
    )
    assert (
        sharpe_ratio(np.array([below, below + gap]), periods_per_year=1.0)["flag_high_sharpe"]
        is False
    )
    assert math.isnan(sharpe_ratio(np.array([0.01, 0.01]))["sharpe"])
    assert sharpe_ratio(np.array([0.01]))["n"] == 1
    with pytest.raises(ValueError, match=_exact("rf_per_period must be finite")):
        sharpe_ratio(np.array([0.01, 0.02]), rf_per_period=float("inf"))
    with pytest.raises(ValueError, match=_exact(_PERIODS)):
        sharpe_ratio(np.array([0.01, 0.02]), periods_per_year=-2.0)


def test_sortino_and_calmar_closed_forms() -> None:
    returns = np.array([0.04, -0.02, 0.01, -0.03])
    mar = 0.0
    downside = downside_deviation(returns, mar=mar, periods_per_year=1.0)
    expected_sortino = float((np.mean(returns) - mar) / downside * math.sqrt(9.0))
    assert sortino_ratio(returns, mar=mar, periods_per_year=9.0) == expected_sortino
    assert math.isnan(sortino_ratio(np.array([0.01, 0.02])))
    assert math.isnan(sortino_ratio(np.array([0.01, math.nan])))
    with pytest.raises(ValueError, match=_exact("mar must be finite")):
        sortino_ratio(np.array([0.01, -0.02]), mar=float("nan"))
    growth = cagr(returns, periods_per_year=4.0)
    mdd = max_drawdown(returns)
    assert calmar_ratio(returns, periods_per_year=4.0) == growth / abs(mdd)
    assert math.isnan(calmar_ratio(np.array([0.01, 0.02, 0.0])))
    assert math.isnan(calmar_ratio(np.array([])))
    with pytest.raises(ValueError, match=_exact(_PERIODS)):
        sortino_ratio(np.array([0.01, -0.02]), periods_per_year=0.0)
    # One losing period is a defined Calmar; a wiped-out path has no finite CAGR.
    assert calmar_ratio(np.array([-0.1]), periods_per_year=1.0) == pytest.approx(-1.0)
    assert math.isnan(calmar_ratio(np.array([-0.5, -1.0])))


def test_defaults_match_252_and_columns_are_flattened() -> None:
    """Default periods and reshape(-1) are part of the numeric contract."""
    column = np.array([[0.1], [-0.2], [0.05]])
    flat = column.reshape(-1)
    assert wealth_index(column).shape == (3,)
    assert wealth_index(column).tolist() == pytest.approx(wealth_index(flat).tolist())
    assert wealth_index(np.array([1, -1], dtype=int)).dtype == np.float64
    path = np.array([0.01, -0.02, 0.015, 0.0])
    assert cagr(path) == cagr(path, periods_per_year=252.0)
    assert annualized_vol(path) == annualized_vol(path, periods_per_year=252.0)
    assert downside_deviation(path) == downside_deviation(path, periods_per_year=252.0)
    assert sortino_ratio(path) == sortino_ratio(path, periods_per_year=252.0)
    assert calmar_ratio(path) == calmar_ratio(path, periods_per_year=252.0)
    assert sharpe_ratio(path)["sharpe"] == sharpe_ratio(path, periods_per_year=252.0)["sharpe"]
    assert sharpe_ratio(path)["annualized"] is True
    # Three observations: the last wealth point is not index 1.
    stepped = np.array([0.0, 0.0, 0.1])
    assert cagr(stepped, periods_per_year=3.0) == pytest.approx(0.1)


def test_short_paths_and_nonzero_mar_keep_their_closed_forms() -> None:
    """Length-2 samples must be scored, and Sortino's mar is not optional."""
    pair = np.array([0.02, -0.02])
    vol = float(np.std(pair, ddof=1) * math.sqrt(1.0))
    assert annualized_vol(pair, periods_per_year=1.0) == vol
    shortfall = np.minimum(pair, 0.0)
    assert downside_deviation(pair, periods_per_year=1.0) == float(math.sqrt(np.mean(shortfall**2)))
    mar = 0.01
    downside = downside_deviation(pair, mar=mar, periods_per_year=1.0)
    assert sortino_ratio(pair, mar=mar, periods_per_year=4.0) == float(
        (np.mean(pair) - mar) / downside * math.sqrt(4.0)
    )
    assert sortino_ratio(pair) == sortino_ratio(pair, mar=0.0)
    excess = pair - 0.0
    periodic = float(np.mean(excess) / np.std(excess, ddof=1))
    short = sharpe_ratio(pair, periods_per_year=4.0)
    assert short["sharpe"] == periodic * math.sqrt(4.0)
    assert math.isnan(calmar_ratio(np.array([0.01, math.nan])))
    # A column and a flat vector of the same weights are one turnover.
    assert turnover(np.array([[0.2], [-0.1]]), np.array([0.0, 0.4])) == pytest.approx(0.7)
    assert turnover(np.array([0.2, -0.1]), np.array([[0.0], [0.4]])) == pytest.approx(0.7)


def test_decimal_strings_are_cast_to_float_before_the_math() -> None:
    """dtype=float is load-bearing: decimal text is not left as a string array."""
    as_float = np.array([0.25, -0.25])
    as_text = ["0.25", "-0.25"]
    assert wealth_index(as_text).tolist() == pytest.approx(wealth_index(as_float).tolist())
    assert annualized_vol(as_text, periods_per_year=1.0) == annualized_vol(
        as_float, periods_per_year=1.0
    )
    assert downside_deviation(as_text, periods_per_year=1.0) == downside_deviation(
        as_float, periods_per_year=1.0
    )
    assert (
        sharpe_ratio(as_text, periods_per_year=1.0)["sharpe"]
        == sharpe_ratio(as_float, periods_per_year=1.0)["sharpe"]
    )
    assert sortino_ratio(as_text, periods_per_year=1.0) == sortino_ratio(
        as_float, periods_per_year=1.0
    )
    assert calmar_ratio(as_text, periods_per_year=1.0) == calmar_ratio(
        as_float, periods_per_year=1.0
    )
    assert turnover(["0.2", "-0.1"], ["0.0", "0.4"]) == pytest.approx(0.7)


def test_turnover_is_the_l1_distance() -> None:
    assert turnover(np.array([0.2, -0.1]), np.array([0.0, 0.4])) == pytest.approx(0.7)
    assert turnover(np.array([0.2]), np.array([-0.1])) == pytest.approx(0.3)
    assert turnover(np.array([]), np.array([])) == 0.0
    assert math.isnan(turnover(np.array([0.1, math.inf]), np.array([0.0, 0.0])))
    assert math.isnan(turnover(np.array([0.1, 0.0]), np.array([0.0, math.nan])))
    with pytest.raises(ValueError, match=r"turnover weight length mismatch: 2 vs 1"):
        turnover(np.array([0.1, 0.2]), np.array([0.1]))
