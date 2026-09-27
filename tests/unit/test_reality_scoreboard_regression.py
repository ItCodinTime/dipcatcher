"""A1 F1 regression: scoreboard/ledger PSR-DSR must not be z-inflated (~15.9x).

Before the fix, `book_economic_scoreboard` and `TrialLedger.dsr_for` fed the
ANNUALIZED Sharpe into `probabilistic_sharpe` / `deflated_sharpe` with a
per-day n, inflating z by sqrt(252) ~ 15.9x. These tests pin the corrected
per-period convention with exact-mean constructions (seed-robust).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.hedge_lab.scoreboard import book_economic_scoreboard
from quant_fund.metrics.overfitting import moments_from_returns
from quant_fund.validation.multiple_testing import TrialLedger


def _exact_mean_series(mu: float, sig: float, n: int, seed: int) -> np.ndarray:
    """iid-shaped series with EXACTLY sample mean mu (seed-robust pins)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, sig, size=n)
    return r - float(np.mean(r)) + mu


def _closed_form_psr(r: np.ndarray) -> float:
    sr_per = float(np.mean(r) / np.std(r, ddof=1))
    _, skew, kurt = moments_from_returns(r)
    se = float(np.sqrt(1.0 - skew * sr_per + ((kurt - 1.0) / 4.0) * sr_per**2))
    return float(norm.cdf(sr_per * np.sqrt(r.size - 1) / se))


def test_zero_edge_iid_book_psr_exactly_half() -> None:
    """Zero-edge daily book, n=252: psr_vs_zero == 0.5 exactly (NOT ~1.0)."""
    r = _exact_mean_series(0.0, 0.01, 252, seed=7)
    blob = book_economic_scoreboard(r, periods_per_year=252.0)
    assert blob["psr_vs_zero"] == pytest.approx(0.5, abs=1e-9)


def test_marginal_book_psr_not_short_circuited() -> None:
    """SR_ann ~ 0.55 daily book: per-period PSR ~ Phi(0.55*sqrt(251)/sqrt(252))
    ~ 0.71 — the old annualized-SR bug returned Phi(0.55*sqrt(251)) ~ 1.0."""
    r = _exact_mean_series(0.0347 * 0.01, 0.01, 252, seed=21)
    blob = book_economic_scoreboard(r, periods_per_year=252.0)
    expected = _closed_form_psr(r)
    assert blob["psr_vs_zero"] == pytest.approx(expected, abs=1e-12)
    assert 0.5 < blob["psr_vs_zero"] < 0.95
    # display Sharpe stays annualized (unchanged behavior)
    assert blob["sharpe"] == pytest.approx(
        float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252.0)), rel=1e-9
    )


def test_min_trl_in_periods_not_years_scale() -> None:
    """Strong daily book: MinTRL must be a period count (tens-hundreds), not
    the ~252x-understated figure the annualized-SR bug produced."""
    r = _exact_mean_series(0.003, 0.01, 500, seed=31)
    blob = book_economic_scoreboard(r, periods_per_year=252.0)
    mtrl = blob["min_trl_days"]
    sr_per = float(np.mean(r) / np.std(r, ddof=1))
    _, skew, kurt = moments_from_returns(r)
    z95 = float(norm.ppf(0.95))
    inside = 1.0 - skew * sr_per + ((kurt - 1.0) / 4.0) * sr_per**2
    expected = 1.0 + max(inside, 1e-12) * (z95 / sr_per) ** 2
    assert mtrl == pytest.approx(expected, rel=1e-9)
    assert mtrl > 10.0  # periods, not the old broken sub-period values


def test_trial_ledger_dsr_for_uses_period_units() -> None:
    """Second F1-class site: validation/multiple_testing.py dsr_for must not
    feed an annualized SR with a per-period n into deflated_sharpe (same
    ~15.9x inflation). Single-trial ledger -> DSR == PSR vs 0 closed form."""
    r = _exact_mean_series(0.0347 * 0.01, 0.01, 252, seed=41)
    ledger = TrialLedger()
    ledger.record("t1", 0.0)
    dsr = ledger.dsr_for(r)
    assert dsr == pytest.approx(_closed_form_psr(r), abs=1e-12)
    assert 0.5 < dsr < 0.95  # old bug: ~1.0
