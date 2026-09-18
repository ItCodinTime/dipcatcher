"""Trial ledger / multiple-testing extremes (research diagnostic only — not live P&L)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.overfitting import probability_of_backtest_overfitting
from quant_fund.validation.multiple_testing import TrialLedger


def test_trial_ledger_tracks_named_trials_and_dsr() -> None:
    ledger = TrialLedger()
    ledger.record("ridge", 0.4)
    ledger.record("tree", 0.5)
    assert ledger.n_trials == 2
    assert ledger.names == ["ridge", "tree"]
    dsr = ledger.dsr_for(np.array([0.01, -0.005, 0.02, 0.01, -0.002]))
    assert np.isfinite(dsr)
    # More trials → tighter (lower) DSR bound for the same return path
    ledger.record("lasso", 0.45)
    dsr_more = ledger.dsr_for(np.array([0.01, -0.005, 0.02, 0.01, -0.002]))
    assert np.isfinite(dsr_more)
    assert dsr_more <= dsr + 1e-12


def test_trial_ledger_empty_is_honest_nan() -> None:
    """Empty ledger must not invent DSR≈1 via max(n_trials, 1)."""
    ledger = TrialLedger()
    assert ledger.n_trials == 0
    varying = np.array([0.01, -0.005, 0.02, 0.01, -0.002, 0.003, 0.004, -0.001])
    assert math.isnan(ledger.dsr_for(varying))
    assert math.isnan(ledger.dsr_for(np.array([0.01, 0.01])))
    with pytest.raises(ValueError, match="n_trials"):
        from quant_fund.metrics.overfitting import expected_max_sharpe

        expected_max_sharpe(0, 0.0)


def test_trial_ledger_rejects_invalid_trial_records() -> None:
    ledger = TrialLedger()
    with pytest.raises(ValueError, match="trial name"):
        ledger.record(" ", 0.1)
    with pytest.raises(ValueError, match="trial name"):
        ledger.record("", 0.1)
    with pytest.raises(ValueError, match="finite"):
        ledger.record("bad", float("nan"))
    with pytest.raises(ValueError, match="finite"):
        ledger.record("bad_inf", float("inf"))
    with pytest.raises(ValueError, match="finite"):
        ledger.record("bad_ninf", float("-inf"))


def test_trial_ledger_rejects_duplicate_names() -> None:
    ledger = TrialLedger()
    ledger.record("ridge", 0.4)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.record("ridge", 0.5)
    # Whitespace-stripped identity
    with pytest.raises(ValueError, match="duplicate"):
        ledger.record("  ridge  ", 0.6)
    assert ledger.n_trials == 1
    assert ledger.names == ["ridge"]


def test_trial_ledger_dsr_nonfinite_returns_are_nan() -> None:
    ledger = TrialLedger()
    ledger.record("a", 0.3)
    ledger.record("b", 0.4)
    assert math.isnan(ledger.dsr_for(np.array([0.01, np.nan, 0.02, 0.01])))
    assert math.isnan(ledger.dsr_for(np.array([0.01, np.inf, -0.02, 0.01])))
    # Too short / constant after clean → nan
    assert math.isnan(ledger.dsr_for(np.array([0.01])))
    assert math.isnan(ledger.dsr_for(np.array([0.02, 0.02, 0.02, 0.02])))


def test_pbo_nonfinite_paths_are_refused_honestly() -> None:
    """Nonfinite IS/OOS paths invalidate PBO rather than being silently dropped."""
    # Any missing score invalidates the model-selection diagnostic.
    is_s = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], dtype=float)
    oos = np.array([[0.5, np.inf], [0.1, 0.0], [0.4, 0.2]], dtype=float)
    pbo = probability_of_backtest_overfitting(is_s, oos)
    assert math.isnan(pbo)
    oos_all_bad = np.array([[0.5, np.nan], [0.1, np.inf], [-0.2, -np.inf]], dtype=float)
    assert math.isnan(probability_of_backtest_overfitting(is_s, oos_all_bad))
    # An all-missing split also invalidates the entire matrix.
    is_mixed = np.array([[np.nan, np.nan], [0.2, 0.8], [0.1, 0.9]], dtype=float)
    oos_ok = np.array([[0.0, 0.0], [0.5, 0.1], [0.0, 0.4]], dtype=float)
    pbo2 = probability_of_backtest_overfitting(is_mixed, oos_ok)
    assert math.isnan(pbo2)
