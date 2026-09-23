"""Wave 44: PSR/DSR/PBO/expected_max edges — honest NaN; research-diagnostic only.

These helpers are NOT live P&L / promotion Sharpe claims.
PBO / moments empty paths also covered in test_inference_edges.py — this file
locks non-finite Sharpe inputs and n_trials boundaries without duplicating sprawl.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.overfitting import (
    deflated_sharpe,
    expected_max_sharpe,
    min_track_record_length,
    moments_from_returns,
    probabilistic_sharpe,
    probability_of_backtest_overfitting,
)


def test_probabilistic_sharpe_nonfinite_and_short() -> None:
    assert math.isnan(probabilistic_sharpe(float("nan"), 0.0, 100, 0.0, 3.0))
    assert math.isnan(probabilistic_sharpe(float("inf"), 0.0, 100, 0.0, 3.0))
    assert math.isnan(probabilistic_sharpe(1.0, float("nan"), 100, 0.0, 3.0))
    assert math.isnan(probabilistic_sharpe(1.0, 0.0, 100, float("nan"), 3.0))
    assert math.isnan(probabilistic_sharpe(1.0, 0.0, 100, 0.0, float("inf")))
    assert math.isnan(probabilistic_sharpe(1.0, 0.0, float("nan"), 0.0, 3.0))
    assert math.isnan(probabilistic_sharpe(1.0, 0.0, 1, 0.0, 3.0))
    assert math.isnan(probabilistic_sharpe(1.0, 0.0, 0, 0.0, 3.0))
    assert math.isnan(probabilistic_sharpe(1.0, 0.0, -5, 0.0, 3.0))
    # Finite diagnostic path (lab only)
    v = probabilistic_sharpe(0.0, 0.0, 50, 0.0, 3.0)
    assert math.isfinite(v) and 0.0 <= v <= 1.0


def test_expected_max_sharpe_n_trials_and_var() -> None:
    assert expected_max_sharpe(1, 0.04) == 0.0
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe(0, 0.04)
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe(-1, 0.04)
    with pytest.raises(ValueError, match="n_trials"):
        expected_max_sharpe(float("nan"), 0.04)
    assert math.isnan(expected_max_sharpe(10, float("nan")))
    assert math.isnan(expected_max_sharpe(10, float("inf")))
    # Negative var_sr clamped → 0 expected max
    assert expected_max_sharpe(10, -1.0) == 0.0
    assert expected_max_sharpe(5, 0.04) > 0.0


def test_deflated_sharpe_nonfinite_and_bad_trials() -> None:
    assert math.isnan(deflated_sharpe(float("nan"), 252, 0.0, 3.0, 10, 0.04))
    assert math.isnan(deflated_sharpe(1.0, 252, float("nan"), 3.0, 10, 0.04))
    assert math.isnan(deflated_sharpe(1.0, 252, 0.0, 3.0, 10, float("nan")))
    assert math.isnan(deflated_sharpe(1.0, 1, 0.0, 3.0, 10, 0.04))
    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe(1.0, 252, 0.0, 3.0, 0, 0.04)
    d = deflated_sharpe(0.5, 50, 0.0, 3.0, 5, 0.04)
    assert math.isfinite(d) and 0.0 <= d <= 1.0


def test_pbo_and_moments_smoke_edges() -> None:
    # Thin smoke — full PBO/moments edges live in test_inference_edges.py
    assert math.isnan(probability_of_backtest_overfitting(np.array([]), np.array([])))
    assert all(math.isnan(x) for x in moments_from_returns(np.array([])))
    assert all(math.isnan(x) for x in moments_from_returns(np.array([np.nan, np.inf])))


def test_min_track_record_length_edges_honest_nan() -> None:
    """Non-finite / bad conf / sr≈sr* → NaN; research-diagnostic only."""
    assert math.isnan(min_track_record_length(float("nan"), 0.0, 3.0))
    assert math.isnan(min_track_record_length(1.0, float("inf"), 3.0))
    assert math.isnan(min_track_record_length(1.0, 0.0, float("nan")))
    assert math.isnan(min_track_record_length(1.0, 0.0, 3.0, sr_star=float("nan")))
    assert math.isnan(min_track_record_length(1.0, 0.0, 3.0, conf=float("nan")))
    for bad_conf in (0.5, 1.0, 0.0, 0.4, 1.01, -0.1):
        assert math.isnan(min_track_record_length(1.0, 0.0, 3.0, conf=bad_conf))
    # sr ≈ sr_star → infinite track length → honest NaN
    assert math.isnan(min_track_record_length(0.0, 0.0, 3.0, sr_star=0.0))
    assert math.isnan(min_track_record_length(0.5, 0.0, 3.0, sr_star=0.5))
    assert math.isnan(min_track_record_length(1.0, 0.0, 3.0, sr_star=1.0 + 1e-20))


def test_min_track_record_length_gaussian_closed_form() -> None:
    """Gaussian moments (skew=0, kurtosis_raw=3) hand-computed MinTRL.

    Bailey–LdP: MinTRL = 1 + [1 - γ3 SR + (γ4−1)/4 SR²] (z_α / (SR−SR*))²
    With γ3=0, γ4=3: inside = 1 + 0.5 SR².
    conf=0.95 → z = Φ^{-1}(0.95) ≈ 1.6448536269514722.
    """
    from scipy.stats import norm

    z = float(norm.ppf(0.95))
    # Case A: SR=1, SR*=0 → inside=1.5 → MinTRL = 1 + 1.5 * z²
    expected_a = 1.0 + 1.5 * (z / 1.0) ** 2
    got_a = min_track_record_length(1.0, 0.0, 3.0, conf=0.95, sr_star=0.0)
    assert got_a == pytest.approx(expected_a, rel=0, abs=1e-12)
    assert got_a == pytest.approx(5.058315181143118, rel=0, abs=1e-9)

    # Case B: SR=0.5, SR*=0 → inside=1.125 → MinTRL = 1 + 1.125 * (z/0.5)²
    expected_b = 1.0 + 1.125 * (z / 0.5) ** 2
    got_b = min_track_record_length(0.5, 0.0, 3.0, conf=0.95, sr_star=0.0)
    assert got_b == pytest.approx(expected_b, rel=0, abs=1e-12)
    assert got_b == pytest.approx(13.174945543429354, rel=0, abs=1e-9)

    # Negative/zero inside clamp: extreme positive skew can drive inside ≤ 0
    clamped = min_track_record_length(2.0, skew=10.0, kurtosis_raw=3.0, conf=0.95)
    assert math.isfinite(clamped) and clamped >= 1.0
    # inside clamped to 1e-18 → MinTRL ≈ 1 + ε (z/diff)²
