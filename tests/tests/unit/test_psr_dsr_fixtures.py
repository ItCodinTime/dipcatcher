"""Wave 8: Bailey–López de Prado PSR/DSR closed-form / boundary fixtures."""

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.metrics.overfitting import (
    _sr_se,
    deflated_sharpe,
    expected_max_sharpe,
    probabilistic_sharpe,
)


def test_psr_sr_equals_sr_star_is_half() -> None:
    """Under normal SE, PSR(sr=sr*) = Φ(0) = 0.5."""
    psr = probabilistic_sharpe(sr=1.0, sr_star=1.0, n_obs=252, skew=0.0, kurtosis_raw=3.0)
    assert psr == pytest.approx(0.5, abs=1e-12)


def test_psr_closed_form_normal() -> None:
    # Keep the z-score away from floating-point CDF saturation at exactly 1.
    sr, sr_star, n, skew, kurt = 0.5, 0.0, 50, 0.0, 3.0
    se = _sr_se(sr, skew, kurt)
    expected = float(norm.cdf((sr - sr_star) * np.sqrt(n - 1) / se))
    got = probabilistic_sharpe(sr, sr_star, n, skew, kurt)
    assert got == pytest.approx(expected, abs=1e-12)
    assert 0.0 < got < 1.0


def test_psr_n_obs_boundary() -> None:
    assert np.isnan(probabilistic_sharpe(1.0, 0.0, 1, 0.0, 3.0))
    assert np.isnan(probabilistic_sharpe(1.0, 0.0, 0, 0.0, 3.0))
    # n_obs=2 is the first finite case
    v = probabilistic_sharpe(0.0, 0.0, 2, 0.0, 3.0)
    assert np.isfinite(v)
    assert v == pytest.approx(0.5, abs=1e-9)


def test_expected_max_sharpe_boundaries() -> None:
    assert expected_max_sharpe(1, 0.04) == 0.0
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 0.04)
    # Monotone in n_trials (more trials → higher expected max)
    e2 = expected_max_sharpe(2, 0.04)
    e10 = expected_max_sharpe(10, 0.04)
    e100 = expected_max_sharpe(100, 0.04)
    assert e2 < e10 < e100
    assert e2 > 0.0


def test_dsr_is_psr_with_sr_star_from_n_trials() -> None:
    sr, n_obs, skew, kurt, n_trials, var_sr = 2.0, 252, 0.0, 3.0, 50, 0.04
    sr_star = expected_max_sharpe(n_trials, var_sr)
    dsr = deflated_sharpe(sr, n_obs, skew, kurt, n_trials, var_sr)
    psr = probabilistic_sharpe(sr, sr_star, n_obs, skew, kurt)
    assert dsr == pytest.approx(psr, abs=1e-12)
    assert 0.0 <= dsr <= 1.0


def test_dsr_more_trials_deflates() -> None:
    """Holding SR fixed, more trials → higher sr* → lower DSR."""
    # Use a moderate z-score so the strict ordering remains representable in float64.
    kwargs = dict(sr=0.5, n_obs=50, skew=0.0, kurtosis_raw=3.0, var_sr=0.04)
    d_few = deflated_sharpe(n_trials=5, **kwargs)
    d_many = deflated_sharpe(n_trials=500, **kwargs)
    assert d_many < d_few
