"""Canon tests: polyhedral post-selection intervals."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.selective_inference import (
    _tg_cdf,
    _truncation_bounds,
    polyhedral_interval,
    sign_conditioned_interval,
)


def test_tg_cdf_standard() -> None:
    assert _tg_cdf(0.0, 0.0, 1.0, -1.0, 1.0) == pytest.approx(0.5)
    assert _tg_cdf(-5.0, 0.0, 1.0, -1.0, 1.0) == 0.0
    assert _tg_cdf(5.0, 0.0, 1.0, -1.0, 1.0) == 1.0
    assert np.isnan(_tg_cdf(0.0, 0.0, 1.0, 2.0, 1.0))  # degenerate truncation


def test_truncation_bounds_sign_event() -> None:
    y = np.array([2.0, 0.5])
    eta = np.array([1.0, 0.0])
    a = -eta[None, :]  # selection: eta^T y >= 0
    v_lo, v_hi, t_hat = _truncation_bounds(a, np.zeros(1), y, eta, 1.0)
    assert v_lo == pytest.approx(0.0)
    assert v_hi == np.inf
    assert t_hat == pytest.approx(2.0)


def test_unconditional_interval_matches_z() -> None:
    # slack constraint on an orthogonal coordinate -> bounds ~ (-inf, inf)
    # in eta^T y units, so the TG interval collapses to the z-interval.
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, 10)
    eta = np.zeros(10)
    eta[0] = 1.0
    a = np.zeros((1, 10))
    a[0, 1] = -1.0  # -y_1 <= big
    b = np.array([1e9])
    out = polyhedral_interval(y, a, b, eta, sigma=1.0, alpha=0.1)
    z = 1.6448536269514722
    assert out["ci_lo"] == pytest.approx(y[0] - z, abs=0.05)
    assert out["ci_hi"] == pytest.approx(y[0] + z, abs=0.05)


def test_sign_conditioned_coverage() -> None:
    # coverage simulation: eta^T y >= 0 events; TG CI should cover mu at ~95%
    rng = np.random.default_rng(17)
    mu, sigma = 0.5, 1.0
    covered = 0
    n_sel = 0
    for _ in range(1000):
        y = rng.normal(mu, sigma, 3)
        eta = np.array([1.0, 0.0, 0.0])
        if eta @ y < 0:
            continue
        n_sel += 1
        out = sign_conditioned_interval(y, eta, sigma, alpha=0.05)
        if out["ci_lo"] <= mu <= out["ci_hi"]:
            covered += 1
    assert n_sel > 500
    assert 0.90 <= covered / n_sel <= 1.0


def test_sign_conditioned_interval_contains_estimate() -> None:
    y = np.array([1.5, -0.2])
    out = sign_conditioned_interval(y, np.array([1.0, 0.0]), sigma=1.0)
    assert out["ci_lo"] <= out["estimate"] <= out["ci_hi"]
    assert out["v_lo"] == pytest.approx(0.0)


def test_polyhedral_two_sided_selection() -> None:
    # selection on |eta^T y| >= c is not polyhedral; but {eta^T y in [a,b]} is
    y = np.array([1.0, 0.3, -0.1])
    eta = np.array([1.0, 0.0, 0.0])
    a = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([0.0, 5.0])  # 0 <= eta^T y <= 5
    out = polyhedral_interval(y, a, b, eta, sigma=0.5, alpha=0.1)
    assert out["v_lo"] == pytest.approx(0.0)
    assert out["v_hi"] == pytest.approx(5.0)
    assert np.isfinite(out["ci_lo"]) and np.isfinite(out["ci_hi"])


def test_polyhedral_violation_rejected() -> None:
    y = np.array([-1.0])
    eta = np.array([1.0])
    with pytest.raises(ValueError, match="selection event"):
        polyhedral_interval(y, -eta[None, :], np.zeros(1), eta, 1.0)


def test_polyhedral_validation() -> None:
    y = np.ones(3)
    eta = np.array([1.0, 0.0, 0.0])
    a = np.zeros((1, 2))
    with pytest.raises(ValueError):
        polyhedral_interval(y, a, np.zeros(1), eta, 1.0)  # A wrong width
    with pytest.raises(ValueError):
        polyhedral_interval(y, np.zeros((1, 3)), np.zeros(1), np.zeros(3), 1.0)
    with pytest.raises(ValueError):
        polyhedral_interval(y, np.zeros((1, 3)), np.zeros(1), eta, 0.0)
    with pytest.raises(ValueError):
        polyhedral_interval(y, np.zeros((1, 3)), np.zeros(1), eta, 1.0, alpha=1.5)
