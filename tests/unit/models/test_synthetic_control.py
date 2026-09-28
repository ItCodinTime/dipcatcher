"""Canon tests: synthetic control + placebo inference."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.synthetic_control import (
    placebo_test,
    simplex_project,
    synthetic_control,
)


def _panel(t: int = 80, j: int = 6, seed: int = 3):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.3, t))
    y0 = common[:, None] + rng.normal(0, 0.05, (t, j))
    return y0


def test_simplex_project_known() -> None:
    np.testing.assert_allclose(simplex_project(np.array([0.5, 0.5, 0.0])), [0.5, 0.5, 0.0])
    np.testing.assert_allclose(simplex_project(np.array([2.0, -1.0, 0.0])), [1.0, 0.0, 0.0])
    p = simplex_project(np.array([0.2, 0.9, -0.4, 0.1]))
    assert p.sum() == pytest.approx(1.0)
    assert (p >= 0).all()


def test_synthetic_control_recovers_weights() -> None:
    y0 = _panel()
    w_true = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
    y1 = y0 @ w_true
    out = synthetic_control(y0, y1, t0=50)
    np.testing.assert_allclose(out["w"][:3], w_true[:3], atol=0.05)
    assert out["rmspe_pre"] < 0.05


def test_synthetic_control_detects_shift() -> None:
    y0 = _panel()
    y1 = y0[:, 0] * 0.7 + y0[:, 1] * 0.3
    y1[60:] += 2.0  # post-treatment shift
    out = synthetic_control(y0, y1, t0=60)
    assert out["rmspe_ratio"] > 3.0
    assert out["att_post"] > 1.0


def test_placebo_extreme_treated() -> None:
    y0 = _panel(j=7)
    treated = y0[:, 0] * 0.5 + y0[:, 1] * 0.5
    treated[55:] += 4.0
    panel = np.column_stack([treated, y0])
    out = placebo_test(panel, treated_idx=0, t0=55, n_iter=800)
    assert 0.0 < out["p_value"] <= 1.0
    assert out["treated_ratio"] > np.median(out["ratios"][1:])


def test_validation_matrix() -> None:
    y0 = _panel()
    y1 = y0[:, 0]
    with pytest.raises(ValueError):
        synthetic_control(y0[:, 0], y1, 40)  # Y0 must be 2-D
    with pytest.raises(ValueError):
        synthetic_control(y0, y1[:-1], 40)  # length mismatch
    with pytest.raises(ValueError):
        synthetic_control(y0, y1, 1)  # T0 too small
    with pytest.raises(ValueError):
        synthetic_control(y0, y1, 80)  # T0 == T
    with pytest.raises(ValueError):
        synthetic_control(y0[:, [0]], y1, 40)  # < 2 donors
    with pytest.raises(ValueError):
        synthetic_control(np.full_like(y0, np.nan), y1, 40)
    with pytest.raises(ValueError):
        synthetic_control(np.zeros_like(y0), y1, 40)  # degenerate donors
    with pytest.raises(ValueError):
        synthetic_control(y0, y1, 40, n_iter=5)
    with pytest.raises(ValueError):
        placebo_test(y0, treated_idx=99, t0=40)
    with pytest.raises(ValueError):
        simplex_project(np.array([np.inf, 1.0]))
