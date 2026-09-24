"""Tests for models/dfm.py — Doz-Giannone-Reichlin DFM."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.dfm import dfm_fit, dfm_forecast


def _dfm_panel(
    t: int = 300, n: int = 30, r: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # factors: persistent AR(1)
    f = np.empty((t, r))
    f[0] = 0.0
    for i in range(1, t):
        f[i] = 0.8 * f[i - 1] + rng.standard_normal(r)
    lam = rng.standard_normal((n, r))
    x = f @ lam.T + 0.3 * rng.standard_normal((t, n))
    return x, f


def test_dfm_recovers_factors() -> None:
    x, f = _dfm_panel()
    out = dfm_fit(x, r=2)
    fs = np.asarray(out["factors"])
    # correlation up to sign/rotation: check each true factor is
    # spanned by the factor space via regression R2
    for j in range(2):
        b = np.linalg.lstsq(np.column_stack([np.ones(fs.shape[0]), fs]), f[:, j], rcond=None)[0]
        resid = f[:, j] - np.column_stack([np.ones(fs.shape[0]), fs]) @ b
        r2 = 1 - resid.var() / f[:, j].var()
        assert r2 > 0.8


def test_dfm_common_r2() -> None:
    x, _ = _dfm_panel(seed=1)
    out = dfm_fit(x, r=2)
    assert out["r2"] > 0.7


def test_dfm_forecast_shape() -> None:
    x, _ = _dfm_panel(seed=2)
    out = dfm_fit(x, r=2)
    fc = dfm_forecast(out)
    assert fc.shape == (30,)
    assert np.isfinite(fc).all()


def test_dfm_var_stationary() -> None:
    x, _ = _dfm_panel(seed=3)
    out = dfm_fit(x, r=2)
    a = np.asarray(out["A"])
    ev = np.linalg.eigvals(a)
    assert np.abs(ev).max() < 1.05


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        dfm_fit(np.random.default_rng(0).standard_normal((20, 30)), 2)
    with pytest.raises(ValueError):
        dfm_fit(np.random.default_rng(0).standard_normal((100, 30)), 0)
    x, _ = _dfm_panel(seed=4)
    x[0, 0] = np.nan
    with pytest.raises(ValueError):
        dfm_fit(x, 2)
