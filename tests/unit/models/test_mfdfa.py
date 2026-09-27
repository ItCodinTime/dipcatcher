"""Tests for models/mfdfa.py — Kantelhardt MF-DFA."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.mfdfa import mfdfa


def _cascade(n: int, seed: int = 3) -> np.ndarray:
    """Random binomial multiplicative cascade — canonical multifractal."""
    rng = np.random.default_rng(seed)
    x = np.ones(n)
    block = n
    while block >= 2:
        block //= 2
        m = rng.random(n // block)  # multiplier per sub-block
        x = x * np.repeat(m, block)
    return x * rng.standard_normal(n) * 0.01 + x  # cascade increments-ish


def test_white_noise_narrow_spectrum() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)
    out = mfdfa(x, order=1)
    assert out["h_at_q2"] == pytest.approx(0.5, abs=0.15)


def test_fq_scales_as_power_law() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4096)
    out = mfdfa(x)
    fq = np.asarray(out["fq"])
    sc = np.asarray(out["scales"])
    # log fq vs log s should be strongly linear for q=2
    j = int(np.argmin(np.abs(np.asarray(out["q"]) - 2.0)))
    col = fq[:, j]
    ok = np.isfinite(col) & (col > 0)
    coef = np.polyfit(np.log(sc[ok]), np.log(col[ok]), 1)
    pred = np.polyval(coef, np.log(sc[ok]))
    ss = 1.0 - np.sum((np.log(col[ok]) - pred) ** 2) / np.sum(
        (np.log(col[ok]) - np.log(col[ok]).mean()) ** 2
    )
    assert ss > 0.98


def test_cascade_wider_than_white() -> None:
    rng = np.random.default_rng(0)
    white = rng.standard_normal(4096)
    casc = _cascade(4096)
    w_white = mfdfa(white)["spectrum_width"]
    w_casc = mfdfa(casc)["spectrum_width"]
    assert w_casc > w_white * 1.2


def test_h_q_decreasing_for_multifractal() -> None:
    casc = _cascade(8192, seed=5)
    out = mfdfa(casc)
    h = np.asarray(out["h_q"])
    ok = np.isfinite(h)
    # h(q) should decline with q for multifractal data
    assert h[ok][0] >= h[ok][-1] - 0.05


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        mfdfa(np.ones(30))
    with pytest.raises(ValueError):
        mfdfa(np.zeros(200))  # zero variance
    with pytest.raises(ValueError):
        mfdfa(np.random.default_rng(0).standard_normal(500), scales=np.array([4, 8, 16]), order=2)
    with pytest.raises(ValueError):
        mfdfa(np.random.default_rng(0).standard_normal(500), q_list=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        mfdfa(np.full(200, np.nan))
