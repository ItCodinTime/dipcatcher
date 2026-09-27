"""Tests for models/jump_diffusion.py — Merton (1976) + Kou (2002)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.models.jump_diffusion import (
    kou_kappa,
    kou_simulate,
    merton_jump_call,
    merton_jump_put,
    merton_jump_simulate,
    merton_kappa,
    merton_log_moments,
)


def _bsm_call(s: float, k: float, t: float, r: float, sig: float) -> float:
    d1 = (math.log(s / k) + (r + 0.5 * sig * sig) * t) / (sig * math.sqrt(t))
    d2 = d1 - sig * math.sqrt(t)
    return s * float(norm.cdf(d1)) - k * math.exp(-r * t) * float(norm.cdf(d2))


def test_no_jump_limit_is_bsm() -> None:
    s, k, t, r, sig = 100.0, 100.0, 0.5, 0.03, 0.2
    jd = merton_jump_call(s, k, t, r, sig, lam=0.0, mu_j=-0.1, s_j=0.3)
    assert jd == pytest.approx(_bsm_call(s, k, t, r, sig), rel=1e-12)


def test_jumps_raise_otm_wing_prices() -> None:
    s, t, r, sig = 100.0, 0.25, 0.02, 0.2
    k_wing = 130.0
    flat = merton_jump_call(s, k_wing, t, r, sig, lam=0.0, mu_j=0.0, s_j=0.2)
    jumpy = merton_jump_call(s, k_wing, t, r, sig, lam=3.0, mu_j=0.0, s_j=0.2)
    assert jumpy > flat


def test_put_call_parity() -> None:
    s, k, t, r = 100.0, 95.0, 1.0, 0.04
    sig, lam, mj, sj = 0.25, 2.0, -0.05, 0.15
    c = merton_jump_call(s, k, t, r, sig, lam, mj, sj)
    p = merton_jump_put(s, k, t, r, sig, lam, mj, sj)
    assert c - p == pytest.approx(s - k * math.exp(-r * t), rel=1e-10)


def test_sim_terminal_mean_martingale() -> None:
    rng = np.random.default_rng(11)
    s0, t, r = 100.0, 0.5, 0.03
    sig, lam, mj, sj = 0.2, 4.0, -0.05, 0.25
    paths = merton_jump_simulate(s0, t, r, sig, lam, mj, sj, n_steps=50, n_paths=20000, rng=rng)
    # E[S_T] = S0 e^{rT} under the risk-neutral drift correction
    assert paths[:, -1].mean() == pytest.approx(s0 * math.exp(r * t), rel=0.04)


def test_log_moments_close_to_sim() -> None:
    rng = np.random.default_rng(3)
    t, r, sig, lam, mj, sj = 1.0, 0.02, 0.2, 5.0, -0.08, 0.2
    mom = merton_log_moments(t, r, sig, lam, mj, sj)
    paths = merton_jump_simulate(100.0, t, r, sig, lam, mj, sj, n_steps=60, n_paths=20000, rng=rng)
    log_r = np.log(paths[:, -1] / paths[:, 0])
    assert log_r.mean() == pytest.approx(mom["mean"], abs=0.03)
    assert mom["skew"] < 0.0  # negative mean jumps -> left skew


def test_kou_sim_martingale() -> None:
    rng = np.random.default_rng(5)
    s0, t, r = 100.0, 0.5, 0.03
    paths = kou_simulate(
        s0,
        t,
        r,
        0.2,
        lam=6.0,
        p_up=0.4,
        eta1=12.0,
        eta2=10.0,
        n_steps=50,
        n_paths=20000,
        rng=rng,
    )
    assert paths[:, -1].mean() == pytest.approx(s0 * math.exp(r * t), rel=0.05)
    assert kou_kappa(0.4, 12.0, 10.0) == pytest.approx(
        0.4 * 12 / 11 + 0.6 * 10 / 11 - 1.0, rel=1e-12
    )


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        merton_jump_call(100.0, 100.0, 1.0, 0.02, 0.2, lam=-1.0, mu_j=0.0, s_j=0.2)
    with pytest.raises(ValueError):
        merton_jump_call(100.0, 100.0, 1.0, 0.02, 0.2, lam=1.0, mu_j=0.0, s_j=0.0)
    with pytest.raises(ValueError):
        merton_jump_simulate(100.0, 1.0, 0.02, 0.2, 1.0, 0.0, 0.2, 0, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        kou_simulate(
            100.0,
            1.0,
            0.02,
            0.2,
            1.0,
            p_up=0.5,
            eta1=0.5,
            eta2=10.0,
            n_steps=10,
            n_paths=10,
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ValueError):
        merton_kappa(0.0, -0.1)
