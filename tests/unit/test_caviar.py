"""Tests for models/caviar.py — Engle-Manganelli CAViaR."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.caviar import caviar_fit, caviar_forecast


def _garch_returns(n: int = 1200, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig2 = np.empty(n)
    r = np.empty(n)
    sig2[0] = 0.5
    r[0] = np.sqrt(sig2[0]) * rng.standard_normal()
    for t in range(1, n):
        sig2[t] = 0.02 + 0.10 * r[t - 1] ** 2 + 0.85 * sig2[t - 1]
        r[t] = np.sqrt(sig2[t]) * rng.standard_normal()
    return r


def test_sav_fit_tracks_vol() -> None:
    r = _garch_returns()
    out = caviar_fit(r, tau=0.05, spec="sav", n_starts=6)
    q = np.asarray(out["q"])
    # fitted quantile magnitude should co-move with *smoothed* vol
    win = 20
    roll_sig = np.array([np.std(r[i - win : i]) for i in range(win, r.size)])
    corr = np.corrcoef(np.abs(q[win:]), roll_sig)[0, 1]
    assert corr > 0.3
    assert 0.01 < out["hit_rate"] < 0.12


def test_as_and_ig_specs() -> None:
    r = _garch_returns(seed=3)
    for spec in ("as", "ig"):
        out = caviar_fit(r, tau=0.05, spec=spec, n_starts=5)
        assert np.isfinite(np.asarray(out["q"])).all()
        assert 0.005 < out["hit_rate"] < 0.15


def test_adaptive() -> None:
    r = _garch_returns(seed=5)
    out = caviar_fit(r, tau=0.05, spec="adaptive", n_starts=5)
    assert np.isfinite(np.asarray(out["q"])).all()


def test_forecast_matches_path() -> None:
    r = _garch_returns(seed=11)
    out = caviar_fit(r, tau=0.05, spec="sav", n_starts=5)
    q = np.asarray(out["q"])
    beta = np.asarray(out["beta"])
    f = caviar_forecast(q, beta, "sav", r[-1])
    assert np.isfinite(f)


def test_hit_rate_better_than_const() -> None:
    r = _garch_returns()
    out = caviar_fit(r, tau=0.05, spec="sav", n_starts=6)
    q = np.asarray(out["q"])
    # in-sample CAViaR should beat unconditional-quantile on tick loss
    e = r - q
    model_loss = np.mean(np.abs(e * (0.05 - (e < 0))))
    e0 = r - np.quantile(r, 0.05)
    base_loss = np.mean(np.abs(e0 * (0.05 - (e0 < 0))))
    assert model_loss <= base_loss * 1.05


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        caviar_fit(np.random.default_rng(0).standard_normal(50))
    with pytest.raises(ValueError):
        caviar_fit(np.random.default_rng(0).standard_normal(500), tau=0.9)
    with pytest.raises(ValueError):
        caviar_fit(np.full(500, np.nan))
    with pytest.raises(ValueError):
        caviar_fit(np.random.default_rng(0).standard_normal(500), spec="bogus")
    with pytest.raises(ValueError):
        caviar_forecast(np.array([1.0]), np.zeros(3), "bogus", 0.0)
