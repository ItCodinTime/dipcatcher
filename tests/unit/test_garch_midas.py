"""Tests for models/garch_midas.py — Engle-Ghysels-Sohn GARCH-MIDAS."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.garch_midas import garch_midas_fit


def _regime_vol(n: int = 1260, seed: int = 7) -> np.ndarray:
    """Two vol regimes: low vol first half, 3x vol second half."""
    rng = np.random.default_rng(seed)
    sig = np.where(np.arange(n) < n // 2, 0.5, 1.5)
    return sig * rng.standard_normal(n)


def test_total_vol_tracks_regime() -> None:
    r = _regime_vol()
    out = garch_midas_fit(r, block=21, k_lag=8, fix_w2=5.0)
    tv = np.asarray(out["total_vol"])
    n = r.size
    v_lo = tv[: n // 2 - 100].mean()
    v_hi = tv[n // 2 + 200 :].mean()
    # true vol ratio is 3x; fitted total should show a clear shift
    assert v_hi > 1.8 * v_lo


def test_g_sane() -> None:
    r = _regime_vol(seed=3)
    out = garch_midas_fit(r, block=21, k_lag=8, fix_w2=5.0)
    g = np.asarray(out["g"])
    # short-run component: positive, finite, not explosive
    assert (g > 0).all()
    assert 0.3 < g.mean() < 5.0
    assert g.max() < 50.0


def test_params_sane() -> None:
    r = _regime_vol(seed=5)
    out = garch_midas_fit(r, block=21, k_lag=8, fix_w2=5.0)
    assert 0 <= out["alpha"] <= 0.5
    assert 0 <= out["beta"] <= 0.999
    assert out["alpha"] + out["beta"] < 1.0
    assert out["theta"] >= 0  # long-run vol responds to past RV


def test_total_vol_reasonable() -> None:
    r = _regime_vol(seed=9)
    out = garch_midas_fit(r, block=21, k_lag=8, fix_w2=5.0)
    tv = np.asarray(out["total_vol"])
    # average fitted vol should bracket the true sigmas
    n = r.size
    assert tv[: n // 2].mean() > 0.2
    assert tv[n // 2 + 100 :].mean() > tv[: n // 2].mean()


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        garch_midas_fit(np.random.default_rng(0).standard_normal(50))
    with pytest.raises(ValueError):
        garch_midas_fit(np.full(500, np.nan))
    with pytest.raises(ValueError):
        garch_midas_fit(np.random.default_rng(0).standard_normal(300), block=3)
    with pytest.raises(ValueError):
        garch_midas_fit(np.random.default_rng(0).standard_normal(400), block=21, k_lag=20)
