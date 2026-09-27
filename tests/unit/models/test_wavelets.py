"""Tests for models/wavelets.py — MODWT + wavelet variance."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.wavelets import (
    modwt,
    modwt_mra,
    wavelet_correlation,
    wavelet_variance,
)


def test_modwt_shapes() -> None:
    x = np.random.default_rng(0).standard_normal(512)
    out = modwt(x, "haar", levels=4)
    assert np.asarray(out["details"]).shape == (4, 512)
    assert np.asarray(out["smooth"]).shape == (512,)


def test_mra_reconstructs() -> None:
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.standard_normal(256))
    mra = modwt_mra(x, "d4", levels=3)
    recon = np.asarray(mra["smooth"]) + np.asarray(mra["details"]).sum(axis=0)
    assert np.abs(recon - x).max() < 1e-8


def test_variance_decomposition() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(512)
    out = wavelet_variance(x, "haar", levels=5)
    total = float(out["total_variance"])
    assert total == pytest.approx(float(x.var()), rel=0.05)
    shares = np.asarray(out["variance_share"])
    assert shares.sum() == pytest.approx(1.0 - float(out["smooth_variance"]) / total)


def test_sinusoid_concentrates_at_scale() -> None:
    n = 512
    t = np.arange(n)
    period = 16  # energy should peak at scale j with 2^j ~ period
    x = np.sin(2 * np.pi * t / period)
    out = wavelet_variance(x, "la8", levels=5)
    shares = np.asarray(out["variance_share"])
    peak = int(np.argmax(shares))
    # expected scale ~ log2(period/2)..log2(period): 3 or 4 (0-indexed)
    assert peak in (2, 3, 4)
    # LA8 leaks to adjacent scales; top-2 scales should hold most energy
    top2 = np.sort(shares)[-2:].sum()
    assert top2 > 0.7


def test_wavelet_correlation() -> None:
    rng = np.random.default_rng(4)
    base = np.cumsum(rng.standard_normal(512))
    corr = wavelet_correlation(base, base, "haar")
    assert np.allclose(np.asarray(corr["correlations"]), 1.0, atol=1e-8)
    indep = wavelet_correlation(base, np.cumsum(rng.standard_normal(512)), "haar")
    c = np.asarray(indep["correlations"])
    assert np.abs(c).max() < 0.6


def test_d4_and_la8_run() -> None:
    x = np.random.default_rng(0).standard_normal(256)
    for wv in ("d4", "la8"):
        out = modwt(x, wv, levels=3)
        assert np.isfinite(np.asarray(out["details"])).all()


def test_fail_closed() -> None:
    x = np.random.default_rng(0).standard_normal(64)
    with pytest.raises(ValueError):
        modwt(x, "bogus")
    with pytest.raises(ValueError):
        modwt(x, "haar", levels=10)  # > log2(n)
    with pytest.raises(ValueError):
        modwt(np.array([1.0, np.nan, 2.0]), "haar")
    with pytest.raises(ValueError):
        wavelet_correlation(np.arange(10.0), np.arange(12.0))
