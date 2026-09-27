"""HAC battery: size/power under autocorrelation, ordering, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.hac import (
    andrews_bandwidth,
    andrews_monahan_lrv,
    dm_hac_tstat,
    hac_mean_covariance,
    hac_mean_test,
    kernel_lrv,
    newey_west_lrv,
)


def _iid(n: int = 500, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, size=n)


def _ar1(n: int = 500, phi: float = 0.7, seed: int = 1) -> np.ndarray:
    r = np.random.default_rng(seed).normal(0.0, 0.01, size=n)
    for t in range(1, n):
        r[t] += phi * r[t - 1]
    return r


def test_kernel_lrv_iid_near_variance() -> None:
    x = _iid(2000)
    lrv = newey_west_lrv(x)
    assert lrv == pytest.approx(float(np.var(x)), rel=0.15)


def test_lrv_grows_with_autocorrelation() -> None:
    x = _ar1(2000, phi=0.7)
    lrv_iid = newey_west_lrv(_iid(2000))
    lrv_ar = newey_west_lrv(x)
    # AR(1) LRV = sigma^2/(1-phi)^2 >> sample variance sigma^2/(1-phi^2).
    assert lrv_ar > lrv_iid


def test_kernel_family_agreement() -> None:
    x = _ar1(1000, phi=0.5)
    b = kernel_lrv(x, "bartlett", lag=10)
    q = kernel_lrv(x, "quadratic_spectral", lag=10)
    p = kernel_lrv(x, "parzen", lag=10)
    assert min(b, q, p) > 0.0
    assert max(b, q, p) / min(b, q, p) < 3.0


def test_andrews_bandwidth_increases_with_rho() -> None:
    bw_iid = andrews_bandwidth(_iid(1000))
    bw_ar = andrews_bandwidth(_ar1(1000, phi=0.8))
    assert bw_ar > bw_iid
    assert bw_iid >= 1.0


def test_andrews_monahan_recovers_ar1_lrv() -> None:
    x = _ar1(2000, phi=0.7)
    am = andrews_monahan_lrv(x)
    # True LRV approx var/(1-rho)^2 with var = sd^2/(1-rho^2).
    true_lrv = 0.01**2 / (1 - 0.7**2) / (1 - 0.7) ** 2
    assert am == pytest.approx(true_lrv, rel=0.6)


def test_hac_mean_test_iid() -> None:
    out = hac_mean_test(_iid(1500))
    assert out["pvalue"] > 0.05
    nz = hac_mean_test(_iid(1500) + 0.005)
    assert nz["pvalue"] < 0.05


def test_dm_hac_tstat_detects_differential() -> None:
    rng = np.random.default_rng(4)
    d = 0.01 + 0.05 * rng.standard_normal(400)
    out = dm_hac_tstat(d)
    assert out["t"] > 2.0
    flat = dm_hac_tstat(0.05 * rng.standard_normal(400))
    assert flat["pvalue"] > 0.05


def test_hac_mean_covariance_psd() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((300, 4)) * 0.01
    cov = hac_mean_covariance(x)
    assert cov.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-14)


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        kernel_lrv(np.ones(5))
    with pytest.raises(ValueError):
        kernel_lrv(_iid(50), kernel="bogus")
    with pytest.raises(ValueError):
        kernel_lrv(_iid(50), lag=0)
    with pytest.raises(ValueError):
        hac_mean_test(_iid(50), mean0=float("nan"))
    with pytest.raises(ValueError):
        hac_mean_covariance(np.ones((4, 2)))
