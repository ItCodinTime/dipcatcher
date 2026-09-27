"""Entropy and fractal batteries: ordering on known processes, fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.entropy import (
    approximate_entropy,
    dispersion_entropy,
    lempel_ziv_complexity,
    permutation_entropy,
    sample_entropy,
    shannon_entropy,
    spectral_entropy,
    transfer_entropy,
)
from quant_fund.metrics.fractal import (
    dfa_hurst,
    higuchi_fd,
    hurst_rs,
    katz_fd,
    roughness_battery,
    sevcik_fd,
)


def _iid(n: int = 1200, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, size=n)


def _periodic(n: int = 1200) -> np.ndarray:
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 50.0) + 0.05 * np.random.default_rng(1).standard_normal(n)


def _random_walk(n: int = 1200, seed: int = 2) -> np.ndarray:
    return np.cumsum(_iid(n, seed))


def test_shannon_entropy_ordering() -> None:
    hi = shannon_entropy(_iid(), bins=12)
    # Concentrated series: half the bins are empty -> strictly lower entropy.
    conc = np.where(np.arange(1200) % 2 == 0, 0.0, 0.005) + _iid(1200, seed=7) * 0.0001
    lo = shannon_entropy(conc, bins=12)
    assert hi["normalized"] > lo["normalized"]
    assert 0.0 <= hi["normalized"] <= 1.05


def test_approximate_entropy_periodic_lower_than_iid() -> None:
    ap_periodic = approximate_entropy(_periodic(400), m=2)
    ap_iid = approximate_entropy(_iid(400), m=2)
    assert ap_periodic < ap_iid
    assert np.isfinite(ap_periodic)


def test_sample_entropy_ordering() -> None:
    se_periodic = sample_entropy(_periodic(400), m=2)
    se_iid = sample_entropy(_iid(400), m=2)
    assert np.isfinite(se_iid)
    assert se_periodic < se_iid


def test_permutation_entropy_bounds_and_ordering() -> None:
    pe_iid = permutation_entropy(_iid(), order=3)
    pe_trend = permutation_entropy(_periodic(), order=3)
    assert 0.0 <= pe_iid["normalized"] <= 1.0
    assert pe_trend["normalized"] < pe_iid["normalized"]
    assert pe_iid["n_patterns"] <= 6.0  # order=3 -> at most 3! = 6 patterns


def test_lempel_ziv_iid_higher_than_periodic() -> None:
    assert lempel_ziv_complexity(_iid()) > lempel_ziv_complexity(_periodic())


def test_spectral_entropy_sine_vs_noise() -> None:
    assert spectral_entropy(_periodic()) < spectral_entropy(_iid())


def test_transfer_entropy_directed() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(2000)
    y = np.zeros(2000)
    for t in range(1, 2000):
        y[t] = 0.8 * x[t - 1] + 0.5 * rng.standard_normal()
    out = transfer_entropy(x, y, k=1, bins=3)
    assert out["te_x_to_y"] > out["te_y_to_x"]


def test_dispersion_entropy_bounds() -> None:
    de = dispersion_entropy(_iid(), classes=5, order=3)
    assert 0.0 < de["normalized"] <= 1.05
    assert dispersion_entropy(_periodic(), classes=5, order=3)["normalized"] < de["normalized"]


def test_entropy_fail_closed() -> None:
    with pytest.raises(ValueError):
        shannon_entropy(np.ones(20))
    with pytest.raises(ValueError):
        approximate_entropy(np.ones(30), m=2)  # sd=0 -> r=0
    with pytest.raises(ValueError):
        permutation_entropy(np.arange(5.0), order=8)
    with pytest.raises(ValueError):
        spectral_entropy(np.ones(64))
    with pytest.raises(ValueError):
        transfer_entropy(np.ones(8), np.ones(8))  # too few observations


def test_hurst_rs_random_walk_vs_mean_reverting() -> None:
    h_walk = hurst_rs(_random_walk())["hurst"]
    h_iid = hurst_rs(_iid())["hurst"]
    assert h_walk > h_iid
    assert 0.4 < h_walk < 1.1


def test_dfa_hurst_walk() -> None:
    # Integrated white noise -> alpha ~ 1.5; white noise -> ~0.5.
    assert dfa_hurst(_random_walk())["hurst"] > dfa_hurst(_iid())["hurst"]
    assert dfa_hurst(_iid())["hurst"] == pytest.approx(0.5, abs=0.2)


def test_katz_higuchi_sevcik_ordering() -> None:
    smooth = np.sin(np.linspace(0, 4 * np.pi, 500))
    rough = _iid(500)
    assert katz_fd(smooth) < katz_fd(rough)
    assert higuchi_fd(smooth, kmax=6) < higuchi_fd(rough, kmax=6)
    assert sevcik_fd(smooth) < sevcik_fd(rough)


def test_roughness_battery_keys() -> None:
    out = roughness_battery(_random_walk())
    assert set(out) == {"hurst_rs", "hurst_dfa", "katz_fd", "higuchi_fd", "sevcik_fd"}
    assert np.isfinite(out["hurst_dfa"])


def test_fractal_fail_closed() -> None:
    with pytest.raises(ValueError):
        hurst_rs(np.arange(20.0))
    with pytest.raises(ValueError):
        dfa_hurst(np.arange(20.0))
    with pytest.raises(ValueError):
        katz_fd(np.ones(50))
    with pytest.raises(ValueError):
        higuchi_fd(np.arange(16.0), kmax=8)
    with pytest.raises(ValueError):
        sevcik_fd(np.ones(50))
