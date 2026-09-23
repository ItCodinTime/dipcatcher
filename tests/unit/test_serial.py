"""Serial-dependence battery: size, power, ordering, fail-closed edges."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.serial import (
    autocorrelation,
    bartels_rank_test,
    box_pierce,
    chow_denning_test,
    engle_arch_lm,
    jarque_bera,
    ljung_box,
    runs_test,
    sign_test,
    unit_root_battery,
    variance_ratio_test,
    wright_variance_ratio,
)

RNG = np.random.default_rng(7)


def _iid(n: int = 400, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, size=n)


def _ar1(n: int = 400, phi: float = 0.6, seed: int = 1) -> np.ndarray:
    r = np.random.default_rng(seed).normal(0.0, 0.01, size=n)
    for t in range(1, n):
        r[t] += phi * r[t - 1]
    return r


def test_autocorrelation_lag1_recovers_ar1() -> None:
    rho = autocorrelation(_ar1(2000), max_lag=5)
    assert rho.shape == (5,)
    assert rho[0] == pytest.approx(0.6, abs=0.15)
    assert abs(rho[0]) > abs(rho[-1])


def test_autocorrelation_white_noise_small() -> None:
    rho = autocorrelation(_iid(2000), max_lag=10)
    assert np.all(np.abs(rho) < 0.1)


def test_ljung_box_rejects_ar1_accepts_iid() -> None:
    assert ljung_box(_ar1(), lag=10)["pvalue"] < 0.01
    assert ljung_box(_iid(2000, seed=3), lag=10)["pvalue"] > 0.01


def test_box_pierce_leq_ljung_box() -> None:
    x = _ar1()
    assert box_pierce(x, lag=10)["stat"] <= ljung_box(x, lag=10)["stat"]


def test_engle_arch_lm_detects_garch() -> None:
    rng = np.random.default_rng(5)
    e = rng.standard_normal(600)
    sig = np.ones(600) * 0.01
    r = np.empty(600)
    for t in range(600):
        r[t] = sig[t] * e[t]
        if t + 1 < 600:
            sig[t + 1] = np.sqrt(1e-6 + 0.2 * r[t] ** 2 + 0.75 * sig[t] ** 2)
    assert engle_arch_lm(r, lags=5)["pvalue"] < 0.01
    assert engle_arch_lm(_iid(600), lags=5)["pvalue"] > 0.01


def test_jarque_bera_normal_vs_fat_tails() -> None:
    assert jarque_bera(_iid(2000))["pvalue"] > 0.05
    fat = np.random.default_rng(9).standard_t(3.0, size=2000) * 0.01
    assert jarque_bera(fat)["pvalue"] < 0.01


def test_variance_ratio_iid_vr_near_one() -> None:
    out = variance_ratio_test(_iid(2000), q=8)
    assert out["vr"] == pytest.approx(1.0, abs=0.15)
    assert out["pvalue_z1"] > 0.05


def test_variance_ratio_persistent_series() -> None:
    rng = np.random.default_rng(2)
    # Positively autocorrelated returns -> VR > 1, rejection expected.
    r = _ar1(2000, phi=0.5, seed=11)
    out = variance_ratio_test(r, q=8)
    assert out["vr"] > 1.3
    assert out["pvalue_z2"] < 0.05
    _ = rng  # determinism


def test_variance_ratio_matches_direct_definition() -> None:
    # VR(q) = Var(q-period sums) / (q * Var(1-period)) on demeaned returns.
    x = _iid(600, seed=4)
    q = 4
    e = x - x.mean()
    num = np.var(np.convolve(e, np.ones(q), mode="valid"))
    ours = variance_ratio_test(x, q=q)
    # ours uses the Lo-MacKinlay small-sample correction m = q(n-q+1)(1-q/n);
    # the plain overlap estimator agrees to O(1/n).
    assert ours["vr"] == pytest.approx(num / (q * np.var(e)), rel=0.05)


def test_chow_denning_joint_rejects_persistent() -> None:
    r = _ar1(2000, phi=0.5, seed=11)
    out = chow_denning_test(r)
    assert out["pvalue"] < 0.05
    assert out["k"] == 4.0
    iid = chow_denning_test(_iid(2000, seed=8))
    assert iid["pvalue"] > 0.01


def test_wright_variance_ratio_shapes() -> None:
    out = wright_variance_ratio(_ar1(500, phi=0.5), q=4, n_boot=200, seed=5)
    assert out["r1"] > 1.1
    assert 0.0 <= out["pvalue_r1"] <= 1.0
    iid = wright_variance_ratio(_iid(500), q=4)
    assert iid["r1"] == pytest.approx(1.0, abs=0.25)


def test_runs_test_trending_vs_iid() -> None:
    trending = np.cumsum(_iid(300, seed=12))
    # Mean-reverting OU-like path: few runs.
    assert runs_test(trending)["runs"] < runs_test(_iid(300, seed=12))["runs"]
    assert 0.0 <= runs_test(_iid(300, seed=12))["pvalue"] <= 1.0


def test_sign_test_balanced_and_skewed() -> None:
    bal = sign_test(_iid(500, seed=6))
    assert bal["share_positive"] == pytest.approx(0.5, abs=0.08)
    sk = sign_test(np.abs(_iid(500, seed=6)))
    assert sk["pvalue"] < 1e-9


def test_bartels_rank_test_bounds() -> None:
    iid = bartels_rank_test(_iid(400, seed=15))
    assert iid["pvalue"] > 0.01
    # Strongly alternating ranks -> large consecutive diffs -> RVN -> 4.
    alt = np.where(np.arange(400) % 2 == 0, 0.01, -0.01) + _iid(400, seed=16) * 0.001
    assert bartels_rank_test(alt)["stat"] > 2.0
    # Monotone trend -> consecutive ranks nearly equal -> RVN -> 0.
    trend = np.linspace(0.0, 1.0, 400) + _iid(400, seed=17) * 0.001
    assert bartels_rank_test(trend)["stat"] < 2.0


def test_unit_root_battery_random_walk_vs_stationary() -> None:
    walk = np.cumsum(_iid(500, seed=21))
    out = unit_root_battery(walk)
    for name in ("adf", "dfgls", "phillips_perron", "kpss", "zivot_andrews"):
        assert set(out[name]) == {"stat", "pvalue"}
    assert out["adf"]["pvalue"] > 0.10  # fail to reject unit root
    stat = unit_root_battery(_ar1(800, phi=0.3, seed=22))
    assert stat["adf"]["pvalue"] < 0.05
    assert stat["kpss"]["pvalue"] > 0.05


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        autocorrelation(np.ones(3))
    with pytest.raises(ValueError):
        autocorrelation(np.array([1.0, 2.0]))  # too few finite observations
    with pytest.raises(ValueError):
        autocorrelation(np.ones(10))  # zero variance
    with pytest.raises(ValueError):
        variance_ratio_test(np.ones(10), q=2)
    with pytest.raises(ValueError):
        variance_ratio_test(_iid(50), q=1)
    with pytest.raises(ValueError):
        runs_test(np.ones(50))
    with pytest.raises(ValueError):
        sign_test(np.zeros(10))
    with pytest.raises(ValueError):
        unit_root_battery(_iid(4))
    with pytest.raises(ValueError):
        chow_denning_test(_iid(50), periods=())


def test_nan_inputs_are_dropped_not_fatal() -> None:
    x = _iid(200)
    x[::7] = np.nan
    rho = autocorrelation(x, 5)
    assert np.isfinite(rho).all()
