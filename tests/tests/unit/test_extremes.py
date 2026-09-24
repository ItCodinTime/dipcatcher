"""EVT battery: recovery on known tails, ordering, fail-closed edges."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.extremes import (
    dekkers_moments,
    empirical_tail_dependence,
    extremogram,
    gev_fit,
    gev_return_level,
    gpd_fit,
    gpd_var_es,
    hill_estimator,
    hill_plot,
    mean_excess,
    moment_ratio_plot_stat,
    pickands_estimator,
)


def _pareto(n: int = 4000, alpha: float = 3.0, seed: int = 0) -> np.ndarray:
    # Pareto tail index gamma = 1/alpha.
    u = np.random.default_rng(seed).uniform(size=n)
    return (1.0 - u) ** (-1.0 / alpha)


def _student(n: int = 4000, df: float = 4.0, seed: int = 1) -> np.ndarray:
    return np.abs(np.random.default_rng(seed).standard_t(df, size=n))


def test_hill_recovers_pareto_index() -> None:
    x = _pareto(alpha=3.0)
    gamma = hill_estimator(x, k=150)
    assert gamma == pytest.approx(1.0 / 3.0, abs=0.1)


def test_hill_student_tail_index() -> None:
    x = _student(df=4.0)
    gamma = hill_estimator(x, k=200)
    assert 0.15 < gamma < 0.5  # xi = 1/df = 0.25 target


def test_pickands_and_moments_pareto() -> None:
    x = _pareto(alpha=2.5, seed=2)
    pk = pickands_estimator(x, k=60)
    dk = dekkers_moments(x, k=150)
    # Pickands uses three order statistics — wide tolerance by design.
    assert pk == pytest.approx(0.4, abs=0.6)
    assert dk == pytest.approx(0.4, abs=0.25)


def test_hill_fails_on_nonpositive() -> None:
    with pytest.raises(ValueError, match="positive"):
        hill_estimator(-np.abs(np.random.default_rng(0).standard_normal(100)), k=10)


def test_gpd_fit_recovers_shape() -> None:
    rng = np.random.default_rng(3)
    excess_true = rng.standard_gamma(2.0, size=3000) * 0.5
    fit = gpd_fit(excess_true, threshold=1.0)
    assert fit["n_exceedances"] >= 10
    assert np.isfinite(fit["xi"]) and fit["sigma"] > 0.0


def test_gpd_var_es_ordering_and_identity() -> None:
    out = gpd_var_es(xi=0.25, sigma=0.4, u=1.0, phi_u=0.1, alpha=0.99)
    assert out["var"] > 1.0
    assert out["es"] > out["var"]
    deeper = gpd_var_es(xi=0.25, sigma=0.4, u=1.0, phi_u=0.1, alpha=0.999)
    assert deeper["var"] > out["var"]


def test_gpd_var_es_fail_closed() -> None:
    with pytest.raises(ValueError):
        gpd_var_es(xi=0.25, sigma=-0.4, u=1.0, phi_u=0.1, alpha=0.99)
    with pytest.raises(ValueError):
        gpd_var_es(xi=0.25, sigma=0.4, u=1.0, phi_u=0.1, alpha=0.5)  # below coverage
    xi_heavy = gpd_var_es(xi=1.5, sigma=0.4, u=1.0, phi_u=0.1, alpha=0.99)
    assert np.isnan(xi_heavy["es"])  # ES undefined for xi >= 1


def test_gev_fit_and_return_level() -> None:
    x = _pareto(n=8000, alpha=2.0, seed=5)
    fit = gev_fit(x, block=40)
    assert fit["xi"] > 0.0  # Frechet tail
    rl = gev_return_level(fit["xi"], fit["mu"], fit["sigma"], period_blocks=100)
    assert rl > fit["mu"]
    assert gev_return_level(fit["xi"], fit["mu"], fit["sigma"], 1000) > rl


def test_empirical_tail_dependence() -> None:
    rng = np.random.default_rng(6)
    z = rng.standard_normal(5000)
    eps = rng.standard_normal(5000)
    y = 0.9 * z + 0.1 * eps  # strongly dependent
    dep = empirical_tail_dependence(z, y, q=0.95)
    assert dep["lambda_upper"] > 0.5
    indep = empirical_tail_dependence(z, rng.standard_normal(5000), q=0.95)
    assert indep["lambda_upper"] < dep["lambda_upper"]


def test_extremogram_clustered_vs_iid() -> None:
    rng = np.random.default_rng(8)
    # GARCH-like: clustering -> extremogram > phi_u at lag 1.
    e = rng.standard_normal(4000)
    sig = np.ones(4000) * 0.01
    r = np.empty(4000)
    for t in range(4000):
        r[t] = sig[t] * e[t]
        if t + 1 < 4000:
            sig[t + 1] = np.sqrt(1e-6 + 0.25 * r[t] ** 2 + 0.7 * sig[t] ** 2)
    ex = extremogram(np.abs(r), max_lag=5)
    iid_ex = extremogram(np.abs(_iid_shaped(4000)), max_lag=5)
    assert ex[0] > iid_ex[0]
    assert iid_ex[0] == pytest.approx(0.05, abs=0.04)


def _iid_shaped(n: int) -> np.ndarray:
    return np.random.default_rng(9).standard_normal(n) * 0.01


def test_mean_excess_monotone_for_pareto() -> None:
    grid, me = mean_excess(_pareto(alpha=2.0, seed=10), n_thresholds=10)
    finite = np.isfinite(me)
    assert grid.shape == me.shape
    # Pareto mean excess grows ~ linearly in u.
    assert np.nanmean(np.diff(me[finite])) > 0.0


def test_hill_plot_grid() -> None:
    ks, gam = hill_plot(_pareto(alpha=3.0, seed=11), ks=np.arange(10, 100, 10))
    assert ks.shape == gam.shape
    assert np.all(np.isfinite(gam))


def test_moment_ratio_heavy_vs_light() -> None:
    heavy = moment_ratio_plot_stat(_pareto(alpha=2.0, seed=12), k=100)
    light = moment_ratio_plot_stat(
        np.abs(np.random.default_rng(13).standard_normal(4000)) + 0.01, k=100
    )
    assert heavy > light


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        hill_estimator(-np.abs(np.random.default_rng(0).standard_normal(100)), k=10)
    with pytest.raises(ValueError):
        hill_estimator(_pareto(100), k=99)  # k out of range
    with pytest.raises(ValueError):
        pickands_estimator(np.random.default_rng(0).standard_normal(50), k=20)  # 4k >= n
    with pytest.raises(ValueError):
        gpd_fit(np.arange(20.0))  # fewer than 10 exceedances
    with pytest.raises(ValueError):
        gev_fit(np.arange(50.0), block=10)  # <8 blocks
    with pytest.raises(ValueError):
        extremogram(np.zeros(100))  # no exceedances
    dep = empirical_tail_dependence(np.ones(50), np.ones(50))
    assert np.isnan(dep["lambda_upper"])  # honest nan, not a fake 0
    with pytest.raises(ValueError):
        mean_excess(np.ones(10))
