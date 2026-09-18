import math

import numpy as np
import pytest

from quant_fund.metrics.inference import (
    benjamini_hochberg,
    bootstrap_sharpe_ci,
    diebold_mariano,
    grouped_mean_tstat,
    mean_tstat,
    newey_west_se,
    two_way_clustered_mean_tstat,
    wild_cluster_bootstrap_two_way_p,
)
from quant_fund.metrics.overfitting import probability_of_backtest_overfitting


def test_newey_west_white_noise_se_near_classical() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    se = newey_west_se(x, lags=0)
    classical = float(np.std(x, ddof=0) / np.sqrt(x.size))
    assert abs(se - classical) / classical < 0.05


def test_mean_tstat_detects_nonzero() -> None:
    rng = np.random.default_rng(1)
    x = 0.2 + rng.normal(scale=0.5, size=400)
    mu, t, p = mean_tstat(x, lags=3)
    assert mu > 0
    assert t > 4
    assert p < 1e-4


def test_grouped_mean_tstat_uses_one_observation_per_group() -> None:
    values = np.array([0.1, 0.3, 0.2, 0.4, 0.1, 0.3])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    mean, _t, _p, n_groups = grouped_mean_tstat(values, groups, target=0.2, lags=0)
    assert mean == pytest.approx(0.2333333333)
    assert n_groups == 3


def test_two_way_clustered_mean_tstat_rejects_constant_edge() -> None:
    dates = np.repeat(np.arange(20), 5)
    names = np.tile(np.arange(5), 20)
    values = 0.02 + 0.002 * (dates % 4) + 0.001 * names.astype(float)
    mean, t_stat, p_value, n, n_a, n_b = two_way_clustered_mean_tstat(values, dates, names)
    assert mean > 0.02
    assert n == 100 and n_a == 20 and n_b == 5
    assert np.isfinite(t_stat)
    assert p_value < 1e-4


def test_two_way_clustered_mean_tstat_fails_closed_one_way() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4])
    dates = np.array([1, 1, 1, 1])
    names = np.array([1, 2, 3, 4])
    _mu, t_stat, p_value, _n, n_a, n_b = two_way_clustered_mean_tstat(values, dates, names)
    assert n_a == 1 and n_b == 4
    assert t_stat != t_stat and p_value != p_value


def test_wild_cluster_bootstrap_two_way_p_is_unit_interval() -> None:
    dates = np.repeat(np.arange(16), 4)
    names = np.tile(np.arange(4), 16)
    values = 0.03 + 0.001 * names.astype(float)
    _mu, t_stat, _p, _n, _na, _nb = two_way_clustered_mean_tstat(values, dates, names)
    wild_p = wild_cluster_bootstrap_two_way_p(
        values, dates, names, observed_t=float(t_stat), n_boot=39, seed=3
    )
    assert 0.0 < wild_p <= 1.0
    assert math.isnan(
        wild_cluster_bootstrap_two_way_p(values, dates, names, observed_t=float("nan"))
    )


def test_diebold_mariano_prefers_better_loss() -> None:
    rng = np.random.default_rng(2)
    y = rng.normal(size=300)
    loss_a = (y - 0.0) ** 2
    loss_b = (y - 1.5) ** 2
    res = diebold_mariano(loss_a, loss_b, name_a="a", name_b="b")
    assert res.preferred == "a"
    assert res.p_value < 1e-6
    assert res.mean_loss_diff < 0


def test_benjamini_hochberg_controls() -> None:
    p = np.array([1e-6, 0.001, 0.02, 0.2, 0.8])
    reject, cutoff = benjamini_hochberg(p, alpha=0.05)
    assert reject[0] and reject[1]
    assert not reject[-1]
    assert cutoff > 0


def test_bootstrap_sharpe_ci_covers_positive_edge() -> None:
    rng = np.random.default_rng(3)
    r = 0.002 + 0.01 * rng.normal(size=400)
    lo, hi, point = bootstrap_sharpe_ci(r, n_boot=200, seed=3)
    assert lo < point < hi
    assert point > 0


def test_bootstrap_sharpe_ci_rejects_invalid_controls() -> None:
    x = np.arange(20.0)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_sharpe_ci(x, n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_sharpe_ci(x, alpha=1.0)
    with pytest.raises(ValueError, match="block"):
        bootstrap_sharpe_ci(x, block=0)
    with pytest.raises(ValueError, match="periods"):
        bootstrap_sharpe_ci(x, periods=0)
    with pytest.raises(ValueError, match="periods"):
        bootstrap_sharpe_ci(x, periods=float("nan"))


def test_pbo_high_when_is_best_fails_oos() -> None:
    is_s = np.array([[2.0, 0.1], [2.0, 0.0], [1.5, 0.2], [3.0, 0.1]])
    oos = np.array([[-1.0, 0.5], [-0.8, 0.4], [-1.2, 0.3], [-0.5, 0.2]])
    pbo = probability_of_backtest_overfitting(is_s, oos)
    assert pbo == 1.0
