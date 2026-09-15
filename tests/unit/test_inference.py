import numpy as np

from quant_fund.metrics.inference import (
    benjamini_hochberg,
    bootstrap_sharpe_ci,
    diebold_mariano,
    mean_tstat,
    newey_west_se,
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


def test_pbo_high_when_is_best_fails_oos() -> None:
    is_s = np.array([[2.0, 0.1], [2.0, 0.0], [1.5, 0.2], [3.0, 0.1]])
    oos = np.array([[-1.0, 0.5], [-0.8, 0.4], [-1.2, 0.3], [-0.5, 0.2]])
    pbo = probability_of_backtest_overfitting(is_s, oos)
    assert pbo == 1.0
