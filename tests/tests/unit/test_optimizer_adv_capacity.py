"""ADV participation: hard optimizer bound vs soft capacity_proxy diagnostic."""

import numpy as np
import pytest

from quant_fund.config.loader import load_config
from quant_fund.metrics.analytics import capacity_proxy
from quant_fund.portfolio.optimizer import optimize_mean_variance


def test_adv_capacity_binds_turnover(tmp_path):
    """HARD: tiny ADV ⇒ |Δw| pinned near zero."""
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.constraints.max_adv_participation = 0.01
    cfg.constraints.turnover_limit = 10.0  # loose
    n = 4
    alpha = np.array([0.05, -0.04, 0.03, -0.02])
    sig = np.eye(n) * 0.02**2
    wp = np.zeros(n)
    # Tiny ADV ⇒ tiny allowed |Δw|
    adv = np.full(n, 1_000.0)  # dollars
    w, diag = optimize_mean_variance(alpha, sig, wp, cfg, adv_dollars=adv, nav=1_000_000.0)
    assert diag.feasible
    # |Δw|_i <= 0.01 * ADV / nav = 0.01 * 1000 / 1e6 = 1e-5
    assert float(np.max(np.abs(w - wp))) <= 1e-5 + 1e-8


def test_adv_capacity_optional_when_null(tmp_path):
    """HARD disabled (None): unconstrained by ADV; can take material exposure."""
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.constraints.max_adv_participation = None
    cfg.optimizer.lambda_risk = 0.5
    cfg.optimizer.lambda_tc = 0.01
    n = 3
    alpha = np.array([0.05, -0.04, 0.03])
    sig = np.eye(n) * 0.02**2
    w, diag = optimize_mean_variance(alpha, sig, np.zeros(n), cfg)
    assert diag.feasible
    assert float(np.sum(np.abs(w))) > 1e-4


def test_adv_hard_larger_adv_allows_more_delta(tmp_path):
    """HARD: larger ADV loosens per-name |Δw| cap."""
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.constraints.max_adv_participation = 0.10
    cfg.constraints.turnover_limit = 10.0
    cfg.optimizer.lambda_risk = 0.1
    cfg.optimizer.lambda_tc = 0.0
    n = 3
    alpha = np.array([0.08, -0.06, 0.04])
    sig = np.eye(n) * 0.02**2
    wp = np.zeros(n)
    nav = 1_000_000.0
    w_small, d1 = optimize_mean_variance(
        alpha, sig, wp, cfg, adv_dollars=np.full(n, 5_000.0), nav=nav
    )
    w_large, d2 = optimize_mean_variance(
        alpha, sig, wp, cfg, adv_dollars=np.full(n, 5_000_000.0), nav=nav
    )
    assert d1.feasible and d2.feasible
    turn_small = float(np.sum(np.abs(w_small - wp)))
    turn_large = float(np.sum(np.abs(w_large - wp)))
    assert turn_large + 1e-12 >= turn_small
    # Small ADV hard cap: |Δw|_i <= 0.10 * 5000 / 1e6 = 5e-4
    assert float(np.max(np.abs(w_small - wp))) <= 5e-4 + 1e-8


def test_adv_soft_capacity_proxy_does_not_change_weights(tmp_path):
    """SOFT diagnostic: capacity_proxy reports participation; optimizer unchanged."""
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.constraints.max_adv_participation = None  # no hard ADV
    cfg.optimizer.lambda_risk = 0.5
    cfg.optimizer.lambda_tc = 0.01
    n = 3
    alpha = np.array([0.05, -0.04, 0.03])
    sig = np.eye(n) * 0.02**2
    wp = np.zeros(n)
    w, diag = optimize_mean_variance(alpha, sig, wp, cfg, nav=1_000_000.0)
    assert diag.feasible
    # Soft proxy on hypothetical trades
    traded = np.abs(w - wp) * 1_000_000.0
    adv = np.full(n, 1_000_000.0)
    prox = capacity_proxy(traded, adv)
    assert prox["n"] == n
    assert prox["mean_participation"] == pytest.approx(float(np.mean(np.abs(w - wp))))
    assert prox["max_participation"] >= prox["mean_participation"] - 1e-12
    # Empty / bad ADV → NaNs, n=0 (documented limit)
    empty = capacity_proxy(np.array([]), np.array([]))
    assert empty["n"] == 0
    assert empty["mean_participation"] != empty["mean_participation"]  # NaN


def test_adv_hard_mismatched_length_raises(tmp_path):
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.constraints.max_adv_participation = 0.1
    n = 3
    alpha = np.ones(n) * 0.01
    sig = np.eye(n) * 0.02**2
    with pytest.raises(ValueError, match="adv_dollars"):
        optimize_mean_variance(alpha, sig, np.zeros(n), cfg, adv_dollars=np.ones(2), nav=1e6)


def test_adv_capacity_rejects_invalid_adv_and_nav(tmp_path):
    cfg = load_config("configs/research.yaml")
    cfg.constraints.max_adv_participation = 0.1
    cfg.constraints.turnover_limit = 1.0
    alpha = np.array([0.1, 0.05])
    sig = np.eye(2)
    with pytest.raises(ValueError, match="adv_dollars"):
        optimize_mean_variance(alpha, sig, np.zeros(2), cfg, adv_dollars=np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="adv_dollars"):
        optimize_mean_variance(alpha, sig, np.zeros(2), cfg, adv_dollars=np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="nav"):
        optimize_mean_variance(alpha, sig, np.zeros(2), cfg, adv_dollars=np.ones(2), nav=0.0)
