"""Turnover soft penalty + expanded stress pack fixtures."""

import numpy as np
import pytest

from quant_fund.config.loader import load_config
from quant_fund.metrics.analytics import stress_scenarios
from quant_fund.portfolio.optimizer import optimize_mean_variance


def test_stress_liquidity_freeze_and_gap_open():
    w = np.array([0.1, -0.05, 0.05])
    out = stress_scenarios(
        w,
        shock_sigma=1.0,
        liquidity_haircut=0.1,
        liquidity_freeze_frac=0.5,
        gap_open_sigma=2.0,
        asset_vols=np.full(3, 0.02),
    )
    assert out["liquidity_freeze_pnl"] < 0
    assert out["gap_open_pnl"] < 0
    assert out["gap_open_pnl"] == pytest.approx(2.0 * out["shock_down_pnl"])
    assert out["liquidity_freeze_frac"] == 0.5
    assert out["note"] == "stylized_hypothetical"


def test_lambda_turnover_soft_reduces_turnover(tmp_path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.constraints.turnover_limit = 10.0  # hard limit loose
    cfg.optimizer.lambda_tc = 0.0
    n = 4
    alpha = np.array([0.05, -0.04, 0.03, -0.02])
    sig = np.eye(n) * 0.02**2
    w_prev = np.zeros(n)
    cfg.optimizer.lambda_turnover = 0.0
    w0, d0 = optimize_mean_variance(alpha, sig, w_prev, cfg)
    cfg.optimizer.lambda_turnover = 5.0
    w1, d1 = optimize_mean_variance(alpha, sig, w_prev, cfg)
    assert d0.feasible and d1.feasible
    # From cash, soft turnover penalty should shrink ||w||_1 (turnover = ||w||_1)
    assert float(np.sum(np.abs(w1))) <= float(np.sum(np.abs(w0))) + 1e-8
