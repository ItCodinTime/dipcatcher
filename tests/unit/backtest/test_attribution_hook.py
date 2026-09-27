"""Wave 7: attribution_hook fixtures — factor/selection/timing + fail-closed."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.analytics import attribution_hook


def test_attribution_hook_shapes_factor_selection_timing() -> None:
    # Construct known means: market 0.01, residual 0.005, cost -0.001, total 0.014
    market = np.full(20, 0.01)
    residual = np.full(20, 0.005)
    cost = np.full(20, -0.001)
    portfolio = market + residual + cost  # 0.014
    out = attribution_hook(portfolio, market, residual, cost)
    assert out["ok"] is True
    assert out["status"] == "ok"
    assert out["live_pnl_claim"] is False
    assert out["research_only"] is True
    assert out["market"] == pytest.approx(0.01)
    assert out["selection_residual"] == pytest.approx(0.005)
    assert out["cost"] == pytest.approx(-0.001)
    assert out["total"] == pytest.approx(0.014)
    assert out["convention"] == 1.0


def test_attribution_hook_empty_fail_closed() -> None:
    empty = np.array([], dtype=float)
    ones = np.ones(5)
    out = attribution_hook(empty, ones, ones, ones)
    assert out["ok"] is False
    assert out["status"] == "empty_input"
    assert out["live_pnl_claim"] is False
    assert out["market"] != out["market"]  # nan
    assert out["total"] != out["total"]


def test_attribution_hook_all_empty_fail_closed() -> None:
    e = np.array([], dtype=float)
    out = attribution_hook(e, e, e, e)
    assert out["ok"] is False
    assert out["status"] == "empty_input"


def test_attribution_hook_length_mismatch_fail_closed() -> None:
    a = np.ones(5)
    b = np.ones(4)
    out = attribution_hook(a, a, a, b)
    assert out["ok"] is False
    assert out["status"] == "length_mismatch"
    assert out["live_pnl_claim"] is False
