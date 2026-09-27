"""optimize_asof passes w_prev and can size on fused alpha."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from quant_fund.config import load_config
from quant_fund.pipeline.dataset import build_gold
from quant_fund.pipeline.forecast import (
    _apply_forecast_interval_caps,
    forecast_asof,
    optimize_asof,
)
from quant_fund.schemas.errors import OptimizationInfeasible
from quant_fund.schemas.forecast import AssetForecast, MarketState


def test_interval_caps_preserve_hard_turnover_limit() -> None:
    cfg = load_config("configs/research.yaml")
    cfg.constraints.name_max = 0.04
    cfg.constraints.turnover_limit = 0.01
    cfg.fusion.apply_interval_caps = True
    asof = datetime(2024, 1, 2)
    state = MarketState(
        asof=asof,
        forecasts=[
            AssetForecast(
                security_id="A",
                symbol="A",
                asof=asof,
                model_version="test",
                interval_lo={"5d": -0.02},
                interval_hi={"5d": 0.04},
                interval_alpha=0.1,
            )
        ],
    )

    with pytest.raises(OptimizationInfeasible, match="interval caps.*turnover"):
        _apply_forecast_interval_caps(np.array([0.05]), state, ["A"], cfg, w_prev=np.array([0.05]))


def test_interval_caps_project_into_remaining_turnover_budget() -> None:
    cfg = load_config("configs/research.yaml")
    cfg.constraints.name_max = 0.04
    cfg.constraints.turnover_limit = 0.02
    cfg.fusion.apply_interval_caps = True
    asof = datetime(2024, 1, 2)
    state = MarketState(
        asof=asof,
        forecasts=[
            AssetForecast(
                security_id="A",
                symbol="A",
                asof=asof,
                model_version="test",
                interval_lo={"5d": -0.02},
                interval_hi={"5d": 0.04},
                interval_alpha=0.1,
            )
        ],
    )

    result = _apply_forecast_interval_caps(
        np.array([-0.05]), state, ["A"], cfg, w_prev=np.array([0.01])
    )

    assert abs(float(result[0])) <= cfg.constraints.name_max + 1e-12
    assert float(np.abs(result - np.array([0.01])).sum()) <= cfg.constraints.turnover_limit + 1e-12
    assert float(result[0]) < -0.005


@pytest.mark.synthetic
def test_optimize_asof_forwards_w_prev(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 4
    cfg.data.synthetic_n_days = 40
    cfg.universe.min_history_bars = 5
    cfg.universe.min_adv = 0.0
    cfg.fusion.apply_interval_caps = False
    build_gold(cfg)
    state = forecast_asof(cfg)
    sid0 = state.forecasts[0].security_id
    seen: dict = {}

    def capture(alpha, sigma, w_prev, config, **kwargs):
        seen["w_prev"] = np.asarray(w_prev, dtype=float).copy()
        n = len(alpha)
        return np.zeros(n), type("D", (), {"feasible": True})()

    with patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=capture):
        optimize_asof(
            cfg,
            state.asof,
            persist=False,
            w_prev={sid0: 0.07},
            use_fused_alpha=False,
        )
    assert "w_prev" in seen
    assert float(np.max(np.abs(seen["w_prev"]))) > 0.0


@pytest.mark.synthetic
def test_optimize_asof_uses_fused_when_enabled(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 4
    cfg.data.synthetic_n_days = 40
    cfg.universe.min_history_bars = 5
    cfg.universe.min_adv = 0.0
    cfg.fusion.use_fused_for_optimize = True
    cfg.fusion.apply_interval_caps = False
    build_gold(cfg)
    seen: dict = {}

    def capture(alpha, sigma, w_prev, config, **kwargs):
        seen["alpha"] = np.asarray(alpha, dtype=float).copy()
        n = len(alpha)
        return np.zeros(n), type("D", (), {"feasible": True})()

    with patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=capture):
        optimize_asof(cfg, persist=False, use_fused_alpha=True)
    assert "alpha" in seen
    assert seen["alpha"].size > 0
