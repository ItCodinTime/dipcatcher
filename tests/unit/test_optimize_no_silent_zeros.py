"""P0: optimize_asof must not silently flatten on OptimizationInfeasible."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from quant_fund.config import load_config
from quant_fund.pipeline.dataset import build_gold
from quant_fund.pipeline.forecast import optimize_asof
from quant_fund.schemas.errors import OptimizationInfeasible


@pytest.mark.synthetic
def test_optimize_asof_reraises_infeasible(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 4
    cfg.data.synthetic_n_days = 40
    build_gold(cfg)

    def boom(*_a, **_k):
        raise OptimizationInfeasible('{"feasible": false}')

    # asof=None resolves to the latest panel decision; an explicit date absent
    # from the panel now fails closed before the optimizer is reached.
    with (
        patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=boom),
        pytest.raises(OptimizationInfeasible),
    ):
        optimize_asof(cfg, persist=False)


@pytest.mark.synthetic
def test_optimize_asof_reraises_generic_optimizer_errors(tmp_path: Path) -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.root = tmp_path
    cfg.data.synthetic_n_assets = 4
    cfg.data.synthetic_n_days = 40
    build_gold(cfg)

    def boom(*_a, **_k):
        raise RuntimeError("solver crashed")

    with (
        patch("quant_fund.pipeline.forecast.optimize_mean_variance", side_effect=boom),
        pytest.raises(RuntimeError, match="solver crashed"),
    ):
        optimize_asof(cfg, persist=False)
