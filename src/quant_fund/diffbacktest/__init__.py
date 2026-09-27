"""Differentiable research backtest. Optional JAX extra; NumPy core always imports.

Research tooling only. Not a live-trading path and not a research-catalog score.
See ``docs/DIFFBACKTEST.md``.
"""

from __future__ import annotations

from typing import Any

from quant_fund.diffbacktest.spec import (
    DATA_SOURCE_SYNTHETIC,
    LIVE_PNL_CLAIM,
    RESEARCH_ONLY,
    STRATEGIES,
    StrategyParams,
)

__all__ = [
    "DATA_SOURCE_SYNTHETIC",
    "LIVE_PNL_CLAIM",
    "RESEARCH_ONLY",
    "STRATEGIES",
    "StrategyParams",
    "adversarial_radius",
    "available",
    "objective_gradients",
    "optimize_flat",
    "run_small_case",
    "sensitivity_map",
    "simulate",
    "simulate_jax",
    "walk_forward_compare",
]


def __getattr__(name: str) -> Any:
    if name == "simulate":
        from quant_fund.diffbacktest.numpy_core import simulate

        return simulate
    if name in {"simulate_jax", "objective_gradients", "available"}:
        from quant_fund.diffbacktest import jax_core

        return getattr(jax_core, name)
    if name in {
        "sensitivity_map",
        "optimize_flat",
        "walk_forward_compare",
        "adversarial_radius",
        "run_small_case",
    }:
        from quant_fund.diffbacktest import study

        return getattr(study, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
