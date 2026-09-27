"""Agent-based limit-order-book simulator.

Research and simulation only. Orders stay inside the process. Reports are
simulation diagnostics, not research-catalog headlines and not live results.
"""

from quant_fund.market_sim.book import OrderBook
from quant_fund.market_sim.config import EcologyConfig, drought_config, validation_config
from quant_fund.market_sim.harness import (
    lightspeed_momentum_weight,
    mean_reversion_weight,
    stress_strategy,
    target_weight_replay,
)
from quant_fund.market_sim.impact import fit_impact_law, measure_impact
from quant_fund.market_sim.native import core_version, matching_benchmark
from quant_fund.market_sim.scenarios import SCENARIOS, run_scenario
from quant_fund.market_sim.simulator import run_ecology
from quant_fund.market_sim.stylized import run_stylized_validation, validate_stylized_facts

__all__ = [
    "SCENARIOS",
    "EcologyConfig",
    "OrderBook",
    "core_version",
    "drought_config",
    "fit_impact_law",
    "lightspeed_momentum_weight",
    "matching_benchmark",
    "mean_reversion_weight",
    "measure_impact",
    "run_ecology",
    "run_scenario",
    "run_stylized_validation",
    "stress_strategy",
    "target_weight_replay",
    "validate_stylized_facts",
    "validation_config",
]
