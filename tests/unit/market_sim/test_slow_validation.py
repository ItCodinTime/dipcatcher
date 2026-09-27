"""Full stylized-fact run. The statuses are observed, not required to pass."""

from __future__ import annotations

import pytest
from quant_fund.market_sim.harness import mean_reversion_weight, stress_strategy
from quant_fund.market_sim.honesty import diagnostic_keys_ok
from quant_fund.market_sim.stylized import run_stylized_validation

_FACTS = (
    "fat_tails",
    "volatility_clustering",
    "no_return_autocorrelation",
    "long_memory_absolute_returns",
    "square_root_impact",
    "spread_and_depth_shape",
)


@pytest.mark.slow
@pytest.mark.synthetic
def test_stylized_validation_reports_every_fact() -> None:
    report = run_stylized_validation()
    assert report["research_only"] is True
    assert report["live_pnl_claim"] is False
    assert report["data_source"] == "SYNTHETIC"
    facts = report["facts"]
    assert isinstance(facts, dict)
    for name in _FACTS:
        status = facts[name]["status"]
        assert status in {"pass", "fail", "inconclusive"}
    assert diagnostic_keys_ok(report)


@pytest.mark.slow
@pytest.mark.synthetic
def test_mean_reversion_stress_covers_the_scenario_library() -> None:
    report = stress_strategy(mean_reversion_weight, name="mean_reversion")
    scenarios = report["scenarios"]
    assert isinstance(scenarios, list)
    assert len(scenarios) == 5
    assert report["live_pnl_claim"] is False
    assert diagnostic_keys_ok(report)
