"""Decision rules and the strategy stress harness. Facts are not forced to pass."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from quant_fund.market_sim.config import EcologyConfig
from quant_fund.market_sim.harness import (
    lightspeed_momentum_weight,
    mean_reversion_weight,
    stress_strategy,
    target_weight_replay,
)
from quant_fund.market_sim.honesty import diagnostic_keys_ok
from quant_fund.market_sim.impact import execute_impact_trial, fit_impact_law
from quant_fund.market_sim.stylized import validate_stylized_facts


def _evidence(report: dict[str, object]) -> None:
    assert report["research_only"] is True
    assert report["live_pnl_claim"] is False
    assert report["data_source"] == "SYNTHETIC"
    assert report["evidence_class"] == "SIMULATION_DIAGNOSTIC"
    assert diagnostic_keys_ok(report)


def test_short_samples_are_inconclusive() -> None:
    report = validate_stylized_facts(
        np.zeros(20),
        np.ones(10),
        np.ones(10),
        impact=None,
    )
    facts = report["facts"]
    assert isinstance(facts, dict)
    for name in (
        "fat_tails",
        "volatility_clustering",
        "no_return_autocorrelation",
        "long_memory_absolute_returns",
        "spread_and_depth_shape",
        "square_root_impact",
    ):
        assert facts[name]["status"] == "inconclusive"
    _evidence(report)


def test_fat_tails_and_autocorrelation_rules() -> None:
    rng = np.random.default_rng(0)
    fat = rng.standard_t(3, size=800)
    fat_report = validate_stylized_facts(fat, np.array([]), np.array([]))
    facts = fat_report["facts"]
    assert isinstance(facts, dict)
    assert facts["fat_tails"]["status"] == "pass"

    bounded = np.tile(np.linspace(-1.0, 1.0, 40), 8)
    thin = validate_stylized_facts(bounded, np.array([]), np.array([]))
    thin_facts = thin["facts"]
    assert isinstance(thin_facts, dict)
    assert thin_facts["fat_tails"]["status"] == "fail"

    shock = rng.normal(size=500)
    ar = np.zeros(500)
    for t in range(1, 500):
        ar[t] = 0.85 * ar[t - 1] + shock[t]
    dependent = validate_stylized_facts(ar, np.array([]), np.array([]))
    dependent_facts = dependent["facts"]
    assert isinstance(dependent_facts, dict)
    assert dependent_facts["no_return_autocorrelation"]["status"] == "fail"


def test_volatility_clustering_on_a_garch_path() -> None:
    rng = np.random.default_rng(1)
    draws = rng.normal(size=900)
    returns = np.zeros(900)
    variance = 1.0
    for t in range(900):
        returns[t] = math_sqrt(variance) * draws[t]
        variance = 0.05 + 0.10 * returns[t] ** 2 + 0.85 * variance
    report = validate_stylized_facts(returns, np.array([]), np.array([]))
    facts = report["facts"]
    assert isinstance(facts, dict)
    assert facts["volatility_clustering"]["status"] == "pass"


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))


def test_impact_law_classifier() -> None:
    participation = np.array(
        [0.02, 0.04, 0.07, 0.1, 0.15, 0.22, 0.3, 0.4, 0.55, 0.7, 0.9, 1.2, 1.6, 2.0]
    )
    noise = np.random.default_rng(2).normal(0.0, 0.01, size=participation.size)
    half = np.exp(np.log(2.5) + 0.5 * np.log(participation) + noise)
    passed = fit_impact_law(participation, half)
    assert passed["status"] == "pass"
    assert float(passed["ci_low"]) > 0.0

    flat = np.exp(np.log(2.5) + 0.02 * np.log(participation))
    failed = fit_impact_law(participation, flat)
    assert failed["status"] == "fail"

    short = fit_impact_law(participation[:4], half[:4])
    assert short["status"] == "inconclusive"


def test_spread_and_depth_lognormal_shape() -> None:
    rng = np.random.default_rng(3)
    spreads = rng.lognormal(mean=0.8, sigma=0.55, size=400)
    depths = rng.lognormal(mean=2.5, sigma=0.45, size=400)
    returns = rng.normal(size=50)
    report = validate_stylized_facts(returns, spreads, depths)
    facts = report["facts"]
    assert isinstance(facts, dict)
    assert facts["spread_and_depth_shape"]["status"] == "pass"


def test_weight_adapters() -> None:
    rising = np.linspace(100.0, 130.0, 40)
    assert lightspeed_momentum_weight(rising) > 0.0
    jump = np.concatenate([np.full(25, 100.0), np.full(5, 120.0)])
    assert mean_reversion_weight(jump) < 0.0
    panel = pl.DataFrame(
        {
            "event_time": [1, 2],
            "security_id": ["SIM", "SIM"],
            "target_weight": [0.1, -0.2],
        }
    )
    replay = target_weight_replay(panel)
    assert replay(np.array([1.0])) == 0.1
    assert replay(np.array([1.0, 2.0, 3.0])) == -0.2
    with pytest.raises(ValueError):
        target_weight_replay(pl.DataFrame({"event_time": [1]}))


@pytest.mark.synthetic
def test_stress_one_scenario_is_a_diagnostic() -> None:
    cfg = replace(EcologyConfig(seed=5), max_events=160, warmup_events=20, bar_events=25)

    def _weight(closes: np.ndarray) -> float:
        return 0.25 if closes.size else 0.0

    report = stress_strategy(_weight, name="constant", cfg=cfg, scenarios=("flash_crash",))
    _evidence(report)
    scenarios = report["scenarios"]
    assert isinstance(scenarios, list)
    block = scenarios[0]
    assert isinstance(block, dict)
    assert block["scenario"] == "flash_crash"
    simulation = block["simulation"]
    assert isinstance(simulation, dict)
    assert simulation["filled_qty"] >= 0
    backtest = block["backtest"]
    assert isinstance(backtest, dict)
    assert backtest["status"] == "ok"
    assert "sharpe" not in backtest
    assert diagnostic_keys_ok(block)


def test_one_impact_trial_returns_a_reason_or_a_point() -> None:
    row = execute_impact_trial(
        40,
        seed=4,
        start_event=80,
        every=4,
        n_slices=5,
        max_events=220,
    )
    assert row["quantity"] == 40
    assert row["dropped"] in {True, False}
    if row["dropped"]:
        assert row["dropped_reason"]
    else:
        assert float(row["adverse_ticks"] or 0.0) > 0.0
