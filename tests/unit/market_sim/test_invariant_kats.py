"""Analytic moment KATs and honesty labels for the synthetic ecology."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from quant_fund.market_sim.agents import Agent, NoiseAgent, _lognormal_size
from quant_fund.market_sim.config import EVIDENCE, EcologyConfig
from quant_fund.market_sim.harness import simulation_bars
from quant_fund.market_sim.quotes import avellaneda_stoikov_quotes
from quant_fund.market_sim.simulator import run_ecology


def test_avellaneda_stoikov_matches_the_closed_form() -> None:
    """reservation = s - q γ σ²; spread = γ σ² + (2/γ) ln(1 + γ/k)."""
    mid, units, gamma, k, sigma2 = 1000.0, 2.0, 0.5, 1.5, 4.0
    reservation = mid - units * gamma * sigma2
    half = max(0.5 * (gamma * sigma2 + (2.0 / gamma) * math.log(1.0 + gamma / k)), 1.0)
    bid, ask, quoted_half = avellaneda_stoikov_quotes(mid, units, gamma, k, sigma2)
    assert quoted_half == pytest.approx(half)
    assert bid == math.floor(reservation - half)
    assert ask == math.ceil(reservation + half)
    assert bid < reservation < ask


def test_half_spread_floors_at_one_tick() -> None:
    _, _, half = avellaneda_stoikov_quotes(100.0, 0.0, 0.2, 1.5, 0.01)
    assert half == 1.0


def test_lognormal_size_matches_analytic_moments() -> None:
    """size ~ capped round(LogNormal(mu, sigma)); median e^mu, mean e^{mu+σ²/2}."""
    rng = np.random.default_rng(0)
    draws = np.asarray([_lognormal_size(rng, 1.1, 1.05) for _ in range(200_000)])
    assert draws.min() >= 1
    assert draws.max() <= 80
    assert abs(draws.mean() - math.exp(1.1 + 0.5 * 1.05 * 1.05)) < 0.15
    assert abs(np.median(draws) - math.exp(1.1)) < 0.1
    assert float((draws == 80).mean()) < 0.01


def test_wait_ns_is_exponential_with_mean_one_over_rate() -> None:
    agent = Agent(kind="probe", agent_id=1, rate=2.0)
    rng = np.random.default_rng(3)
    waits = np.asarray([agent.wait_ns(rng) for _ in range(50_000)], dtype=float)
    assert waits.min() >= 1
    assert abs(waits.mean() - 1e9 / 2.0) < 8e6


def test_noise_size_and_offsets_are_seeded_deterministically() -> None:
    first = NoiseAgent(kind="noise", agent_id=2, rate=1.0)
    second = NoiseAgent(kind="noise", agent_id=2, rate=1.0)
    rng_a = np.random.default_rng(9)
    rng_b = np.random.default_rng(9)
    assert _lognormal_size(rng_a, first.size_mu, first.size_sigma) == _lognormal_size(
        rng_b, second.size_mu, second.size_sigma
    )


def test_evidence_dict_labels_the_tape_as_simulation_diagnostic() -> None:
    assert EVIDENCE == {
        "research_only": True,
        "live_pnl_claim": False,
        "evidence_class": "SIMULATION_DIAGNOSTIC",
        "data_source": "SYNTHETIC",
        "claim": "simulation_diagnostic_only",
    }


def test_same_seed_replays_identical_fills_spreads_and_depths() -> None:
    cfg = replace(EcologyConfig(seed=9), max_events=160, warmup_events=10)
    left = run_ecology(cfg)
    right = run_ecology(cfg)
    assert left.fills == right.fills
    assert np.array_equal(left.spreads, right.spreads)
    assert np.array_equal(left.depths, right.depths)
    assert left.checksum == right.checksum != 0


def test_simulation_bars_carry_the_synthetic_source_label() -> None:
    cfg = replace(EcologyConfig(seed=2), max_events=120, warmup_events=10, bar_events=20)
    bars = simulation_bars(run_ecology(cfg))
    assert bars["source"].unique().to_list() == ["synthetic"]
    assert bars.height == len(bars["event_time"].unique())
