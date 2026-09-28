"""Metaorder scheduling and impact-trial guard regressions."""

from __future__ import annotations

from dataclasses import replace

import pytest

from quant_fund.market_sim.config import EcologyConfig
from quant_fund.market_sim.impact import execute_impact_trial
from quant_fund.market_sim.simulator import META_AGENT, Metaorder, run_ecology


def test_metaorders_sharing_an_agent_id_each_run_to_completion() -> None:
    """Sent counters are per metaorder, not per agent id."""
    cfg = replace(EcologyConfig(seed=5), max_events=400, warmup_events=50)
    first = Metaorder(start_event=100, side=1, qty=8, slice_qty=4, every=10)
    second = Metaorder(start_event=300, side=1, qty=8, slice_qty=4, every=10)
    result = run_ecology(cfg, metaorders=(first, second))
    fills = [fill for fill in result.fills if fill.agent == META_AGENT]
    assert sum(fill.qty for fill in fills) == 16


def test_impact_schedule_guard_counts_actual_children() -> None:
    """51/20 slices to qty 2 -> 26 children; the last lands at start + 25*every."""
    with pytest.raises(ValueError, match="finish before max_events"):
        execute_impact_trial(
            51,
            seed=7,
            n_slices=20,
            every=8,
            start_event=100,
            max_events=290,
        )


def test_noise_probabilities_may_not_exceed_one() -> None:
    with pytest.raises(ValueError, match="sum to at most 1.0"):
        EcologyConfig(noise_market_prob=0.6, noise_cancel_prob=0.5)
