"""A small multi-seed swarm. The CI workflow runs a larger bounded swarm."""

from __future__ import annotations

import pytest

from quant_fund.simtest.swarm import run_swarm

pytestmark = pytest.mark.slow


def test_slow_swarm_of_sixteen_seeds_holds_invariants() -> None:
    report = run_swarm(16, n_days=4, base_seed=100, max_faults=3, shrink=True)
    assert report.ok, [(item.seed, item.reason, item.minimized) for item in report.failures]
    assert report.research_only is True
    assert report.live_pnl_claim is False
    assert report.data_source == "SYNTHETIC"
    assert report.elapsed_seconds > 0.0
    assert report.sessions_per_second > 0.0
    assert report.log_bytes_max >= report.log_bytes_min > 0
