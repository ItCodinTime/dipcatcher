"""Single-threaded matching throughput. One measured replay, not a tuned loop."""

from __future__ import annotations

from quant_fund.market_sim.native import matching_benchmark


def test_matching_core_exceeds_one_million_events_per_second() -> None:
    first = matching_benchmark(1_000_000, seed=1)
    second = matching_benchmark(200_000, seed=1)
    again = matching_benchmark(200_000, seed=1)
    assert first["rc"] == 0
    assert first["list_ok"] is True
    assert int(first["n_rejects"]) == 0
    assert int(first["n_trades"]) > 1_000_000 / 50
    assert float(first["match_events_per_s"]) >= 1_000_000.0
    assert "pass-2" in str(first["timed_region"])
    assert second["checksum"] == again["checksum"]
    assert second["n_trades"] == again["n_trades"]
