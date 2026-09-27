"""Wave 8: ReplayClock deterministic sequence + paper fill/NAV identity."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.paper.clock import ReplayClock
from quant_fund.paper.loop import run_paper_loop


def test_replay_clock_sequence_deterministic() -> None:
    times = [datetime(2024, 1, 1 + i, tzinfo=UTC) for i in range(5)]
    c1 = ReplayClock(event_times=list(times))
    c2 = ReplayClock(event_times=list(times))
    seq1 = list(c1)
    seq2 = list(c2)
    assert seq1 == times
    assert seq2 == times
    assert seq1 == seq2
    assert c1.exhausted() and c2.exhausted()
    assert c1.remaining() == 0


def test_replay_clock_tick_now_alignment() -> None:
    times = [datetime(2024, 2, i, tzinfo=UTC) for i in range(1, 4)]
    clock = ReplayClock(event_times=list(times))
    assert clock.now() == times[0]
    assert clock.tick() == times[0]
    assert clock.now() == times[1]
    assert clock.tick() == times[1]
    assert clock.tick() == times[2]
    assert clock.tick() is None
    assert clock.exhausted()


def test_replay_clock_empty_raises() -> None:
    clock = ReplayClock(event_times=[])
    with pytest.raises(RuntimeError):
        clock.now()
    assert clock.tick() is None
    assert clock.exhausted()


def _bars(n_days: int = 8) -> pl.DataFrame:
    rows = []
    for d in range(n_days):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        for sid, px0 in [("A", 100.0 + d), ("B", 50.0 + 0.5 * d)]:
            rows.append(
                {
                    "security_id": sid,
                    "event_time": t,
                    "open": px0,
                    "close": px0,
                    "close_total_return": px0,
                    "volume": 1_000_000.0,
                    "adv": 100_000_000.0,
                    "vol_20": 0.02,
                    "source": "synthetic",
                }
            )
    return pl.DataFrame(rows)


def _weights(n_days: int = 7, a: float = 0.1, b: float = -0.05) -> pl.DataFrame:
    rows = []
    for d in range(n_days):
        t = datetime(2024, 1, 1 + d, tzinfo=UTC)
        rows.append({"event_time": t, "security_id": "A", "target_weight": a})
        rows.append({"event_time": t, "security_id": "B", "target_weight": b})
    return pl.DataFrame(rows)


def _cfg(tmp_path: Path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 0.5
    cfg.risk_gate.max_net = 0.5
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    return cfg


def test_identical_replay_inputs_identical_fills_and_nav(tmp_path: Path) -> None:
    """Same ReplayClock calendar + same broker inputs → identical fills/NAV."""
    cfg = _cfg(tmp_path)
    bars = _bars(8)
    weights = _weights(7)

    r1 = run_paper_loop(
        bars,
        cfg,
        champion_weights=weights,
        initial_nav=100_000.0,
        max_steps=5,
        prefer_latest=False,
        use_wall_clock=False,
        run_id="replay-ident-a",
    )
    r2 = run_paper_loop(
        bars,
        cfg,
        champion_weights=weights,
        initial_nav=100_000.0,
        max_steps=5,
        prefer_latest=False,
        use_wall_clock=False,
        run_id="replay-ident-b",
    )
    assert r1.metrics["n_steps"] == r2.metrics["n_steps"]
    assert r1.metrics["n_fills"] == r2.metrics["n_fills"]
    assert r1.metrics["n_fills"] >= 1
    assert r1.metrics["live_pnl_claim"] is False
    assert r1.metrics["research_only"] is True

    nav1 = r1.champion_equity["nav"].to_list()
    nav2 = r2.champion_equity["nav"].to_list()
    assert nav1 == pytest.approx(nav2, abs=1e-9)

    # Fill notional identity via orders ledger (reject/fill statuses)
    o1 = r1.orders.sort(["order_id"]) if r1.orders.height else r1.orders
    o2 = r2.orders.sort(["order_id"]) if r2.orders.height else r2.orders
    if o1.height and "status" in o1.columns:
        assert o1["status"].to_list() == o2["status"].to_list()
