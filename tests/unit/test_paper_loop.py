"""Paper / shadow loop: fills, kill halt, risk reject, champion vs shadow divergence."""

from datetime import UTC, datetime

import polars as pl

from quant_fund.config.loader import load_config
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.monitoring.kill_switch import HALT_NEW_ORDERS
from quant_fund.paper.loop import (
    StaleValuationError,
    _bounded_valuation_marks,
    _require_valuation_marks,
    run_paper_loop,
)


def test_bounded_valuation_marks_expire_stale_carry_forward(tmp_path) -> None:
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.stale_price_bars = 1
    broker = SimulatedBroker(config=cfg)
    broker.last_marks = {"A": 100.0, "B": 50.0}
    ages: dict[str, int] = {}

    first = _bounded_valuation_marks(broker, {"A": 101.0}, ages, max_stale_bars=1)
    second = _bounded_valuation_marks(broker, {"A": 102.0}, ages, max_stale_bars=1)
    third = _bounded_valuation_marks(broker, {"A": 103.0}, ages, max_stale_bars=1)

    assert first == {"A": 101.0, "B": 50.0}
    assert second == {"A": 102.0, "B": 50.0}
    assert third == {"A": 103.0}


def test_expired_held_mark_raises_controlled_valuation_error(tmp_path) -> None:
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    broker = SimulatedBroker(config=cfg)
    broker.shares = {"B": 1.0}
    broker.last_marks = {"B": 50.0}
    ages: dict[str, int] = {}

    _bounded_valuation_marks(broker, {}, ages, max_stale_bars=0)
    expired = _bounded_valuation_marks(broker, {}, ages, max_stale_bars=0)

    assert expired == {}
    try:
        _require_valuation_marks(broker, expired)
    except StaleValuationError as exc:
        assert "B" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expired held mark did not fail closed")


def _bars():
    times = [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
        datetime(2024, 1, 4, tzinfo=UTC),
    ]
    rows = []
    for t in times:
        for sid, px0 in [("A", 100.0), ("B", 50.0)]:
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


def _weights(scale: float = 1.0):
    times = [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    ]
    rows = []
    for t in times:
        rows.append({"event_time": t, "security_id": "A", "target_weight": 0.1 * scale})
        rows.append({"event_time": t, "security_id": "B", "target_weight": -0.05 * scale})
    return pl.DataFrame(rows)


def _cfg(tmp_path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 0.5
    cfg.risk_gate.max_net = 0.5
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.frictionless = True
    return cfg


def test_paper_loop_fills_and_ledger(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_paper_loop(
        _bars(),
        cfg,
        champion_weights=_weights(1.0),
        shadow_weights=_weights(0.5),
        initial_nav=100_000.0,
        max_steps=3,
    )
    assert result.source_note == "SYNTHETIC"
    assert result.metrics["live_pnl_claim"] is False
    assert result.metrics["research_only"] is True
    assert result.metrics["n_fills"] >= 1
    assert result.champion_equity.height >= 1
    assert result.divergence["mean_l1"] > 0.0
    assert (tmp_path / "metadata" / "paper" / result.run_id / "meta.json").is_file()


def test_paper_loop_kill_switch_halt(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.kill_switch.state = HALT_NEW_ORDERS
    result = run_paper_loop(
        _bars(),
        cfg,
        champion_weights=_weights(),
        initial_nav=100_000.0,
        max_steps=2,
    )
    assert result.metrics["kill_switch_halts"] >= 1
    assert result.metrics["n_fills"] == 0


def test_paper_loop_risk_reject(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_order_notional = 1.0  # force reject
    result = run_paper_loop(
        _bars(),
        cfg,
        champion_weights=_weights(),
        initial_nav=100_000.0,
        max_steps=2,
    )
    assert result.metrics["risk_gate_rejects"] >= 1


def test_shadow_vs_champion_divergence(tmp_path):
    cfg = _cfg(tmp_path)
    result = run_paper_loop(
        _bars(),
        cfg,
        champion_weights=_weights(1.0),
        shadow_weights=_weights(0.0),  # flat shadow
        initial_nav=100_000.0,
        max_steps=2,
    )
    assert result.metrics["has_shadow"] is True
    assert result.divergence["max_l1"] > 0.0
