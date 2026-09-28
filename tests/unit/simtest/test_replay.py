"""Bit-for-bit replay of a multi-day paper session."""

from __future__ import annotations

from datetime import UTC, datetime

from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus
from quant_fund.simtest.eventlog import EventLog
from quant_fund.simtest.faults import Fault, FaultSchedule
from quant_fund.simtest.invariants import check_invariants
from quant_fund.simtest.session import run_session


def test_record_twice_and_replay_share_every_state_hash() -> None:
    schedule = FaultSchedule(())
    first = run_session(11, n_days=4, schedule=schedule)
    second = run_session(11, n_days=4, schedule=schedule)
    replay = run_session(11, n_days=4, schedule=schedule, replay_log=first.log_bytes)

    assert first.state_hashes
    assert first.state_hashes == second.state_hashes == replay.state_hashes
    assert first.log_bytes == second.log_bytes
    assert first.cash == replay.cash
    assert first.shares == replay.shares
    assert first.order_ids == replay.order_ids
    assert first.fill_ids == replay.fill_ids
    assert EventLog.from_bytes(first.log_bytes).to_bytes() == first.log_bytes
    report = check_invariants(first)
    assert report.ok, report.failures
    assert first.research_only is True
    assert first.live_pnl_claim is False
    assert first.data_source == "SYNTHETIC"
    assert first.idempotent_resume is True
    assert first.ledger_ok is True
    assert first.receipt_errors == []


def test_faulted_session_replays_and_differs_from_a_clean_run() -> None:
    clean = run_session(3, n_days=4, schedule=FaultSchedule(()))
    schedule = FaultSchedule((Fault(step=0, kind="partial_fill", target="AAA", magnitude=0.25),))
    faulted = run_session(3, n_days=4, schedule=schedule)
    replay = run_session(3, n_days=4, schedule=schedule, replay_log=faulted.log_bytes)

    assert faulted.state_hashes == replay.state_hashes
    assert faulted.state_hashes != clean.state_hashes
    assert check_invariants(faulted).ok
    faulted_qty = [
        float(item["fill_qty"]) for item in faulted.fills if item["security_id"] == "AAA"
    ]
    clean_qty = [float(item["fill_qty"]) for item in clean.fills if item["security_id"] == "AAA"]
    assert faulted_qty and clean_qty
    assert faulted_qty[0] < clean_qty[0]


def test_default_broker_ids_stay_uuid_shaped(tmp_path) -> None:
    from quant_fund.config.loader import load_config

    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    moment = datetime(2024, 1, 2, tzinfo=UTC)
    order = Order(
        order_id="ignored",
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=1.0,
        signal_time=moment,
        decision_time=moment,
        order_time=moment,
        status=OrderStatus.NEW,
    )
    minted = broker.target_to_orders(
        {"A": 0.01}, {"A": 100.0}, signal_time=moment, order_time=moment
    )
    assert minted[0].order_id.startswith("champion-")
    assert len(minted[0].order_id.split("-", 1)[1]) == 10
    record = broker.submit(order, price=100.0, nav=100_000.0, adv_dollars=1e9, sigma=0.02)
    assert record.fill is not None
    assert record.fill.fill_id.startswith("fill-")
    assert len(record.fill.fill_id.removeprefix("fill-")) == 12
