"""Shadow adapter leaves the simulated broker's results unchanged."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.paper.loop import run_paper_loop
from quant_fund.pretrade.config import load_pretrade_config
from quant_fund.pretrade.engine import PretradeEngine
from quant_fund.pretrade.shadow import ShadowRiskAdapter, business_session_id
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus
from tests.unit.pretrade.support import HMAC_KEY

ROOT = Path(__file__).resolve().parents[3]
WHEN = datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York"))


def _broker_config(tmp_path: Path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 3.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.frictionless = True
    return cfg


def _order(qty: float, oid: str) -> Order:
    return Order(
        order_id=oid,
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=qty,
        signal_time=WHEN,
        decision_time=WHEN,
        order_time=WHEN,
        status=OrderStatus.NEW,
    )


def _submit_all(broker: SimulatedBroker) -> list[tuple[str, str | None, float | None]]:
    rows = []
    for index, qty in enumerate((10.0, 5.0, 1_000_000.0)):
        record = broker.submit(
            _order(qty, f"o{index}"),
            price=100.0,
            nav=broker.nav(),
            adv_dollars=1e12,
            sigma=0.02,
        )
        fill_px = None if record.fill is None else float(record.fill.price)
        rows.append((record.order.status.value, record.reject_reason, fill_px))
    return rows


def test_attach_does_not_change_broker_results(tmp_path: Path) -> None:
    plain = SimulatedBroker(config=_broker_config(tmp_path / "plain"), initial_cash=100_000)
    plain.mark({"A": 100.0})
    plain_rows = _submit_all(plain)

    observed = SimulatedBroker(config=_broker_config(tmp_path / "obs"), initial_cash=100_000)
    observed.mark({"A": 100.0})
    config, _, _ = load_pretrade_config(ROOT / "configs" / "pretrade_risk.yaml", hmac_key=HMAC_KEY)
    adapter = ShadowRiskAdapter(config, hmac_key=HMAC_KEY, log_path=tmp_path / "shadow.jsonl")
    adapter.attach(observed)
    observed_rows = _submit_all(observed)
    adapter.close()

    assert observed_rows == plain_rows
    assert observed.cash == pytest.approx(plain.cash)
    assert observed.shares == plain.shares
    assert len(adapter.events) == 3
    assert all(event["behavior_changed"] is False for event in adapter.events)
    assert all(event["config_sha256"] == adapter.config_sha256 for event in adapter.events)
    assert all(
        event["config_hmac_sha256"] == adapter.config_hmac_sha256 for event in adapter.events
    )
    assert adapter.events[-1]["broker_reject_reason"] is not None
    lines = (tmp_path / "shadow.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line)["config_sha256"] == adapter.config_sha256 for line in lines)


def test_observer_exception_does_not_block_the_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("shadow failed")

    monkeypatch.setattr(PretradeEngine, "decide", boom)
    broker = SimulatedBroker(config=_broker_config(tmp_path), initial_cash=100_000)
    broker.mark({"A": 100.0})
    config, _, _ = load_pretrade_config(ROOT / "configs" / "pretrade_risk.yaml", hmac_key=HMAC_KEY)
    adapter = ShadowRiskAdapter(config, hmac_key=HMAC_KEY)
    adapter.attach(broker)
    record = broker.submit(
        _order(10.0, "x"),
        price=100.0,
        nav=broker.nav(),
        adv_dollars=1e12,
        sigma=0.02,
    )
    assert record.fill is not None
    assert broker.shares["A"] == pytest.approx(10.0)
    assert adapter.events[-1]["allowed"] is False
    assert adapter.events[-1]["reasons"] == ["internal_error"]
    assert adapter.events[-1]["config_sha256"] == adapter.config_sha256
    assert adapter.errors


def test_cancel_and_amend_are_logged_without_changing_the_broker(tmp_path: Path) -> None:
    def resting(oid: str, qty: float = 10.0) -> Order:
        return _order(qty, oid).model_copy(update={"limit_price": 90.0})

    def drive(broker: SimulatedBroker) -> tuple[str, str, str]:
        submitted = broker.submit(
            resting("lim"),
            price=100.0,
            nav=broker.nav(),
            adv_dollars=1e12,
            sigma=0.02,
        )
        amended = broker.amend_order("lim", quantity=8.0, limit_price=91.0)
        cancelled = broker.cancel_order("lim")
        with pytest.raises(ValueError, match="not working"):
            broker.cancel_order("missing")
        return (
            submitted.order.status.value,
            amended.order.status.value,
            cancelled.order.status.value,
        )

    plain = SimulatedBroker(config=_broker_config(tmp_path / "plain"), initial_cash=100_000)
    plain.mark({"A": 100.0})
    plain_rows = drive(plain)

    observed = SimulatedBroker(config=_broker_config(tmp_path / "obs"), initial_cash=100_000)
    observed.mark({"A": 100.0})
    config, _, _ = load_pretrade_config(ROOT / "configs" / "pretrade_risk.yaml", hmac_key=HMAC_KEY)
    adapter = ShadowRiskAdapter(config, hmac_key=HMAC_KEY, log_path=tmp_path / "shadow.jsonl")
    adapter.attach(observed)
    with pytest.raises(RuntimeError, match="already"):
        adapter.attach(observed)
    observed_rows = drive(observed)
    adapter.close()

    assert observed_rows == plain_rows
    assert observed.shares == plain.shares
    assert observed.cash == pytest.approx(plain.cash)
    kinds = [event["kind"] for event in adapter.events]
    assert kinds == ["order", "replace", "cancel", "cancel"]
    assert all(event["behavior_changed"] is False for event in adapter.events)
    assert adapter.events[-1]["reasons"] == ["unknown_symbol"]
    assert adapter.events[-1]["allowed"] is False
    lines = [
        json.loads(line)
        for line in (tmp_path / "shadow.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(line.get("kind") == "broker_result" for line in lines)


def test_class_patch_rejects_a_second_observer_and_instance_attach(tmp_path: Path) -> None:
    config, _, _ = load_pretrade_config(ROOT / "configs" / "pretrade_risk.yaml", hmac_key=HMAC_KEY)
    adapter = ShadowRiskAdapter(config, hmac_key=HMAC_KEY)
    broker = SimulatedBroker(config=_broker_config(tmp_path), initial_cash=10_000)
    with adapter.observe_simulated_broker():
        with pytest.raises(RuntimeError, match="already observed"):
            adapter.attach(broker)
        with pytest.raises(RuntimeError, match="already observed"):  # noqa: SIM117
            with adapter.observe_simulated_broker():
                pass
    adapter.attach(broker)


def test_business_session_id_folds_the_weekend_onto_friday() -> None:
    from datetime import date

    friday = business_session_id(date(2024, 1, 5))
    assert business_session_id(date(2024, 1, 6)) == friday
    assert business_session_id(date(2024, 1, 7)) == friday
    assert business_session_id(date(2024, 1, 8)) == friday + 1


def test_class_observer_is_restored_and_paper_loop_is_unchanged(tmp_path: Path) -> None:
    import polars as pl

    from quant_fund.execution import simulated_broker as broker_mod

    original = broker_mod.SimulatedBroker.submit
    config, _, _ = load_pretrade_config(ROOT / "configs" / "pretrade_risk.yaml", hmac_key=HMAC_KEY)
    adapter = ShadowRiskAdapter(config, hmac_key=HMAC_KEY)

    def run(root: Path):
        cfg = _broker_config(root)
        times = [datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2, 3, 4)]
        rows = []
        for stamp in times:
            for security, px in (("A", 100.0), ("B", 50.0)):
                rows.append(
                    {
                        "security_id": security,
                        "event_time": stamp,
                        "open": px,
                        "close": px,
                        "close_total_return": px,
                        "volume": 1_000_000.0,
                        "adv": 100_000_000.0,
                        "vol_20": 0.02,
                        "source": "synthetic",
                    }
                )
        bars = pl.DataFrame(rows)
        weights = pl.DataFrame(
            [
                {"event_time": stamp, "security_id": security, "target_weight": weight}
                for stamp in times[:3]
                for security, weight in (("A", 0.1), ("B", -0.05))
            ]
        )
        return run_paper_loop(
            bars,
            cfg,
            champion_weights=weights,
            shadow_weights=weights,
            initial_nav=100_000.0,
            max_steps=3,
        )

    alone = run(tmp_path / "alone")
    with adapter.observe_simulated_broker():
        assert broker_mod.SimulatedBroker.submit is not original
        watched = run(tmp_path / "watched")
    assert broker_mod.SimulatedBroker.submit is original
    assert watched.metrics["n_fills"] == alone.metrics["n_fills"]
    assert watched.metrics["reject_total"] == alone.metrics["reject_total"]
    assert watched.champion_equity["nav"].to_list() == alone.champion_equity["nav"].to_list()
    assert adapter.events
    assert all(event["mode"] == "shadow" for event in adapter.events)
    assert all(event["config_sha256"] == adapter.config_sha256 for event in adapter.events)
    source = Path(broker_mod.__file__).read_text(encoding="utf-8")
    loop_source = Path(run_paper_loop.__code__.co_filename).read_text(encoding="utf-8")
    for body in (source, loop_source):
        assert "quant_fund.pretrade" not in body
        assert "ShadowRiskAdapter" not in body
