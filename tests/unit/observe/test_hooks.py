"""Passive hooks are off by default and do not change simulated execution."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.observe.install import install_passive_hooks, uninstall_passive_hooks, wrap_callable
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus

_PRODUCTION = (
    "src/quant_fund/data/ingest.py",
    "src/quant_fund/features/engine.py",
    "src/quant_fund/pipeline/forecast/decide.py",
    "src/quant_fund/portfolio/risk_gate.py",
    "src/quant_fund/execution/simulated_broker.py",
    "src/quant_fund/paper/ledger.py",
    "src/quant_fund/research/verify.py",
)


def test_production_modules_do_not_reference_audit_or_observe() -> None:
    for relative in _PRODUCTION:
        text = Path(relative).read_text(encoding="utf-8")
        assert "quant_fund.observe" not in text
        assert "quant_fund.audit" not in text


def test_disabled_install_keeps_function_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
    uninstall_passive_hooks()
    original = SimulatedBroker.submit

    def add(value: int) -> int:
        return value + 1

    assert wrap_callable(add, "ingest") is add
    assert install_passive_hooks() == 0
    assert SimulatedBroker.submit is original


def test_wrapper_preserves_return_and_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")

    def add(left: int, right: int) -> int:
        return left + right

    assert wrap_callable(add, "features")(2, 3) == 5

    def explode() -> None:
        raise KeyError("same")

    with pytest.raises(KeyError, match="same"):
        wrap_callable(explode, "decision")()


def test_wrapped_submit_matches_direct_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    monkeypatch.delenv("DIPCATCHER_METRICS_PORT", raising=False)
    uninstall_passive_hooks()
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    stamp = datetime(2024, 1, 2, tzinfo=UTC)
    order = Order(
        order_id="o1",
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=10.0,
        signal_time=stamp,
        decision_time=stamp,
        order_time=stamp,
        status=OrderStatus.NEW,
    )
    direct = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    direct.mark({"A": 100.0})
    direct_record = direct.submit(order, price=100.0, nav=100_000.0, adv_dollars=1e9, sigma=0.02)
    try:
        wrapped_count = install_passive_hooks()
        assert wrapped_count >= 5
        assert getattr(SimulatedBroker.submit, "__audit_wrapped__", False)
        wrapped = SimulatedBroker(config=cfg, initial_cash=100_000.0)
        wrapped.mark({"A": 100.0})
        wrapped_record = wrapped.submit(
            order, price=100.0, nav=100_000.0, adv_dollars=1e9, sigma=0.02
        )
        assert wrapped_record.order.status == direct_record.order.status
        assert wrapped_record.fill is not None
        assert direct_record.fill is not None
        assert wrapped_record.fill.price == direct_record.fill.price
        assert wrapped_record.fill.quantity == direct_record.fill.quantity
        assert wrapped.cash == direct.cash
        assert wrapped.shares == direct.shares
        assert wrapped.reject_count == direct.reject_count
    finally:
        uninstall_passive_hooks()
    assert not getattr(SimulatedBroker.submit, "__audit_wrapped__", False)


def test_cli_import_follows_the_flag() -> None:
    off = (
        "import os\n"
        "os.environ.pop('DIPCATCHER_OBSERVE', None)\n"
        "os.environ.pop('DIPCATCHER_METRICS_PORT', None)\n"
        "from quant_fund.execution.simulated_broker import SimulatedBroker\n"
        "original = SimulatedBroker.submit\n"
        "import quant_fund.cli.main\n"
        "assert SimulatedBroker.submit is original\n"
    )
    subprocess.check_call([sys.executable, "-c", off])
    on = (
        "import os\n"
        "os.environ['DIPCATCHER_OBSERVE'] = '1'\n"
        "os.environ.pop('DIPCATCHER_METRICS_PORT', None)\n"
        "import quant_fund.cli.main\n"
        "from quant_fund.execution.simulated_broker import SimulatedBroker\n"
        "assert getattr(SimulatedBroker.submit, '__audit_wrapped__', False)\n"
    )
    subprocess.check_call([sys.executable, "-c", on])
