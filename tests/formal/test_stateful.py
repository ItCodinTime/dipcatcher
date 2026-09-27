"""Hypothesis stateful tests: broker cash versus the reference account."""

from datetime import UTC, datetime

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test

from quant_fund.config.loader import load_config
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.formal.accounting import Account
from quant_fund.formal.order_lifecycle import check_trace
from quant_fund.formal.traces import TraceSession
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus


def _permissive(tmp_name: str = "stateful"):
    cfg = load_config("configs/paper.yaml")
    cfg.costs.frictionless = False
    cfg.costs.commission_bps = 1.0
    cfg.costs.half_spread_bps = 2.0
    cfg.costs.impact_y = 0.1
    cfg.costs.bps_per_turnover = 4.0
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_gross = 3.0
    cfg.risk_gate.max_net = 3.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    return cfg


class AccountingMachine(RuleBasedStateMachine):
    """Random buys, sells, marks, and restarts against ``Account``."""

    def __init__(self) -> None:
        super().__init__()
        self.cfg = _permissive()
        self.broker = SimulatedBroker(config=self.cfg, initial_cash=1_000_000.0)
        self.account = Account(initial_cash=1_000_000.0)
        self.marks = {"A": 100.0, "B": 50.0}
        self.broker.mark(self.marks)
        self.n = 0
        self.when = datetime(2024, 6, 3, tzinfo=UTC)

    @rule(
        qty=st.floats(min_value=1.0, max_value=25.0, allow_nan=False, allow_infinity=False),
        buy=st.booleans(),
        name=st.sampled_from(["A", "B"]),
        with_decision=st.booleans(),
    )
    def trade(self, qty: float, buy: bool, name: str, with_decision: bool) -> None:
        side = OrderSide.BUY if buy else OrderSide.SELL
        self.n += 1
        order = Order(
            order_id=f"t{self.n}",
            security_id=name,
            symbol=name,
            side=side,
            quantity=float(qty),
            signal_time=self.when,
            decision_time=self.when,
            order_time=self.when,
            status=OrderStatus.NEW,
        )
        kwargs = {
            "price": self.marks[name],
            "nav": self.broker.nav(self.marks),
            "adv_dollars": 1e12,
            "sigma": 0.02,
        }
        if with_decision:
            kwargs["decision_price"] = self.marks[name] * 0.97
        record = self.broker.submit(order, **kwargs)
        fill = record.fill
        if fill is None:
            return
        signed = fill.quantity if side is OrderSide.BUY else -fill.quantity
        cost = fill.fee + fill.spread_cost + fill.impact_cost + fill.turnover_cost
        self.account.apply(name, signed, fill.price, cost)

    @rule(
        price_a=st.floats(min_value=10.0, max_value=250.0, allow_nan=False, allow_infinity=False),
        price_b=st.floats(min_value=10.0, max_value=250.0, allow_nan=False, allow_infinity=False),
    )
    def mark(self, price_a: float, price_b: float) -> None:
        self.marks = {"A": float(price_a), "B": float(price_b)}
        self.broker.mark(self.marks)

    @rule()
    def restart(self) -> None:
        state = self.broker.to_dict()
        if self.broker.fills:
            duplicated = dict(state)
            duplicated["history"] = list(state["history"]) + [state["history"][-1]]
            duplicated["n_orders"] = len(duplicated["history"])
            restored = SimulatedBroker.from_state(self.cfg, duplicated)
            assert abs(restored.cash - self.broker.cash) < 1e-6
            assert len(restored.fills) == len(self.broker.fills)
        self.broker = SimulatedBroker.from_state(self.cfg, state)

    @invariant()
    def books_match(self) -> None:
        for name, quantity in self.broker.shares.items():
            if abs(quantity) > 1e-12:
                assert name in self.marks
        assert abs(self.account.cash - self.broker.cash) < 1e-6
        assert abs(self.account.nav(self.marks) - self.broker.nav(self.marks)) < 1e-6
        assert abs(self.account.identity_gap(self.marks)) < 1e-6
        for name in ("A", "B"):
            assert abs(self.account.qty.get(name, 0.0) - self.broker.shares.get(name, 0.0)) < 1e-6


class LifecycleMachine(RuleBasedStateMachine):
    """Random limit lifecycle on the simulated broker, checked against the spec."""

    def __init__(self) -> None:
        super().__init__()
        cfg = _permissive()
        cfg.costs.frictionless = True
        cfg.costs.participation_limit = 0.5
        self.session = TraceSession(SimulatedBroker(config=cfg, initial_cash=1_000_000.0))
        self.session.broker.mark({"A": 100.0})
        self.n = 0
        self.when = datetime(2024, 6, 3, tzinfo=UTC)

    def _order(self, qty: float, limit: float, *, expire=None) -> Order:
        self.n += 1
        return Order(
            order_id=f"L{self.n}",
            security_id="A",
            symbol="A",
            side=OrderSide.BUY,
            quantity=qty,
            signal_time=self.when,
            decision_time=self.when,
            order_time=self.when,
            status=OrderStatus.NEW,
            limit_price=limit,
            expire_time=expire,
        )

    @rule(
        qty=st.floats(min_value=1.0, max_value=8.0, allow_nan=False, allow_infinity=False),
        limit=st.floats(min_value=90.0, max_value=110.0, allow_nan=False, allow_infinity=False),
    )
    def rest(self, qty: float, limit: float) -> None:
        self.session.submit(
            self._order(float(qty), float(limit)),
            price=100.0,
            nav=self.session.broker.nav(),
            adv_dollars=800.0,
            sigma=0.0,
        )

    @rule(low=st.floats(min_value=80.0, max_value=120.0, allow_nan=False, allow_infinity=False))
    def sweep(self, low: float) -> None:
        high = max(low, 100.0)
        self.session.process_bar(
            "A",
            bar_open=100.0,
            bar_high=high,
            bar_low=min(low, 100.0),
            bar_time=self.when,
            adv_dollars=800.0,
        )

    @rule()
    def cancel_oldest(self) -> None:
        if not self.session.broker.open_orders:
            return
        order_id = sorted(self.session.broker.open_orders)[0]
        self.session.cancel(order_id)

    @rule(
        qty=st.floats(min_value=1.0, max_value=6.0, allow_nan=False, allow_infinity=False),
    )
    def expire_on_touch(self, qty: float) -> None:
        expire = self.when
        self.session.submit(
            self._order(float(qty), 99.0, expire=expire),
            price=100.0,
            nav=self.session.broker.nav(),
            adv_dollars=1e12,
            sigma=0.0,
        )
        self.session.process_bar(
            "A",
            bar_open=100.0,
            bar_high=101.0,
            bar_low=98.0,
            bar_time=expire,
            adv_dollars=1e12,
        )

    @rule()
    def restart(self) -> None:
        self.session.restart()

    @invariant()
    def trace_is_allowed(self) -> None:
        checked = check_trace(self.session.events)
        assert checked.ok, checked.violations
        for order in checked.book.orders.values():
            assert order.filled <= order.qty + 1e-8


test_accounting_state_machine = run_state_machine_as_test(
    AccountingMachine,
    settings=settings(max_examples=12, stateful_step_count=8, deadline=None),
)
test_lifecycle_state_machine = run_state_machine_as_test(
    LifecycleMachine,
    settings=settings(max_examples=10, stateful_step_count=8, deadline=None),
)
