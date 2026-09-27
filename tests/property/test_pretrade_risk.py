"""Hard limits commute, never pass, and the circuit breaker stays latched."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.pretrade.codes import HARD_LIMIT_MASK, KIND_ORDER, decision_allowed
from quant_fund.pretrade.spec import HARD_PREDICATES, HardView, hard_limit_bits
from tests.unit.pretrade.support import TS_NS, arm, make_engine, order

_FINITE = dict(allow_nan=False, allow_infinity=False)


def _engine_for(
    *,
    qty: float,
    px: float,
    pos: float,
    ref: float,
    side: int,
    is_limit: int,
    max_qty: float,
    max_notional: float,
    max_pos_qty: float,
    max_pos_notional: float,
    max_gross: float,
    max_net: float,
) -> tuple[int, HardView]:
    engine = make_engine(
        nav=100_000.0,
        cash=1e12,
        max_order_quantity=max_qty,
        max_order_notional=max_notional,
        max_position_quantity=max_pos_qty,
        max_position_notional=max_pos_notional,
        max_gross_notional=max_gross,
        max_net_notional=max_net,
        max_name_concentration=1.0,
        price_collar_bps=100.0,
        cash_account=False,
        restrict_to_settled_cash=False,
        duplicate_window_ns=1,
        max_orders_per_window=40,
        max_messages_per_window=40,
    )
    sid = arm(engine, ref=ref, pos=pos, bid=ref)
    sym = engine.book.syms[sid]
    view = HardView(
        qty=qty,
        px=px,
        side=side,
        is_limit=is_limit,
        pos=pos,
        ref=ref,
        ref_ok=1,
        hi=sym.hi,
        lo=sym.lo,
        gross=engine.book.gross,
        net=engine.book.net,
        max_qty=engine.book.max_qty,
        max_notional=engine.book.max_notional,
        max_pos_qty=engine.book.max_pos_qty,
        max_pos_notional=engine.book.max_pos_notional,
        max_gross=engine.book.max_gross,
        max_net=engine.book.max_net,
        name_limit=engine.book.name_limit,
    )
    bits = engine.check(
        order(sid, side=side, qty=qty, px=px, ts_ns=TS_NS, is_limit=is_limit),
        apply=False,
    )
    return bits, view


@given(
    qty=st.floats(0.2, 80, **_FINITE),
    px=st.floats(1.0, 250, **_FINITE),
    pos=st.floats(-40, 40, **_FINITE),
    ref=st.floats(1.0, 250, **_FINITE),
    side=st.sampled_from((1, -1)),
    is_limit=st.sampled_from((0, 1)),
    max_qty=st.floats(0.5, 100, **_FINITE),
    max_notional=st.floats(1.0, 50_000, **_FINITE),
    max_pos_qty=st.floats(1.0, 100, **_FINITE),
    max_pos_notional=st.floats(1.0, 50_000, **_FINITE),
    max_gross=st.floats(1.0, 80_000, **_FINITE),
    max_net=st.floats(1.0, 80_000, **_FINITE),
    perm=st.permutations(tuple(range(len(HARD_PREDICATES)))),
)
@settings(max_examples=40, deadline=None)
def test_hard_limits_never_pass_and_commute(
    qty: float,
    px: float,
    pos: float,
    ref: float,
    side: int,
    is_limit: int,
    max_qty: float,
    max_notional: float,
    max_pos_qty: float,
    max_pos_notional: float,
    max_gross: float,
    max_net: float,
    perm: tuple[int, ...],
) -> None:
    bits, view = _engine_for(
        qty=qty,
        px=px,
        pos=pos,
        ref=ref,
        side=side,
        is_limit=is_limit,
        max_qty=max_qty,
        max_notional=max_notional,
        max_pos_qty=max_pos_qty,
        max_pos_notional=max_pos_notional,
        max_gross=max_gross,
        max_net=max_net,
    )
    spec = hard_limit_bits(view)
    assert hard_limit_bits(view, tuple(perm)) == spec
    assert bits & HARD_LIMIT_MASK == spec
    if spec:
        assert decision_allowed(bits, KIND_ORDER) is False


@given(depth=st.floats(0.02, 0.8, **_FINITE))
@settings(max_examples=20, deadline=None)
def test_circuit_breaker_latches_until_reset(depth: float) -> None:
    start = 1_000.0
    engine = make_engine(
        nav=start,
        cash=start,
        max_daily_loss_fraction=0.01,
        max_trailing_drawdown_fraction=0.99,
    )
    sid = arm(engine)
    stressed = start * (1.0 - 0.01) - depth
    engine.update_account(nav=stressed, mark_ts_ns=TS_NS, session_start_nav=start, peak_nav=start)
    first = engine.check(order(sid, qty=1, px=100), apply=False)
    assert not decision_allowed(first, KIND_ORDER)
    assert engine.book.killed == 1
    engine.update_account(nav=start, mark_ts_ns=TS_NS + 5, session_start_nav=start, peak_nav=start)
    second = engine.check(order(sid, qty=2, px=100, ts_ns=TS_NS + 5), apply=False)
    assert engine.book.killed == 1
    assert not decision_allowed(second, KIND_ORDER)
    engine.reset("risk", "manual clear after review", TS_NS + 6)
    third = engine.check(order(sid, qty=3, px=100, ts_ns=TS_NS + 6), apply=False)
    assert third == 0
    assert engine.book.killed == 0


def test_full_check_sequence_is_deterministic() -> None:
    def run() -> list[int]:
        engine = make_engine(duplicate_window_ns=10_000, max_orders_per_window=10)
        sid = arm(engine)
        return [
            engine.check(order(sid, qty=1 + index * 0.1, ts_ns=TS_NS + index * 100), apply=False)
            for index in range(6)
        ]

    assert run() == run()
