"""Random valid order streams keep the price-time book consistent."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.market_sim.book import OrderBook


@given(
    st.lists(
        st.tuples(
            st.sampled_from(["limit", "market", "ioc", "fok", "cancel", "halt", "uncross"]),
            st.sampled_from([1, -1]),
            st.integers(90, 110),
            st.integers(1, 8),
        ),
        min_size=1,
        max_size=25,
    )
)
@settings(max_examples=40, deadline=None)
def test_random_ops_keep_list_and_touch(ops: list[tuple[str, int, int, int]]) -> None:
    book = OrderBook(debug=True)
    live: list[int] = []
    try:
        ts = 1_000
        for kind, side, price, qty in ops:
            ts += 1
            if kind == "halt":
                book.halt()
            elif kind == "uncross":
                if book.halted:
                    book.uncross(ts, 100)
            elif kind == "cancel":
                if live:
                    book.cancel(live.pop(0), ts)
            elif kind == "limit":
                result = book.limit(side, price, qty, ts, 7)
                if result.status in {"resting", "partial_rest"}:
                    live.append(result.order_id)
            elif kind == "market":
                result = book.market(side, qty, ts, 7)
                if result.status in {"resting", "partial_rest"}:
                    live.append(result.order_id)
            elif kind == "ioc":
                book.ioc(side, price, qty, ts, 7)
            else:
                book.fok(side, price, qty, ts, 7)
            audit = book.audit()
            assert audit.list_ok
            bid, ask, _, _ = book.touch()
            if bid is not None and ask is not None and not audit.halted:
                assert bid < ask
            if not audit.halted:
                assert not audit.crossed
    finally:
        book.close()
