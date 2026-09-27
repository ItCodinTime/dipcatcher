"""Price-time matching core. Simulation only."""

from __future__ import annotations

import pytest

from quant_fund.market_sim.book import OrderBook
from quant_fund.market_sim.native import core_version


def test_core_version() -> None:
    assert core_version() == "lob-core-1"


def test_price_time_priority() -> None:
    book = OrderBook(debug=True)
    first = book.limit(1, 100, 5, 1, 1)
    second = book.limit(1, 100, 7, 1, 2)
    sell = book.market(-1, 5, 2, 3)
    assert sell.filled_qty == 5
    assert sell.trades[0].maker_agent == 1
    assert sell.trades[0].maker_order_id == first.order_id
    assert book.order_qty(first.order_id) == -1
    assert book.order_qty(second.order_id) == 7
    assert book.level_qty(1, 100) == 7
    book.close()


def test_same_timestamp_keeps_submission_order() -> None:
    book = OrderBook(debug=True)
    book.limit(1, 50, 2, 10, 11)
    book.limit(1, 50, 2, 10, 12)
    sell = book.market(-1, 2, 10, 13)
    assert sell.trades[0].maker_agent == 11
    book.close()


def test_nanosecond_timestamp_round_trip() -> None:
    book = OrderBook(debug=True)
    ts = 1_700_000_000_000_000_000
    book.limit(-1, 100, 1, ts, 1)
    filled = book.market(1, 1, ts, 2)
    assert filled.trades[0].ts_ns == ts
    book.close()


def test_ioc_does_not_rest() -> None:
    book = OrderBook(debug=True)
    book.limit(-1, 100, 3, 1, 1)
    result = book.ioc(1, 100, 10, 2, 2)
    assert result.status == "ioc_done"
    assert result.filled_qty == 3
    assert result.resting_qty == 0
    assert book.level_qty(-1, 100) == 0
    assert book.touch()[0] is None
    book.close()


def test_fok_does_not_partially_fill() -> None:
    book = OrderBook(debug=True)
    book.limit(-1, 100, 3, 1, 1)
    before = book.level_qty(-1, 100)
    result = book.fok(1, 100, 10, 2, 2)
    assert result.status == "fok_unfilled"
    assert result.filled_qty == 0
    assert result.trades == ()
    assert book.level_qty(-1, 100) == before
    book.close()


def test_cancel_removes_the_order() -> None:
    book = OrderBook(debug=True)
    resting = book.limit(1, 90, 4, 1, 1)
    cancelled = book.cancel(resting.order_id, 2)
    assert cancelled.status == "cancelled"
    assert book.order_qty(resting.order_id) == -1
    assert book.level_qty(1, 90) == 0
    missing = book.cancel(resting.order_id, 3)
    assert missing.reason == "unknown_order"
    book.close()


def test_halt_suppresses_continuous_trades() -> None:
    book = OrderBook(debug=True)
    book.limit(1, 100, 5, 1, 1)
    book.limit(-1, 101, 5, 1, 2)
    book.halt()
    trades = book.audit().n_trades
    market = book.market(-1, 5, 2, 3)
    assert market.status == "resting"
    assert market.filled_qty == 0
    immediate = book.ioc(1, 101, 1, 3, 4)
    assert immediate.status == "rejected"
    assert immediate.reason == "halted_immediate"
    crossed = book.limit(1, 105, 2, 4, 5)
    assert crossed.status == "resting"
    assert book.audit().n_trades == trades
    book.close()


def test_auction_clears_at_the_maximum_volume_price() -> None:
    """Bid 10 @ 105; asks 4 @ 100 and 6 @ 103; reference 100.

    Volume is 10 from 103 through 105. The closest price to 100 is 103.
    """
    book = OrderBook(debug=True)
    book.halt()
    book.limit(1, 105, 10, 1, 1)
    book.limit(-1, 100, 4, 1, 2)
    book.limit(-1, 103, 6, 1, 3)
    auction = book.uncross(10, 100)
    assert auction.price_tick == 103
    assert sum(trade.qty for trade in auction.trades) == 10
    assert all(trade.price_tick == 103 for trade in auction.trades)
    assert all(trade.aggressor_side == 0 for trade in auction.trades)
    assert book.audit().n_trades == len(auction.trades)
    assert not book.halted
    audit = book.audit()
    assert audit.list_ok
    assert not audit.crossed
    book.close()


def test_rejects_bad_price_and_qty_and_closed_book() -> None:
    book = OrderBook(debug=True)
    assert book.limit(1, 0, 1, 1, 1).reason == "price"
    assert book.limit(1, 100, 0, 1, 1).reason == "qty"
    with pytest.raises(RuntimeError, match="halted"):
        book.uncross(1, 100)
    book.close()
    with pytest.raises(RuntimeError, match="closed"):
        book.limit(1, 100, 1, 1, 1)


def test_two_streams_share_a_checksum() -> None:
    def _run() -> int:
        book = OrderBook(debug=True)
        book.limit(1, 100, 4, 1, 1)
        book.limit(-1, 101, 4, 1, 2)
        book.market(1, 2, 2, 3)
        book.limit(-1, 103, 6, 3, 4)
        checksum = book.audit().checksum
        book.close()
        return checksum

    assert _run() == _run()
    assert _run() != 0
