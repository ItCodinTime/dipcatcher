"""Python facade over the price-time matching core.

Nanosecond timestamps are caller-supplied integers. The engine is
deterministic given the same event stream. It is a simulator: orders
never leave the process.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from types import TracebackType

from quant_fund.market_sim.native import (
    AuditC,
    EventC,
    EventResultC,
    UncrossResultC,
    load_library,
)

EV_LIMIT = 1
EV_MARKET = 2
EV_CANCEL = 3
EV_IOC = 4
EV_FOK = 5

_STATUS = {
    1: "rejected",
    2: "filled",
    3: "partial_rest",
    4: "resting",
    5: "cancelled",
    6: "ioc_done",
    7: "fok_unfilled",
}
_REASON = {
    0: "",
    1: "qty",
    2: "price",
    3: "book_full",
    4: "unknown_order",
    5: "halted_immediate",
    6: "not_halted",
    7: "bad_type",
}


@dataclass(frozen=True)
class Trade:
    ts_ns: int
    price_tick: int
    qty: int
    aggressor_side: int
    taker_agent: int
    maker_agent: int
    taker_order_id: int
    maker_order_id: int


@dataclass(frozen=True)
class SubmitResult:
    status: str
    reason: str
    order_id: int
    filled_qty: int
    resting_qty: int
    filled_notional_ticks: int
    trades: tuple[Trade, ...]


@dataclass(frozen=True)
class AuctionResult:
    price_tick: int
    trades: tuple[Trade, ...]
    n_moo_cancelled: int


@dataclass(frozen=True)
class BookAudit:
    bid_qty: int
    ask_qty: int
    trade_qty: int
    trade_notional: int
    checksum: int
    n_bid_orders: int
    n_ask_orders: int
    n_live: int
    best_bid: int
    best_ask: int
    crossed: bool
    halted: bool
    n_trades: int
    list_ok: bool


class OrderBook:
    """Price-time priority book for one instrument.

    Limit prices are integer ticks in ``[1, price_max)``. Time priority at a
    price is insertion order. A nanosecond timestamp is stored on each order
    and copied onto prints; it does not reorder the queue.
    """

    def __init__(self, *, debug: bool = False) -> None:
        self._lib: ctypes.CDLL | None = None
        self._ptr: int | None = None
        self.debug = debug
        lib = load_library()
        self._lib = lib
        ptr = lib.lob_new()
        if not ptr:
            raise MemoryError("lob_new failed")
        self._ptr = int(ptr)
        self.price_max = int(lib.lob_price_max())

    def close(self) -> None:
        ptr = self._ptr
        lib = self._lib
        if ptr is not None and lib is not None:
            lib.lob_free(ptr)
            self._ptr = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return

    def __enter__(self) -> OrderBook:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def halted(self) -> bool:
        return self.audit().halted

    def _dll(self) -> ctypes.CDLL:
        lib = self._lib
        if lib is None or self._ptr is None:
            raise RuntimeError("order book is closed")
        return lib

    def _require(self) -> int:
        self._dll()
        if self._ptr is None:
            raise RuntimeError("order book is closed")
        return self._ptr

    def _trades(self) -> tuple[Trade, ...]:
        n = ctypes.c_int32()
        ptr = self._dll().lob_trades(self._require(), ctypes.byref(n))
        count = int(n.value)
        if count <= 0 or not ptr:
            return ()
        out: list[Trade] = []
        for i in range(count):
            item = ptr[i]
            out.append(
                Trade(
                    ts_ns=int(item.ts),
                    price_tick=int(item.price),
                    qty=int(item.qty),
                    aggressor_side=int(item.aggressor_side),
                    taker_agent=int(item.taker_agent),
                    maker_agent=int(item.maker_agent),
                    taker_order_id=int(item.taker_order_id),
                    maker_order_id=int(item.maker_order_id),
                )
            )
        return tuple(out)

    def _check(self, truncated: int) -> None:
        if truncated:
            raise RuntimeError("trade log truncated; agent inventory would be wrong")
        if self.debug:
            audit = self.audit()
            if not audit.list_ok:
                raise RuntimeError(f"book invariant failed: {audit}")

    def _submit(
        self,
        ev_type: int,
        side: int,
        price_tick: int,
        qty: int,
        ts_ns: int,
        agent: int,
        order_id: int = 0,
    ) -> SubmitResult:
        if ts_ns < 0 or ts_ns > 2**63 - 1:
            raise ValueError("ts_ns must fit in a non-negative int64")
        ev = EventC(
            ts=int(ts_ns),
            type=int(ev_type),
            side=int(side),
            price=int(price_tick),
            qty=int(qty),
            agent=int(agent),
            _pad=0,
            order_id=int(order_id),
        )
        result = EventResultC()
        self._dll().lob_submit(self._require(), ctypes.byref(ev), ctypes.byref(result))
        trades = self._trades()
        self._check(int(result.truncated))
        return SubmitResult(
            status=_STATUS.get(int(result.status), "rejected"),
            reason=_REASON.get(int(result.reason), "unknown"),
            order_id=int(result.order_id),
            filled_qty=int(result.filled_qty),
            resting_qty=int(result.resting_qty),
            filled_notional_ticks=int(result.filled_notional),
            trades=trades,
        )

    def limit(self, side: int, price_tick: int, qty: int, ts_ns: int, agent: int) -> SubmitResult:
        """Rest a limit order, matching anything it crosses first."""
        return self._submit(EV_LIMIT, side, price_tick, qty, ts_ns, agent)

    def market(self, side: int, qty: int, ts_ns: int, agent: int) -> SubmitResult:
        """Market order. Unfilled size is cancelled. Nothing rests."""
        return self._submit(EV_MARKET, side, 0, qty, ts_ns, agent)

    def ioc(self, side: int, price_tick: int, qty: int, ts_ns: int, agent: int) -> SubmitResult:
        """Immediate-or-cancel limit. Unfilled size is cancelled."""
        return self._submit(EV_IOC, side, price_tick, qty, ts_ns, agent)

    def fok(self, side: int, price_tick: int, qty: int, ts_ns: int, agent: int) -> SubmitResult:
        """Fill-or-kill. Either the full quantity trades or nothing does."""
        return self._submit(EV_FOK, side, price_tick, qty, ts_ns, agent)

    def cancel(self, order_id: int, ts_ns: int) -> SubmitResult:
        return self._submit(EV_CANCEL, 0, 0, 0, ts_ns, 0, order_id=order_id)

    def order_qty(self, order_id: int) -> int:
        """Remaining shares, or -1 if the id is not live."""
        return int(self._dll().lob_order_qty(self._require(), int(order_id)))

    def halt(self) -> None:
        """Stop continuous matching. Limits rest, including through the opposite side."""
        self._dll().lob_halt(self._require(), 1)

    def touch(self) -> tuple[int | None, int | None, int, int]:
        """Best bid, best ask, and the quantities at those prices.

        ``None`` means that side of the book is empty. Sentinel market-on-open
        orders used during a halt are not the touch.
        """
        bid = ctypes.c_int32()
        ask = ctypes.c_int32()
        bq = ctypes.c_int64()
        aq = ctypes.c_int64()
        self._dll().lob_touch(
            self._require(),
            ctypes.byref(bid),
            ctypes.byref(ask),
            ctypes.byref(bq),
            ctypes.byref(aq),
        )
        bid_px = int(bid.value)
        ask_px = int(ask.value)
        return (
            None if bid_px < 0 else bid_px,
            None if ask_px < 0 else ask_px,
            int(bq.value),
            int(aq.value),
        )

    def depth(self, side: int, n_levels: int = 5) -> int:
        """Total shares on the first ``n_levels`` occupied prices from the touch."""
        return int(self._dll().lob_depth(self._require(), int(side), int(n_levels)))

    def level_qty(self, side: int, price_tick: int) -> int:
        return int(self._dll().lob_level_qty(self._require(), int(side), int(price_tick)))

    def mid_tick(self) -> float | None:
        bid, ask, _, _ = self.touch()
        if bid is None or ask is None:
            return None
        return 0.5 * (bid + ask)

    def spread_ticks(self) -> int | None:
        bid, ask, _, _ = self.touch()
        if bid is None or ask is None:
            return None
        return ask - bid

    def uncross(self, ts_ns: int, ref_tick: int) -> AuctionResult:
        """Open a halted book with a single-price call auction.

        The clearing price maximises matched volume. Ties break toward the
        reference price, then toward the lower price. The book then resumes
        continuous trading. Market orders resting through the halt are
        market-on-open and are cancelled if they do not trade.
        """
        if ts_ns < 0:
            raise ValueError("ts_ns must be non-negative")
        out = UncrossResultC()
        self._dll().lob_uncross(self._require(), int(ts_ns), int(ref_tick), ctypes.byref(out))
        if int(out.status) == 1:
            raise RuntimeError("uncross requires a halted book")
        if int(out.status) != 0:
            raise RuntimeError("uncross failed")
        trades = self._trades()
        self._check(int(out.truncated))
        return AuctionResult(
            price_tick=int(out.price),
            trades=trades,
            n_moo_cancelled=int(out.n_moo_cancelled),
        )

    def audit(self) -> BookAudit:
        raw = AuditC()
        self._dll().lob_audit(self._require(), ctypes.byref(raw))
        return BookAudit(
            bid_qty=int(raw.bid_qty),
            ask_qty=int(raw.ask_qty),
            trade_qty=int(raw.trade_qty),
            trade_notional=int(raw.trade_notional),
            checksum=int(raw.checksum),
            n_bid_orders=int(raw.n_bid_orders),
            n_ask_orders=int(raw.n_ask_orders),
            n_live=int(raw.n_live),
            best_bid=int(raw.best_bid),
            best_ask=int(raw.best_ask),
            crossed=bool(raw.crossed),
            halted=bool(raw.halted),
            n_trades=int(raw.n_trades),
            list_ok=bool(raw.list_ok),
        )
