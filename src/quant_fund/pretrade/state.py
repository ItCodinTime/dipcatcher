"""Preallocated hot-path state. Rings and lot tables are fixed at build time."""

from __future__ import annotations

from quant_fund.pretrade.config import PretradeConfig


class SymbolSlot:
    __slots__ = (
        "pos",
        "ref",
        "hi",
        "lo",
        "deadline",
        "ref_ok",
        "halted",
        "sho",
        "locate",
        "bid",
        "opened_session",
        "lq",
        "ls",
        "lu",
        "nlot",
    )

    def __init__(self, max_lots: int) -> None:
        self.pos = 0.0
        self.ref = 0.0
        self.hi = 0.0
        self.lo = 0.0
        self.deadline = 0
        self.ref_ok = 0
        self.halted = 0
        self.sho = 0
        self.locate = 0
        self.bid = 0.0
        self.opened_session = 0
        self.lq = [0.0] * max_lots
        self.ls = [0] * max_lots
        self.lu = [0] * max_lots
        self.nlot = 0


class HotBook:
    """Mutable check state. The allow path writes into these preallocated slots."""

    __slots__ = (
        "syms",
        "max_lots",
        "nav",
        "gross",
        "net",
        "settled",
        "session_start",
        "peak",
        "daily_floor",
        "dd_floor",
        "mark_deadline",
        "name_limit",
        "pdt_tight",
        "killed",
        "kill_reason",
        "pending_trip",
        "last_ts",
        "session_open",
        "open_ns",
        "close_ns",
        "day_lo",
        "day_hi",
        "max_qty",
        "max_notional",
        "max_pos_qty",
        "max_pos_notional",
        "max_gross",
        "max_net",
        "max_orders",
        "max_msgs",
        "window_ns",
        "dup_window",
        "pdt_max",
        "pdt_window",
        "cash_account",
        "restrict_settled",
        "settlement_ns",
        "qty_tick",
        "price_tick",
        "collar_frac",
        "ots",
        "o_head",
        "o_tail",
        "o_live",
        "mts",
        "m_head",
        "m_tail",
        "m_live",
        "ring_cap",
        "ring_mask",
        "h_fp",
        "h_ts",
        "r_fp",
        "r_ts",
        "r_head",
        "r_tail",
        "r_live",
        "dt_sess",
        "dt_n",
        "dt_cap",
        "uc_amt",
        "uc_ts",
        "uc_head",
        "uc_tail",
        "uc_live",
        "uc_cap",
        "uc_mask",
    )

    def __init__(self, config: PretradeConfig) -> None:
        limits = config.limits
        cap = limits.ring_capacity
        lots = limits.max_lots_per_symbol
        self.syms = [SymbolSlot(lots) for _ in range(limits.symbol_capacity)]
        self.max_lots = lots
        self.nav = 0.0
        self.gross = 0.0
        self.net = 0.0
        self.settled = 0.0
        self.session_start = 0.0
        self.peak = 0.0
        self.daily_floor = 0.0
        self.dd_floor = 0.0
        self.mark_deadline = 0
        self.name_limit = 0.0
        self.pdt_tight = 1
        self.killed = 0
        self.kill_reason = 0
        self.pending_trip = 0
        self.last_ts = 0
        self.session_open = 0
        self.open_ns = 0
        self.close_ns = 0
        self.day_lo = 0
        self.day_hi = 0
        self.max_qty = float(limits.max_order_quantity)
        self.max_notional = float(limits.max_order_notional)
        self.max_pos_qty = float(limits.max_position_quantity)
        self.max_pos_notional = float(limits.max_position_notional)
        self.max_gross = float(limits.max_gross_notional)
        self.max_net = float(limits.max_net_notional)
        self.max_orders = int(limits.max_orders_per_window)
        self.max_msgs = int(limits.max_messages_per_window)
        self.window_ns = int(limits.rate_window_ns)
        self.dup_window = int(limits.duplicate_window_ns)
        self.pdt_max = int(limits.pdt_max_day_trades)
        self.pdt_window = int(limits.pdt_window_sessions)
        self.cash_account = 1 if limits.cash_account else 0
        self.restrict_settled = 1 if limits.restrict_to_settled_cash else 0
        self.settlement_ns = int(limits.settlement_ns)
        self.qty_tick = float(limits.qty_tick)
        self.price_tick = float(limits.price_tick)
        self.collar_frac = float(limits.price_collar_bps) / 10_000.0
        self.ots = [0] * cap
        self.o_head = 0
        self.o_tail = 0
        self.o_live = 0
        self.mts = [0] * cap
        self.m_head = 0
        self.m_tail = 0
        self.m_live = 0
        self.ring_cap = cap
        self.ring_mask = cap - 1
        self.h_fp = [0] * cap
        self.h_ts = [0] * cap
        self.r_fp = [0] * cap
        self.r_ts = [0] * cap
        self.r_head = 0
        self.r_tail = 0
        self.r_live = 0
        dt_cap = limits.day_trade_capacity
        self.dt_sess = [0] * dt_cap
        self.dt_n = 0
        self.dt_cap = dt_cap
        uc = limits.unsettled_capacity
        self.uc_amt = [0.0] * uc
        self.uc_ts = [0] * uc
        self.uc_head = 0
        self.uc_tail = 0
        self.uc_live = 0
        self.uc_cap = uc
        self.uc_mask = uc - 1
