"""Scenario library for the agent market.

Each scenario is a hook on the same seeded ecology. Effects are orders and
halts inside the simulator. There is no exchange connection and no live
order. Reports are simulation diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from quant_fund.market_sim.agents import FundamentalAgent
from quant_fund.market_sim.config import EVIDENCE, EcologyConfig, drought_config
from quant_fund.market_sim.simulator import (
    SimResult,
    Simulator,
    WeightFunction,
    run_ecology,
)

SCENARIOS = (
    "flash_crash",
    "halt_auction",
    "liquidity_drought",
    "gap_open",
    "crowded_unwind",
)


@dataclass
class _Hook:
    name: str
    fired: bool = False
    extra: dict[str, float | int | str | None] = field(default_factory=dict)

    def on_step(self, sim: Simulator) -> None:
        raise NotImplementedError

    def finish(self, sim: Simulator) -> dict[str, float | int | str | None]:
        mid = sim.book.mid_tick()
        self.extra["final_mid_tick"] = None if mid is None else float(mid)
        self.extra["scenario"] = self.name
        return dict(self.extra)


class FlashCrash(_Hook):
    """Market makers pull quotes and a market sell walks the bid."""

    def __init__(self) -> None:
        super().__init__("flash_crash")
        self.trigger = 0
        self.resume = 0

    def on_step(self, sim: Simulator) -> None:
        if self.trigger == 0:
            self.trigger = max(10, int(0.35 * sim.cfg.max_events))
            self.resume = self.trigger + max(20, int(0.08 * sim.cfg.max_events))
        if sim.event_index == self.trigger:
            bid, _ask, _bq, _aq = sim.book.touch()
            depth = sim.book.depth(1, 30)
            self.extra["mid_before"] = sim.book.mid_tick()
            self.extra["bid_before"] = -1 if bid is None else bid
            self.extra["depth_before"] = depth
            pulled = sim.cancel_kind("mm")
            self.extra["mm_orders_pulled"] = pulled
            sim.paused_kinds.add("mm")
            shock = max(depth + 5, 40)
            self.extra["shock_qty"] = shock
            result = sim.inject("market", -1, shock)
            self.extra["shock_filled"] = result.filled_qty
            self.extra["shock_last_price"] = result.trades[-1].price_tick if result.trades else -1
            self.fired = True
        if sim.event_index == self.resume:
            sim.paused_kinds.discard("mm")
            self.extra["mm_resumed"] = 1

    def finish(self, sim: Simulator) -> dict[str, float | int | str | None]:
        out = super().finish(sim)
        before = out.get("mid_before")
        after = out.get("final_mid_tick")
        last = out.get("shock_last_price")
        if isinstance(before, float) and isinstance(last, int) and last > 0:
            out["shock_drop_ticks"] = float(before) - float(last)
        if isinstance(before, float) and isinstance(after, float) and isinstance(last, int):
            drop = float(before) - float(last)
            if drop > 0.0 and after is not None:
                out["recovery_fraction"] = (float(after) - float(last)) / drop
        return out


class HaltAuction(_Hook):
    """Continuous trading stops, interest accumulates, a call auction reopens."""

    def __init__(self) -> None:
        super().__init__("halt_auction")
        self.trigger = 0
        self.reopen = 0
        self.trades_at_halt = 0
        self._halted = False

    def on_step(self, sim: Simulator) -> None:
        if self.trigger == 0:
            self.trigger = max(10, int(0.4 * sim.cfg.max_events))
            self.reopen = self.trigger + max(30, int(0.1 * sim.cfg.max_events))
        if sim.event_index == self.trigger:
            self.trades_at_halt = sim.book.audit().n_trades
            mid = sim.book.mid_tick() or float(sim.cfg.initial_mid_tick)
            center = int(round(mid))
            sim.book.halt()
            self._halted = True
            # Crossed interest so the auction has something to clear even
            # before agents wake: a bid through the old ask and an ask
            # through the old bid, plus size at the touch.
            sim.inject("limit", 1, 25, price_tick=center + 2, agent=900_010)
            sim.inject("limit", -1, 25, price_tick=center - 2, agent=900_011)
            self.extra["halt_center"] = center
            self.fired = True
        if self._halted and sim.event_index < self.reopen:
            # IOC/FOK are rejected by the engine while halted. Limits rest.
            self.extra["still_halted"] = 1
        if sim.event_index == self.reopen and self._halted:
            trades_during = sim.book.audit().n_trades - self.trades_at_halt
            self.extra["trades_while_halted"] = trades_during
            center = int(self.extra.get("halt_center") or sim.cfg.initial_mid_tick)
            auction = sim.book.uncross(sim.clock_ns, center)
            for trade in auction.trades:
                sim._apply_trade(trade)
            self._halted = False
            self.extra["auction_price"] = auction.price_tick
            self.extra["auction_trades"] = len(auction.trades)
            self.extra["auction_qty"] = sum(trade.qty for trade in auction.trades)
            self.fired = True


class GapOpen(_Hook):
    """Halt, jump the latent price, reopen in a call auction around the new level."""

    def __init__(self, gap_ticks: int = 400) -> None:
        super().__init__("gap_open")
        self.gap_ticks = gap_ticks
        self.trigger = 0
        self.reopen = 0
        self._halted = False

    def on_step(self, sim: Simulator) -> None:
        if self.trigger == 0:
            self.trigger = max(10, int(0.3 * sim.cfg.max_events))
            self.reopen = self.trigger + max(25, int(0.08 * sim.cfg.max_events))
        if sim.event_index == self.trigger:
            old = sim.book.mid_tick()
            self.extra["mid_before"] = None if old is None else float(old)
            sim.cancel_all()
            sim.book.halt()
            self._halted = True
            for agent in sim.agents:
                if isinstance(agent, FundamentalAgent):
                    agent.value += self.gap_ticks
            new_center = int(round(old or sim.cfg.initial_mid_tick)) + self.gap_ticks
            new_center = min(sim.book.price_max - 2, max(2, new_center))
            sim.anchor_override = new_center
            sim.force_anchor = True
            for i in range(1, 8):
                sim.inject("limit", 1, 12, price_tick=new_center - i, agent=900_020)
                sim.inject("limit", -1, 12, price_tick=new_center + i, agent=900_021)
            # A small crossed quantity so the print is not exactly the
            # symmetric mid if agents add imbalance while halted.
            sim.inject("limit", 1, 8, price_tick=new_center + 1, agent=900_022)
            self.extra["new_center"] = new_center
            self.fired = True
        if sim.event_index == self.reopen and self._halted:
            center = int(self.extra.get("new_center") or sim.cfg.initial_mid_tick)
            auction = sim.book.uncross(sim.clock_ns, center)
            for trade in auction.trades:
                sim._apply_trade(trade)
            sim.force_anchor = False
            self._halted = False
            self.extra["auction_price"] = auction.price_tick
            self.extra["auction_qty"] = sum(trade.qty for trade in auction.trades)
            before = self.extra.get("mid_before")
            if isinstance(before, float) and auction.price_tick:
                self.extra["gap_ticks"] = float(auction.price_tick) - before


class CrowdedUnwind(_Hook):
    """A buy burst lifts the book, then a larger sell burst takes it back."""

    def __init__(self) -> None:
        super().__init__("crowded_unwind")
        self.entry = 0
        self.exit = 0
        self.peak: float | None = None

    def on_step(self, sim: Simulator) -> None:
        if self.entry == 0:
            self.entry = max(10, int(0.4 * sim.cfg.max_events))
            self.exit = self.entry + max(15, int(0.05 * sim.cfg.max_events))
        if sim.event_index == self.entry:
            depth = sim.book.depth(-1, 20)
            qty = max(30, depth + 10)
            before = sim.book.mid_tick()
            result = sim.inject("market", 1, qty)
            self.extra["entry_qty"] = qty
            self.extra["entry_filled"] = result.filled_qty
            self.extra["mid_before"] = None if before is None else float(before)
            after_entry = sim.book.mid_tick()
            if after_entry is not None:
                self.peak = after_entry
            self.fired = True
        if self.entry < sim.event_index < self.exit:
            mid = sim.book.mid_tick()
            if mid is not None and (self.peak is None or mid > self.peak):
                self.peak = mid
        if sim.event_index == self.exit:
            depth = sim.book.depth(1, 25)
            qty = max(40, depth + 15)
            result = sim.inject("market", -1, qty)
            self.extra["exit_qty"] = qty
            self.extra["exit_filled"] = result.filled_qty
            self.extra["exit_last"] = result.trades[-1].price_tick if result.trades else -1
            self.extra["peak_mid"] = None if self.peak is None else float(self.peak)


def run_scenario(
    name: str,
    cfg: EcologyConfig | None = None,
    *,
    strategy: WeightFunction | None = None,
) -> SimResult:
    """Run one named scenario. ``liquidity_drought`` changes the population."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name}")
    base = cfg or EcologyConfig()
    if name == "liquidity_drought":
        return run_ecology(drought_config(base), strategy=strategy)
    hooks: dict[str, _Hook] = {
        "flash_crash": FlashCrash(),
        "halt_auction": HaltAuction(),
        "gap_open": GapOpen(),
        "crowded_unwind": CrowdedUnwind(),
    }
    return run_ecology(base, strategy=strategy, hook=hooks[name])


def scenario_spread(cfg: EcologyConfig | None = None) -> dict[str, float | int | str | bool]:
    """Median spread of the drought population and of the baseline, same seed."""
    base = cfg or EcologyConfig()
    normal = run_ecology(base)
    dry = run_ecology(drought_config(base))

    def _median(values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        return float(np.median(values))

    out: dict[str, float | int | str | bool] = dict(EVIDENCE)
    out.update(
        {
            "baseline_median_spread_ticks": _median(normal.spreads),
            "drought_median_spread_ticks": _median(dry.spreads),
            "baseline_n": int(normal.spreads.size),
            "drought_n": int(dry.spreads.size),
        }
    )
    return out
