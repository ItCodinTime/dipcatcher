"""Event-time simulator that wakes heterogeneous agents against one book.

Timestamps are integer nanoseconds. The run is a pure function of
``EcologyConfig.seed`` plus the optional strategy and hooks. Cash and
shares are accounted inside the process. Nothing is routed to a broker.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.random import Generator

from quant_fund.market_sim.agents import (
    Action,
    Agent,
    ExecutionAgent,
    FundamentalAgent,
    InformedAgent,
    MarketMaker,
    MarketView,
    MeanRevertAgent,
    MomentumAgent,
    NoiseAgent,
)
from quant_fund.market_sim.book import OrderBook, SubmitResult, Trade
from quant_fund.market_sim.config import EVIDENCE, EcologyConfig

SEED_AGENT = 900_001
STRATEGY_AGENT = 900_003
META_AGENT = 910_000


class WeightFunction(Protocol):
    def __call__(self, closes: np.ndarray) -> float: ...


class MarketHook(Protocol):
    def on_step(self, sim: Simulator) -> None: ...

    def finish(self, sim: Simulator) -> dict[str, float | int | str | None]: ...


@dataclass(frozen=True)
class Metaorder:
    """Scheduled child orders injected by the impact experiment."""

    start_event: int
    side: int
    qty: int
    slice_qty: int
    every: int
    agent_id: int = META_AGENT


@dataclass(frozen=True)
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Account:
    cash_ticks: int = 0
    position: int = 0

    def apply(self, price_tick: int, qty: int, side: int) -> None:
        self.position += int(side) * int(qty)
        self.cash_ticks -= int(side) * int(price_tick) * int(qty)

    def mtm_ticks(self, mid_tick: float) -> float:
        return float(self.cash_ticks) + float(self.position) * float(mid_tick)


@dataclass
class FillRec:
    event_index: int
    agent: int
    price_tick: int
    qty: int
    side: int


@dataclass
class SimResult:
    """Synthetic tape. Mark-to-market fields are simulation accounting."""

    seed: int
    n_events: int
    n_trades: int
    returns: np.ndarray
    spreads: np.ndarray
    depths: np.ndarray
    bars: list[Bar]
    checksum: int
    cash_ticks_sum: int
    position_sum: int
    final_mid_tick: float | None
    strategy_equity: np.ndarray
    strategy_slippage_bps: float
    strategy_filled_qty: int
    strategy_requested_qty: int
    fills: list[FillRec] = field(default_factory=list)
    hook: dict[str, float | int | str | None] = field(default_factory=dict)

    @property
    def research_only(self) -> bool:
        return True

    @property
    def live_pnl_claim(self) -> bool:
        return False

    def summary(self) -> dict[str, float | int | str | bool | None]:
        out: dict[str, float | int | str | bool | None] = dict(EVIDENCE)
        out.update(
            {
                "seed": self.seed,
                "n_events": self.n_events,
                "n_trades": self.n_trades,
                "n_returns": int(self.returns.size),
                "n_spread_samples": int(self.spreads.size),
                "n_bars": len(self.bars),
                "checksum": self.checksum,
                "cash_ticks_sum": self.cash_ticks_sum,
                "position_sum": self.position_sum,
                "final_mid_tick": self.final_mid_tick,
                "strategy_slippage_bps": self.strategy_slippage_bps,
                "strategy_filled_qty": self.strategy_filled_qty,
                "strategy_requested_qty": self.strategy_requested_qty,
            }
        )
        return out


def build_agents(cfg: EcologyConfig, rng: Generator, price_max: int) -> list[Agent]:
    """Construct the population. Parameter draws happen in a fixed order."""
    agents: list[Agent] = []
    next_id = 1
    agents.append(
        FundamentalAgent(
            kind="fundamental",
            agent_id=next_id,
            rate=cfg.fundamental_rate,
            value=cfg.initial_mid_tick,
            base_sigma=cfg.fundamental_sigma,
            price_max=price_max,
        )
    )
    next_id += 1
    lookbacks = (8, 16, 32, 64, 128, 48)
    for _i in range(cfg.n_mm):
        agents.append(
            MarketMaker(
                kind="mm",
                agent_id=next_id,
                rate=cfg.mm_rate,
                gamma=cfg.mm_gamma,
                k=cfg.mm_k,
                size=cfg.mm_size,
                inventory_cap=cfg.mm_inventory_cap,
            )
        )
        next_id += 1
    for i in range(cfg.n_momentum):
        agents.append(
            MomentumAgent(
                kind="momentum",
                agent_id=next_id,
                rate=cfg.momentum_rate,
                lookback=lookbacks[i % len(lookbacks)],
                size=3 + (i % 3),
            )
        )
        next_id += 1
    for i in range(cfg.n_mean_revert):
        agents.append(
            MeanRevertAgent(
                kind="mean_revert",
                agent_id=next_id,
                rate=cfg.mean_revert_rate,
                lookback=20 + 5 * (i % 4),
            )
        )
        next_id += 1
    for i in range(cfg.n_noise):
        agents.append(
            NoiseAgent(
                kind="noise",
                agent_id=next_id,
                rate=cfg.noise_rate,
                limit_prob=cfg.noise_limit_prob,
                market_prob=cfg.noise_market_prob,
                cancel_prob=cfg.noise_cancel_prob,
                size_mu=cfg.noise_size_mu,
                size_sigma=cfg.noise_size_sigma,
                offset_lo=cfg.noise_offset_lo,
                offset_hi=cfg.noise_offset_hi,
                persistence=float(rng.uniform(0.45, 0.92)),
                last_side=1 if i % 2 == 0 else -1,
            )
        )
        next_id += 1
    for i in range(cfg.n_informed):
        agents.append(
            InformedAgent(
                kind="informed",
                agent_id=next_id,
                rate=cfg.informed_rate,
                noise_ticks=1.5 + float(i),
                _fundamental=cfg.initial_mid_tick,
            )
        )
        next_id += 1
    start = max(1, int(cfg.execution_start_frac * cfg.max_events))
    for _i in range(cfg.n_execution):
        side = 1 if float(rng.random()) < 0.5 else -1
        agents.append(
            ExecutionAgent(
                kind="execution",
                agent_id=next_id,
                rate=cfg.execution_rate,
                target_qty=cfg.execution_qty,
                slice_qty=max(1, cfg.execution_qty // max(1, cfg.execution_slices)),
                side=side,
                start_event=start,
            )
        )
        next_id += 1
    return agents


class Simulator:
    """One run. ``hook.on_step`` may halt, cancel, or inject orders."""

    def __init__(
        self,
        cfg: EcologyConfig,
        *,
        strategy: WeightFunction | None = None,
        hook: MarketHook | None = None,
        metaorders: tuple[Metaorder, ...] = (),
        debug_book: bool = False,
    ) -> None:
        self.cfg = cfg
        self.strategy = strategy
        self.hook = hook
        self.metaorders = metaorders
        self.meta_sent: dict[int, int] = {i: 0 for i in range(len(metaorders))}
        self.rng = np.random.default_rng(cfg.seed)
        self.book = OrderBook(debug=debug_book)
        self.agents = build_agents(cfg, self.rng, self.book.price_max)
        self.accounts: dict[int, Account] = {}
        self.paused_kinds: set[str] = set()
        self.force_anchor = False
        self.anchor_override: int | None = None
        self.event_index = 0
        self.clock_ns = 0
        self.last_trade_tick: int | None = None
        self.seed_ids: list[int] = []
        self._heap: list[tuple[int, int, int, int]] = []
        self.mid_history: list[int] = []
        self.trailing_var = 1.0
        self._seq = 1
        self._returns: list[float] = []
        self._spreads: list[int] = []
        self._depths: list[int] = []
        self._bars: list[Bar] = []
        self._bar_prices: list[float] = []
        self._bar_volume = 0
        self._prev_sample_mid: float | None = None
        self._trade_marks = 0
        self._closes: list[float] = []
        self._strategy_equity: list[float] = []
        self._slip_notional = 0.0
        self._slip_cost = 0.0
        self._strategy_filled = 0
        self._strategy_requested = 0
        self._decision_mid: float | None = None
        self.fills: list[FillRec] = []
        if strategy is not None:
            cash = int(round(cfg.strategy_nav / cfg.tick_size))
            self.accounts[STRATEGY_AGENT] = Account(cash_ticks=cash, position=0)
        self._seed_book()

    def close(self) -> None:
        self.book.close()

    @property
    def fundamental(self) -> int:
        for agent in self.agents:
            if isinstance(agent, FundamentalAgent):
                return agent.value
        return self.cfg.initial_mid_tick

    def _seed_book(self) -> None:
        mid = self.cfg.initial_mid_tick
        for i in range(1, 13):
            for side, price in ((1, mid - i), (-1, mid + i)):
                result = self.book.limit(side, price, 10, 0, SEED_AGENT)
                if result.status in {"resting", "partial_rest"}:
                    self.seed_ids.append(result.order_id)
        self.mid_history.append(mid)

    def _anchor(self) -> int:
        if self.anchor_override is not None:
            return self.anchor_override
        mid = self.book.mid_tick()
        if mid is not None:
            return int(round(mid))
        if self.last_trade_tick is not None:
            return self.last_trade_tick
        if self.mid_history:
            return self.mid_history[-1]
        return self.cfg.initial_mid_tick

    def _view(self, agent: Agent) -> MarketView:
        bid, ask, bq, aq = self.book.touch()
        mid = self.book.mid_tick()
        spread = self.book.spread_ticks()
        account = self.accounts.get(agent.agent_id, Account())
        return MarketView(
            bid=bid,
            ask=ask,
            bid_qty=bq,
            ask_qty=aq,
            mid=mid,
            spread=spread,
            anchor=self._anchor(),
            mid_history=self.mid_history,
            trailing_var=self.trailing_var,
            halted=self.book.halted,
            inventory=account.position,
            event_index=self.event_index,
            force_anchor=self.force_anchor,
            price_max=self.book.price_max,
        )

    def _remember_mid(self) -> None:
        mid = self.book.mid_tick()
        if mid is None:
            return
        tick = int(round(mid))
        self.mid_history.append(tick)
        if len(self.mid_history) > 512:
            del self.mid_history[:-512]
        if len(self.mid_history) >= 8:
            window = np.asarray(self.mid_history[-33:], dtype=float)
            changes = np.diff(window)
            var = float(np.var(changes))
            self.trailing_var = max(var, 0.25)

    def _account(self, agent_id: int) -> Account:
        acc = self.accounts.get(agent_id)
        if acc is None:
            acc = Account()
            self.accounts[agent_id] = acc
        return acc

    def _apply_trade(self, trade: Trade) -> None:
        if trade.aggressor_side == 0:
            self._account(trade.taker_agent).apply(trade.price_tick, trade.qty, 1)
            self._account(trade.maker_agent).apply(trade.price_tick, trade.qty, -1)
            self._record_fill(trade.taker_agent, trade, 1)
            self._record_fill(trade.maker_agent, trade, -1)
        else:
            self._account(trade.taker_agent).apply(
                trade.price_tick, trade.qty, trade.aggressor_side
            )
            self._account(trade.maker_agent).apply(
                trade.price_tick, trade.qty, -trade.aggressor_side
            )
            self._record_fill(trade.taker_agent, trade, trade.aggressor_side)
            self._record_fill(trade.maker_agent, trade, -trade.aggressor_side)
        for agent in self.agents:
            if isinstance(agent, ExecutionAgent) and agent.agent_id == trade.taker_agent:
                agent.filled += trade.qty
        self.last_trade_tick = trade.price_tick
        price = trade.price_tick * self.cfg.tick_size
        self._bar_prices.append(price)
        self._bar_volume += trade.qty
        self._sample_return()
        if trade.qty >= self.cfg.hawkes_qty and self.cfg.hawkes_extra > 0 and not self.book.halted:
            self._excite_noise(trade.qty)

    def _record_fill(self, agent_id: int, trade: Trade, side: int) -> None:
        if agent_id == STRATEGY_AGENT and self._decision_mid is not None and side != 0:
            mid = self._decision_mid
            adverse = (trade.price_tick - mid) if side > 0 else (mid - trade.price_tick)
            self._slip_cost += adverse * trade.qty
            self._slip_notional += mid * trade.qty
            self._strategy_filled += trade.qty
        self.fills.append(
            FillRec(
                event_index=self.event_index,
                agent=agent_id,
                price_tick=trade.price_tick,
                qty=trade.qty,
                side=side,
            )
        )

    def _sample_return(self) -> None:
        self._trade_marks += 1
        if self.event_index < self.cfg.warmup_events:
            return
        if self._trade_marks % self.cfg.return_stride != 0:
            return
        mid = self.book.mid_tick()
        if mid is None or mid <= 0.0:
            return
        if self._prev_sample_mid is not None and self._prev_sample_mid > 0.0:
            self._returns.append(float(np.log(mid / self._prev_sample_mid)))
        self._prev_sample_mid = mid

    def _excite_noise(self, qty: int) -> None:
        del qty
        noise = [i for i, agent in enumerate(self.agents) if agent.kind == "noise"]
        if not noise:
            return
        delays = (1_000_000, 4_000_000, 9_000_000)
        for k in range(self.cfg.hawkes_extra):
            index = noise[(self.event_index + k) % len(noise)]
            wake = self.clock_ns + delays[k % len(delays)]
            heapq.heappush(self._heap, (wake, self._seq, index, 0))
            self._seq += 1

    def _apply_action(self, agent: Agent, action: Action) -> SubmitResult:
        ts = self.clock_ns
        if action.kind == "cancel":
            return self.book.cancel(action.order_id, ts)
        if action.kind == "limit":
            result = self.book.limit(action.side, action.price_tick, action.qty, ts, agent.agent_id)
        elif action.kind == "market":
            result = self.book.market(action.side, action.qty, ts, agent.agent_id)
        elif action.kind == "ioc":
            result = self.book.ioc(action.side, action.price_tick, action.qty, ts, agent.agent_id)
        elif action.kind == "fok":
            result = self.book.fok(action.side, action.price_tick, action.qty, ts, agent.agent_id)
        else:
            raise ValueError(f"unknown action {action.kind}")
        for trade in result.trades:
            self._apply_trade(trade)
        if result.status in {"resting", "partial_rest"} and result.order_id:
            agent.live_ids.append(result.order_id)
        return result

    def inject(
        self,
        kind: str,
        side: int,
        qty: int,
        *,
        price_tick: int = 0,
        agent: int = SEED_AGENT,
    ) -> SubmitResult:
        """Submit an exogenous order. Used by scenarios, still inside the book."""
        action = Action(kind, side=side, price_tick=price_tick, qty=qty)
        holder = Agent(kind="inject", agent_id=agent, rate=1.0)
        return self._apply_action(holder, action)

    def cancel_kind(self, kind: str) -> int:
        n = 0
        for agent in self.agents:
            if agent.kind != kind:
                continue
            for order_id in list(agent.live_ids):
                result = self.book.cancel(order_id, self.clock_ns)
                if result.status == "cancelled":
                    n += 1
            agent.live_ids.clear()
        return n

    def cancel_all(self) -> int:
        """Cancel tracked resting orders, including the opening ladder."""
        n = 0
        for kind in ("mm", "momentum", "mean_revert", "noise", "informed", "execution"):
            n += self.cancel_kind(kind)
        for order_id in list(self.seed_ids):
            result = self.book.cancel(order_id, self.clock_ns)
            if result.status == "cancelled":
                n += 1
        self.seed_ids.clear()
        return n

    def _act(self, agent: Agent) -> None:
        if isinstance(agent, InformedAgent):
            agent.observe(self.fundamental)
        view = self._view(agent)
        for action in agent.propose(view, self.rng):
            self._apply_action(agent, action)

    def _meta(self) -> None:
        for i, order in enumerate(self.metaorders):
            if self.event_index < order.start_event:
                continue
            sent = self.meta_sent.get(i, 0)
            if sent >= order.qty:
                continue
            if (self.event_index - order.start_event) % order.every != 0:
                continue
            qty = min(order.slice_qty, order.qty - sent)
            self.meta_sent[i] = sent + qty
            self.inject("market", order.side, qty, agent=order.agent_id)

    def _strategy_bar(self) -> None:
        if self.event_index % self.cfg.bar_events != 0:
            return
        self._close_bar()
        if self.strategy is None:
            return
        mid = self.book.mid_tick()
        if mid is None:
            return
        closes = np.asarray(self._closes, dtype=float)
        if closes.size == 0:
            return
        weight = float(self.strategy(closes))
        if not np.isfinite(weight):
            raise ValueError("strategy weight must be finite")
        cap = self.cfg.strategy_max_weight
        weight = float(np.clip(weight, -cap, cap))
        account = self._account(STRATEGY_AGENT)
        equity_ticks = account.mtm_ticks(mid)
        if equity_ticks <= 0.0:
            return
        target = int(round(weight * equity_ticks / mid))
        delta = target - account.position
        self._decision_mid = mid
        equity = equity_ticks * self.cfg.tick_size
        self._strategy_equity.append(equity)
        if delta == 0:
            return
        self._strategy_requested += abs(delta)
        self.inject("market", 1 if delta > 0 else -1, abs(delta), agent=STRATEGY_AGENT)

    def _close_bar(self) -> None:
        mid = self.book.mid_tick()
        if self._bar_prices:
            prices = self._bar_prices
            bar = Bar(
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=self._bar_volume,
            )
        elif mid is not None:
            price = mid * self.cfg.tick_size
            bar = Bar(open=price, high=price, low=price, close=price, volume=0)
        else:
            self._bar_prices = []
            self._bar_volume = 0
            return
        self._bars.append(bar)
        self._closes.append(bar.close)
        self._bar_prices = []
        self._bar_volume = 0

    def _sample_book(self) -> None:
        if self.event_index < self.cfg.warmup_events:
            return
        if self.event_index % self.cfg.sample_every != 0:
            return
        if self.book.halted:
            return
        spread = self.book.spread_ticks()
        if spread is None or spread <= 0:
            return
        depth = self.book.depth(1, self.cfg.depth_levels) + self.book.depth(
            -1, self.cfg.depth_levels
        )
        self._spreads.append(int(spread))
        self._depths.append(int(depth))

    def run(self) -> SimResult:
        self._heap = []
        for index, agent in enumerate(self.agents):
            heapq.heappush(self._heap, (agent.wait_ns(self.rng), self._seq, index, 1))
            self._seq += 1
        try:
            while self._heap and self.event_index < self.cfg.max_events:
                ts, _seq, index, reschedule = heapq.heappop(self._heap)
                self.clock_ns = int(ts)
                self.event_index += 1
                if self.hook is not None:
                    self.hook.on_step(self)
                agent = self.agents[index]
                if agent.kind not in self.paused_kinds:
                    self._act(agent)
                before_meta = getattr(self.hook, "before_meta", None)
                if before_meta is not None:
                    before_meta(self)
                self._meta()
                self._strategy_bar()
                self._remember_mid()
                self._sample_book()
                if reschedule and self.event_index < self.cfg.max_events:
                    wake = self.clock_ns + agent.wait_ns(self.rng)
                    heapq.heappush(self._heap, (wake, self._seq, index, 1))
                    self._seq += 1
            if self._bar_prices or not self._bars:
                self._close_bar()
            if self.strategy is not None and STRATEGY_AGENT in self.accounts:
                mid = self.book.mid_tick()
                if mid is not None:
                    ticks = self.accounts[STRATEGY_AGENT].mtm_ticks(mid)
                    self._strategy_equity.append(ticks * self.cfg.tick_size)
            hook_out: dict[str, float | int | str | None] = {}
            if self.hook is not None:
                hook_out = self.hook.finish(self)
            audit = self.book.audit()
            cash = sum(acc.cash_ticks for acc in self.accounts.values())
            pos = sum(acc.position for acc in self.accounts.values())
            slip = 0.0
            if self._slip_notional > 0.0:
                slip = self._slip_cost / self._slip_notional * 1e4
            return SimResult(
                seed=self.cfg.seed,
                n_events=self.event_index,
                n_trades=audit.n_trades,
                returns=np.asarray(self._returns, dtype=float),
                spreads=np.asarray(self._spreads, dtype=float),
                depths=np.asarray(self._depths, dtype=float),
                bars=list(self._bars),
                checksum=audit.checksum,
                cash_ticks_sum=int(cash),
                position_sum=int(pos),
                final_mid_tick=self.book.mid_tick(),
                strategy_equity=np.asarray(self._strategy_equity, dtype=float),
                strategy_slippage_bps=float(slip),
                strategy_filled_qty=self._strategy_filled,
                strategy_requested_qty=self._strategy_requested,
                fills=list(self.fills),
                hook=hook_out,
            )
        finally:
            self.close()


def run_ecology(
    cfg: EcologyConfig | None = None,
    *,
    strategy: WeightFunction | None = None,
    hook: MarketHook | None = None,
    metaorders: tuple[Metaorder, ...] = (),
    debug_book: bool = False,
) -> SimResult:
    """Run one seeded ecology. The book is closed before this returns."""
    sim = Simulator(
        cfg or EcologyConfig(),
        strategy=strategy,
        hook=hook,
        metaorders=metaorders,
        debug_book=debug_book,
    )
    return sim.run()
