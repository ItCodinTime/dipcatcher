"""Heterogeneous agents for the limit-order-book simulator.

Each agent proposes orders from public book state. Informed agents also
see a noisy copy of the latent fundamental. Nobody here can reach a broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.random import Generator
from quant_fund.market_sim.quotes import avellaneda_stoikov_quotes


@dataclass(frozen=True)
class Action:
    kind: str
    side: int = 0
    price_tick: int = 0
    qty: int = 0
    order_id: int = 0


@dataclass
class MarketView:
    """Public state at one wake. ``anchor`` is the mid, or a scenario override."""

    bid: int | None
    ask: int | None
    bid_qty: int
    ask_qty: int
    mid: float | None
    spread: int | None
    anchor: int
    mid_history: list[int]
    trailing_var: float
    halted: bool
    inventory: int
    event_index: int
    force_anchor: bool
    price_max: int


@dataclass
class Agent:
    kind: str
    agent_id: int
    rate: float
    live_ids: list[int] = field(default_factory=list)

    def observe(self, fundamental: int) -> None:
        del fundamental

    def propose(self, view: MarketView, rng: Generator) -> list[Action]:
        del view, rng
        return []

    def wait_ns(self, rng: Generator) -> int:
        draw = float(rng.exponential(1.0 / self.rate))
        return max(1, int(draw * 1_000_000_000.0))


def _clamp_price(price: int, price_max: int) -> int:
    return min(price_max - 1, max(1, int(price)))


def _lognormal_size(rng: Generator, mu: float, sigma: float, cap: int = 80) -> int:
    draw = float(rng.lognormal(mu, sigma))
    return max(1, min(cap, int(round(draw))))


@dataclass
class FundamentalAgent(Agent):
    """Latent efficient price. Two volatility regimes, switched rarely."""

    value: int = 20_000
    base_sigma: float = 0.85
    price_max: int = 100_000
    regime: int = 0

    def propose(self, view: MarketView, rng: Generator) -> list[Action]:
        del view
        if float(rng.random()) < 0.02:
            self.regime = 1 - self.regime
        sigma = self.base_sigma if self.regime == 0 else self.base_sigma * 4.0
        shock = int(round(float(rng.normal(0.0, sigma))))
        lo = 1_000
        hi = self.price_max - 1_000
        self.value = min(hi, max(lo, self.value + shock))
        return []


@dataclass
class MarketMaker(Agent):
    """Inventory-aware quote, Avellaneda–Stoikov with a one-unit horizon."""

    gamma: float = 0.45
    k: float = 1.5
    size: int = 6
    inventory_cap: float = 20.0

    def propose(self, view: MarketView, rng: Generator) -> list[Action]:
        del rng
        actions = [Action("cancel", order_id=oid) for oid in self.live_ids]
        self.live_ids.clear()
        center = float(view.anchor if view.force_anchor or view.mid is None else view.mid)
        units = float(view.inventory) / float(self.size)
        bid, ask, _half = avellaneda_stoikov_quotes(
            center, units, self.gamma, self.k, view.trailing_var
        )
        bid = _clamp_price(bid, view.price_max)
        ask = _clamp_price(ask, view.price_max)
        if ask <= bid:
            ask = _clamp_price(bid + 1, view.price_max)
        if units < self.inventory_cap:
            actions.append(Action("limit", side=1, price_tick=bid, qty=self.size))
        if units > -self.inventory_cap:
            actions.append(Action("limit", side=-1, price_tick=ask, qty=self.size))
        return actions


@dataclass
class MomentumAgent(Agent):
    """Trend follower on a fixed mid lookback. Horizons differ across the crowd."""

    lookback: int = 16
    size: int = 4
    threshold: int = 2
    inventory_cap: int = 48

    def propose(self, view: MarketView, rng: Generator) -> list[Action]:
        del rng
        hist = view.mid_history
        if len(hist) <= self.lookback:
            return []
        signal = hist[-1] - hist[-1 - self.lookback]
        if signal >= self.threshold and view.inventory < self.inventory_cap:
            return [Action("market", side=1, qty=self.size)]
        if signal <= -self.threshold and view.inventory > -self.inventory_cap:
            return [Action("market", side=-1, qty=self.size)]
        return []


@dataclass
class MeanRevertAgent(Agent):
    """Trades toward a trailing mid. Does not observe the latent fundamental."""

    lookback: int = 30
    size: int = 3
    band: float = 2.5

    def propose(self, view: MarketView, rng: Generator) -> list[Action]:
        del rng
        hist = view.mid_history
        if len(hist) < self.lookback:
            return []
        window = hist[-self.lookback :]
        mean = float(sum(window)) / float(self.lookback)
        gap = mean - float(hist[-1])
        if gap >= self.band and view.inventory < 40:
            return [Action("market", side=1, qty=self.size)]
        if gap <= -self.band and view.inventory > -40:
            return [Action("market", side=-1, qty=self.size)]
        return []


@dataclass
class NoiseAgent(Agent):
    """Zero-intelligence limit, market, and cancel flow with a persistent sign."""

    limit_prob: float = 0.70
    market_prob: float = 0.18
    cancel_prob: float = 0.12
    size_mu: float = 1.1
    size_sigma: float = 1.05
    offset_lo: int = 0
    offset_hi: int = 5
    persistence: float = 0.6
    last_side: int = 1

    def propose(self, view: MarketView, rng: Generator) -> list[Action]:
        draw = float(rng.random())
        if draw < self.cancel_prob and self.live_ids:
            index = int(rng.integers(0, len(self.live_ids)))
            order_id = self.live_ids.pop(index)
            return [Action("cancel", order_id=order_id)]
        if float(rng.random()) < self.persistence:
            side = self.last_side
        else:
            side = 1 if float(rng.random()) < 0.5 else -1
        self.last_side = side
        qty = _lognormal_size(rng, self.size_mu, self.size_sigma)
        if draw < self.cancel_prob + self.market_prob:
            return [Action("market", side=side, qty=qty)]
        off = int(rng.integers(self.offset_lo, self.offset_hi + 1))
        if side > 0:
            touch = view.bid if view.bid is not None else view.anchor
            price = _clamp_price(touch - off, view.price_max)
        else:
            touch = view.ask if view.ask is not None else view.anchor
            price = _clamp_price(touch + off, view.price_max)
        if len(self.live_ids) >= 4:
            return [Action("cancel", order_id=self.live_ids.pop(0))]
        return [Action("limit", side=side, price_tick=price, qty=qty)]


@dataclass
class InformedAgent(Agent):
    """Trades a private noisy observation of the latent fundamental."""

    noise_ticks: float = 2.0
    _fundamental: int = 20_000

    def observe(self, fundamental: int) -> None:
        self._fundamental = int(fundamental)

    def propose(self, view: MarketView, rng: Generator) -> list[Action]:
        observed = self._fundamental + int(round(float(rng.normal(0.0, self.noise_ticks))))
        if view.ask is not None and observed > view.ask + 1 and view.inventory < 30:
            qty = max(1, min(10, int(observed - view.ask)))
            return [Action("market", side=1, qty=qty)]
        if view.bid is not None and observed < view.bid - 1 and view.inventory > -30:
            qty = max(1, min(10, int(view.bid - observed)))
            return [Action("market", side=-1, qty=qty)]
        return []


@dataclass
class ExecutionAgent(Agent):
    """Child slices of one parent order. A TWAP-style schedule, not a broker algo."""

    target_qty: int = 60
    slice_qty: int = 6
    side: int = 1
    start_event: int = 1_000
    filled: int = 0

    def propose(self, view: MarketView, rng: Generator) -> list[Action]:
        del rng
        if view.event_index < self.start_event:
            return []
        left = self.target_qty - self.filled
        if left <= 0:
            return []
        qty = min(self.slice_qty, left)
        return [Action("market", side=self.side, qty=qty)]
