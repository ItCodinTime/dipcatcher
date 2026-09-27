"""Shared strategy contract used by the backtest reference and the shadow.

Both runners call :meth:`Strategy.decide`. They do not keep a private copy
of the decision rule. A strategy that branches on ``ctx.origin`` is a
code-path divergence; the call-trace guard is there to catch it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, cast

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """Point-in-time view handed to the strategy.

    ``history`` contains only bars with ``available_time <= decision_time``.
    ``origin`` is ``backtest`` or ``shadow`` so a dishonest branch is
    observable. A parity-clean strategy ignores it.
    """

    origin: str
    decision_time: datetime
    bar_time: datetime
    history: pl.DataFrame
    positions: dict[str, float]
    cash: float
    marked_value: float
    restarted: bool
    restart_generation: int


@dataclass(frozen=True, slots=True)
class TargetDecision:
    """Target portfolio weights at one decision time. Omitted names are flat."""

    weights: dict[str, float]

    def __post_init__(self) -> None:
        cleaned: dict[str, float] = {}
        for sid, weight in self.weights.items():
            value = float(weight)
            if not np.isfinite(value):
                raise ValueError(f"target weight for {sid!r} must be finite")
            cleaned[str(sid)] = value
        object.__setattr__(self, "weights", cleaned)


class Strategy(Protocol):
    """Decision rule shared by the reference replay and the shadow replay."""

    def decide(self, ctx: DecisionContext) -> TargetDecision:
        """Return target weights from ``ctx`` only."""
        ...


DecideFn = Callable[[DecisionContext], TargetDecision]


@dataclass
class FixedWeightStrategy:
    """Constant weights. Useful as a zero-divergence control, not a forecast."""

    weights: dict[str, float]
    name: str = "fixed"

    def decide(self, ctx: DecisionContext) -> TargetDecision:
        del ctx
        return TargetDecision(weights=dict(self.weights))

    def export_state(self) -> dict[str, Any]:
        return {"name": self.name}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.name = str(state.get("name", self.name))


def resolve_decide(strategy: Strategy | DecideFn) -> DecideFn:
    """Return the callable both runners must invoke."""
    method = getattr(strategy, "decide", None)
    if callable(method):
        return cast(DecideFn, method)
    if callable(strategy):
        return cast(DecideFn, strategy)
    raise TypeError("strategy must be callable or provide decide()")


def export_strategy_state(strategy: Strategy | DecideFn) -> dict[str, Any]:
    fn = getattr(strategy, "export_state", None)
    if not callable(fn):
        return {}
    state = fn()
    if not isinstance(state, dict):
        raise TypeError("export_state() must return a dict")
    return dict(state)


def load_strategy_state(strategy: Strategy | DecideFn, state: Mapping[str, Any]) -> None:
    fn = getattr(strategy, "load_state", None)
    if not callable(fn):
        if state:
            raise TypeError("strategy exported state but has no load_state(); cannot restart")
        return
    fn(dict(state))


def round_weight(weight: float, lot_size: float) -> float:
    """Snap a weight to a lot grid. ``lot_size <= 0`` leaves the weight unchanged."""
    if not np.isfinite(weight):
        raise ValueError("weight must be finite")
    if not np.isfinite(lot_size) or lot_size < 0.0:
        raise ValueError("lot_size must be finite and non-negative")
    if lot_size == 0.0:
        return float(weight)
    steps = round(float(weight) / float(lot_size))
    return float(steps * float(lot_size))
