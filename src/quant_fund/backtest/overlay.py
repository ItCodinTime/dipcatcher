"""Portfolio overlay scalers for the perp book — all causal.

Each scaler implements the ``OverlayScaler`` protocol: the engine reports the
realized NAV each bar (``observe``) and the scaler rewrites queued target
weights (``scale``). Nothing here sees future data.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class VolTargetScaler:
    """Scale the whole book to a target annualized volatility.

    Realized vol is the std of the last ``window`` bar returns annualized by
    ``periods_per_year``. The scale factor is clipped to ``[min_scale,
    max_scale]`` so a quiet regime can lever up to ``max_scale`` and a hot
    regime de-risks toward ``min_scale``.
    """

    target_ann_vol: float = 0.15
    window: int = 168
    periods_per_year: float = 8766.0
    min_scale: float = 0.0
    max_scale: float = 3.0
    _navs: deque[float] = field(default_factory=lambda: deque(maxlen=10_000))

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_ann_vol) or self.target_ann_vol <= 0:
            raise ValueError("target_ann_vol must be finite and positive")
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if not 0 <= self.min_scale <= self.max_scale:
            raise ValueError("need 0 <= min_scale <= max_scale")

    def observe(self, dt: datetime, nav: float) -> None:
        self._navs.append(float(nav))

    def factor(self) -> float:
        n = len(self._navs)
        if n < self.window + 1:
            return 1.0
        navs = np.asarray(self._navs, dtype=float)[-self.window - 1 :]
        rets = np.diff(navs) / navs[:-1]
        rets = rets[np.isfinite(rets)]
        if rets.size < 2:
            return 1.0
        vol = float(np.std(rets, ddof=1) * np.sqrt(self.periods_per_year))
        if vol <= 0:
            return 1.0
        return float(np.clip(self.target_ann_vol / vol, self.min_scale, self.max_scale))

    def scale(self, dt: datetime, targets: dict[str, float]) -> dict[str, float]:
        f = self.factor()
        if f == 1.0:
            return targets
        return {sid: w * f for sid, w in targets.items()}


@dataclass
class DrawdownGovernor:
    """Linear de-risking between a soft and hard drawdown bound.

    scale = 1.0 above ``dd_soft``; ramps linearly to ``floor`` at ``dd_hard``;
    stays at ``floor`` beyond it. Pure function of the realized NAV path.
    """

    dd_soft: float = 0.03
    dd_hard: float = 0.05
    floor: float = 0.25
    _peak: float = 0.0
    _dd: float = 0.0

    def __post_init__(self) -> None:
        if not 0 <= self.dd_soft < self.dd_hard <= 1:
            raise ValueError("need 0 <= dd_soft < dd_hard <= 1")
        if not 0 <= self.floor <= 1:
            raise ValueError("floor must be in [0, 1]")

    def observe(self, dt: datetime, nav: float) -> None:
        nav = float(nav)
        if nav > self._peak:
            self._peak = nav
        self._dd = 1.0 - nav / self._peak if self._peak > 0 else 0.0

    def factor(self) -> float:
        if self._dd <= self.dd_soft:
            return 1.0
        if self._dd >= self.dd_hard:
            return self.floor
        frac = (self._dd - self.dd_soft) / (self.dd_hard - self.dd_soft)
        return float(1.0 - frac * (1.0 - self.floor))

    def scale(self, dt: datetime, targets: dict[str, float]) -> dict[str, float]:
        f = self.factor()
        if f == 1.0:
            return targets
        return {sid: w * f for sid, w in targets.items()}


class CompositeScaler:
    """Chain scalers: all observe; scale factors multiply."""

    def __init__(self, scalers: list) -> None:
        self._scalers = list(scalers)

    def observe(self, dt: datetime, nav: float) -> None:
        for s in self._scalers:
            s.observe(dt, nav)

    def scale(self, dt: datetime, targets: dict[str, float]) -> dict[str, float]:
        out = targets
        for s in self._scalers:
            out = s.scale(dt, out)
        return out
