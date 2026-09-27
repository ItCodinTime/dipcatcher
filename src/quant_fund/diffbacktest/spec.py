"""Tolerances, parameter boxes, and honesty labels for the differentiable backtest.

Research tooling only. Nothing here is a live-trading claim or a sealed
research-catalog score. See ``docs/DIFFBACKTEST.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

# Honesty labels. Reports must carry these. They are not research-catalog keys.
RESEARCH_ONLY = True
LIVE_PNL_CLAIM = False
DATA_SOURCE_SYNTHETIC = "SYNTHETIC"

STRATEGIES: tuple[str, ...] = (
    "tsmom",
    "topk",
    "antonacci",
    "risk_parity",
    "momentum",
    "reversal",
    "equal_weight",
)

# Calendar controls. They stay integers so a decision at t cannot read P_t
# when delay >= 1, and so the rebalance clock matches the NumPy books.
STRUCTURAL: tuple[str, ...] = ("delay", "rebalance_every", "periods_per_year")

# Hard JAX vs this package's NumPy core, and vs the existing research books
# (band = 0, impact off). Float64, no relaxation.
HARD_PARITY_ATOL = 1e-8
HARD_PARITY_RTOL = 1e-8

# Smooth / softplus forward pass. Large beta is the sharpened limit.
PARITY_BETA = 128.0
# Moderate beta used when finite differences must see curvature (a saturated
# tanh has a gradient that is numerically zero, which is exact but useless).
GRAD_BETA = 8.0
# Absolute gap allowed between the sharpened smooth net series and the hard
# net series on the strong-trend fixture (N <= 4, one_way_cost <= 1e-3,
# band = 0). The leading term is the soft-abs gap at exact-zero turnover:
# one_way_cost * N * 2 * log(2) / beta per bar. Tests also require the gap
# to shrink as beta grows; this constant is not a licence to hide a sign bug.
SMOOTH_NET_ATOL = 5e-4

# Fixed synthetic DGP for the CI small case. Do not retune after seeing
# out-of-sample diagnostics.
SMALL_CASE_SEED = 0
SMALL_CASE_MU = 0.0002
SMALL_CASE_SIGMA = 0.01
SMALL_CASE_T = 96
SMALL_CASE_N = 3
SMALL_CASE_TRAIN = 64
SMALL_CASE_TEST = 16
SMALL_CASE_LAMBDA = 0.1
SMALL_CASE_ADAM_STEPS = 12
SMALL_CASE_BETA = 8.0
SMALL_CASE_RHO_MAX = 1.5
SMALL_CASE_PGD_STEPS = 8
SMALL_CASE_BISECTIONS = 6

Box = tuple[float, float]

# Closed intervals for every differentiable knob. Defaults below lie inside.
BOXES: dict[str, Box] = {
    "target_vol": (0.05, 1.0),
    "max_gross": (0.2, 1.5),
    "rebalance_band": (0.0, 0.08),
    "lookback": (8.0, 63.0),
    "skip": (0.0, 15.0),
    "vol_lookback": (8.0, 40.0),
    "long_only": (0.0, 1.0),
    "top_k": (1.0, 5.0),
    "require_positive": (0.0, 1.0),
    "fraction": (0.2, 0.8),
    "long_short": (0.0, 1.0),
    "gross_limit": (0.2, 1.5),
    "max_name_weight": (0.05, 1.0),
    "target_buffer": (0.01, 0.3),
    "one_way_cost": (0.0, 0.02),
    "commission_bps": (0.0, 20.0),
    "half_spread_bps": (0.0, 20.0),
    "impact_y": (0.0, 1.0),
}

_COST = ("one_way_cost", "commission_bps", "half_spread_bps", "impact_y")

ACTIVE: dict[str, tuple[str, ...]] = {
    "tsmom": (
        "target_vol",
        "max_gross",
        "rebalance_band",
        "lookback",
        "skip",
        "vol_lookback",
        "long_only",
        *_COST,
    ),
    "topk": (
        "rebalance_band",
        "lookback",
        "skip",
        "top_k",
        "require_positive",
        *_COST,
    ),
    "antonacci": ("rebalance_band", "lookback", "skip", *_COST),
    "risk_parity": ("rebalance_band", "vol_lookback", "max_gross", *_COST),
    "momentum": (
        "rebalance_band",
        "lookback",
        "fraction",
        "long_short",
        "gross_limit",
        "max_name_weight",
        "target_buffer",
        *_COST,
    ),
    "reversal": (
        "rebalance_band",
        "lookback",
        "fraction",
        "long_short",
        "gross_limit",
        "max_name_weight",
        "target_buffer",
        *_COST,
    ),
    "equal_weight": (
        "rebalance_band",
        "gross_limit",
        "max_name_weight",
        "target_buffer",
        *_COST,
    ),
}

# Knobs the small-case optimizer and the grid are both allowed to move.
# Same space, so the walk-forward comparison is not an expanded search.
FREE: dict[str, tuple[str, ...]] = {
    "tsmom": ("target_vol", "rebalance_band", "max_gross"),
    "topk": ("top_k", "rebalance_band"),
    "antonacci": ("lookback", "rebalance_band"),
    "risk_parity": ("vol_lookback", "rebalance_band", "max_gross"),
    "momentum": ("fraction", "rebalance_band", "gross_limit"),
    "reversal": ("fraction", "rebalance_band", "gross_limit"),
    "equal_weight": ("gross_limit", "rebalance_band"),
}

OBJECTIVES: tuple[str, ...] = ("pnl", "pnl_sum", "sharpe", "drawdown")


def active_parameters(strategy: str) -> tuple[str, ...]:
    """Differentiable knobs for ``strategy``, including cost coefficients."""
    _require_strategy(strategy)
    return ACTIVE[strategy]


def free_parameters(strategy: str) -> tuple[str, ...]:
    """Subset optimized in the small case. Grid search uses the same names."""
    _require_strategy(strategy)
    return FREE[strategy]


def _require_strategy(strategy: str) -> None:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")


@dataclass(frozen=True)
class StrategyParams:
    """Continuous knobs plus the two causal calendar integers.

    Floats are the differentiable parameters. ``delay`` and
    ``rebalance_every`` are structural: delay must stay an integer >= 1 so
    the weight at t cannot depend on P_t, and the rebalance clock matches
    the existing books. ``periods_per_year`` only rescales vol targeting and
    Sharpe annualization; it is not a strategy choice.
    """

    target_vol: float = 0.40
    max_gross: float = 1.0
    rebalance_band: float = 0.0
    lookback: float = 21.0
    skip: float = 5.0
    vol_lookback: float = 16.0
    long_only: float = 1.0
    top_k: float = 2.0
    require_positive: float = 1.0
    fraction: float = 0.5
    long_short: float = 0.0
    gross_limit: float = 1.0
    max_name_weight: float = 0.5
    target_buffer: float = 0.05
    one_way_cost: float = 0.001
    commission_bps: float = 0.0
    half_spread_bps: float = 0.0
    impact_y: float = 0.0
    rebalance_every: int = 5
    delay: int = 1
    periods_per_year: float = 252.0

    def with_updates(self, **kwargs: Any) -> StrategyParams:
        return replace(self, **kwargs)


def as_int(value: float, *, name: str, lo: int, hi: int) -> int:
    """Round half away from zero (positive knobs) and reject out-of-range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    rounded = int(np.floor(number + 0.5))
    if rounded < lo or rounded > hi:
        raise ValueError(f"{name} rounds to {rounded}, outside [{lo}, {hi}]")
    return rounded


def flag(value: float) -> bool:
    """Map a [0, 1] mix to the hard book. The boundary 0.5 is on."""
    return float(value) >= 0.5


def validate_params(strategy: str, params: StrategyParams) -> None:
    """Reject non-causal or non-finite configurations before a backtest."""
    _require_strategy(strategy)
    if isinstance(params.delay, bool) or not isinstance(params.delay, int) or params.delay < 1:
        raise ValueError("delay must be an integer >= 1 (weights at t use prices through t-delay)")
    if (
        isinstance(params.rebalance_every, bool)
        or not isinstance(params.rebalance_every, int)
        or params.rebalance_every < 1
    ):
        raise ValueError("rebalance_every must be an integer >= 1")
    if not np.isfinite(params.periods_per_year) or params.periods_per_year <= 0:
        raise ValueError("periods_per_year must be finite and positive")
    for name in active_parameters(strategy):
        value = float(getattr(params, name))
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        lo, hi = BOXES[name]
        # Allow a hair outside so a sigmoid projection at the boundary is safe,
        # but reject values that would change the hard book's meaning.
        if value < lo - 1e-8 or value > hi + 1e-8:
            raise ValueError(f"{name}={value} outside [{lo}, {hi}]")
    for name in ("one_way_cost", "commission_bps", "half_spread_bps", "impact_y", "rebalance_band"):
        if float(getattr(params, name)) < 0:
            raise ValueError(f"{name} must be non-negative")
    lookback = as_int(params.lookback, name="lookback", lo=2, hi=10_000)
    skip = as_int(params.skip, name="skip", lo=0, hi=10_000)
    if strategy in {"tsmom", "topk", "antonacci", "momentum", "reversal"} and lookback <= skip:
        raise ValueError("lookback must exceed skip")
    if strategy == "tsmom":
        vol_lookback = as_int(params.vol_lookback, name="vol_lookback", lo=2, hi=10_000)
        if lookback + 1 < vol_lookback:
            raise ValueError("tsmom lookback must cover the vol window")
    if strategy == "risk_parity":
        as_int(params.vol_lookback, name="vol_lookback", lo=2, hi=10_000)
    if not 0.0 < float(params.target_buffer) < 0.5:
        raise ValueError("target_buffer must be in (0, 0.5)")
    if float(params.gross_limit) <= 0 or float(params.max_name_weight) <= 0:
        raise ValueError("gross_limit and max_name_weight must be positive")
    if float(params.max_gross) <= 0 or float(params.target_vol) <= 0:
        raise ValueError("max_gross and target_vol must be positive")


def pack(params: StrategyParams, names: tuple[str, ...]) -> np.ndarray:
    return np.array([float(getattr(params, name)) for name in names], dtype=np.float64)


def unpack(theta: np.ndarray, names: tuple[str, ...], base: StrategyParams) -> StrategyParams:
    values = np.asarray(theta, dtype=np.float64).reshape(-1)
    if values.shape != (len(names),):
        raise ValueError(f"theta length {values.shape[0]} != {len(names)} names")
    if not np.all(np.isfinite(values)):
        raise ValueError("theta must be finite")
    updates: dict[str, Any] = {name: float(values[i]) for i, name in enumerate(names)}
    # replace() rejects a splatted dict because some fields are ints.
    return cast(StrategyParams, replace(base, **updates))


def project_box(params: StrategyParams, names: tuple[str, ...]) -> StrategyParams:
    """Clip named knobs into ``BOXES``. Used by the grid and the optimizer."""
    updates: dict[str, Any] = {}
    for name in names:
        lo, hi = BOXES[name]
        updates[name] = float(np.clip(float(getattr(params, name)), lo, hi))
    return cast(StrategyParams, replace(params, **updates))


def harden(params: StrategyParams) -> StrategyParams:
    """Snap integer windows and {0, 1} gates for the hard out-of-sample book.

    Continuous scale, cost, fraction, and band knobs are unchanged. Fraction
    still enters the hard rank book as ``int(N * fraction)`` (truncation),
    matching ``net_replay._weights``.
    """
    return replace(
        params,
        lookback=float(as_int(params.lookback, name="lookback", lo=2, hi=10_000)),
        skip=float(as_int(params.skip, name="skip", lo=0, hi=10_000)),
        vol_lookback=float(as_int(params.vol_lookback, name="vol_lookback", lo=2, hi=10_000)),
        top_k=float(as_int(params.top_k, name="top_k", lo=1, hi=10_000)),
        long_only=1.0 if flag(params.long_only) else 0.0,
        require_positive=1.0 if flag(params.require_positive) else 0.0,
        long_short=1.0 if flag(params.long_short) else 0.0,
    )


def linear_cost_rate(params: StrategyParams) -> float:
    """NAV-fraction cost per unit of L1 turnover.

    ``one_way_cost`` is the directional-book coefficient. Commission and
    half-spread are the same linear term in basis points, matching the
    planning cost in ``docs/COST_AWARE_CONSTRUCTION.md``.
    """
    return (
        float(params.one_way_cost)
        + (float(params.commission_bps) + float(params.half_spread_bps)) / 1e4
    )


def limitations() -> tuple[str, ...]:
    return (
        "SYNTHETIC paths are a correctness and mechanics check, not market evidence.",
        "Not a live P&L claim and not a broker, order-router, or execution simulator.",
        "The core is the delay-1 close-to-close weight book, not backtest.engine "
        "and not the net_replay share/cash ledger.",
        "Smooth-mode gradients are exact for the relaxation. Straight-through "
        "gradients are surrogates for discrete trades. Hard sign/abs/top-k maps "
        "are not differentiable at the kinks.",
        "The adversarial radius is the smallest volatility-scaled L-infinity ball "
        "on which projected gradient found a destroying path. It is an upper bound "
        "on the true radius, not a certificate.",
        "Walk-forward on the CI path has no power to claim an edge. Grid search "
        "and the flat regularizer are both scored out of sample with the hard book; "
        "neither is selected after seeing the test fold.",
        "delay and rebalance_every stay integers. They are causal controls, not "
        "relaxed parameters.",
    )
