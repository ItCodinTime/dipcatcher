"""Pre-registered ecology parameters.

Numbers below are literature-scale choices (Avellaneda–Stoikov risk
aversion and arrival slope, a noise-dominated population in the spirit of
Farmer's zero-intelligence markets, heterogeneous momentum horizons).
They were written down before the stylized-fact measurement. They are not
a fit to this repository's market data, and they are not a claim that the
simulated tape is that data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from numbers import Integral, Real

EVIDENCE: dict[str, bool | str] = {
    "research_only": True,
    "live_pnl_claim": False,
    "evidence_class": "SIMULATION_DIAGNOSTIC",
    "data_source": "SYNTHETIC",
    "claim": "simulation_diagnostic_only",
}


@dataclass(frozen=True)
class EcologyConfig:
    """One seeded run of the single-instrument agent market."""

    seed: int = 7
    tick_size: float = 0.01
    initial_mid_tick: int = 20_000
    max_events: int = 2_500
    warmup_events: int = 250
    bar_events: int = 40
    return_stride: int = 5
    sample_every: int = 4
    n_mm: int = 6
    n_momentum: int = 6
    n_mean_revert: int = 8
    n_noise: int = 28
    n_informed: int = 3
    n_execution: int = 1
    fundamental_sigma: float = 0.85
    fundamental_rate: float = 4.0
    mm_gamma: float = 0.45
    mm_k: float = 1.5
    mm_size: int = 6
    mm_rate: float = 3.0
    mm_inventory_cap: float = 20.0
    momentum_rate: float = 0.8
    mean_revert_rate: float = 1.2
    noise_rate: float = 2.4
    informed_rate: float = 1.4
    execution_rate: float = 2.0
    noise_limit_prob: float = 0.70
    noise_market_prob: float = 0.18
    noise_cancel_prob: float = 0.12
    noise_size_mu: float = 1.1
    noise_size_sigma: float = 1.05
    noise_offset_lo: int = 0
    noise_offset_hi: int = 5
    hawkes_qty: int = 20
    hawkes_extra: int = 2
    depth_levels: int = 5
    execution_qty: int = 60
    execution_slices: int = 10
    execution_start_frac: float = 0.4
    strategy_nav: float = 1_000_000.0
    strategy_max_weight: float = 0.5

    def __post_init__(self) -> None:
        # Type hints on a dataclass do not validate runtime callers.
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type == "int":
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(f"{field.name} must be an integer")
            elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        for name in (
            "fundamental_rate",
            "mm_rate",
            "momentum_rate",
            "mean_revert_rate",
            "noise_rate",
            "informed_rate",
            "execution_rate",
            "mm_size",
            "depth_levels",
            "execution_qty",
            "execution_slices",
            "mm_inventory_cap",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("fundamental_sigma", "noise_size_sigma", "hawkes_qty", "hawkes_extra"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 1_000 <= self.initial_mid_tick < 99_000:
            raise ValueError("initial_mid_tick must leave room for seed quotes")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.tick_size <= 0.0:
            raise ValueError("tick_size must be positive")
        if self.max_events < 1 or self.warmup_events < 0:
            raise ValueError("event counts must be positive (warmup may be zero)")
        if self.warmup_events >= self.max_events:
            raise ValueError("warmup_events must be smaller than max_events")
        if self.bar_events < 1 or self.return_stride < 1 or self.sample_every < 1:
            raise ValueError("sampling strides must be positive")
        for name in (
            "n_mm",
            "n_momentum",
            "n_mean_revert",
            "n_noise",
            "n_informed",
            "n_execution",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.mm_gamma <= 0.0 or self.mm_k <= 0.0:
            raise ValueError("mm_gamma and mm_k must be positive")
        if self.noise_offset_lo < 0 or self.noise_offset_hi < self.noise_offset_lo:
            raise ValueError("noise offsets must satisfy 0 <= lo <= hi")
        probs = (self.noise_limit_prob, self.noise_market_prob, self.noise_cancel_prob)
        if any(p < 0.0 for p in probs) or sum(probs) <= 0.0:
            raise ValueError("noise probabilities must be non-negative and not all zero")
        if not 0.0 < self.execution_start_frac < 1.0:
            raise ValueError("execution_start_frac must be in (0, 1)")
        if self.strategy_nav <= 0.0 or self.strategy_max_weight <= 0.0:
            raise ValueError("strategy nav and max weight must be positive")


def validation_config(seed: int = 7) -> EcologyConfig:
    """Longer run used for the stylized-fact report. Still synthetic."""
    return replace(
        EcologyConfig(seed=seed),
        max_events=24_000,
        warmup_events=2_000,
        bar_events=25,
        return_stride=4,
        sample_every=3,
    )


def drought_config(cfg: EcologyConfig) -> EcologyConfig:
    """Liquidity providers step back and noise quotes sit away from the touch."""
    return replace(
        cfg,
        n_mm=1,
        mm_gamma=1.6,
        mm_size=2,
        noise_offset_lo=4,
        noise_offset_hi=12,
        noise_cancel_prob=0.40,
        noise_limit_prob=0.48,
        noise_market_prob=0.12,
        n_execution=0,
    )
