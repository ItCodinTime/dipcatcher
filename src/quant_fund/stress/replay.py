"""Apply catalog shocks and public-domain paths to a research strategy.

Published factor shocks and public-domain paths are reported as separate
losses. They are not added together when they describe the same episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quant_fund.stress.bundle import (
    YIELD_FACTORS,
    eurchf_levels,
    max_simple_drawdown,
    max_yield_increase_pp,
    series_for_window,
    window_series,
)
from quant_fund.stress.catalog import CRISIS_CATALOG, Crisis, FactorShock
from quant_fund.stress.strategy import Position, ResearchStrategy


@dataclass(frozen=True)
class Contribution:
    factor: str
    kind: str
    portfolio_return: float
    loss: float
    detail: str


@dataclass(frozen=True)
class ScenarioLoss:
    kind: str
    loss: float | None
    contributions: tuple[Contribution, ...]
    partial: bool
    note: str


@dataclass
class CrisisReplay:
    crisis_id: str
    name: str
    historical: bool
    summary: str
    limitations: tuple[str, ...]
    shocks: tuple[FactorShock, ...]
    factor_loss: ScenarioLoss
    path_loss: ScenarioLoss
    path_stats: dict[str, float] = field(default_factory=dict)


def _apply_shock(position: Position, shock: FactorShock) -> Contribution | None:
    """Map one primary shock through one position. Inapplicable pairs return None."""
    if position.factor != shock.factor or shock.role != "primary":
        return None
    if position.mapping == "return" and shock.unit == "simple_return":
        portfolio_return = position.weight * shock.value
    elif position.mapping == "duration" and shock.unit == "yield_change":
        portfolio_return = -position.weight * shock.value
    else:
        return None
    return Contribution(
        factor=shock.factor,
        kind="published_factor_shock",
        portfolio_return=float(portfolio_return),
        loss=float(-portfolio_return),
        detail=f"{shock.qualifier}; {shock.unit}={shock.value}",
    )


def factor_shock_loss(strategy: ResearchStrategy, crisis: Crisis) -> ScenarioLoss:
    """Loss from primary published shocks only."""
    positions = strategy.weight_map()
    contributions: list[Contribution] = []
    used: set[str] = set()
    for shock in crisis.shocks:
        if shock.role != "primary":
            continue
        position = positions.get(shock.factor)
        if position is None:
            continue
        applied = _apply_shock(position, shock)
        if applied is None:
            continue
        contributions.append(applied)
        used.add(shock.factor)
    relevant = [
        position.factor
        for position in strategy.positions
        if any(
            shock.factor == position.factor and shock.role == "primary" for shock in crisis.shocks
        )
    ]
    partial = len(used) < len(relevant)
    if not contributions:
        return ScenarioLoss(
            kind="published_factor_shock",
            loss=None,
            contributions=(),
            partial=True,
            note="no primary shock maps to this strategy",
        )
    loss = float(sum(item.loss for item in contributions))
    return ScenarioLoss(
        kind="published_factor_shock",
        loss=loss,
        contributions=tuple(contributions),
        partial=partial,
        note="primary published shocks; context shocks are not included",
    )


def _yield_contribution(position: Position, values: list[float], series_id: str) -> Contribution:
    increase_pp = max_yield_increase_pp(np.asarray(values, dtype=float))
    dy = increase_pp / 100.0
    portfolio_return = -position.weight * dy
    return Contribution(
        factor=position.factor,
        kind="public_domain_path",
        portfolio_return=float(portfolio_return),
        loss=float(-portfolio_return),
        detail=(
            f"{series_id} largest yield increase {increase_pp:.4f} percentage points; "
            "loss uses modified duration"
        ),
    )


def _price_contribution(position: Position, levels: list[float], label: str) -> Contribution:
    drawdown = max_simple_drawdown(np.asarray(levels, dtype=float))
    portfolio_return = position.weight * drawdown
    return Contribution(
        factor=position.factor,
        kind="public_domain_path",
        portfolio_return=float(portfolio_return),
        loss=float(-portfolio_return),
        detail=f"{label} peak-to-trough simple return {drawdown:.6f}",
    )


def path_shock_loss(
    strategy: ResearchStrategy, crisis: Crisis
) -> tuple[ScenarioLoss, dict[str, float]]:
    """Loss from the public-domain path, separate from published scalars."""
    if crisis.window_id is None:
        return (
            ScenarioLoss(
                kind="public_domain_path",
                loss=None,
                contributions=(),
                partial=False,
                note="no public-domain window for this entry",
            ),
            {},
        )
    positions = strategy.weight_map()
    contributions: list[Contribution] = []
    stats: dict[str, float] = {}
    available = set(series_for_window(crisis.window_id))
    for series_id, factor in YIELD_FACTORS.items():
        if series_id not in available:
            continue
        observations = window_series(series_id, crisis.window_id)
        values = [value for _, value in observations]
        stats[f"{series_id}_start"] = values[0]
        stats[f"{series_id}_end"] = values[-1]
        stats[f"{series_id}_min"] = float(min(values))
        stats[f"{series_id}_max"] = float(max(values))
        stats[f"{series_id}_max_increase_pp"] = max_yield_increase_pp(
            np.asarray(values, dtype=float)
        )
        position = positions.get(factor)
        if position is not None and position.mapping == "duration":
            contributions.append(_yield_contribution(position, values, series_id))
    if "DEXUSEU" in available and "DEXSZUS" in available:
        levels = [level for _, level in eurchf_levels(crisis.window_id)]
        stats["eurchf_start"] = levels[0]
        stats["eurchf_end"] = levels[-1]
        stats["eurchf_min"] = float(min(levels))
        stats["eurchf_max_drawdown"] = max_simple_drawdown(np.asarray(levels, dtype=float))
        position = positions.get("eurchf")
        if position is not None and position.mapping == "return":
            contributions.append(_price_contribution(position, levels, "H.10 EURCHF"))
    if not contributions:
        return (
            ScenarioLoss(
                kind="public_domain_path",
                loss=None,
                contributions=(),
                partial=False,
                note="public-domain series in this window do not map to the strategy",
            ),
            stats,
        )
    return (
        ScenarioLoss(
            kind="public_domain_path",
            loss=float(sum(item.loss for item in contributions)),
            contributions=tuple(contributions),
            partial=False,
            note="H.15/H.10 path stress. Not added to the published factor-shock loss.",
        ),
        stats,
    )


def replay_crisis(strategy: ResearchStrategy, crisis: Crisis) -> CrisisReplay:
    """Replay one catalog entry. Research simulation only."""
    if not strategy.research_only:
        raise ValueError("stress replay refuses a strategy that is not research_only")
    path_loss, stats = path_shock_loss(strategy, crisis)
    return CrisisReplay(
        crisis_id=crisis.crisis_id,
        name=crisis.name,
        historical=crisis.historical,
        summary=crisis.summary,
        limitations=crisis.limitations,
        shocks=crisis.shocks,
        factor_loss=factor_shock_loss(strategy, crisis),
        path_loss=path_loss,
        path_stats=stats,
    )


def replay_portfolio(
    strategy: ResearchStrategy,
    *,
    historical_only: bool = False,
) -> tuple[CrisisReplay, ...]:
    """Replay every catalog entry, optionally skipping hypothetical scenarios."""
    crises = [crisis for crisis in CRISIS_CATALOG if crisis.historical or not historical_only]
    return tuple(replay_crisis(strategy, crisis) for crisis in crises)
