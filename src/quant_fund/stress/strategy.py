"""Research strategy definition for the stress engine.

A strategy is a set of factor exposures plus an optional decimal-return panel.
``research_only`` must stay true. This object does not place orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

_MAPPINGS = frozenset({"return", "duration"})


@dataclass(frozen=True)
class Position:
    factor: str
    weight: float
    mapping: str


@dataclass(frozen=True)
class ResearchStrategy:
    name: str
    positions: tuple[Position, ...]
    asset_weights: tuple[tuple[str, float], ...] = ()
    returns_csv: str | None = None
    research_only: bool = True

    def weight_map(self) -> dict[str, Position]:
        return {position.factor: position for position in self.positions}


def _require_mapping(mapping: str) -> str:
    if mapping not in _MAPPINGS:
        raise ValueError(f"mapping must be one of {sorted(_MAPPINGS)}")
    return mapping


def strategy_from_mapping(
    payload: dict[str, Any], *, base_dir: Path | None = None
) -> ResearchStrategy:
    """Build a strategy from a parsed YAML/JSON object."""
    if payload.get("research_only") is False:
        raise ValueError("stress strategies must be research_only")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("strategy name is required")
    raw_positions = payload.get("positions")
    if not isinstance(raw_positions, list) or not raw_positions:
        raise ValueError("strategy positions must be a non-empty list")
    positions: list[Position] = []
    seen: set[str] = set()
    for row in raw_positions:
        if not isinstance(row, dict):
            raise ValueError("each position must be a mapping")
        factor = str(row.get("factor") or "").strip()
        if not factor or factor in seen:
            raise ValueError(f"duplicate or empty factor in positions: {factor!r}")
        seen.add(factor)
        weight = float(row["weight"])
        if not np.isfinite(weight):
            raise ValueError(f"weight for {factor} must be finite")
        positions.append(
            Position(factor, weight, _require_mapping(str(row.get("mapping", "return"))))
        )
    asset_weights: list[tuple[str, float]] = []
    raw_assets = payload.get("asset_weights") or {}
    if raw_assets:
        if not isinstance(raw_assets, dict):
            raise ValueError("asset_weights must be a mapping")
        for key, value in raw_assets.items():
            weight = float(value)
            if not np.isfinite(weight):
                raise ValueError(f"asset weight for {key} must be finite")
            asset_weights.append((str(key), weight))
    returns_csv = payload.get("returns_csv")
    returns_text = str(returns_csv) if returns_csv else None
    if returns_text and base_dir is not None and not Path(returns_text).is_absolute():
        returns_text = str((base_dir / returns_text).resolve())
    return ResearchStrategy(
        name=name,
        positions=tuple(positions),
        asset_weights=tuple(asset_weights),
        returns_csv=returns_text,
        research_only=True,
    )


def load_strategy(path: Path) -> ResearchStrategy:
    """Load a YAML strategy file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"strategy file must be a mapping: {path}")
    return strategy_from_mapping(cast(dict[str, Any], raw), base_dir=path.parent)


def load_return_panel(path: Path) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Load a decimal-return CSV.

    An ISO ``date`` or ``observation_date`` column must increase strictly in
    time, so expanding-window forecasts use chronological observations. Rows
    with a non-finite return are dropped. At least ten finite rows are required.
    """
    text = path.read_text(encoding="utf-8").splitlines()
    if not text:
        raise ValueError(f"return panel is empty: {path}")
    header = [cell.strip() for cell in text[0].split(",")]
    if len(header) < 2:
        raise ValueError("return panel needs a header and at least one series")
    drop = {i for i, name in enumerate(header) if name.lower() in {"date", "observation_date"}}
    if len(drop) > 1:
        raise ValueError("return panel must have at most one date column")
    names = tuple(name for i, name in enumerate(header) if i not in drop)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("return panel series names must be non-empty and unique")
    if len(names) < 1:
        raise ValueError("return panel has no series columns")
    rows: list[list[float]] = []
    previous_date: datetime | None = None
    for line in text[1:]:
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != len(header):
            raise ValueError("return panel has a ragged row")
        if drop:
            raw_date = cells[next(iter(drop))]
            try:
                parsed_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("return panel has a non-ISO date") from exc
            if parsed_date.tzinfo is not None:
                parsed_date = parsed_date.astimezone(UTC)
            if previous_date is not None:
                if (previous_date.tzinfo is None) != (parsed_date.tzinfo is None):
                    raise ValueError("return panel mixes naive and timezone-aware dates")
                if parsed_date <= previous_date:
                    raise ValueError("return panel dates must increase strictly")
            previous_date = parsed_date
        try:
            values = [float(cells[i]) for i in range(len(header)) if i not in drop]
        except ValueError as exc:
            raise ValueError("return panel contains a non-numeric value") from exc
        if all(np.isfinite(values)):
            rows.append(values)
    if len(rows) < 10:
        raise ValueError("return panel needs at least 10 finite rows")
    panel = np.asarray(rows, dtype=np.float64)
    return names, panel


def portfolio_returns(
    names: tuple[str, ...],
    panel: NDArray[np.float64],
    asset_weights: tuple[tuple[str, float], ...],
) -> NDArray[np.float64]:
    """Weighted sum of decimal asset returns. Missing weights are equal-weighted."""
    if asset_weights:
        weight_map = dict(asset_weights)
        missing = [name for name in names if name not in weight_map]
        if missing:
            raise ValueError(f"asset_weights missing columns: {missing}")
        weights = np.asarray([weight_map[name] for name in names], dtype=np.float64)
    else:
        weights = np.full(len(names), 1.0 / len(names), dtype=np.float64)
    return np.asarray(panel @ weights, dtype=np.float64)
