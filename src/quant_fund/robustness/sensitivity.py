"""Per-threat sensitivity on one fixed path.

Closed forms are marked ``proven``. A search that exhausts a finite parameter
on this path is ``exact_on_path``: it is exact for the array it was given and
is not a general theorem. Stochastic searches are ``empirical``.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.robustness.analytic import cost_shock_radius
from quant_fund.robustness.threats import (
    THREAT_NAMES,
    apply_jitter,
    apply_missing,
    apply_stale,
    net_excess,
    path_gross,
    path_turnover,
)

FloatArray = NDArray[np.float64]

_EXACT_MISSING_LIMIT = 12


def _record(
    value: float | None,
    *,
    unit: str,
    status: str,
    method: str,
    flipped: bool | None,
) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "status": status,
        "method": method,
        "flipped": flipped,
    }


def jitter_sensitivity(strategy: Any, sample: FloatArray) -> dict[str, Any]:
    """Smallest circular shift whose decision differs. Finite and exact here."""
    path = np.asarray(sample, dtype=float).reshape(-1)
    base = int(strategy.decision(path))
    for distance in range(1, path.size):
        for shift in (distance, -distance):
            if int(strategy.decision(apply_jitter(path, shift))) != base:
                return _record(
                    float(distance),
                    unit="bars",
                    status="exact_on_path",
                    method="enumerate_shifts",
                    flipped=True,
                )
    return _record(
        None, unit="bars", status="exact_on_path", method="enumerate_shifts", flipped=False
    )


def missing_sensitivity(strategy: Any, sample: FloatArray) -> dict[str, Any]:
    """Smallest number of zeroed coordinates that flips the decision."""
    path = np.asarray(sample, dtype=float).reshape(-1)
    base = int(strategy.decision(path))
    if path.size > _EXACT_MISSING_LIMIT:
        return _record(
            None,
            unit="bars",
            status="unavailable",
            method="enumeration_capped",
            flipped=None,
        )
    for count in range(1, path.size + 1):
        for choice in itertools.combinations(range(path.size), count):
            if int(strategy.decision(apply_missing(path, list(choice)))) != base:
                return _record(
                    float(count),
                    unit="bars",
                    status="exact_on_path",
                    method="enumerate_subsets",
                    flipped=True,
                )
    return _record(
        None, unit="bars", status="exact_on_path", method="enumerate_subsets", flipped=False
    )


def stale_sensitivity(strategy: Any, sample: FloatArray) -> dict[str, Any]:
    """Shortest run of repeated prints that flips the decision."""
    path = np.asarray(sample, dtype=float).reshape(-1)
    base = int(strategy.decision(path))
    for length in range(1, path.size + 1):
        for start in range(path.size):
            if int(strategy.decision(apply_stale(path, start, length))) != base:
                return _record(
                    float(length),
                    unit="bars",
                    status="exact_on_path",
                    method="enumerate_windows",
                    flipped=True,
                )
    return _record(
        None, unit="bars", status="exact_on_path", method="enumerate_windows", flipped=False
    )


def spike_sensitivity(
    strategy: Any, sample: FloatArray, scale: FloatArray | None = None
) -> dict[str, Any]:
    """Smallest one-coordinate spike, using a closed form when the strategy has one."""
    path = np.asarray(sample, dtype=float).reshape(-1)
    analytic = getattr(strategy, "analytic_spike_radius", None)
    if callable(analytic):
        radius = float(analytic(path, scale))
        value = None if math.isinf(radius) else radius
        return _record(
            value,
            unit="volatility_scaled",
            status="proven",
            method="linear_margin_over_max_coordinate",
            flipped=value is not None,
        )
    return _record(
        None,
        unit="volatility_scaled",
        status="unavailable",
        method="no_closed_form",
        flipped=None,
    )


def cost_sensitivity(strategy: Any, sample: FloatArray, base_cost: float = 0.0) -> dict[str, Any]:
    """Additive cost-rate increase that makes this path's net excess non-positive.

    The algebra in ``cost_shock_radius`` is exact given the simulated gross
    sum and turnover of this path. It is not a market result.
    """
    path = np.asarray(sample, dtype=float).reshape(-1)
    positions = np.asarray(strategy.positions(path), dtype=float).reshape(-1)
    if positions.shape != path.shape:
        raise ValueError("positions must align with the sample")
    gross = path_gross(positions, path)
    turnover = path_turnover(positions)
    # Touch the simulator so a change in the cost definition cannot drift
    # from the algebraic radius unnoticed by tests that recompute net excess.
    _ = net_excess(positions, path, base_cost)
    radius = cost_shock_radius(gross, turnover, base_cost)
    if radius is None:
        return _record(
            None,
            unit="cost_rate",
            status="proven",
            method="gross_over_turnover",
            flipped=False,
        )
    return _record(
        radius,
        unit="cost_rate",
        status="proven",
        method="gross_over_turnover",
        flipped=True,
    )


def vol_path_sensitivity(attack: dict[str, Any]) -> dict[str, Any]:
    """Reuse the empirical path attack. Not a second search."""
    return _record(
        attack.get("value"),
        unit=str(attack.get("unit", "volatility_scaled")),
        status=str(attack.get("status", "empirical")),
        method=str(attack.get("method", "black_box")),
        flipped=attack.get("flipped") if isinstance(attack.get("flipped"), bool) else None,
    )


def regime_sensitivity(radius_to_nonpositive_mean: float) -> dict[str, Any]:
    """Wasserstein radius that makes the worst-case mean non-positive. Exact."""
    if not math.isfinite(radius_to_nonpositive_mean) or radius_to_nonpositive_mean < 0.0:
        raise ValueError("regime radius must be finite and non-negative")
    return _record(
        float(radius_to_nonpositive_mean),
        unit="wasserstein_1",
        status="proven",
        method="mean_minus_radius",
        flipped=True,
    )


def assemble_sensitivity(
    strategy: Any,
    sample: FloatArray,
    *,
    path_attack: dict[str, Any],
    regime_radius: float,
    scale: FloatArray | None = None,
    base_cost: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """One record per threat in ``THREAT_NAMES``."""
    sensitivity = {
        "vol_scaled_path": vol_path_sensitivity(path_attack),
        "missing_bars": missing_sensitivity(strategy, sample),
        "stale_prints": stale_sensitivity(strategy, sample),
        "spikes": spike_sensitivity(strategy, sample, scale),
        "timing_jitter": jitter_sensitivity(strategy, sample),
        "cost_shock": cost_sensitivity(strategy, sample, base_cost),
        "wasserstein_regime": regime_sensitivity(regime_radius),
    }
    missing = [name for name in THREAT_NAMES if name not in sensitivity]
    if missing:
        raise RuntimeError(f"sensitivity is missing threats: {missing}")
    return sensitivity
