"""Square-root impact experiment on the agent market.

A sliced buy metaorder walks the book. Adverse implementation shortfall is
regressed on log participation. The pass rule was fixed before the run:

- at least 12 strictly positive adverse-impact points, else inconclusive
- ordinary least squares of log(adverse ticks) on log(filled/volume)
- pass only if the 95% Student-t interval contains 0.5 and its lower bound
  is strictly positive
- fail if that interval excludes 0.5
- inconclusive if the interval contains both 0.5 and 0

Points with non-positive adverse impact are dropped and counted. Nothing
here is tuned after seeing the slope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from quant_fund.market_sim.config import EVIDENCE, EcologyConfig
from quant_fund.market_sim.simulator import META_AGENT, Metaorder, Simulator, run_ecology
from scipy import stats

# Pre-registered trial design. Do not edit these to chase a slope of 1/2.
IMPACT_QUANTITIES: tuple[int, ...] = (50, 100, 200, 400, 800, 1600)
IMPACT_REPS = 3
IMPACT_START = 600
IMPACT_EVERY = 8
IMPACT_SLICES = 20
IMPACT_MAX_EVENTS = 2_000
IMPACT_SEED = 7
MIN_IMPACT_POINTS = 12


@dataclass
class _Arrival:
    """Book mid immediately before the first child of the metaorder."""

    start_event: int
    mid: float | None = None

    def on_step(self, sim: Simulator) -> None:
        del sim

    def before_meta(self, sim: Simulator) -> None:
        if sim.event_index == self.start_event and self.mid is None:
            mid = sim.book.mid_tick()
            self.mid = None if mid is None else float(mid)

    def finish(self, sim: Simulator) -> dict[str, float | int | str | None]:
        del sim
        return {"arrival_mid_tick": self.mid}


def impact_trial_config(seed: int, max_events: int = IMPACT_MAX_EVENTS) -> EcologyConfig:
    """Ecology used by the impact trials. No separate execution agent."""
    if max_events < 50:
        raise ValueError("impact trials need at least 50 events")
    return replace(
        EcologyConfig(seed=seed),
        max_events=max_events,
        warmup_events=min(200, max_events // 5),
        n_execution=0,
        bar_events=max(50, max_events),
    )


def execute_impact_trial(
    quantity: int,
    *,
    seed: int = IMPACT_SEED,
    start_event: int = IMPACT_START,
    every: int = IMPACT_EVERY,
    n_slices: int = IMPACT_SLICES,
    max_events: int = IMPACT_MAX_EVENTS,
    side: int = 1,
) -> dict[str, float | int | str | bool | None]:
    """Run one sliced metaorder and return its impact point, or a drop reason."""
    if quantity < 1 or n_slices < 1 or every < 1 or start_event < 1:
        raise ValueError("quantity, slices, stride, and start must be positive")
    if side not in (1, -1):
        raise ValueError("side must be +1 or -1")
    last = start_event + (n_slices - 1) * every
    if last >= max_events:
        raise ValueError("metaorder schedule must finish before max_events")
    hook = _Arrival(start_event)
    order = Metaorder(
        start_event=start_event,
        side=side,
        qty=quantity,
        slice_qty=max(1, quantity // n_slices),
        every=every,
        agent_id=META_AGENT,
    )
    result = run_ecology(
        impact_trial_config(seed, max_events),
        hook=hook,
        metaorders=(order,),
    )
    row: dict[str, float | int | str | bool | None] = {
        "quantity": int(quantity),
        "seed": int(seed),
        "side": int(side),
        "start_event": int(start_event),
        "dropped": True,
        "dropped_reason": "no_arrival",
        "filled_qty": 0,
        "volume": 0,
        "participation": None,
        "adverse_ticks": None,
        "arrival_mid_tick": hook.mid,
    }
    if hook.mid is None or hook.mid <= 0.0:
        return row
    meta = [
        fill
        for fill in result.fills
        if fill.agent == META_AGENT and fill.side == side and fill.event_index >= start_event
    ]
    if not meta:
        row["dropped_reason"] = "no_fill"
        return row
    end = max(fill.event_index for fill in meta)
    filled = sum(fill.qty for fill in meta)
    notional = sum(fill.price_tick * fill.qty for fill in meta)
    volume = sum(
        fill.qty
        for fill in result.fills
        if fill.side > 0 and start_event <= fill.event_index <= end
    )
    row["filled_qty"] = int(filled)
    row["volume"] = int(volume)
    if filled <= 0 or volume <= 0:
        row["dropped_reason"] = "no_volume"
        return row
    vwap = notional / filled
    adverse = float(side) * (vwap - float(hook.mid))
    row["participation"] = float(filled) / float(volume)
    row["adverse_ticks"] = float(adverse)
    if adverse <= 0.0:
        row["dropped_reason"] = "nonpositive_impact"
        return row
    row["dropped"] = False
    row["dropped_reason"] = ""
    return row


def fit_impact_law(
    participation: np.ndarray,
    adverse_ticks: np.ndarray,
    *,
    min_points: int = MIN_IMPACT_POINTS,
) -> dict[str, float | int | str | None]:
    """OLS log-log fit and the pre-registered pass / fail / inconclusive rule."""
    x_raw = np.asarray(participation, dtype=float).reshape(-1)
    y_raw = np.asarray(adverse_ticks, dtype=float).reshape(-1)
    if x_raw.size != y_raw.size:
        raise ValueError("participation and adverse impact must have the same length")
    keep = np.isfinite(x_raw) & np.isfinite(y_raw) & (x_raw > 0.0) & (y_raw > 0.0)
    x = np.log(x_raw[keep])
    y = np.log(y_raw[keep])
    n = int(x.size)
    out: dict[str, float | int | str | None] = {
        "n": n,
        "slope": None,
        "intercept": None,
        "ci_low": None,
        "ci_high": None,
        "r_squared": None,
        "status": "inconclusive",
        "reason": "too_few_points" if n < min_points else "",
    }
    if n < min_points or n < 3:
        out["status"] = "inconclusive"
        out["reason"] = "too_few_points"
        return out
    x_bar = float(x.mean())
    y_bar = float(y.mean())
    dx = x - x_bar
    sxx = float(np.dot(dx, dx))
    if sxx <= 0.0:
        out["reason"] = "no_participation_variation"
        return out
    slope = float(np.dot(dx, y - y_bar) / sxx)
    intercept = y_bar - slope * x_bar
    resid = y - (intercept + slope * x)
    dof = n - 2
    sigma2 = float(np.dot(resid, resid) / dof)
    if not math.isfinite(sigma2) or sigma2 < 0.0:
        out["reason"] = "non_finite_residual"
        return out
    se = math.sqrt(sigma2 / sxx)
    tcrit = float(stats.t.ppf(0.975, dof))
    lo = slope - tcrit * se
    hi = slope + tcrit * se
    ss_tot = float(np.dot(y - y_bar, y - y_bar))
    r2 = 1.0 if ss_tot <= 0.0 else 1.0 - float(np.dot(resid, resid)) / ss_tot
    out.update(
        {
            "slope": slope,
            "intercept": intercept,
            "ci_low": lo,
            "ci_high": hi,
            "r_squared": r2,
        }
    )
    if not (math.isfinite(lo) and math.isfinite(hi)):
        out["reason"] = "non_finite_interval"
        return out
    contains_half = lo <= 0.5 <= hi
    contains_zero = lo <= 0.0 <= hi
    if not contains_half:
        out["status"] = "fail"
        out["reason"] = "interval_excludes_one_half"
        return out
    if contains_zero:
        out["status"] = "inconclusive"
        out["reason"] = "interval_contains_zero_and_one_half"
        return out
    out["status"] = "pass"
    out["reason"] = "interval_contains_one_half_and_excludes_zero"
    return out


def measure_impact(
    *,
    quantities: tuple[int, ...] = IMPACT_QUANTITIES,
    reps: int = IMPACT_REPS,
    seed: int = IMPACT_SEED,
) -> dict[str, object]:
    """Repeat the pre-registered buy schedule and fit the impact law."""
    if reps < 1:
        raise ValueError("reps must be positive")
    rows: list[dict[str, float | int | str | bool | None]] = []
    for rep in range(reps):
        for quantity in quantities:
            rows.append(execute_impact_trial(quantity, seed=seed + rep))
    kept_x: list[float] = []
    kept_y: list[float] = []
    dropped = 0
    reasons: dict[str, int] = {}
    for row in rows:
        if row["dropped"]:
            dropped += 1
            reason = str(row["dropped_reason"] or "dropped")
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        participation = row["participation"]
        adverse = row["adverse_ticks"]
        if isinstance(participation, float) and isinstance(adverse, float):
            kept_x.append(participation)
            kept_y.append(adverse)
    fit = fit_impact_law(np.asarray(kept_x, dtype=float), np.asarray(kept_y, dtype=float))
    report: dict[str, object] = dict(EVIDENCE)
    report.update(
        {
            "quantities": list(quantities),
            "reps": int(reps),
            "seed": int(seed),
            "side": 1,
            "start_event": IMPACT_START,
            "every": IMPACT_EVERY,
            "n_slices": IMPACT_SLICES,
            "max_events": IMPACT_MAX_EVENTS,
            "n_attempted": len(rows),
            "n_dropped": dropped,
            "n_positive": len(kept_x),
            "drop_reasons": reasons,
            "fit": fit,
            "points": rows,
            "definition": (
                "Buy metaorder, sliced. Arrival mid is the touch immediately "
                "before the first child. Adverse impact is vwap minus that mid, "
                "in ticks. Volume counts each print once (positive-side fills) "
                "from the first child through the last metaorder fill."
            ),
        }
    )
    return report
