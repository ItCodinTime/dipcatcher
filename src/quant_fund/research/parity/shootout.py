"""Runtime shootout: wall-clock each engine on the same labeled panel.

``time.perf_counter`` around each engine adapter call; first run is reported
separately from steady-state min/median because vectorbt JIT-compiles on first
use (a real, reportable cost — not hidden). Commands to reproduce are emitted
with the numbers.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import polars as pl

from quant_fund.research.parity.engines import run_engine
from quant_fund.research.parity.strategies import StrategyPlan


def time_engines(
    panel: pl.DataFrame,
    plans: list[StrategyPlan],
    engines: list[str],
    *,
    initial_nav: float = 1_000_000.0,
    commission_bps: float = 0.0,
    reps: int = 3,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for plan in plans:
        for engine in engines:
            times: list[float] = []
            status = "ok"
            last_notes: list[str] = []
            for rep in range(reps):
                t0 = time.perf_counter()
                run = run_engine(
                    engine,
                    panel,
                    plan,
                    initial_nav=initial_nav,
                    commission_bps=commission_bps,
                )
                dt = time.perf_counter() - t0
                status = run.status
                last_notes = run.notes
                if run.status != "ok":
                    break
                times.append(dt)
            if times:
                rows.append(
                    {
                        "engine": engine,
                        "strategy": plan.name,
                        "status": status,
                        "first": float(times[0]),
                        "min": float(min(times)),
                        "median": float(statistics.median(times)),
                        "reps_completed": len(times),
                        "notes": last_notes[:2],
                    }
                )
            else:
                rows.append(
                    {
                        "engine": engine,
                        "strategy": plan.name,
                        "status": status,
                        "first": float("nan"),
                        "min": float("nan"),
                        "median": float("nan"),
                        "reps_completed": 0,
                        "notes": last_notes[:2],
                    }
                )
    return {
        "rows": rows,
        "reps": reps,
        "command": (
            "uv run --no-sync python -m quant_fund.research.parity shootout "
            "--symbols 50 --days 1260 --reps 3"
        ),
    }
