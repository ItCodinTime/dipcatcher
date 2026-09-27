"""Seeded swarm runner and failure shrinker.

Each seed builds a fault schedule, runs one session, and checks invariants.
Failures are subset-minimized with :func:`~quant_fund.simtest.faults.shrink_schedule`.
Unexpected exceptions count as failures: a faulted session is supposed to
finish with ``ok``, ``fail_closed``, or ``recovered``.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field

from quant_fund.simtest.faults import FaultSchedule, schedule_from_seed, shrink_schedule
from quant_fund.simtest.invariants import check_invariants
from quant_fund.simtest.session import run_session


@dataclass
class SwarmFailure:
    seed: int
    reason: str
    schedule: tuple[dict[str, object], ...]
    minimized: tuple[dict[str, object], ...]


@dataclass
class SwarmReport:
    """Measured swarm summary. No return or risk headline metrics."""

    n_seeds: int
    n_days: int
    max_faults: int
    failures: list[SwarmFailure] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    sessions_per_second: float = 0.0
    outcome_counts: dict[str, int] = field(default_factory=dict)
    log_bytes_min: int = 0
    log_bytes_max: int = 0
    research_only: bool = True
    live_pnl_claim: bool = False
    data_source: str = "SYNTHETIC"

    @property
    def ok(self) -> bool:
        return not self.failures


def run_swarm(
    n_seeds: int,
    *,
    n_days: int = 6,
    base_seed: int = 0,
    max_faults: int = 3,
    shrink: bool = True,
) -> SwarmReport:
    """Run ``n_seeds`` sessions. Seeds are ``base_seed .. base_seed + n_seeds - 1``."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    report = SwarmReport(n_seeds=n_seeds, n_days=n_days, max_faults=max_faults)
    sizes: list[int] = []
    started = time.perf_counter()
    for offset in range(n_seeds):
        seed = base_seed + offset
        schedule = schedule_from_seed(seed, n_days, max_faults=max_faults)
        try:
            result = run_session(seed, n_days=n_days, schedule=schedule, max_faults=max_faults)
        except Exception as exc:
            minimized = _minimize(seed, n_days, schedule, max_faults) if shrink else schedule
            report.failures.append(
                SwarmFailure(
                    seed=seed,
                    reason=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
                    schedule=tuple(fault.as_dict() for fault in schedule.faults),
                    minimized=tuple(fault.as_dict() for fault in minimized.faults),
                )
            )
            continue
        sizes.append(len(result.log_bytes))
        for outcome in result.outcomes:
            report.outcome_counts[outcome] = report.outcome_counts.get(outcome, 0) + 1
        findings = check_invariants(result)
        if findings.ok:
            continue
        minimized = _minimize(seed, n_days, schedule, max_faults) if shrink else schedule
        report.failures.append(
            SwarmFailure(
                seed=seed,
                reason="; ".join(findings.failures),
                schedule=tuple(fault.as_dict() for fault in schedule.faults),
                minimized=tuple(fault.as_dict() for fault in minimized.faults),
            )
        )
    elapsed = time.perf_counter() - started
    report.elapsed_seconds = elapsed
    report.sessions_per_second = n_seeds / elapsed if elapsed > 0 else 0.0
    if sizes:
        report.log_bytes_min = min(sizes)
        report.log_bytes_max = max(sizes)
    return report


def _minimize(
    seed: int,
    n_days: int,
    schedule: FaultSchedule,
    max_faults: int,
) -> FaultSchedule:
    def fails(candidate: FaultSchedule) -> bool:
        try:
            result = run_session(seed, n_days=n_days, schedule=candidate, max_faults=max_faults)
        except Exception:
            return True
        return not check_invariants(result).ok

    try:
        return shrink_schedule(schedule, fails)
    except ValueError:
        return schedule
