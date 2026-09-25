"""Statistical ship-gate comparison between base and candidate models.

"Beats the base" is a statistical claim, not a vibes claim: pass-rate deltas
carry bootstrap confidence intervals, and per-task paired outcomes are tested
with McNemar. The ship gate requires the domain delta CI to exclude zero on
the positive side.
"""

from __future__ import annotations

import random

from pydantic import BaseModel


class ComparisonResult(BaseModel):
    n_tasks: int
    base_pass_rate: float
    candidate_pass_rate: float
    delta: float
    delta_ci_low: float
    delta_ci_high: float
    mcnemar_statistic: float
    significant_improvement: bool


def bootstrap_delta_ci(
    base: list[bool],
    candidate: list[bool],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float]:
    """Paired bootstrap CI for the pass-rate difference (candidate - base)."""
    if len(base) != len(candidate) or not base:
        raise ValueError("paired inputs must be non-empty and equal length")
    rng = random.Random(seed)
    n = len(base)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(candidate[i] for i in idx) / n - sum(base[i] for i in idx) / n
        deltas.append(d)
    deltas.sort()
    low = deltas[int(alpha / 2 * n_boot)]
    high = deltas[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return low, high


def mcnemar_statistic(base: list[bool], candidate: list[bool]) -> float:
    """McNemar chi-square (with continuity correction) on discordant pairs."""
    b = sum(1 for x, y in zip(base, candidate, strict=True) if not x and y)
    c = sum(1 for x, y in zip(base, candidate, strict=True) if x and not y)
    if b + c == 0:
        return 0.0
    return (abs(b - c) - 1) ** 2 / (b + c)


def compare_runs(
    base_pass: list[bool], candidate_pass: list[bool], *, seed: int = 7
) -> ComparisonResult:
    n = len(base_pass)
    base_rate = sum(base_pass) / n
    cand_rate = sum(candidate_pass) / n
    low, high = bootstrap_delta_ci(base_pass, candidate_pass, seed=seed)
    stat = mcnemar_statistic(base_pass, candidate_pass)
    return ComparisonResult(
        n_tasks=n,
        base_pass_rate=base_rate,
        candidate_pass_rate=cand_rate,
        delta=cand_rate - base_rate,
        delta_ci_low=low,
        delta_ci_high=high,
        mcnemar_statistic=stat,
        significant_improvement=low > 0.0,
    )
