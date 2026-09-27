"""SPA / White reality check driver over the audit-verified
``metrics/snooping.py`` — no reimplementation.

Standardizes trial-ledger return series into the ``T x K`` differential
matrix convention (strategy minus benchmark; larger is better), makes the
stationary-bootstrap parameters explicit (``n_boot``, ``seed``, and the
Politis–White automatic block length via ``snooping._resolve_block`` when
``block`` is None), and returns the ``metrics.snooping`` result dataclasses
unchanged.

Research diagnostics only — never a live P&L / promotion claim.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.snooping import (
    SnoopingResult,
    SpaResult,
    StepMResult,
    reality_check,
    spa_test,
    stepm,
)
from quant_fund.proofcore.contracts import RealityFilterError

Array = NDArray[np.float64]


def _differential_matrix(
    returns_by_trial: dict[str, Array],
    *,
    benchmark: str | None,
) -> Array:
    """Stack per-trial return series into a T x K differential matrix.

    All series must share one length (the ledger aligns trials on a common
    evaluation window). With ``benchmark=None`` the differentials are the raw
    returns (H0: mean <= 0); otherwise every column is minus the named
    benchmark trial's series, and the benchmark column itself is dropped.
    """
    if len(returns_by_trial) < 1:
        raise RealityFilterError("spa driver requires at least one trial series")
    names = sorted(returns_by_trial)
    cols = []
    n_obs: int | None = None
    for name in names:
        r = np.asarray(returns_by_trial[name], dtype=float).reshape(-1)
        if n_obs is None:
            n_obs = int(r.size)
        elif int(r.size) != n_obs:
            raise RealityFilterError(
                f"trial series length mismatch: {name} has {r.size}, expected {n_obs}"
            )
        cols.append(r)
    mat = np.column_stack(cols) if cols else np.empty((0, 0))
    if benchmark is not None:
        if benchmark not in names:
            raise RealityFilterError(f"benchmark trial {benchmark!r} not in ledger series")
        bidx = names.index(benchmark)
        bench = mat[:, bidx]
        keep = [i for i in range(len(names)) if i != bidx]
        if not keep:
            raise RealityFilterError("no non-benchmark trials to test")
        mat = mat[:, keep] - bench[:, None]
    return mat


def spa_from_trials(
    returns_by_trial: dict[str, Array],
    *,
    benchmark: str | None = None,
    n_boot: int = 2000,
    block: float | None = None,
    seed: int = 7,
) -> SpaResult:
    """Hansen's SPA over the trial-ledger return series (driver only)."""
    return spa_test(
        _differential_matrix(returns_by_trial, benchmark=benchmark),
        n_boot=int(n_boot),
        block=block,
        seed=int(seed),
    )


def reality_check_from_trials(
    returns_by_trial: dict[str, Array],
    *,
    benchmark: str | None = None,
    n_boot: int = 2000,
    block: float | None = None,
    seed: int = 7,
) -> SnoopingResult:
    """White's Reality Check over the trial-ledger return series (driver only)."""
    return reality_check(
        _differential_matrix(returns_by_trial, benchmark=benchmark),
        n_boot=int(n_boot),
        block=block,
        seed=int(seed),
    )


def stepm_from_trials(
    returns_by_trial: dict[str, Array],
    *,
    benchmark: str | None = None,
    n_boot: int = 2000,
    block: float | None = None,
    alpha: float = 0.05,
    seed: int = 7,
) -> StepMResult:
    """Romano–Wolf StepM over the trial-ledger return series (driver only)."""
    return stepm(
        _differential_matrix(returns_by_trial, benchmark=benchmark),
        n_boot=int(n_boot),
        block=block,
        alpha=float(alpha),
        seed=int(seed),
    )
