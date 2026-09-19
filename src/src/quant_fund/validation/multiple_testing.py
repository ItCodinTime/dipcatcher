"""Trial accounting for multiple-testing corrections.

``TrialLedger`` records trial Sharpe scores (DSR accounting) and, optionally,
per-trial period-level performance series for the data-snooping battery
(Reality Check / SPA / StepM / MCS over the full universe of trials).
Research diagnostic only — never a live P&L claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.overfitting import deflated_sharpe, moments_from_returns
from quant_fund.metrics.returns import sharpe_ratio
from quant_fund.metrics.snooping import (
    model_confidence_set,
    reality_check,
    spa_test,
    stepm,
)

Array = NDArray[np.float64]


@dataclass
class TrialLedger:
    """Named trial Sharpe ledger for DSR / multiple-testing accounting.

    Research diagnostic only — never a live P&L claim. Empty ledger refuses
    to invent a DSR (returns NaN). Duplicate names and non-finite scores fail closed.
    """

    sharpes: list[float] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    series: dict[str, Array] = field(default_factory=dict)

    def record(self, name: str, sharpe: float) -> None:
        key = str(name).strip()
        if not key:
            raise ValueError("trial name must be non-empty")
        if key in self.names:
            raise ValueError(f"duplicate trial name: {key!r}")
        if not math.isfinite(float(sharpe)):
            raise ValueError("trial score must be finite")
        self.names.append(key)
        self.sharpes.append(float(sharpe))

    def record_series(self, name: str, returns: Array) -> None:
        """Record a trial's period-level performance series for data-snooping.

        The series is the per-period performance differential (larger is better),
        e.g. excess return or date-level IC. Fail-closed: empty name, duplicate
        series name, fewer than 2 observations, or non-finite values raise
        ValueError. All recorded series must share one length (checked in
        :meth:`snooping`).
        """
        key = str(name).strip()
        if not key:
            raise ValueError("trial name must be non-empty")
        if key in self.series:
            raise ValueError(f"duplicate trial series: {key!r}")
        arr = np.asarray(returns, dtype=float).reshape(-1)
        if arr.size < 2:
            raise ValueError("trial series must have at least 2 observations")
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError("trial series must be finite")
        self.series[key] = arr

    @property
    def n_trials(self) -> int:
        return len(self.sharpes)

    @property
    def n_series(self) -> int:
        return len(self.series)

    def dsr_for(self, returns, name: str | None = None) -> float:
        """Deflated Sharpe given the recorded trial budget.

        Empty ledger → NaN (honest: no trials means no DSR bound).
        Non-finite / degenerate returns → NaN via underlying Sharpe moments.
        ``name`` is reserved for future per-trial lookup; unused today.
        """
        _ = name  # reserved
        if self.n_trials < 1:
            return float("nan")
        arr = np.asarray(returns, dtype=float).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return float("nan")
        sr_info = sharpe_ratio(arr)
        sr = float(sr_info["sharpe"])
        n = int(sr_info["n"])
        if not math.isfinite(sr) or n < 2:
            return float("nan")
        _, skew, kurt = moments_from_returns(arr)
        if not (math.isfinite(skew) and math.isfinite(kurt)):
            return float("nan")
        var_sr = float(np.var(self.sharpes, ddof=1)) if len(self.sharpes) > 1 else 0.0
        return deflated_sharpe(sr, n, skew, kurt, self.n_trials, var_sr)

    def snooping(
        self,
        *,
        n_boot: int = 2000,
        block: float | None = None,
        alpha: float = 0.05,
        mcs_alpha: float = 0.10,
        seed: int = 7,
    ) -> dict[str, Any]:
        """Data-snooping battery over the recorded trial series.

        Runs White's Reality Check, Hansen's SPA (lower / consistent / upper),
        Romano–Wolf StepM (FWER at ``alpha``) and the Hansen–Lunde–Nason model
        confidence set (at ``mcs_alpha``) on the ``T × K`` matrix of recorded
        series (larger is better). Fail-closed: fewer than two series → ``{}``
        (honest: no universe to snoop); unequal lengths → ValueError.
        """
        names = list(self.series)
        if len(names) < 2:
            return {}
        lengths = {name: int(self.series[name].size) for name in names}
        if len(set(lengths.values())) != 1:
            parts = ", ".join(f"{k}={v}" for k, v in lengths.items())
            raise ValueError(f"trial series must align in length: {parts}")
        matrix = np.column_stack([self.series[name] for name in names])
        rc = reality_check(matrix, n_boot=n_boot, block=block, seed=seed)
        spa = spa_test(matrix, n_boot=n_boot, block=block, seed=seed)
        st = stepm(matrix, n_boot=n_boot, block=block, alpha=alpha, seed=seed)
        mcs = model_confidence_set(matrix, n_boot=n_boot, block=block, alpha=mcs_alpha, seed=seed)
        best = names[rc.best_index] if rc.best_index >= 0 else None
        return {
            "claim": "research_diagnostic_only",
            "research_only": True,
            "n_trials": len(names),
            "n_obs": rc.n_obs,
            "n_boot": int(n_boot),
            "block": rc.block,
            "best_trial": best,
            "best_mean": rc.best_mean,
            "reality_check": {"statistic": rc.statistic, "p_value": rc.p_value},
            "spa": {
                "statistic": spa.statistic,
                "p_lower": spa.p_lower,
                "p_consistent": spa.p_consistent,
                "p_upper": spa.p_upper,
            },
            "stepm": {
                "alpha": st.alpha,
                "n_rejected": st.n_rejected,
                "rejected": [names[i] for i, flag in enumerate(st.rejected) if flag],
                "adjusted_p": {names[i]: st.adjusted_p[i] for i in range(len(names))},
            },
            "mcs": {
                "alpha": mcs.alpha,
                "n_included": mcs.n_included,
                "included": [names[i] for i, flag in enumerate(mcs.included) if flag],
                "p_values": {names[i]: mcs.p_values[i] for i in range(len(names))},
            },
        }
