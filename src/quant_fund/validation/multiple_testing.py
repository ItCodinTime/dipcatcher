"""Trial accounting for multiple-testing corrections."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_fund.metrics.overfitting import deflated_sharpe, moments_from_returns
from quant_fund.metrics.returns import sharpe_ratio


@dataclass
class TrialLedger:
    sharpes: list[float] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    def record(self, name: str, sharpe: float) -> None:
        self.names.append(name)
        self.sharpes.append(float(sharpe))

    @property
    def n_trials(self) -> int:
        return len(self.sharpes)

    def dsr_for(self, returns, name: str | None = None) -> float:
        import numpy as np

        sr_info = sharpe_ratio(np.asarray(returns, dtype=float))
        sr = float(sr_info["sharpe"])
        n = int(sr_info["n"])
        _, skew, kurt = moments_from_returns(np.asarray(returns, dtype=float))
        var_sr = float(np.var(self.sharpes, ddof=1)) if len(self.sharpes) > 1 else 0.0
        return deflated_sharpe(sr, n, skew, kurt, max(self.n_trials, 1), var_sr)
