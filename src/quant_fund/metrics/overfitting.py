"""Probabilistic and Deflated Sharpe. Bailey & Lopez de Prado."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Array = NDArray[np.float64]

EULER_MASCHERONI = 0.5772156649015329


def _sr_se(sr: float, skew: float, kurtosis_raw: float) -> float:
    """Non-normal SE of Sharpe (Lo 2002). kurtosis_raw is the raw fourth moment / sigma^4."""
    inside = 1.0 - skew * sr + ((kurtosis_raw - 1.0) / 4.0) * sr**2
    return float(np.sqrt(max(inside, 1e-18)))


def probabilistic_sharpe(
    sr: float,
    sr_star: float,
    n_obs: int,
    skew: float,
    kurtosis_raw: float,
) -> float:
    if n_obs < 2:
        return float("nan")
    se = _sr_se(sr, skew, kurtosis_raw)
    return float(norm.cdf((sr - sr_star) * np.sqrt(n_obs - 1) / se))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    n = float(n_trials)
    gamma = EULER_MASCHERONI
    term = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n) + gamma * norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(max(var_sr, 0.0)) * term)


def deflated_sharpe(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis_raw: float,
    n_trials: int,
    var_sr: float,
) -> float:
    sr_star = expected_max_sharpe(n_trials, var_sr)
    return probabilistic_sharpe(sr, sr_star, n_obs, skew, kurtosis_raw)


def probability_of_backtest_overfitting(is_sharpes: Array, oos_sharpes: Array) -> float:
    """CSCV PBO (López de Prado): share of splits where the IS-best trial is OOS-below-median.

    Both arrays are shape (n_splits, n_trials). Higher Sharpe is better.
    """
    ins = np.asarray(is_sharpes, dtype=float)
    oos = np.asarray(oos_sharpes, dtype=float)
    if ins.shape != oos.shape or ins.ndim != 2 or ins.shape[0] < 2 or ins.shape[1] < 2:
        return float("nan")
    flags: list[float] = []
    for i in range(ins.shape[0]):
        if not np.isfinite(ins[i]).any():
            continue
        best = int(np.nanargmax(ins[i]))
        oos_row = oos[i]
        if not np.isfinite(oos_row[best]):
            continue
        med = float(np.nanmedian(oos_row))
        flags.append(1.0 if oos_row[best] < med else 0.0)
    if not flags:
        return float("nan")
    return float(np.mean(flags))


def moments_from_returns(returns: Array) -> tuple[float, float, float]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return float("nan"), float("nan"), float("nan")
    mu = float(np.mean(r))
    sig = float(np.std(r, ddof=1))
    if sig == 0:
        return 0.0, 0.0, 3.0
    z = (r - mu) / sig
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))  # raw
    return sig, skew, kurt
