"""Probabilistic and Deflated Sharpe. Bailey & Lopez de Prado.

Research-diagnostic only — never a live P&L / promotion Sharpe claim.
Empty or non-finite inputs → honest NaN (or fail-closed ValueError for
invalid ``n_trials``).
"""

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


def _finite_sharpe_inputs(*vals: float) -> bool:
    return all(np.isfinite(v) for v in vals)


def probabilistic_sharpe(
    sr: float,
    sr_star: float,
    n_obs: int,
    skew: float,
    kurtosis_raw: float,
) -> float:
    """PSR: P(true SR > sr_star | observed moments). Research diagnostic only."""
    if not _finite_sharpe_inputs(sr, sr_star, skew, kurtosis_raw) or not np.isfinite(n_obs):
        return float("nan")
    if int(n_obs) < 2:
        return float("nan")
    se = _sr_se(float(sr), float(skew), float(kurtosis_raw))
    return float(norm.cdf((float(sr) - float(sr_star)) * np.sqrt(int(n_obs) - 1) / se))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected max of ``n_trials`` zero-mean Sharpes (Bailey–LdP). Diagnostic only."""
    if not np.isfinite(n_trials) or n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if not np.isfinite(var_sr):
        return float("nan")
    if n_trials == 1:
        return 0.0
    n = float(n_trials)
    gamma = EULER_MASCHERONI
    term = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n) + gamma * norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(np.sqrt(max(float(var_sr), 0.0)) * term)


def deflated_sharpe(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis_raw: float,
    n_trials: int,
    var_sr: float,
) -> float:
    """DSR = PSR with sr* = E[max SR under n_trials]. Research diagnostic only — not live P&L."""
    if not _finite_sharpe_inputs(sr, skew, kurtosis_raw, var_sr) or not np.isfinite(n_obs):
        return float("nan")
    sr_star = expected_max_sharpe(n_trials, var_sr)
    return probabilistic_sharpe(sr, sr_star, n_obs, skew, kurtosis_raw)


def probability_of_backtest_overfitting(is_sharpes: Array, oos_sharpes: Array) -> float:
    """CSCV PBO (López de Prado): share of splits where the IS-best trial is OOS-below-median.

    Both arrays are shape (n_splits, n_trials). Higher Sharpe is better.
    Research-diagnostic only — not a live Sharpe / P&L claim.
    Empty, mismatched, too-small (need ≥2 splits and ≥2 trials), 1×N / N×1,
    or partially non-finite paths → honest NaN (never a silent 0.0).
    A PBO computed after silently dropping a split is not a valid model-selection
    diagnostic, so the function refuses the entire matrix when any score is
    missing or non-finite. Valid finite matrices may still return exact 0.0 or 1.0.
    """
    ins = np.asarray(is_sharpes, dtype=float)
    oos = np.asarray(oos_sharpes, dtype=float)
    if ins.shape != oos.shape or ins.ndim != 2 or ins.shape[0] < 2 or ins.shape[1] < 2:
        return float("nan")
    if not np.isfinite(ins).all() or not np.isfinite(oos).all():
        return float("nan")
    flags: list[float] = []
    for i in range(ins.shape[0]):
        best = int(np.nanargmax(ins[i]))
        oos_row = oos[i]
        med = float(np.median(oos_row))
        flags.append(1.0 if oos_row[best] < med else 0.0)
    if not flags:
        return float("nan")
    return float(np.mean(flags))


def moments_from_returns(returns: Array) -> tuple[float, float, float]:
    """Return (sigma, skew, raw_kurtosis) for PSR/DSR. Research diagnostic only.

    Needs ≥4 finite observations; else honest (NaN, NaN, NaN).
    """
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


def min_track_record_length(
    sr: float,
    skew: float,
    kurtosis_raw: float,
    *,
    conf: float = 0.95,
    sr_star: float = 0.0,
) -> float:
    """Minimum track-record length (Bailey & López de Prado) for PSR ≥ conf.

    Research-diagnostic only — not a live Sharpe / promotion claim.
    Non-finite inputs, ``conf`` ∉ (0.5, 1), or ``sr == sr_star`` → honest NaN.
    """
    if not _finite_sharpe_inputs(sr, skew, kurtosis_raw, sr_star, conf):
        return float("nan")
    if not (0.5 < float(conf) < 1.0):
        return float("nan")
    diff = float(sr) - float(sr_star)
    if abs(diff) < 1e-18:
        return float("nan")
    z = float(norm.ppf(float(conf)))
    if not np.isfinite(z):
        return float("nan")
    # Bailey–LdP: MinTRL = 1 + [1 - γ3 SR + (γ4−1)/4 SR²] (z/(SR−SR*))²
    # Negative/zero SE-variance term is clamped (same as _sr_se) for honest finite MinTRL.
    inside = 1.0 - float(skew) * float(sr) + ((float(kurtosis_raw) - 1.0) / 4.0) * float(sr) ** 2
    # Keep the minimum track length strictly above one even when extreme
    # higher-moment inputs drive the asymptotic variance term toward zero.
    inside = max(inside, 1e-12)
    return float(1.0 + inside * (z / diff) ** 2)


def min_trl_from_returns(
    returns: Array,
    *,
    conf: float = 0.95,
    sr_star: float = 0.0,
) -> dict[str, float | int | bool | str]:
    """Research-only Bailey–LdP MinTRL smoke from return moments.

    Computes per-period mean/std SR plus ``moments_from_returns`` skew/kurtosis,
    then ``min_track_record_length``. Payload keys intentionally omit underscore
    tokens ``sharpe`` / ``sortino`` / ``calmar`` / ``pnl`` / ``nav`` so research
    family-blob hygiene stays green.

    Not a live performance / promotion claim. Short, non-finite, or zero-vol
    paths → honest NaN MinTRL fields with ``n_obs`` still reported.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    n = int(r.size)
    out: dict[str, float | int | bool | str] = {
        "n_obs": n,
        "conf": float(conf),
        "min_track_record_length": float("nan"),
        "min_trl": float("nan"),
        "track_record_bars": float("nan"),
        "research_only": True,
        # Intentionally omit live_pnl_claim — "pnl" is a forbidden key token.
        "claim": "bailey_ldp_min_trl_diagnostic_only",
    }
    if n < 4:
        return out
    sig, skew, kurt = moments_from_returns(r)
    if not (np.isfinite(sig) and np.isfinite(skew) and np.isfinite(kurt)) or sig == 0.0:
        return out
    # Per-period observed ratio (same scale as moments); not annualized headline.
    observed = float(np.mean(r) / sig)
    mtrl = min_track_record_length(
        observed, float(skew), float(kurt), conf=float(conf), sr_star=float(sr_star)
    )
    out["min_track_record_length"] = float(mtrl)
    out["min_trl"] = float(mtrl)
    if np.isfinite(mtrl):
        out["track_record_bars"] = float(np.ceil(mtrl))
    return out
