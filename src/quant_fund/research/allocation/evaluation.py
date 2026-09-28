"""Walk-forward evaluation of allocation engines — risk-targeting quality only.

For every decision index ``t`` (stepping by ``step``) the engine is fitted on
``returns[t - window : t]`` — strictly in the past — and the resulting
constrained weights are scored on the *forward* block
``returns[t : t + horizon]``. Reported quantities measure calibration and
stability, never P&L:

- ``predicted_vol[t]``: ex-ante ``sqrt(w' Σ_t w)`` from the fitting window;
- ``realized_vol[t]``: root-mean-square of forward per-period portfolio
  returns ``sqrt(mean((r w)²))`` over the horizon;
- ``turnover[t]``: gross weight change ``Σ|w_t − w_{t-1}|`` (first decision
  is measured against an empty book);
- ``erc_residual[t]``: max deviation of risk-contribution shares from
  ``1/n`` — the acceptance residual for the risk-parity engine and a
  descriptive diagnostic for the others;
- ``constraint_violations``: count of decisions where the post-constraint
  book fails :func:`weights_satisfy` — expected to be zero; nonzero means a
  bug in the constraint layer, not a market event.

No Sharpe, mean return, or NAV is computed anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quant_fund.research.allocation.constraints import (
    AllocationConstraints,
    Array,
    weights_satisfy,
)
from quant_fund.research.allocation.engines import (
    ENGINE_NAMES,
    MIN_WINDOW,
    EngineName,
    estimate_covariance,
    fit_weights,
    portfolio_vol,
    risk_contributions,
)


@dataclass(frozen=True)
class AllocationEvaluation:
    """Result of a strictly causal walk-forward allocation evaluation.

    Arrays are aligned to ``decision_index``: ``weights[i]`` was fitted on
    ``returns[decision_index[i]-window : decision_index[i]]`` and scored on
    the forward block starting at ``decision_index[i]``.
    """

    engine: str
    n_assets: int
    window: int
    step: int
    horizon: int
    target_vol: float | None
    decision_index: Array
    weights: Array
    predicted_vol: Array
    realized_vol: Array
    turnover: Array
    erc_residual: Array
    constraint_violations: int
    metrics: dict[str, float]


def _rms_vol(portfolio_returns: Array) -> float:
    """Root-mean-square forward per-period return — a vol proxy valid for
    horizon = 1 (unlike ddof-adjusted std)."""
    return float(np.sqrt(np.mean(portfolio_returns * portfolio_returns)))


def _erc_residual(weights: Array, cov: Array) -> float:
    """Max |RC share − 1/n| — 0 exactly for equal risk contribution."""
    rc = risk_contributions(weights, cov)
    total = float(rc.sum())
    if total <= 0.0:
        return 0.0
    shares = rc / total
    return float(np.abs(shares - 1.0 / shares.size).max())


def run_walk_forward(
    returns: Array,
    engine: EngineName,
    *,
    window: int,
    constraints: AllocationConstraints | None = None,
    step: int = 1,
    horizon: int | None = None,
    kelly_fraction: float = 0.5,
    target_vol: float | None = None,
    base_engine: EngineName = "inverse_volatility",
    erc_tol: float = 1e-10,
    erc_max_iter: int = 200,
) -> AllocationEvaluation:
    """Strictly causal walk-forward evaluation of one engine.

    ``returns`` is a finite ``T x N`` matrix of simple per-period returns.
    Decisions are taken at ``t = window, window+step, ...``; weights at
    ``t`` use only rows ``[t-window, t)``. ``horizon`` (default ``step``)
    is the forward block length used for realized-vol measurement — a
    *measurement* window, not information available at the decision.
    """
    if engine not in ENGINE_NAMES:
        raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINE_NAMES}")
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or not np.all(np.isfinite(r)):
        raise ValueError("returns must be a finite T x N matrix")
    t_total, n = r.shape
    if not isinstance(window, int) or isinstance(window, bool) or window < MIN_WINDOW:
        raise ValueError(f"window must be an integer >= {MIN_WINDOW}")
    if window >= t_total:
        raise ValueError("window must be shorter than the sample")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    h = step if horizon is None else horizon
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise ValueError("horizon must be a positive integer")
    cons = constraints if constraints is not None else AllocationConstraints()

    decisions = list(range(window, t_total, step))
    weights = np.empty((len(decisions), n), dtype=float)
    predicted = np.empty(len(decisions), dtype=float)
    realized = np.empty(len(decisions), dtype=float)
    turnover = np.empty(len(decisions), dtype=float)
    erc_res = np.empty(len(decisions), dtype=float)
    violations = 0

    previous = np.zeros(n, dtype=float)
    for i, t in enumerate(decisions):
        window_slice = r[t - window : t]
        w = fit_weights(
            engine,
            window_slice,
            cons,
            kelly_fraction=kelly_fraction,
            target_vol=target_vol,
            base_engine=base_engine,
            erc_tol=erc_tol,
            erc_max_iter=erc_max_iter,
        )
        violations += len(weights_satisfy(w, cons))
        cov = estimate_covariance(window_slice)
        predicted[i] = portfolio_vol(w, cov)
        forward = r[t : t + h] @ w
        realized[i] = _rms_vol(forward)
        turnover[i] = float(np.abs(w - previous).sum())
        erc_res[i] = _erc_residual(w, cov)
        weights[i] = w
        previous = w

    metrics = _summarize(
        predicted=predicted,
        realized=realized,
        turnover=turnover,
        erc_res=erc_res,
        target_vol=target_vol,
    )
    return AllocationEvaluation(
        engine=engine,
        n_assets=n,
        window=window,
        step=step,
        horizon=h,
        target_vol=target_vol,
        decision_index=np.asarray(decisions, dtype=np.int64),
        weights=weights,
        predicted_vol=predicted,
        realized_vol=realized,
        turnover=turnover,
        erc_residual=erc_res,
        constraint_violations=violations,
        metrics=metrics,
    )


def _summarize(
    *,
    predicted: Array,
    realized: Array,
    turnover: Array,
    erc_res: Array,
    target_vol: float | None,
) -> dict[str, float]:
    """Aggregate per-decision arrays into research-quality metrics.

    Keys deliberately avoid every forbidden headline token
    (sharpe/sortino/calmar/pnl/nav) — these are calibration and stability
    statistics only.
    """
    n_dec = int(predicted.size)
    error = realized - predicted
    metrics: dict[str, float] = {
        "n_rebalances": float(n_dec),
        "predicted_vol_mean": float(predicted.mean()),
        "realized_vol_mean": float(realized.mean()),
        # Realized-vs-predicted vol calibration (risk targeting quality).
        "vol_rmse": float(np.sqrt(np.mean(error * error))),
        "vol_mae": float(np.abs(error).mean()),
        "vol_bias": float(error.mean()),
        # Weight turnover / stability. ``mean_turnover`` excludes the
        # initial buy-in measured against an empty book.
        "initial_turnover": float(turnover[0]) if n_dec else 0.0,
        "mean_turnover": float(turnover[1:].mean()) if n_dec > 1 else 0.0,
        "max_turnover": float(turnover.max()) if n_dec else 0.0,
        "mean_turnover_one_way": float(0.5 * turnover[1:].mean()) if n_dec > 1 else 0.0,
        # Equal-risk-contribution residual (0 == perfect ERC).
        "erc_residual_mean": float(erc_res.mean()),
        "erc_residual_max": float(erc_res.max()),
    }
    if target_vol is not None:
        t_error = realized - float(target_vol)
        metrics["target_vol_rmse"] = float(np.sqrt(np.mean(t_error * t_error)))
        metrics["target_vol_mae"] = float(np.abs(t_error).mean())
        metrics["target_vol_bias"] = float(t_error.mean())
    return metrics


def compare_engines(
    returns: Array,
    engines: list[EngineName] | tuple[EngineName, ...],
    **kwargs: Any,
) -> dict[str, AllocationEvaluation]:
    """Run :func:`run_walk_forward` for several engines on the same input."""
    out: dict[str, AllocationEvaluation] = {}
    for name in engines:
        out[name] = run_walk_forward(returns, name, **kwargs)
    return out
