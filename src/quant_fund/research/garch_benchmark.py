"""Fixed-protocol per-series GARCH comparison, not a SOTA certification.

Origin indexes the first unobserved return: fit r[:origin], score
sum(r[origin:origin+h]**2). Refits assimilate only observed returns.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl

from quant_fund.metrics.inference import diebold_mariano

CANDIDATES = ("garch_normal", "garch_t", "gjr_t", "egarch_t")
BASELINES = ("rolling", "ewma", "har")


def realizable(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("forecasts must be finite and positive")
    return values


def build_origins(
    *, n_dates: int, h: int, min_history: int, stride: int, n_origins: int
) -> np.ndarray:
    for value in (n_dates, h, min_history, stride, n_origins):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("protocol sizes must be positive integers")
    if stride < h or min_history < 20:
        raise ValueError("stride must cover the horizon; history must cover rolling20")
    origins = np.arange(min_history, n_dates - h + 1, stride)
    if origins.size < n_origins:
        raise ValueError("insufficient history for the frozen origin schedule")
    return origins[:n_origins]


def aggregate_by_date(dates: np.ndarray, values: np.ndarray) -> np.ndarray:
    dates, values = np.asarray(dates), np.asarray(values, dtype=float)
    if dates.ndim != 1 or dates.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("complete aligned finite losses required")
    return np.asarray([np.mean(values[dates == date]) for date in np.unique(dates)])


def chronological_origin_schedule(
    *, n_dates: int, h: int, min_history: int, stride: int, n_validation: int, n_test: int
) -> tuple[np.ndarray, np.ndarray]:
    """Validation origins first, then a disjoint test block at the series end.

    Validation targets and test targets never share a return index and the
    test block starts at least ``h`` bars after the last validation target.
    """
    validation = build_origins(
        n_dates=n_dates, h=h, min_history=min_history, stride=stride, n_origins=n_validation
    )
    val_last_used = int(validation[-1]) + h - 1  # last return index in a validation target
    test = np.arange(val_last_used + h, n_dates - h + 1, stride)  # strict chronological gap
    if test.size < n_test:
        raise ValueError("insufficient history for the frozen test schedule")
    return validation, test[-n_test:]


def select_on_validation(losses: dict[str, np.ndarray]) -> str:
    if not losses:
        raise ValueError("no validation candidates")
    sizes = {np.asarray(v).size for v in losses.values()}
    if len(sizes) != 1 or min(sizes) < 1:
        raise ValueError("validation losses must align and be nonempty")
    if any(not np.isfinite(v).all() for v in losses.values()):
        raise ValueError("nonfinite validation loss")
    return min(sorted(losses), key=lambda name: float(np.mean(losses[name])))


def summarize_losses(
    validation: dict[str, np.ndarray], test: dict[str, np.ndarray], *, source: str
) -> dict[str, Any]:
    selected = select_on_validation(validation)
    if selected not in test or any(b not in test for b in BASELINES):
        raise ValueError("selected candidate and all configured baselines required")
    if any(not np.isfinite(v).all() for v in test.values()):
        raise ValueError("nonfinite test losses")
    comparisons = {}
    for baseline in BASELINES:
        dm = diebold_mariano(test[selected], test[baseline], name_a=selected, name_b=baseline)
        comparisons[baseline] = {
            "mean_loss_difference": dm.mean_loss_diff,
            "p_two_sided_bonferroni": min(1.0, dm.p_value * len(BASELINES))
            if np.isfinite(dm.p_value)
            else None,
            "n_dates": dm.n,
            "significant_improvement": bool(
                dm.n >= 30
                and dm.mean_loss_diff < 0
                and np.isfinite(dm.p_value)
                and dm.p_value * len(BASELINES) < 0.05
            ),
        }
    return {
        "selected": selected,
        "source": source,
        "sota_proven": False,
        "beats_all_baselines": all(c["significant_improvement"] for c in comparisons.values()),
        "validation_qlike": {k: float(np.mean(v)) for k, v in validation.items()},
        "test_qlike": {k: float(np.mean(v)) for k, v in test.items()},
        "comparisons": comparisons,
        "scope": "Fixed local benchmark only; not a comprehensive SOTA comparison",
    }


def har_forecast(history: np.ndarray, h: int) -> float:
    """HAR-style squared-daily-return proxy, not intraday HAR-RV.

    Design rows are causal: row t uses squared returns observed through bar
    t and predicts their sum over bars t+1..t+h. The final forecast uses
    the full observed history. Daily squared returns are a noisy RV proxy.
    """
    squared = history**2
    n = squared.size
    if n < 45:  # need enough rows for a stable 4-parameter OLS
        return max(float(np.mean(squared[-20:])) * h, 1e-12)
    daily = squared
    cum = np.concatenate([[0.0], np.cumsum(daily)])
    # weekly[t] = mean of daily over bars t-4..t (needs t >= 4)
    weekly = np.full(n, np.nan)
    weekly[4:] = (cum[5:] - cum[:-5]) / 5.0
    rows_y, rows_x = [], []
    for t in range(25, n - h):  # target window t+1..t+h must be observed
        rows_x.append([1.0, daily[t], weekly[t], (cum[t + 1] - cum[t - 21]) / 22.0])
        rows_y.append(cum[t + h + 1] - cum[t + 1])
    if len(rows_y) < 20:
        return max(float(np.mean(squared[-20:])) * h, 1e-12)
    beta, *_ = np.linalg.lstsq(np.asarray(rows_x), np.asarray(rows_y), rcond=None)
    last = np.array([1.0, daily[n - 1], weekly[n - 1], (cum[n] - cum[n - 22]) / 22.0])
    return max(float(last @ beta), 1e-12)


def evaluate_models(
    *,
    prices: np.ndarray,
    h: int = 5,
    min_history: int = 200,
    stride: int = 5,
    n_origins: int = 40,
    seed: int = 42,
    candidates: tuple[str, ...] = (*BASELINES, *CANDIDATES),
) -> pl.DataFrame:
    """Refit each candidate on observed log returns at nonoverlapping origins.

    Zero-mean specifications forecast second moments matching the RV target.
    Failed fits use rolling second moments and remain in the scored sample.
    EGARCH includes a leverage term and uses a seeded simulation distribution.
    """
    from arch import arch_model
    from arch.univariate import Normal, StudentsT

    prices = realizable(prices)
    if prices.ndim != 1:
        raise ValueError("one security price series required")
    if not candidates or any(c not in (*BASELINES, *CANDIDATES) for c in candidates):
        raise ValueError("unknown or empty candidates")
    returns = np.diff(np.log(prices))
    origins = build_origins(
        n_dates=len(returns), h=h, min_history=min_history, stride=stride, n_origins=n_origins
    )
    rows = []
    for origin in origins:
        history = returns[:origin]
        rolling = max(float(np.mean(history[-20:] ** 2)) * h, 1e-12)
        ewma = float(history[0] ** 2)
        for value in history[1:]:
            ewma = 0.94 * ewma + 0.06 * float(value**2)
        row: dict[str, Any] = {
            "date": int(origin),
            "target": float(np.sum(returns[origin : origin + h] ** 2)),
        }
        for name in candidates:
            pred, status = rolling, "baseline"
            if name == "ewma":
                pred = max(h * ewma, 1e-12)
            elif name == "har":
                pred = har_forecast(history, h)
            elif name in CANDIDATES:
                model = arch_model(
                    history * 100,
                    mean="Zero",
                    p=1,
                    q=1,
                    vol="EGARCH" if name == "egarch_t" else "GARCH",
                    o=int(name in {"gjr_t", "egarch_t"}),
                    rescale=False,
                )
                model.distribution = (
                    Normal(seed=seed + int(origin))
                    if name == "garch_normal"
                    else StudentsT(seed=seed + int(origin))
                )
                status = "fallback"
                try:
                    fit = model.fit(disp="off", show_warning=False, options={"maxiter": 500})
                    if fit.convergence_flag == 0 and np.isfinite(fit.params).all():
                        forecast_method: Literal["analytic", "simulation", "bootstrap"] = (
                            "simulation" if name == "egarch_t" and h > 1 else "analytic"
                        )
                        forecast = fit.forecast(
                            horizon=h, method=forecast_method, simulations=2000, reindex=False
                        )
                        variance = realizable(np.asarray(forecast.variance)[-1] / 10000)
                        pred, status = float(np.sum(variance)), "fitted"
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    status = "numerical_failure"
            row[name], row[f"{name}_status"] = pred, status
        rows.append(row)
    return pl.DataFrame(rows)
