"""VaR and Expected Shortfall with intervals and backtests.

Forecast conventions, translated inside this module:

- ``level`` is the VaR confidence level (0.95 means a 95 percent VaR).
- :func:`quant_fund.metrics.var_backtest.kupiec_test` and
  :func:`quant_fund.metrics.var_backtest.christoffersen_test` receive ``level``.
- :func:`quant_fund.metrics.probability.acerbi_szekely_z1` receives the tail
  probability ``1 - level``.
- :func:`quant_fund.metrics.es_backtest.acerbi_szekely_test` is the bootstrap
  Z2 test and does not take a separate alpha.

Losses are positive when the portfolio return is negative. Research
diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.es_backtest import acerbi_szekely_test
from quant_fund.metrics.inference import stationary_bootstrap_indices
from quant_fund.metrics.probability import acerbi_szekely_z1, acerbi_szekely_z2
from quant_fund.metrics.risk import gaussian_es, gaussian_var, historical_es, historical_var
from quant_fund.metrics.risk_parametric import parametric_var_es
from quant_fund.metrics.var_backtest import christoffersen_test, kupiec_test
from quant_fund.stress.bootstrap import resolve_block_length
from quant_fund.stress.garch_copula import fit_variance_targeted_garch11

Array = NDArray[np.float64]


def _losses(losses: Array, min_n: int = 30) -> Array:
    data = np.asarray(losses, dtype=float).reshape(-1)
    if data.size < min_n or not np.isfinite(data).all():
        raise ValueError(f"losses must be finite with length >= {min_n}")
    return np.asarray(data, dtype=np.float64)


def _level(level: float) -> float:
    if not np.isfinite(level) or not 0.5 < float(level) < 1.0:
        raise ValueError("level must be a VaR confidence level in (0.5, 1)")
    return float(level)


@dataclass(frozen=True)
class IntervalEstimate:
    estimate: float
    low: float
    high: float
    confidence: float
    method: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
            "method": self.method,
        }


def point_estimates(losses: Array, level: float = 0.95) -> dict[str, dict[str, float]]:
    """Historical, Gaussian, Student-t, and Cornish–Fisher VaR and ES."""
    data = _losses(losses)
    alpha = _level(level)
    hist_var = historical_var(data, alpha)
    hist_es = historical_es(data, alpha)
    out: dict[str, dict[str, float]] = {
        "historical": {"var": hist_var, "es": hist_es},
        "gaussian": {"var": gaussian_var(data, alpha), "es": gaussian_es(data, alpha)},
    }
    for method in ("student_t", "cornish_fisher"):
        fitted = parametric_var_es(data, alpha, method=method)
        out[method] = {"var": float(fitted["var"]), "es": float(fitted["es"])}
    filtered = garch_filtered_var_es(data, alpha)
    out["garch_filtered"] = {"var": filtered["var"], "es": filtered["es"]}
    return out


def garch_filtered_var_es(losses: Array, level: float = 0.95) -> dict[str, float]:
    """Filtered historical VaR/ES on the loss series.

    A variance-targeted GARCH(1,1) is fit to losses. Standardized residuals are
    scored with the historical VaR and ES, then scaled by the last conditional
    sigma. This is a one-step conditional risk measure.
    """
    data = _losses(losses)
    alpha = _level(level)
    fit = fit_variance_targeted_garch11(data)
    sigma = _conditional_sigma(data, fit["omega"], fit["alpha"], fit["beta"])
    residual = data / sigma
    var = float(sigma[-1] * historical_var(residual, alpha))
    es = float(sigma[-1] * historical_es(residual, alpha))
    return {"var": var, "es": es, "sigma_last": float(sigma[-1])}


def _conditional_sigma(values: Array, omega: float, alpha: float, beta: float) -> Array:
    sigma = np.empty(values.size, dtype=np.float64)
    prev_s = float(np.mean(values**2))
    prev_r2 = prev_s
    for t, value in enumerate(values):
        sigma2 = omega + alpha * prev_r2 + beta * prev_s
        sigma[t] = float(np.sqrt(max(sigma2, 1e-18)))
        prev_r2 = float(value) ** 2
        prev_s = sigma2
    return sigma


def bootstrap_var_es_interval(
    losses: Array,
    level: float = 0.95,
    *,
    n_boot: int = 400,
    ci_level: float = 0.95,
    mean_block: float | None = None,
    seed: int = 0,
) -> dict[str, IntervalEstimate]:
    """Stationary-bootstrap percentile intervals for historical VaR and ES."""
    data = _losses(losses)
    alpha = _level(level)
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 50:
        raise ValueError("n_boot must be an integer >= 50")
    if not np.isfinite(ci_level) or not 0.5 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0.5, 1)")
    block = resolve_block_length(data, mean_block)
    rng = np.random.default_rng(seed)
    index = stationary_bootstrap_indices(data.size, n_boot, block, rng)
    var_draws = np.empty(n_boot, dtype=float)
    es_draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = data[index[b]]
        var_draws[b] = historical_var(sample, alpha)
        es_draws[b] = historical_es(sample, alpha)
    tail = (1.0 - ci_level) / 2.0
    out: dict[str, IntervalEstimate] = {}
    for name, draws, estimate in (
        ("var", var_draws, historical_var(data, alpha)),
        ("es", es_draws, historical_es(data, alpha)),
    ):
        low = float(np.quantile(draws, tail))
        high = float(np.quantile(draws, 1.0 - tail))
        out[name] = IntervalEstimate(
            float(estimate), low, high, ci_level, "stationary_bootstrap_percentile"
        )
    return out


def causal_historical_forecasts(
    losses: Array,
    level: float = 0.95,
    *,
    min_history: int = 50,
) -> tuple[Array, Array, Array]:
    """Expanding-window historical VaR and ES. Forecast at t uses losses before t."""
    data = _losses(losses, min_n=min_history + 30)
    alpha = _level(level)
    if min_history < 30:
        raise ValueError("min_history must be >= 30")
    var = np.empty(data.size - min_history, dtype=np.float64)
    es = np.empty(data.size - min_history, dtype=np.float64)
    for t in range(min_history, data.size):
        window = data[:t]
        var[t - min_history] = historical_var(window, alpha)
        es[t - min_history] = historical_es(window, alpha)
    return data[min_history:], var, es


def _finite_test(name: str, payload: dict[str, float]) -> dict[str, object]:
    return {"status": "ok", "name": name, **payload}


def backtest_var_es(
    losses: Array,
    var: Array,
    es: Array,
    level: float = 0.95,
    *,
    n_boot: int = 300,
    seed: int = 0,
) -> dict[str, object]:
    """Kupiec, Christoffersen, and Acerbi–Székely tests on aligned loss forecasts.

    A test that the underlying implementation defines as unidentified is
    returned with ``status="undefined"`` and the reason. It is not replaced
    with a passing p-value.
    """
    alpha = _level(level)
    realized = np.asarray(losses, dtype=float).reshape(-1)
    var_f = np.asarray(var, dtype=float).reshape(-1)
    es_f = np.asarray(es, dtype=float).reshape(-1)
    if not (realized.size == var_f.size == es_f.size) or realized.size < 30:
        raise ValueError("backtest series must share a length of at least 30")
    if not (np.isfinite(realized).all() and np.isfinite(var_f).all() and np.isfinite(es_f).all()):
        raise ValueError("backtest series must be finite")
    hits = (realized > var_f).astype(float)
    out: dict[str, object] = {
        "level": alpha,
        "tail_probability": 1.0 - alpha,
        "n": int(realized.size),
        "n_hits": int(hits.sum()),
    }
    try:
        out["kupiec"] = _finite_test(
            "kupiec", {k: float(v) for k, v in kupiec_test(hits, alpha).items()}
        )
    except ValueError as exc:
        out["kupiec"] = {"status": "undefined", "reason": str(exc)}
    try:
        raw = christoffersen_test(hits, alpha)
        out["christoffersen"] = _finite_test(
            "christoffersen", {k: float(v) for k, v in raw.items()}
        )
    except ValueError as exc:
        out["christoffersen"] = {"status": "undefined", "reason": str(exc)}
    z1, n_hits = acerbi_szekely_z1(realized, var_f, es_f, 1.0 - alpha)
    z2, _n2 = acerbi_szekely_z2(realized, var_f, es_f)
    out["acerbi_szekely_z1"] = {
        "status": "ok" if np.isfinite(z1) else "undefined",
        "z1": float(z1),
        "n_hits": int(n_hits),
    }
    out["acerbi_szekely_z2"] = {
        "status": "ok" if np.isfinite(z2) else "undefined",
        "z2": float(z2),
        "n_hits": int(n_hits),
    }
    try:
        as_boot = acerbi_szekely_test(realized, var_f, es_f, n_boot=n_boot, seed=seed)
        out["acerbi_szekely_bootstrap"] = _finite_test(
            "acerbi_szekely_z2_bootstrap", {k: float(v) for k, v in as_boot.items()}
        )
    except ValueError as exc:
        out["acerbi_szekely_bootstrap"] = {"status": "undefined", "reason": str(exc)}
    return out
