"""Rerunnable SOTA evaluation: dipcatcher causal distribution baselines vs
published foundation forecasters (Kronos, Chronos-2, Chronos-Bolt, TimesFM)
on real Binance bars.

Protocol (declared up-front, fixed before looking at results):

- Per asset, walk-forward over the last ``--origins`` bars. At origin ``i`` the
  predictors see only ``bars[: i + 1]`` (all ``available_time`` <= the origin
  bar's close time, enforced by the collector dropping in-progress bars) and
  forecast the next bar's close-to-close return ``r_{i+1}``.
- Published targets (any combination, each named for the receipt):
  - ``--kronos name=model_dir,tokenizer_dir`` (repeatable): i.i.d. sample
    ensemble — ``--samples`` independent ``predict(sample_count=1)`` calls per
    origin (upstream ``predict`` averages its internal samples, so repeated
    calls are the honest way to obtain a distribution). Only the predicted
    close is scored.
  - ``--chronos2 name=dir``: Chronos-2 native quantiles on a 0.01..0.99 grid
    of the next close, context = trailing close prices only.
  - ``--bolt name=dir``: Chronos-Bolt native quantiles, same grid/contract.
  - ``--timesfm name=dir``: TimesFM-2.5 native quantiles (0.1..0.9 + mean).
  - Legacy flags kept for rerunnability: ``--target kronos`` +
    ``--kronos-repo/--model-dir/--tokenizer-dir`` (target name ``kronos``) and
    ``--target chronos`` + ``--chronos-model`` (target name ``chronos``).
- Quantile-forecast targets are scored through the quantile-pinball identity
  ``CRPS = 2 * integral pinball_tau d tau`` on a completed quantile function:
  trapezoid over the native grid, with exponential-tail completion outside the
  grid's coverage (``quantile_completion`` in the receipt). Pinball at the
  requested taus interpolates inside the grid and uses the same completed tail
  outside it (disclosed per-target as ``tail_extrapolated_taus``).
- Dipcatcher challengers are causal univariate fits on trailing returns — a
  strict subset of the information the targets receive (Kronos additionally
  sees open/high/low/volume/amount and calendar stamps; all challengers are
  deterministic, no sampling noise):
  - ``dip_gauss`` / ``dip_student_t`` / ``dip_ewma_t`` / ``dip_empirical``:
    Gaussian, fitted Student-t, RiskMetrics EWMA-t(5), and empirical on the
    trailing ``--window`` returns.
  - ``dip_empirical_long``: empirical on ``--garch-window`` returns.
  - ``dip_garch_t``: GARCH(1,1)-t (arch) on ``--garch-window`` returns;
    one-step forecast sigma and fitted df drive a Student-t predictive.
  - ``dip_fhs``: filtered historical simulation — GJR-GARCH(1,1,1) normal vol
    path on ``--garch-window`` returns; standardized residuals rescaled by the
    one-step sigma forecast form the predictive ensemble (full residual set,
    no resampling).
  - ``dip_ewma_emp``: EWMA-weighted empirical distribution (lambda=0.97,
    recent observations weighted up) on ``--garch-window`` returns.
  - ``dip_lgbm_q``: LightGBM quantile forecaster at 9 levels, trained causally
    per origin on lagged-return/vol features over ``--garch-window``; fixed
    hyperparameters declared up-front (no test-window tuning).
  - ``dip_blend``: deterministic 50/50 mixture of the window empirical sample
    and a quantile-grid copy of the fitted Student-t.
- Proper scores only: CRPS and pinball at tau in {0.05, 0.5, 0.95}. No P&L or
  Sharpe.
- Multiplicity-aware inference: DM, SPA and MCS operate on equal-weight
  cross-asset losses at each aligned target timestamp. Inference requires a
  regular balanced panel; missing timestamps/gaps produce descriptive results
  only. Coverage and common-observation score tables are reported explicitly.
- ``--merge-parts a.npz b.npz ... --merge-out receipt.json`` recomputes pooled
  inference over concatenated per-asset shards (each shard stores the aligned
  loss cube); shard runs are deterministic and concatenate cleanly.

Research evaluation on public data. This is not a live-P&L or deployability
claim; ``live_pnl_claim`` is always false in the emitted receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as st

from quant_fund.research.sota_evidence import (
    aligned_inference_losses,
    block_sensitivity,
    coverage_summary,
    paired_effect_intervals,
    validate_bars,
    validate_losses,
)

TAUS = (0.05, 0.5, 0.95)
SCORING_CONTRACT = "native_shapes_timesfm_point_first.v2"
# v2-native plus the splice marker for legacy shards whose timesfm column was
# regenerated under v2 (other adapters verified equivalent across checkouts).
VERIFIED_SCORING_CONTRACTS = {
    SCORING_CONTRACT,
    "native_shapes_timesfm_point_first.v2+spliced",
}
BASELINES = (
    "dip_gauss",
    "dip_student_t",
    "dip_ewma_t",
    "dip_empirical",
    "dip_empirical_long",
    "dip_garch_t",
    "dip_fhs",
    "dip_ewma_emp",
    "dip_lgbm_q",
    "dip_blend",
    "dip_gmm_k",
    "dip_skt",
    "dip_qar",
    "dip_conf_t",
    "dip_regime",
)
# dip_gmm_k: BIC-selected Gaussian mixture (K in GMM_COMPONENT_COUNTS) fitted on
# --garch-window returns; scored by the closed-form mixture CRPS. Fixed init
# seed keeps the column bit-reproducible across reruns.
GMM_COMPONENT_COUNTS = (1, 2, 3)
GMM_SEED = 7
LGBM_TAUS = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
CHRONOS_LEVELS = np.round(np.arange(0.01, 1.0, 0.01), 2)
# Chronos-Bolt and TimesFM only emit their native decile grid; out-of-range
# requests are clamped by Bolt, so request the native grid and let the shared
# exponential-tail completion cover the outer deciles.
TIMESFM_LEVELS = np.round(np.arange(0.1, 1.0, 0.1), 2)
BOLT_LEVELS = TIMESFM_LEVELS
# Dense interior grid for the quantile-integral CRPS; tails are completed
# exponentially so every quantile target is integrated over the same domain.
DENSE_GRID = np.concatenate(
    [
        np.linspace(1e-4, 0.01, 40, endpoint=False),
        np.linspace(0.01, 0.99, 400),
        np.linspace(0.99, 1.0 - 1e-4, 40)[1:],
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(_sha256(child).encode("ascii"))
    return digest.hexdigest()


def _ensure_new_output(path: Path) -> None:
    for destination in (path, path.with_suffix(".losses.npz")):
        if destination.exists():
            raise FileExistsError(
                f"preserving existing evidence: {destination}; choose a new output"
            )


def _json_safe(value: Any) -> Any:
    """JSON null denotes unavailable statistics; nonstandard NaN tokens do not."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_result(path: Path, receipt: dict[str, Any], *, meta: dict[str, Any], **arrays) -> None:
    """Publish complete, hash-bound evidence without replacing a prior run.

    Both temporary files are complete before publication. Hard links create
    each destination exclusively; the JSON receipt is the last commit marker.
    """
    _ensure_new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{path.stem}-", dir=path.parent) as stage:
        staged_losses = Path(stage) / "losses.npz"
        np.savez_compressed(
            staged_losses,
            **arrays,
            meta_json=np.array(json.dumps(_json_safe(meta), sort_keys=True, allow_nan=False)),
        )
        receipt["losses_sha256"] = _sha256(staged_losses)
        receipt["losses_file"] = path.with_suffix(".losses.npz").name
        staged_receipt = Path(stage) / "receipt.json"
        staged_receipt.write_text(
            json.dumps(_json_safe(receipt), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.link(staged_losses, path.with_suffix(".losses.npz"))
        os.link(staged_receipt, path)


def ewma_next_sigma(returns: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA variance recursion; forecast sigma for the next bar."""
    var = float(returns[0] ** 2)
    for r in returns[1:]:
        var = lam * var + (1.0 - lam) * float(r) * float(r)
    return float(np.sqrt(var))


# --------------------------------------------------------------------------
# Published-target predictors
# --------------------------------------------------------------------------


def kronos_samples(predictor, history: pd.DataFrame, x_ts, y_ts, samples: int) -> np.ndarray:
    """S i.i.d. one-step close draws -> return samples relative to last close."""
    last_close = float(history["close"].iloc[-1])
    closes = np.empty(samples, dtype=float)
    for s in range(samples):
        out = predictor.predict(
            history, x_ts, y_ts, 1, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=False
        )
        closes[s] = float(out["close"].iloc[0])
    return closes / last_close - 1.0


def chronos2_quantiles(pipe, closes_hist: np.ndarray) -> np.ndarray:
    """Chronos-2 next-close quantiles on CHRONOS_LEVELS -> return quantiles."""
    import torch

    # Chronos-2 expects (n_series, n_variates, history_length).
    context = torch.from_numpy(np.asarray(closes_hist, dtype=np.float32))[None, None, :]
    quantiles, _mean = pipe.predict_quantiles(
        context, prediction_length=1, quantile_levels=CHRONOS_LEVELS.tolist()
    )
    # One input's output is (n_variates, horizon, n_quantiles).
    q = np.asarray(quantiles[0], dtype=np.float64)
    if q.shape != (1, 1, len(CHRONOS_LEVELS)):
        raise ValueError(f"unexpected Chronos-2 quantile shape: {q.shape}")
    next_close = np.sort(q[0, 0, :])
    return next_close / float(closes_hist[-1]) - 1.0


def bolt_quantiles(pipe, closes_hist: np.ndarray) -> np.ndarray:
    """Chronos-Bolt next-close quantiles on its native decile grid.

    Bolt was trained on levels 0.1..0.9 only; requesting a wider grid makes the
    pipeline clamp edge levels, so we request the native grid and rely on the
    shared tail completion for the outer deciles.
    """
    import torch

    # Chronos-Bolt is univariate: (n_series, history_length).
    context = torch.from_numpy(np.asarray(closes_hist, dtype=np.float32))[None, :]
    quantiles, _mean = pipe.predict_quantiles(
        context, prediction_length=1, quantile_levels=BOLT_LEVELS.tolist()
    )
    arr = np.asarray(quantiles, dtype=np.float64)
    # Native output is (batch, horizon, n_quantiles).
    if arr.shape != (1, 1, len(BOLT_LEVELS)):
        raise ValueError(f"unexpected Chronos-Bolt quantile shape: {arr.shape}")
    next_close = np.sort(arr[0, 0, :])
    return next_close / float(closes_hist[-1]) - 1.0


def timesfm_quantiles(model, closes_hist: np.ndarray) -> np.ndarray:
    """TimesFM-2.5 next-close quantiles on TIMESFM_LEVELS -> return quantiles.

    ``forecast`` returns ``(points, quantiles)`` where the quantile tensor's
    last axis is ``[q50, q10, q20, q30, q40, point, q60, q70, q80, q90]``:
    upstream ``TimesFM_2p5`` writes ``full_forecast[..., 5]`` as the point
    output and spread-corrects channels ``{1,2,3,4,6,7,8,9}``. Verified
    empirically 2026-09-22: ``quant[..., 5] == point`` exactly. The nine
    deciles are therefore all channels except index 5. Including the point
    channel (v1 ``q[1:]``, legacy remote ``q[:9]``) injects a central
    pseudo-quantile and drops a real decile, inflating CRPS.
    """
    _point, quant = model.forecast(horizon=1, inputs=[np.asarray(closes_hist, dtype=np.float64)])
    q = np.asarray(quant, dtype=np.float64)
    if q.shape != (1, 1, len(TIMESFM_LEVELS) + 1):
        raise ValueError(f"unexpected TimesFM quantile shape: {q.shape}")
    next_close = np.sort(np.delete(q[0, 0], 5))
    return next_close / float(closes_hist[-1]) - 1.0


# --------------------------------------------------------------------------
# Quantile completion + quantile-domain scoring
# --------------------------------------------------------------------------


def complete_quantiles(levels: np.ndarray, q: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Extend a quantile function defined on ``levels`` onto ``grid``.

    Interior values interpolate linearly. Outside the native grid the tails
    are completed exponentially: with tail slope ``s0 = dq/dtau`` at the edge,
    the exponential quantile tail is ``Q(tau) = q0 + l0*s0*ln(tau/l0)`` on the
    left and ``Q(tau) = qn - (1-ln)*sn*ln((1-tau)/(1-ln))`` on the right. The
    result is made non-decreasing. This is a fixed, disclosed completion rule
    applied identically to every quantile target.
    """
    levels = np.asarray(levels, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    order = np.argsort(levels)
    levels, q = levels[order], np.maximum.accumulate(q[order])
    l0, ln = float(levels[0]), float(levels[-1])
    s_l = max(float(q[1] - q[0]) / (levels[1] - levels[0]), 1e-12)
    s_r = max(float(q[-1] - q[-2]) / (levels[-1] - levels[-2]), 1e-12)
    out = np.interp(grid, levels, q)
    left = grid < l0
    right = grid > ln
    out[left] = q[0] + l0 * s_l * np.log(grid[left] / l0)
    out[right] = q[-1] - (1.0 - ln) * s_r * np.log((1.0 - grid[right]) / (1.0 - ln))
    return np.maximum.accumulate(out)


def crps_from_quantiles(y: float, levels: np.ndarray, q: np.ndarray) -> float:
    """CRPS from a quantile function: 2 * integral_0^1 pinball_tau on a
    completed dense grid (trapezoid)."""
    qd = complete_quantiles(levels, q, DENSE_GRID)
    pin = np.maximum(DENSE_GRID * (y - qd), (DENSE_GRID - 1.0) * (y - qd))
    return float(2.0 * np.trapezoid(pin, DENSE_GRID))


def qf_at(levels: np.ndarray, q: np.ndarray, tau: float) -> float:
    """Quantile function value at ``tau`` under the completion rule."""
    levels = np.asarray(levels, dtype=np.float64)
    q = np.maximum.accumulate(np.asarray(q, dtype=np.float64)[np.argsort(levels)])
    levels = np.sort(levels)
    l0, ln = float(levels[0]), float(levels[-1])
    if l0 <= tau <= ln:
        return float(np.interp(tau, levels, q))
    s_l = max(float(q[1] - q[0]) / (levels[1] - levels[0]), 1e-12)
    s_r = max(float(q[-1] - q[-2]) / (levels[-1] - levels[-2]), 1e-12)
    if tau < l0:
        return float(q[0] + l0 * s_l * np.log(tau / l0))
    return float(q[-1] - (1.0 - ln) * s_r * np.log((1.0 - tau) / (1.0 - ln)))


# --------------------------------------------------------------------------
# Dipcatcher challengers (deterministic, causal)
# --------------------------------------------------------------------------


def _arch_fit(rets_pct: np.ndarray, vol: str, dist: str, o: int):
    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(
            rets_pct, mean="Constant", vol=vol, p=1, o=o, q=1, dist=dist, rescale=False
        )
        return model.fit(disp="off", show_warning=False, options={"maxiter": 300})


def _ewma_weighted_empirical(rets_long: np.ndarray, y: float, lam: float = 0.97):
    """EWMA-weighted empirical distribution CRPS + quantiles at TAUS.

    Weights decay as lam^(age) so recent observations dominate; the weighted
    CRPS uses the normalized-weight ensemble identity.
    """
    x_raw = np.asarray(rets_long, dtype=float)
    n = x_raw.size
    w = lam ** np.arange(n - 1, -1, -1.0)
    w /= w.sum()
    order = np.argsort(x_raw)
    x = x_raw[order]
    w = w[order]
    term1 = float(np.sum(w * np.abs(x - y)))
    pair = np.abs(x[:, None] - x[None, :])
    term2 = 0.5 * float(np.sum((w[:, None] * w[None, :]) * pair))
    cumw = np.cumsum(w)
    qs = np.array([x[min(np.searchsorted(cumw, t), n - 1)] for t in TAUS])
    return term1 - term2, qs


def _lgbm_features(rets_all: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """Causal per-origin features: ``X[j]`` uses only ``rets_all[:j]`` plus the
    (known-ahead) weekday of the bar being forecast. Warmup rows are NaN."""
    n = rets_all.size
    X = np.full((n, 8), np.nan)
    ew = np.full(n, np.nan)
    var = float(rets_all[0] ** 2) if n else np.nan
    for j in range(n):
        if j > 0:
            var = 0.94 * var + 0.06 * float(rets_all[j - 1] ** 2)
        ew[j] = np.sqrt(var)
    for j in range(20, n):
        w5 = rets_all[j - 5 : j]
        w20 = rets_all[j - 20 : j]
        X[j] = (
            rets_all[j - 1],
            rets_all[j - 2],
            float(np.mean(w5)),
            float(np.mean(w20)),
            float(np.std(w5)),
            float(np.std(w20)),
            ew[j],
            dow[j],
        )
    return X


def lgbm_quantiles(X: np.ndarray, rets_all: np.ndarray, i: int, window: int) -> np.ndarray:
    """LightGBM quantile forecast at LGBM_TAUS for target ``rets_all[i]``,
    trained on rows j in [i-window, i) with X[j] -> rets_all[j]."""
    from lightgbm import LGBMRegressor

    lo = max(0, i - window)
    rows = np.arange(lo, i)
    Xtr, ytr = X[rows], rets_all[rows]
    ok = np.isfinite(Xtr).all(axis=1) & np.isfinite(ytr)
    Xtr, ytr = Xtr[ok], ytr[ok]
    if Xtr.shape[0] < 200:
        raise ValueError("lgbm train set too small")
    qs = np.empty(LGBM_TAUS.size)
    for k, tau in enumerate(LGBM_TAUS):
        m = LGBMRegressor(
            objective="quantile",
            alpha=float(tau),
            n_estimators=150,
            learning_rate=0.06,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.9,
            random_state=7,
            n_jobs=2,
            verbose=-1,
        )
        m.fit(Xtr, ytr)
        qs[k] = float(m.predict(X[i].reshape(1, -1))[0])
    return np.maximum.accumulate(qs)


def _fit_student_t(rets: np.ndarray) -> tuple[float, float, float]:
    """(nu, loc, scale) for a Student-t on a return window.

    MLE (`t.fit`) is fit on x100-scaled returns — the same rescale the GARCH
    path uses — because the optimizer diverges on small-magnitude 4h windows.
    A method-of-moments estimate (nu from excess kurtosis, Gaussian when there
    is none) is the fallback; only a truly zero-variance window yields NaN.
    """
    sd = float(np.std(rets, ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return np.nan, np.nan, np.nan
    try:
        nu, loc, scale = (float(v) for v in st.t.fit(rets * 100.0))
        if np.isfinite(nu) and nu > 2.0 and np.isfinite(scale) and scale > 0.0:
            return nu, loc / 100.0, scale / 100.0
    except Exception:  # noqa: BLE001 - fall through to moments
        pass
    kurt = float(st.kurtosis(rets, fisher=True, bias=False))
    nu = float(np.clip(6.0 / kurt + 4.0, 3.0, 60.0)) if np.isfinite(kurt) and kurt > 0 else 60.0
    return nu, float(np.mean(rets)), sd * np.sqrt((nu - 2.0) / nu)


def _fit_gmm(rets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BIC-selected Gaussian mixture on returns. Returns (weights, mu, sigma)."""
    from sklearn.mixture import GaussianMixture

    x = np.asarray(rets, dtype=float).reshape(-1, 1)
    if x.shape[0] < 60 or not np.isfinite(x).all():
        raise ValueError("gmm needs >=60 finite returns")
    best = None
    for k in GMM_COMPONENT_COUNTS:
        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-8,
            max_iter=200,
            n_init=4,
            random_state=GMM_SEED,
        ).fit(x)
        if best is None or gm.bic(x) < best.bic(x):
            best = gm
    w = np.asarray(best.weights_, dtype=float)
    mu = np.asarray(best.means_, dtype=float).ravel()
    sig = np.sqrt(np.asarray(best.covariances_, dtype=float).reshape(-1))
    return w, mu, sig


def _skt_pdf(z: np.ndarray, alpha: float, nu: float) -> np.ndarray:
    """Azzalini skew-t pdf in standardized units."""
    return (
        2.0 * st.t.pdf(z, nu) * st.t.cdf(alpha * z * np.sqrt((nu + 1.0) / (nu + z * z)), nu + 1.0)
    )


def _fit_skt(rets: np.ndarray) -> tuple[float, float, float, float]:
    """Nelder-Mead MLE of the skew-t (xi, omega, alpha, nu) on returns."""
    from scipy.optimize import minimize

    x = np.asarray(rets, dtype=float)
    if x.size < 60 or not np.isfinite(x).all():
        raise ValueError("skt needs >=60 finite returns")

    def nll(th: np.ndarray) -> float:
        xi, om, al, lnnu = th
        nu = np.exp(lnnu)
        z = (x - xi) / om
        lp = (
            np.log(2.0)
            - np.log(om)
            + st.t.logpdf(z, nu)
            + st.t.logcdf(al * z * np.sqrt((nu + 1.0) / (nu + z * z)), nu + 1.0)
        )
        return -float(lp.sum())

    th0 = np.array([np.mean(x), np.std(x), 0.0, np.log(8.0)])
    r = minimize(
        nll,
        th0,
        method="Nelder-Mead",
        options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8},
    )
    xi, om, al, lnnu = r.x
    nu = float(np.clip(np.exp(lnnu), 2.05, 300.0))
    if not np.isfinite(r.fun) or om <= 0.0 or not np.isfinite(al):
        raise ValueError("skt fit failed")
    return float(xi), float(om), float(np.clip(al, -80.0, 80.0)), nu


def _skt_table(xi: float, om: float, al: float, nu: float) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative-trapezoid CDF table of the fitted skew-t (z-grid, cdf)."""
    from scipy.integrate import cumulative_trapezoid

    half = 12.0  # standardized units; |alpha| shifts mass but tails stay t-nu
    z = np.linspace(-half, half, 8193)
    pdf = _skt_pdf(z, al, nu)
    cdf = cumulative_trapezoid(pdf, z, initial=0.0)
    if cdf[-1] <= 0.0:
        raise ValueError("skt quadrature degenerate")
    cdf /= cdf[-1]
    return z, cdf


def _skt_quantiles(xi: float, om: float, al: float, nu: float, taus: np.ndarray) -> np.ndarray:
    """Quantiles of the fitted skew-t by monotone interpolation of the CDF."""
    z, cdf = _skt_table(xi, om, al, nu)
    return xi + om * np.interp(taus, cdf, z)


def _qar_quantiles(rets: np.ndarray) -> np.ndarray:
    """Quantile autoregression at LGBM_TAUS: linear quantile model on lag
    features [1, r_{t-1}, |r_{t-1}|, mean(r_{t-5..t-1})] — all causal at the
    origin. Rearranged monotone; crossing is not hidden from diagnostics."""
    from statsmodels.regression.quantile_regression import QuantReg

    x = np.asarray(rets, dtype=float)
    if x.size < 100 or not np.isfinite(x).all():
        raise ValueError("qar needs >=100 finite returns")
    n = x.size
    Xf = np.empty((n - 1, 4))
    Xf[:, 0] = 1.0
    Xf[:, 1] = x[:-1]
    Xf[:, 2] = np.abs(x[:-1])
    cs = np.cumsum(np.concatenate([[0.0], x]))
    idx = np.arange(1, n)
    lo = np.maximum(idx - 5, 0)
    Xf[:, 3] = (cs[idx] - cs[lo]) / (idx - lo)
    pred = np.array([1.0, x[-1], abs(x[-1]), (cs[n] - cs[max(0, n - 5)]) / min(5, n)])
    q = np.full(LGBM_TAUS.size, np.nan)
    for k, tau in enumerate(LGBM_TAUS):
        try:
            fit = QuantReg(x[1:], Xf).fit(q=float(tau), max_iter=2000)
            q[k] = float(pred @ fit.params)
        except Exception:  # noqa: BLE001 - per-tau failure leaves NaN
            pass
    if not np.isfinite(q).all():
        raise ValueError("qar fits incomplete")
    return np.sort(q)  # rearrangement


def _conf_t_quantiles(rets: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Split-conformal PIT-warped Student-t quantiles.

    Fit Student-t on the first two-thirds of the window; the trailing third's
    empirical PIT quantile function warps the fitted quantiles (Gneiting-style
    distributional calibration). Causal by construction — the calibration block
    is strictly pre-origin — and deterministic.
    """
    x = np.asarray(rets, dtype=float)
    n = x.size
    if n < 120 or not np.isfinite(x).all():
        raise ValueError("conf_t needs >=120 finite returns")
    n_tr = (2 * n) // 3
    nu, loc, sc = _fit_student_t(x[:n_tr])
    if not (np.isfinite(nu) and np.isfinite(sc) and sc > 0.0):
        raise ValueError("conf_t base fit failed")
    u = st.t.cdf(x[n_tr:], nu, loc=loc, scale=sc)
    u = u[np.isfinite(u)]
    if u.size < 30:
        raise ValueError("conf_t calibration set too small")
    warped = np.quantile(u, taus)
    return st.t.ppf(np.clip(warped, 1e-9, 1.0 - 1e-9), nu, loc=loc, scale=sc)


def _weighted_quantile(vals: np.ndarray, ws: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Weighted empirical quantiles (midpoint interpolation rule)."""
    order = np.argsort(vals)
    v, w = np.asarray(vals)[order], np.asarray(ws)[order]
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cw /= cw[-1]
    return np.interp(taus, 0.5 * (cw[:-1] + cw[1:]), v)


def _regime_quantiles(rets: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Two-state vol-regime mixture on the window.

    Regime labels come from the EWMA-variance state *before* each observation
    (same recursion as ``ewma_next_sigma``); the mixture weight is the
    empirical next-state transition probability given the current state —
    regime persistence rather than a static 50/50, clipped to [0.05, 0.95]
    so both regime cells always contribute (boundary robustness).
    """
    x = np.asarray(rets, dtype=float)
    n = x.size
    if n < 120 or not np.isfinite(x).all():
        raise ValueError("regime needs >=120 finite returns")
    var = float(x[0] ** 2)
    sig = np.empty(n)  # sig[j] = vol forecast for x[j] made from x[:j]
    for j in range(1, n):
        sig[j] = np.sqrt(var)
        var = 0.94 * var + 0.06 * float(x[j] ** 2)
    med = float(np.median(sig[1:]))
    lbl = sig[1:] > med  # regime label per obs j in [1, n)
    if lbl.size < 2 or not (lbl.any() and (~lbl).any()):
        raise ValueError("regime split degenerate")
    cur_hi = bool(np.sqrt(var) > med)  # regime generating the target obs
    prev, nxt = lbl[:-1], lbl[1:]
    pool = nxt[prev == cur_hi]
    w_hi = float(np.clip(pool.mean() if pool.size else 0.5, 0.05, 0.95))
    hi_x, lo_x = x[1:][lbl], x[1:][~lbl]
    if hi_x.size < 10 or lo_x.size < 10:
        raise ValueError("regime cells too small")
    vals = np.concatenate([hi_x, lo_x])
    ws = np.concatenate(
        [np.full(hi_x.size, w_hi / hi_x.size), np.full(lo_x.size, (1.0 - w_hi) / lo_x.size)]
    )
    return _weighted_quantile(vals, ws, taus)


def dip_challengers(
    rets: np.ndarray, rets_long: np.ndarray, y: float
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """All lab challengers at one origin. Returns (crps, quantiles@TAUS)."""
    from quant_fund.metrics.scoring import (
        crps_empirical,
        crps_gaussian,
        crps_gaussian_mixture,
        crps_student_t,
        gaussian_mixture_quantiles,
    )

    crps: dict[str, float] = {}
    quant: dict[str, np.ndarray] = {}
    nan3 = np.full(len(TAUS), np.nan)

    mu, sd = float(np.mean(rets)), float(np.std(rets, ddof=1))
    nu_hat, loc_hat, scale_hat = _fit_student_t(rets)
    ewma_sd = ewma_next_sigma(rets)

    crps["dip_gauss"] = float(crps_gaussian(np.array([y]), np.array([mu]), np.array([sd]))[0])
    quant["dip_gauss"] = st.norm.ppf(TAUS, loc=mu, scale=sd)
    crps["dip_student_t"] = float(
        crps_student_t(np.array([y]), np.array([loc_hat]), np.array([scale_hat]), nu_hat)[0]
    )
    quant["dip_student_t"] = (
        st.t.ppf(TAUS, nu_hat, loc=loc_hat, scale=scale_hat) if np.isfinite(nu_hat) else nan3
    )
    crps["dip_ewma_t"] = float(
        crps_student_t(np.array([y]), np.array([0.0]), np.array([ewma_sd]), 5.0)[0]
    )
    quant["dip_ewma_t"] = st.t.ppf(TAUS, 5.0, loc=0.0, scale=ewma_sd)
    crps["dip_empirical"] = crps_empirical(y, rets)
    quant["dip_empirical"] = np.quantile(rets, TAUS)
    crps["dip_empirical_long"] = crps_empirical(y, rets_long)
    quant["dip_empirical_long"] = np.quantile(rets_long, TAUS)

    try:
        fit = _arch_fit(rets_long * 100.0, vol="GARCH", dist="t", o=0)
        nu_g = float(fit.params["nu"])
        mu_g = float(fit.params.get("mu", 0.0)) / 100.0
        sig_g = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])) / 100.0
        scale_g = sig_g * np.sqrt((nu_g - 2.0) / nu_g) if nu_g > 2.0 else np.nan
        crps["dip_garch_t"] = float(
            crps_student_t(np.array([y]), np.array([mu_g]), np.array([scale_g]), nu_g)[0]
        )
        quant["dip_garch_t"] = (
            st.t.ppf(TAUS, nu_g, loc=mu_g, scale=scale_g) if np.isfinite(scale_g) else nan3
        )
    except Exception:  # noqa: BLE001 - non-convergence -> honest NaN
        crps["dip_garch_t"] = float("nan")
        quant["dip_garch_t"] = nan3

    try:
        fit = _arch_fit(rets_long * 100.0, vol="GARCH", dist="normal", o=1)
        sig_next = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])) / 100.0
        mu_f = float(fit.params.get("mu", 0.0)) / 100.0
        cond_vol = np.asarray(fit.conditional_volatility, dtype=float) / 100.0
        resid = np.asarray(fit.resid, dtype=float) / 100.0
        ok = cond_vol > 0.0
        std_resid = resid[ok] / cond_vol[ok]
        std_resid = std_resid[np.isfinite(std_resid)]
        if std_resid.size < 20 or not np.isfinite(sig_next):
            raise ValueError("fhs residual set too small")
        samp = mu_f + sig_next * std_resid
        crps["dip_fhs"] = crps_empirical(y, samp)
        quant["dip_fhs"] = np.quantile(samp, TAUS)
    except Exception:  # noqa: BLE001
        crps["dip_fhs"] = float("nan")
        quant["dip_fhs"] = nan3

    try:
        c_ew, q_ew = _ewma_weighted_empirical(rets_long, y)
        crps["dip_ewma_emp"] = float(c_ew)
        quant["dip_ewma_emp"] = q_ew
    except Exception:  # noqa: BLE001
        crps["dip_ewma_emp"] = float("nan")
        quant["dip_ewma_emp"] = nan3

    try:
        gw, gmu, gsig = _fit_gmm(rets_long)
        crps["dip_gmm_k"] = float(crps_gaussian_mixture(np.array([y]), gw, gmu, gsig)[0])
        quant["dip_gmm_k"] = gaussian_mixture_quantiles(gw, gmu, gsig, TAUS)
    except Exception:  # noqa: BLE001
        crps["dip_gmm_k"] = float("nan")
        quant["dip_gmm_k"] = nan3

    try:
        xi, om, al, nu_s = _fit_skt(rets_long)
        grid = (np.arange(512) + 0.5) / 512.0
        samp = _skt_quantiles(xi, om, al, nu_s, grid)
        crps["dip_skt"] = crps_empirical(y, samp)
        quant["dip_skt"] = _skt_quantiles(xi, om, al, nu_s, TAUS)
    except Exception:  # noqa: BLE001
        crps["dip_skt"] = float("nan")
        quant["dip_skt"] = nan3

    try:
        q_qar = _qar_quantiles(rets_long)
        crps["dip_qar"] = crps_from_quantiles(y, LGBM_TAUS, q_qar)
        quant["dip_qar"] = np.array([qf_at(LGBM_TAUS, q_qar, t) for t in TAUS])
    except Exception:  # noqa: BLE001
        crps["dip_qar"] = float("nan")
        quant["dip_qar"] = nan3

    try:
        grid = (np.arange(512) + 0.5) / 512.0
        crps["dip_conf_t"] = crps_empirical(y, _conf_t_quantiles(rets_long, grid))
        quant["dip_conf_t"] = _conf_t_quantiles(rets_long, np.asarray(TAUS))
    except Exception:  # noqa: BLE001
        crps["dip_conf_t"] = float("nan")
        quant["dip_conf_t"] = nan3

    try:
        grid = (np.arange(512) + 0.5) / 512.0
        crps["dip_regime"] = crps_empirical(y, _regime_quantiles(rets_long, grid))
        quant["dip_regime"] = _regime_quantiles(rets_long, np.asarray(TAUS))
    except Exception:  # noqa: BLE001
        crps["dip_regime"] = float("nan")
        quant["dip_regime"] = nan3

    if np.isfinite(nu_hat):
        grid = (np.arange(rets.size) + 0.5) / rets.size
        t_copy = st.t.ppf(grid, nu_hat, loc=loc_hat, scale=scale_hat)
        blend = np.concatenate([rets, t_copy[np.isfinite(t_copy)]])
        crps["dip_blend"] = crps_empirical(y, blend)
        quant["dip_blend"] = np.quantile(blend, TAUS)
    else:
        crps["dip_blend"] = float("nan")
        quant["dip_blend"] = nan3

    return crps, quant


# --------------------------------------------------------------------------
# Receipt / inference (shared by run mode and merge mode)
# --------------------------------------------------------------------------


def summarize(
    loss_matrix: np.ndarray,
    pinball_cube: np.ndarray,
    asset_ids: np.ndarray,
    asset_names: list[str],
    model_names: list[str],
    targets: list[str],
    *,
    seed: int,
    n_boot: int,
    target_time_ns: np.ndarray | None = None,
    bar_interval_ns: int | None = None,
    requested_per_asset: int | None = None,
    scores_verified: bool = True,
) -> dict[str, Any]:
    from quant_fund.metrics.inference import diebold_mariano
    from quant_fund.metrics.snooping import model_confidence_set, spa_test

    validate_losses(loss_matrix, pinball_cube, asset_ids, asset_names, model_names, len(TAUS))
    if len(set(targets)) != len(targets) or not set(targets).issubset(model_names):
        raise ValueError("targets must be unique members of model_names")
    complete = np.isfinite(loss_matrix).all(axis=1) & np.isfinite(pinball_cube).all(axis=(1, 2))
    lm, inference = aligned_inference_losses(
        loss_matrix, complete, asset_ids, asset_names, target_time_ns, bar_interval_ns
    )
    if not scores_verified:
        lm = np.empty((0, len(model_names)))
        inference = inference | {
            "status": "unavailable",
            "reason": "unverified_scoring_contract",
            "n_observations": 0,
        }
    pc = pinball_cube[complete]
    aids = asset_ids[complete]

    dm: dict[str, dict[str, Any]] = {}
    spa_by_target: dict[str, Any] = {}
    for target in targets if inference["status"] == "computed" else []:
        t_idx = model_names.index(target)
        dm[target] = {}
        for j, m in enumerate(model_names):
            if m == target:
                continue
            res = diebold_mariano(lm[:, t_idx], lm[:, j], name_a=target, name_b=m)
            dm[target][m] = {
                "mean_loss_diff_target_minus_challenger": res.mean_loss_diff,
                "statistic": res.statistic,
                "p_value": res.p_value,
                "preferred": res.preferred,
                "n": res.n,
            }
        others = [j for j in range(len(model_names)) if j != t_idx]
        spa = spa_test(lm[:, [t_idx]] - lm[:, others], n_boot=n_boot, seed=seed)
        spa_by_target[target] = {
            "statistic": spa.statistic,
            "p_lower": spa.p_lower,
            "p_consistent": spa.p_consistent,
            "p_upper": spa.p_upper,
        }
    # model_confidence_set uses a higher-is-better convention (it eliminates the
    # smallest demeaned mean); feed negative losses so superior = retained.
    mcs = (
        model_confidence_set(-lm, n_boot=n_boot, seed=seed)
        if inference["status"] == "computed"
        else None
    )
    mcs_valid = mcs is not None and np.isfinite(mcs.p_values).all()

    def finite_mean(values: np.ndarray) -> float | None:
        valid = values[np.isfinite(values)]
        return float(np.mean(valid)) if valid.size else None

    means = {m: finite_mean(loss_matrix[:, i]) for i, m in enumerate(model_names)}
    per_asset_means = {
        name: {m: finite_mean(loss_matrix[asset_ids == a, i]) for i, m in enumerate(model_names)}
        for a, name in enumerate(asset_names)
        if np.any(asset_ids == a)
    }
    pinball_means = {
        m: {str(tau): finite_mean(pinball_cube[:, i, k]) for k, tau in enumerate(TAUS)}
        for i, m in enumerate(model_names)
    }
    pinball_per_asset = {
        name: {
            m: {str(tau): finite_mean(pc[aids == a, i, k]) for k, tau in enumerate(TAUS)}
            for i, m in enumerate(model_names)
        }
        for a, name in enumerate(asset_names)
        if np.any(aids == a)
    }
    return {
        "n_rows": int(loss_matrix.shape[0]),
        "n_complete": int(complete.sum()),
        "n_dropped_incomplete": int((~complete).sum()),
        "inference": inference,
        "effect_sizes": paired_effect_intervals(lm, model_names, targets, n_boot=n_boot, seed=seed)
        if inference["status"] == "computed" and targets
        else {"status": "unavailable"},
        "bootstrap_sensitivity": block_sensitivity(
            lm, model_names, targets, n_boot=n_boot, seed=seed
        )
        if inference["status"] == "computed"
        else [],
        "coverage": coverage_summary(
            loss_matrix, pinball_cube, asset_ids, asset_names, model_names, requested_per_asset
        ),
        "score_table_scope": "available_per_model; use common_sample for paired comparisons",
        "common_sample": {
            "n_rows": int(complete.sum()),
            "mean_crps": {
                m: finite_mean(loss_matrix[complete, i]) for i, m in enumerate(model_names)
            },
            "mean_pinball": {
                m: {str(tau): finite_mean(pc[:, i, k]) for k, tau in enumerate(TAUS)}
                for i, m in enumerate(model_names)
            },
        },
        "inference_mean_crps": {m: finite_mean(lm[:, i]) for i, m in enumerate(model_names)},
        "n_scored_per_asset": {
            name: int(np.sum(asset_ids == a))
            for a, name in enumerate(asset_names)
            if np.any(asset_ids == a)
        },
        "mean_crps_pooled": means,
        "mean_crps_per_asset": per_asset_means,
        "mean_pinball_pooled": pinball_means,
        "mean_pinball_per_asset": pinball_per_asset,
        "diebold_mariano": dm,
        "spa": spa_by_target,
        "mcs_status": "computed" if mcs_valid else "unavailable",
        "mcs_included": {m: bool(mcs.included[i]) for i, m in enumerate(model_names)}
        if mcs_valid
        else {},
        "mcs_p_values": {m: float(mcs.p_values[i]) for i, m in enumerate(model_names)}
        if mcs_valid
        else {},
    }


def _print_table(receipt: dict[str, Any], model_names: list[str], targets: list[str]) -> None:
    def fmt(value: float | None) -> str:
        return f"{value:.6f}" if value is not None else "unavailable"

    print(f"n_rows={receipt['n_rows']} n_complete={receipt['n_complete']}")
    print(f"{'model':<20}{'mean_crps':>12}   per-asset mean CRPS")
    for m in model_names:
        detail = "  ".join(
            f"{s}={fmt(receipt['mean_crps_per_asset'][s][m])}"
            for s in receipt["mean_crps_per_asset"]
        )
        print(f"{m:<20}{fmt(receipt['mean_crps_pooled'][m]):>12}   {detail}")
    if receipt["inference"]["status"] != "computed":
        print(f"Inference unavailable: {receipt['inference']['reason']}")
        return
    print(f"Inference: {receipt['inference']['n_observations']} aligned target timestamps")
    for target in targets:
        for m, d in receipt["diebold_mariano"][target].items():
            print(
                f"DM {target} vs {m}: diff={d['mean_loss_diff_target_minus_challenger']:+.6f} "
                f"t={d['statistic']:+.3f} p={d['p_value']:.4f} preferred={d['preferred']}"
            )
        s = receipt["spa"][target]
        print(
            f"SPA (any challenger beats {target}): p_lower={s['p_lower']:.4f} "
            f"p_cons={s['p_consistent']:.4f} p_upper={s['p_upper']:.4f}"
        )
    if receipt["mcs_status"] == "computed":
        print(f"MCS @0.10: {[m for m in model_names if receipt['mcs_included'][m]]}")
    else:
        print("MCS unavailable: unstudentizable loss differences")


# --------------------------------------------------------------------------
# Target specs
# --------------------------------------------------------------------------


@dataclass
class TargetSpec:
    name: str
    kind: str  # kronos | chronos2 | bolt | timesfm
    model_dir: Path
    tokenizer_dir: Path | None = None


def parse_specs(args: argparse.Namespace) -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    for entry in args.kronos or []:
        name, _, rest = entry.partition("=")
        mdir, _, tdir = rest.partition(",")
        if not (name and mdir and tdir):
            raise SystemExit(f"--kronos expects name=model_dir,tokenizer_dir; got {entry!r}")
        specs.append(TargetSpec(name, "kronos", Path(mdir), Path(tdir)))
    for attr, kind in (("chronos2", "chronos2"), ("bolt", "bolt"), ("timesfm", "timesfm")):
        for entry in getattr(args, attr) or []:
            name, _, mdir = entry.partition("=")
            if not (name and mdir):
                raise SystemExit(f"--{attr} expects name=dir; got {entry!r}")
            specs.append(TargetSpec(name, kind, Path(mdir)))
    for t in dict.fromkeys(args.target or []):
        if t == "kronos":
            if not (args.kronos_repo and args.model_dir and args.tokenizer_dir):
                raise SystemExit(
                    "--target kronos requires --kronos-repo, --model-dir, --tokenizer-dir"
                )
            specs.append(
                TargetSpec("kronos", "kronos", Path(args.model_dir), Path(args.tokenizer_dir))
            )
        elif t == "chronos":
            if not args.chronos_model:
                raise SystemExit("--target chronos requires --chronos-model")
            specs.append(TargetSpec("chronos", "chronos2", Path(args.chronos_model)))
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise SystemExit(f"duplicate target names: {names}")
    if not specs:
        raise SystemExit("no targets: pass --kronos/--chronos2/--bolt/--timesfm or --target")
    return specs


# --------------------------------------------------------------------------
# Merge mode
# --------------------------------------------------------------------------


def merge_parts(args: argparse.Namespace) -> int:
    _ensure_new_output(args.merge_out)
    paths = [p.resolve() for p in args.merge_parts]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate shard paths")
    part_hashes = {str(p): _sha256(p) for p in paths}
    if len(set(part_hashes.values())) != len(paths):
        raise ValueError("duplicate shard contents")
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            parts.append({key: archive[key] for key in archive.files})
    model_names = [str(x) for x in parts[0]["model_names"]]
    for p in parts[1:]:
        if [str(x) for x in p["model_names"]] != model_names:
            raise SystemExit("part model_names mismatch")
    metas = [json.loads(str(p["meta_json"])) for p in parts]
    protocol_keys = (
        "lookback",
        "window",
        "garch_window",
        "samples_per_origin",
        "origins_per_asset",
        "seed",
        "taus",
    )
    for meta in metas:
        if meta.get("partial") or not meta.get("bars_sha256") or not meta.get("artifact_sha256"):
            raise ValueError("partial or unprovenanced shard cannot be merged")
        if any(key not in meta["config"] for key in protocol_keys):
            raise ValueError("shard is missing the complete evaluation protocol")
        for key in protocol_keys:
            if meta["config"][key] != metas[0]["config"][key]:
                raise ValueError(f"part protocol mismatch: {key}")
        if meta.get("scoring_contract") != metas[0].get("scoring_contract"):
            raise ValueError("part scoring contract mismatch; rerun targets with the same adapter")
        if meta["config"]["taus"] != list(TAUS):
            raise ValueError("part quantile levels do not match evaluator")
    asset_names: list[str] = []
    remap = []
    for p, meta in zip(parts, metas, strict=True):
        local_ids = np.asarray(p["asset_ids"])
        local_names = [str(x) for x in meta["asset_names"]]
        validate_losses(
            p["crps_matrix"], p["pinball_cube"], local_ids, local_names, model_names, len(TAUS)
        )
        if set(local_names).intersection(asset_names):
            raise ValueError("duplicate asset across shards; do not pool seeds or repeated runs")
        offset = len(asset_names)
        remap.append(np.asarray([offset + int(i) for i in local_ids]))
        asset_names.extend(local_names)
    loss_matrix = np.concatenate([p["crps_matrix"] for p in parts], axis=0)
    pinball_cube = np.concatenate([p["pinball_cube"] for p in parts], axis=0)
    asset_ids = np.concatenate(remap, axis=0)
    targets = [str(t) for t in metas[0]["targets"]]
    for meta in metas[1:]:
        if [str(t) for t in meta["targets"]] != targets:
            raise SystemExit("part targets mismatch")

    bars_sha: dict[str, str] = {}
    artifact_sha: dict[str, str] = {}
    for meta in metas:
        for field, destination in (("bars_sha256", bars_sha), ("artifact_sha256", artifact_sha)):
            for key, digest in meta[field].items():
                if key in destination and destination[key] != digest:
                    raise ValueError(f"conflicting {field}: {key}")
                destination[key] = digest
        if not set(targets).issubset(meta["artifact_sha256"]):
            raise ValueError("missing target artifact hash")

    times_by_part = []
    intervals = []
    timestamp_source = "recorded"
    for part, meta in zip(parts, metas, strict=True):
        if "target_time_ns" in part and meta.get("bar_interval_ns"):
            times_by_part.append(part["target_time_ns"])
            intervals.append(int(meta["bar_interval_ns"]))
        elif getattr(args, "bars_root", None) is not None:
            recovered, interval = _recover_target_times(part, meta, args.bars_root)
            times_by_part.append(recovered)
            intervals.append(interval)
            timestamp_source = "reconstructed_from_hash_verified_bars"
    if times_by_part and len(times_by_part) != len(parts):
        raise ValueError(
            "only some shards have timestamps; provide --bars-root for all legacy shards"
        )
    if len(set(intervals)) > 1:
        raise ValueError("mixed bar intervals cannot share a pooled inference claim")
    target_times = np.concatenate(times_by_part) if times_by_part else None
    interval = intervals[0] if intervals else None

    summary = summarize(
        loss_matrix,
        pinball_cube,
        asset_ids,
        asset_names,
        model_names,
        targets,
        seed=args.seed,
        n_boot=args.n_boot,
        target_time_ns=target_times,
        bar_interval_ns=interval,
        requested_per_asset=metas[0]["config"]["origins_per_asset"],
        scores_verified=metas[0].get("scoring_contract") in VERIFIED_SCORING_CONTRACTS,
    )
    receipt = {
        "schema": "sota_eval.v4",
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": metas[0]["config"]
        | {"merged_parts": len(parts), "inference_seed": args.seed, "n_boot": args.n_boot},
        "scoring_contract": metas[0].get("scoring_contract", "legacy_unverified"),
        "legacy_scores_require_rerun": metas[0].get("scoring_contract")
        not in VERIFIED_SCORING_CONTRACTS,
        "source_parts_sha256": part_hashes,
        "timestamp_source": timestamp_source if target_times is not None else "unavailable",
        "bar_interval_ns": interval,
        "implementation_sha256": _implementation_hashes(),
        "artifact_sha256": artifact_sha,
        "bars_sha256": bars_sha,
        "quantile_completion": (
            "trapezoid on dense grid; exponential tails outside native quantile grid"
        ),
        "challengers": [m for m in model_names if m not in targets],
        **summary,
        "research_only": True,
        "live_pnl_claim": False,
        "disclaimer": (
            "Causal walk-forward evaluation on public Binance klines with proper "
            "scores only. Not a live-trading or deployability claim."
        ),
    }
    _write_result(
        args.merge_out,
        receipt,
        crps_matrix=loss_matrix,
        pinball_cube=pinball_cube,
        asset_ids=asset_ids,
        model_names=np.asarray(model_names),
        **({"target_time_ns": target_times} if target_times is not None else {}),
        meta={
            "asset_names": asset_names,
            "targets": targets,
            "config": receipt["config"],
            "bars_sha256": bars_sha,
            "artifact_sha256": artifact_sha,
            "bar_interval_ns": interval,
            "scoring_contract": metas[0].get("scoring_contract"),
            "source_parts_sha256": part_hashes,
            "timestamp_source": receipt["timestamp_source"],
        },
    )
    _print_table(receipt, model_names, targets)
    print(f"receipt: {args.merge_out}")
    return 0


def _implementation_hashes() -> dict[str, str]:
    from quant_fund.metrics import inference, snooping
    from quant_fund.research import sota_evidence

    paths = [
        Path(__file__),
        Path(inference.__file__),
        Path(snooping.__file__),
        Path(sota_evidence.__file__),
    ]
    return {p.name: _sha256(p) for p in paths}


def _recover_target_times(part: dict, meta: dict, bars_root: Path) -> tuple[np.ndarray, int]:
    """Recover legacy row times only from the exact hashed input and origin rule."""
    if len(meta["asset_names"]) != 1 or len(meta["bars_sha256"]) != 1:
        raise ValueError("legacy timestamp recovery requires a single-asset raw shard")
    filename, digest = next(iter(meta["bars_sha256"].items()))
    if Path(filename).name != filename:
        raise ValueError("bar hash key must be a filename")
    path = bars_root / filename
    if not path.is_file() or _sha256(path) != digest:
        raise ValueError(f"missing or hash-mismatched bars: {path}")
    frame = pl.read_parquet(path).sort("event_time")
    times, interval = validate_bars(frame)
    # Shards may relabel assets with an interval suffix (e.g. "XRPUSDT-4h") to
    # disambiguate pooled merges; the underlying bar security is the prefix.
    shard_asset = meta["asset_names"][0]
    bar_security = str(frame["security_id"][0])
    if shard_asset != bar_security and not shard_asset.startswith(bar_security + "-"):
        raise ValueError("bar security does not match shard asset")
    config = meta["config"]
    start = max(
        config["lookback"], config["window"] + 1, frame.height - config["origins_per_asset"] - 1
    )
    target_times = times[start + 1 :]
    if len(target_times) != len(part["crps_matrix"]):
        raise ValueError("shard row count does not match the recorded origin protocol")
    return target_times, interval


# --------------------------------------------------------------------------
# Run mode
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="append",
        choices=["kronos", "chronos"],
        help="Legacy target selector (repeatable)",
    )
    parser.add_argument(
        "--kronos-repo",
        type=Path,
        help="Path to the upstream Kronos checkout (for --target kronos)",
    )
    parser.add_argument("--model-dir", type=Path, help="Kronos model dir (legacy)")
    parser.add_argument("--tokenizer-dir", type=Path, help="Kronos tokenizer dir (legacy)")
    parser.add_argument(
        "--chronos-model", type=Path, help="Local Chronos-2 artifact dir (legacy --target chronos)"
    )
    parser.add_argument(
        "--kronos",
        action="append",
        metavar="NAME=MODEL_DIR,TOK_DIR",
        help="Kronos target (repeatable)",
    )
    parser.add_argument(
        "--chronos2", action="append", metavar="NAME=DIR", help="Chronos-2 target (repeatable)"
    )
    parser.add_argument(
        "--bolt", action="append", metavar="NAME=DIR", help="Chronos-Bolt target (repeatable)"
    )
    parser.add_argument(
        "--timesfm", action="append", metavar="NAME=DIR", help="TimesFM-2.5 target (repeatable)"
    )
    parser.add_argument(
        "--bars", type=Path, nargs="+", help="Collected source parquet files (one symbol each)"
    )
    parser.add_argument("--origins", type=int, default=120)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--lookback", type=int, default=400)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--garch-window", type=int, default=750)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="torch.set_num_threads cap per process (0 = library default)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="write a partial .npz every N origins per asset (0 = off)",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--merge-parts", type=Path, nargs="+", help="Merge mode: concatenate partial .npz shards"
    )
    parser.add_argument("--merge-out", type=Path, help="Merge mode output receipt path")
    parser.add_argument(
        "--bars-root",
        type=Path,
        help="Recover legacy shard timestamps from exact SHA256-matching input bars",
    )
    args = parser.parse_args()

    if args.merge_parts:
        if not args.merge_out:
            parser.error("--merge-parts requires --merge-out")
        return merge_parts(args)
    if not args.bars or not args.out:
        parser.error("run mode requires --bars and --out")
    _ensure_new_output(args.out)
    if (
        min(args.origins, args.window, args.garch_window, args.n_boot) < 1
        or min(args.samples, args.lookback) < 2
    ):
        parser.error(
            "origins/windows/bootstrap count must be positive; samples/lookback must be >=2"
        )
    if len({p.name for p in args.bars}) != len(args.bars):
        parser.error("bar filenames must be unique")

    specs = parse_specs(args)
    model_names = [s.name for s in specs] + list(BASELINES)
    if len(set(model_names)) != len(model_names):
        parser.error("target names cannot collide with challenger names")
    for spec in specs:
        if not spec.model_dir.is_dir() or (
            spec.tokenizer_dir is not None and not spec.tokenizer_dir.is_dir()
        ):
            parser.error(f"target {spec.name} requires existing local model/tokenizer directories")
    initial_bars_sha = {p.name: _sha256(p) for p in args.bars}

    import torch

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    predictors: dict[str, Any] = {}
    if any(s.kind == "kronos" for s in specs):
        if not args.kronos_repo:
            parser.error("--kronos targets require --kronos-repo")
        sys.path.insert(0, str(args.kronos_repo.resolve()))
        from model import Kronos, KronosPredictor, KronosTokenizer

        torch.manual_seed(args.seed)
        for s in specs:
            if s.kind != "kronos":
                continue
            tokenizer = KronosTokenizer.from_pretrained(str(s.tokenizer_dir), local_files_only=True)
            kmodel = Kronos.from_pretrained(str(s.model_dir), local_files_only=True)
            kmodel.eval()
            tokenizer.eval()
            predictors[s.name] = KronosPredictor(
                kmodel, tokenizer, device="cpu", max_context=2048, clip=5.0
            )
    for s in specs:
        if s.kind == "chronos2":
            from chronos import Chronos2Pipeline

            predictors[s.name] = Chronos2Pipeline.from_pretrained(
                str(s.model_dir), device_map="cpu"
            )
        elif s.kind == "bolt":
            from chronos import ChronosBoltPipeline

            predictors[s.name] = ChronosBoltPipeline.from_pretrained(
                str(s.model_dir), device_map="cpu"
            )
        elif s.kind == "timesfm":
            import timesfm

            fm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                str(s.model_dir), torch_compile=False
            )
            fm.compile(
                timesfm.ForecastConfig(
                    max_context=max(args.lookback, 512),
                    max_horizon=1,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    infer_is_positive=False,
                    fix_quantile_crossing=True,
                )
            )
            predictors[s.name] = fm

    from quant_fund.metrics.scoring import crps_empirical, pinball_loss

    crps_rows: list[np.ndarray] = []
    pin_rows: list[np.ndarray] = []
    asset_ids: list[int] = []
    asset_names: list[str] = []
    target_times: list[int] = []
    bar_interval_ns: int | None = None
    t0 = time.time()

    def _write_partial() -> None:
        """Checkpoint the aligned loss cube so a killed run keeps its rows."""
        if not crps_rows:
            return
        part = args.out.with_suffix(".partial.npz")
        part.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            part,
            crps_matrix=np.vstack(crps_rows),
            pinball_cube=np.stack(pin_rows),
            asset_ids=np.asarray(asset_ids),
            model_names=np.asarray(model_names),
            target_time_ns=np.asarray(target_times, dtype=np.int64),
            meta_json=np.array(
                json.dumps(
                    {
                        "asset_names": asset_names,
                        "targets": [s.name for s in specs],
                        "config": {"targets": [s.name for s in specs], "seed": args.seed},
                        "bars_sha256": {},
                        "artifact_sha256": {},
                        "partial": True,
                        "bar_interval_ns": bar_interval_ns,
                        "scoring_contract": SCORING_CONTRACT,
                    }
                )
            ),
        )

    for a_idx, bars_path in enumerate(args.bars):
        frame = pl.read_parquet(bars_path).sort("event_time")
        event_times, interval = validate_bars(frame)
        if bar_interval_ns is not None and interval != bar_interval_ns:
            raise ValueError("run separate evaluations for different bar intervals")
        bar_interval_ns = interval
        symbol = str(frame["security_id"][0])
        if symbol in asset_names:
            raise ValueError(f"duplicate asset: {symbol}")
        asset_names.append(symbol)
        required = {"event_time", "available_time", "open", "high", "low", "close", "volume"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{bars_path}: missing columns {sorted(missing)}")
        if not (frame["event_time"].to_numpy()[:-1] < frame["event_time"].to_numpy()[1:]).all():
            raise ValueError(f"{bars_path}: event_time not strictly increasing")
        closes = frame["close"].cast(pl.Float64).to_numpy()
        n = frame.height
        # Causal feature table for the LightGBM challenger: X[j] is built from
        # returns strictly before j plus the weekday of the bar being forecast.
        rets_all = np.diff(closes) / closes[:-1]
        dow = pd.to_datetime(event_times[1:], unit="ns", utc=True).dayofweek.to_numpy(dtype=float)
        X = _lgbm_features(rets_all, dow)
        lgbm_col = model_names.index("dip_lgbm_q")
        first_origin = max(args.lookback, args.window + 1, n - args.origins - 1)
        if first_origin >= n - 1:
            raise ValueError(f"{symbol}: not enough bars ({n}) for the requested protocol")

        for i in range(first_origin, n - 1):
            hist = frame.slice(0, i + 1)
            closes_hist = closes[: i + 1]
            rets = np.diff(closes_hist[-(args.window + 1) :]) / closes_hist[-(args.window + 1) : -1]
            long_start = max(0, closes_hist.size - (args.garch_window + 1))
            rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
            y = float(closes[i + 1] / closes[i] - 1.0)

            row = np.full(len(model_names), np.nan)
            prow = np.full((len(model_names), len(TAUS)), np.nan)

            for s in specs:
                col = model_names.index(s.name)
                try:
                    if s.kind == "kronos":
                        pdf = (
                            hist.select(
                                ["open", "high", "low", "close", "volume"]
                                + (["amount"] if "amount" in hist.columns else [])
                            )
                            .tail(args.lookback)
                            .to_pandas()
                        )
                        # Upstream calc_time_stamps requires Series (.dt accessor).
                        x_ts = pd.Series(
                            pd.to_datetime(hist["event_time"].tail(args.lookback).to_numpy())
                        )
                        step = x_ts.iloc[-1] - x_ts.iloc[-2]
                        y_ts = pd.Series([x_ts.iloc[-1] + step])
                        draws = kronos_samples(predictors[s.name], pdf, x_ts, y_ts, args.samples)
                        row[col] = crps_empirical(y, draws)
                        prow[col, :] = np.quantile(draws, TAUS)
                    elif s.kind == "chronos2":
                        q = chronos2_quantiles(predictors[s.name], closes_hist[-args.lookback :])
                        row[col] = crps_from_quantiles(y, CHRONOS_LEVELS, q)
                        prow[col, :] = [qf_at(CHRONOS_LEVELS, q, t) for t in TAUS]
                    elif s.kind == "bolt":
                        q = bolt_quantiles(predictors[s.name], closes_hist[-args.lookback :])
                        row[col] = crps_from_quantiles(y, BOLT_LEVELS, q)
                        prow[col, :] = [qf_at(BOLT_LEVELS, q, t) for t in TAUS]
                    elif s.kind == "timesfm":
                        q = timesfm_quantiles(predictors[s.name], closes_hist[-args.lookback :])
                        row[col] = crps_from_quantiles(y, TIMESFM_LEVELS, q)
                        prow[col, :] = [qf_at(TIMESFM_LEVELS, q, t) for t in TAUS]
                except Exception:  # noqa: BLE001 - target failure -> NaN, disclosed
                    row[col] = np.nan
                    prow[col, :] = np.nan

            try:
                q_l = lgbm_quantiles(X, rets_all, i, args.garch_window)
                row[lgbm_col] = crps_from_quantiles(y, LGBM_TAUS, q_l)
                for k, tau in enumerate(TAUS):
                    qk = qf_at(LGBM_TAUS, q_l, tau)
                    prow[lgbm_col, k] = (
                        float(pinball_loss(np.array([y]), np.array([qk]), tau)[0])
                        if np.isfinite(qk)
                        else np.nan
                    )
            except Exception:  # noqa: BLE001
                row[lgbm_col] = np.nan
                prow[lgbm_col, :] = np.nan

            dcrps, dq = dip_challengers(rets, rets_long, y)
            for name, val in dcrps.items():
                col = model_names.index(name)
                row[col] = val
                for k in range(len(TAUS)):
                    yq = dq[name][k]
                    prow[col, k] = (
                        float(pinball_loss(np.array([y]), np.array([yq]), TAUS[k])[0])
                        if np.isfinite(yq)
                        else np.nan
                    )

            # Targets report raw quantiles at TAUS; convert to pinball losses.
            for s in specs:
                col = model_names.index(s.name)
                if np.isfinite(row[col]) and np.isfinite(prow[col, :]).all():
                    qs = prow[col, :].copy()
                    for k, tau in enumerate(TAUS):
                        prow[col, k] = float(pinball_loss(np.array([y]), np.array([qs[k]]), tau)[0])

            crps_rows.append(row)
            pin_rows.append(prow)
            asset_ids.append(a_idx)
            target_times.append(int(event_times[i + 1]))
            done = i - first_origin + 1
            total_i = n - 1 - first_origin
            if done % 25 == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"[{symbol}] {done}/{total_i} origins {rate:.2f}/s", flush=True)
            if args.checkpoint_every and done % args.checkpoint_every == 0:
                _write_partial()
        print(f"[{symbol}] scored {sum(1 for x in asset_ids if x == a_idx)} origins", flush=True)
        if args.checkpoint_every:
            _write_partial()

    loss_matrix = np.vstack(crps_rows)
    pinball_cube = np.stack(pin_rows)
    aids = np.asarray(asset_ids)
    targets = [s.name for s in specs]

    summary = summarize(
        loss_matrix,
        pinball_cube,
        aids,
        asset_names,
        model_names,
        targets,
        seed=args.seed,
        n_boot=args.n_boot,
        target_time_ns=np.asarray(target_times, dtype=np.int64),
        bar_interval_ns=bar_interval_ns,
        requested_per_asset=args.origins,
    )
    artifact_sha = {
        s.name: _dir_digest(s.model_dir)
        for s in specs
        if s.model_dir is not None and s.model_dir.exists()
    }
    for s in specs:
        if s.tokenizer_dir is not None and s.tokenizer_dir.exists():
            artifact_sha[f"{s.name}_tokenizer"] = _dir_digest(s.tokenizer_dir)
    bars_sha = {p.name: _sha256(p) for p in args.bars}
    if bars_sha != initial_bars_sha:
        raise ValueError(
            "bar files changed during evaluation; refusing to publish mixed provenance"
        )
    receipt = {
        "schema": "sota_eval.v4",
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "config": {
            "targets": targets,
            "target_specs": {
                s.name: {
                    "kind": s.kind,
                    "model_dir": str(s.model_dir),
                    "tokenizer_dir": str(s.tokenizer_dir) if s.tokenizer_dir else None,
                }
                for s in specs
            },
            "origins_per_asset": args.origins,
            "samples_per_origin": args.samples,
            "lookback": args.lookback,
            "window": args.window,
            "garch_window": args.garch_window,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "taus": list(TAUS),
        },
        "artifact_sha256": artifact_sha,
        "bars_sha256": bars_sha,
        "bar_interval_ns": bar_interval_ns,
        "scoring_contract": SCORING_CONTRACT,
        "implementation_sha256": _implementation_hashes(),
        "quantile_completion": (
            "trapezoid on dense grid; exponential tails outside native quantile grid"
        ),
        "challengers": list(BASELINES),
        **summary,
        "research_only": True,
        "live_pnl_claim": False,
        "disclaimer": (
            "Causal walk-forward evaluation on public Binance klines with proper "
            "scores only. Not a live-trading or deployability claim."
        ),
    }
    _write_result(
        args.out,
        receipt,
        crps_matrix=loss_matrix,
        pinball_cube=pinball_cube,
        asset_ids=aids,
        model_names=np.asarray(model_names),
        target_time_ns=np.asarray(target_times, dtype=np.int64),
        meta={
            "asset_names": asset_names,
            "targets": targets,
            "config": receipt["config"],
            "bars_sha256": bars_sha,
            "artifact_sha256": artifact_sha,
            "bar_interval_ns": bar_interval_ns,
            "scoring_contract": SCORING_CONTRACT,
            "implementation_sha256": receipt["implementation_sha256"],
        },
    )

    print(f"elapsed={receipt['elapsed_seconds']}s", flush=True)
    _print_table(receipt, model_names, targets)
    print(f"receipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
