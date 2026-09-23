"""Conformal Risk Control for monotone tail losses. No Sharpe.

Angelopoulos, Bates, Malik, Jordan (2022): choose the smallest threshold λ
such that the finite-sample CRC statistic is at most α,

    (n * L̂_n(λ) + B) / (n + 1) ≤ α,

where L̂_n is the empirical mean of a bounded monotone loss on calibration.
For 0-1 VaR-hit / drawdown-exceedance, L = 1{loss > bound} and B = 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.metrics.probability import kupiec_pof
from quant_fund.models.base import JoblibMixin, ModelMeta

Array = NDArray[np.float64]


def loss_hit(y_loss: Array, bound: Array) -> Array:
    """Indicator loss: 1 if realized loss strictly exceeds the bound."""
    return (np.asarray(y_loss, dtype=float) > np.asarray(bound, dtype=float)).astype(float)


def _finite_1d(values: Array) -> Array:
    v = np.asarray(values, dtype=float).reshape(-1)
    return v[np.isfinite(v)]


def _crc_stat(lhat: Array | float, n: int, b: float) -> Array | float:
    return (float(n) * lhat + float(b)) / float(n + 1)


def crc_threshold(losses: Array, candidates: Array, alpha: float, B: float = 1.0) -> float:
    """Smallest candidate λ with CRC statistic ≤ α. Empty set → λ_max.

    For hit loss, L̂_n(λ) is the empirical mean of 1{loss > λ} and `candidates`
    are candidate bounds (sorted unique losses or a grid).
    """
    if not np.isfinite(B) or float(B) <= 0.0:
        raise ValueError("B must be a finite positive upper bound on the loss")
    if not 0.0 < float(alpha) < float(B):
        raise ValueError("alpha must be in (0, B)")
    y = _finite_1d(losses)
    cands = np.unique(_finite_1d(candidates))
    if y.size == 0:
        raise ValueError("losses must contain at least one finite value")
    if cands.size == 0:
        raise ValueError("candidates must contain at least one finite value")
    n = int(y.size)
    lhat = np.mean(y[:, None] > cands[None, :], axis=0)
    ok = _crc_stat(lhat, n, B) <= float(alpha)
    if not bool(np.any(ok)):
        return float(cands[-1])
    return float(cands[ok][0])


def _paired(losses: Array, bounds: Array) -> tuple[Array, Array]:
    y = np.asarray(losses, dtype=float).reshape(-1)
    b = np.asarray(bounds, dtype=float)
    if b.size == 1:
        b = np.full_like(y, float(np.reshape(b, -1)[0]))
    else:
        b = b.reshape(-1)
    if y.size != b.size:
        raise ValueError("losses and bounds must have the same length")
    mask = np.isfinite(y) & np.isfinite(b)
    return y[mask], b[mask]


def _expansion_candidates(residuals: Array, alpha: float) -> Array:
    """Non-negative expansions. Empirical hit risk is piecewise-constant on residuals."""
    r = _finite_1d(residuals)
    pts = [0.0]
    if r.size:
        pts.extend(float(v) for v in r[r >= 0.0])
        qhat = conformal_quantile(r, alpha)
        if np.isfinite(qhat) and qhat >= 0.0:
            pts.append(float(qhat))
    return np.unique(np.asarray(pts, dtype=float))


class ConformalRiskControl(JoblibMixin):
    """Expand a base VaR / drawdown bound until CRC expected hit risk ≤ α."""

    def __init__(self, alpha: float = 0.05, B: float = 1.0) -> None:
        if not np.isfinite(B) or float(B) <= 0.0:
            raise ValueError("B must be a finite positive upper bound on the loss")
        if not 0.0 < float(alpha) < float(B):
            raise ValueError("alpha must be in (0, B)")
        self.alpha = float(alpha)
        self.B = float(B)
        self.lambda_hat = 0.0
        self.n_cal = 0

    def calibrate(self, losses: Array, bounds: Array) -> ConformalRiskControl:
        y, b = _paired(losses, bounds)
        self.n_cal = int(y.size)
        if self.n_cal == 0:
            self.lambda_hat = 0.0
            return self
        resid = y - b
        cands = _expansion_candidates(resid, self.alpha)
        self.lambda_hat = crc_threshold(resid, cands, self.alpha, self.B)
        return self

    def predict_bound(self, bounds: Array) -> Array:
        return np.asarray(bounds, dtype=float) + float(self.lambda_hat)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="conformal",
            name="crc",
            version="v1",
            extra={"alpha": self.alpha, "B": self.B, "lambda_hat": self.lambda_hat},
        )


def bench_crc_var(losses: Array, base_bound: Array, alpha: float = 0.05) -> dict[str, float]:
    """Calibration CRC diagnostic for a VaR-style bound. Keys are risk, not Sharpe."""
    crc = ConformalRiskControl(alpha=alpha).calibrate(losses, base_bound)
    y, b = _paired(losses, base_bound)
    n = int(y.size)
    if n == 0:
        return {
            "risk": float("nan"),
            "nominal": float(alpha),
            "n": 0.0,
            "lambda_hat": float(crc.lambda_hat),
        }
    pred = crc.predict_bound(b)
    hits = loss_hit(y, pred)
    risk = float(np.mean(hits))
    _rate, lr, p_val = kupiec_pof(hits, alpha)
    return {
        "risk": risk,
        "nominal": float(alpha),
        "n": float(n),
        "lambda_hat": float(crc.lambda_hat),
        "crc_stat": float(_crc_stat(risk, n, crc.B)),
        "kupiec_lr": float(lr),
        "kupiec_p": float(p_val),
    }
