"""Online conformal risk control: Gibbs–Candès update on the CRC threshold.

Batch CRC (Angelopoulos, Bates, Malik, Jordan 2022) picks one λ on a
calibration window. Adaptive conformal inference (Gibbs & Candès 2021)
tracks coverage, not a monotone tail loss. This wrapper initializes λ with
CRC, then applies the dual ACI integrator once per timestamp

    λ_{t+1} = max(0, λ_t + γ (L_t − α)),

so expected hit-risk tracks α over time. Cross-sectional names on the same
date share one update; dates are never stacked. Lab scores are mean hit
risk, nominal α, and n. No Sharpe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.cross_section import _date_keys
from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.crc import ConformalRiskControl, loss_hit

Array = NDArray[np.float64]


def _as_1d(values: Array | float) -> Array:
    return np.asarray(values, dtype=float).reshape(-1)


def _align_pair(loss: Array | float, base_bound: Array | float) -> tuple[Array, Array]:
    y = _as_1d(loss)
    b = _as_1d(base_bound)
    if b.size == 1 and y.size != 1:
        b = np.full(y.size, float(b[0]))
    elif y.size == 1 and b.size != 1:
        y = np.full(b.size, float(y[0]))
    if y.size != b.size:
        raise ValueError("loss and base_bound must have the same length")
    return y, b


def _date_sort_value(date: object, key: str) -> tuple[int, float | str]:
    if isinstance(date, np.datetime64):
        return (0, float(date.astype("datetime64[ns]").astype(np.int64)))
    if isinstance(date, (np.integer, int)):
        return (0, float(date))
    if isinstance(date, (np.floating, float)):
        value = float(date)
        if np.isfinite(value):
            return (0, value)
        return (1, key)
    stamp = getattr(date, "timestamp", None)
    if callable(stamp):
        try:
            return (0, float(stamp()))
        except (OSError, TypeError, ValueError):
            pass
    try:
        return (0, float(key))
    except ValueError:
        return (1, key)


def _ordered_date_keys(dates: NDArray[Any] | list[object]) -> tuple[list[str], list[str]]:
    raw = list(dates)
    keys = _date_keys(raw)
    first: dict[str, object] = {}
    for key, date in zip(keys, raw, strict=False):
        if key not in first:
            first[key] = date
    order = sorted(first, key=lambda key: _date_sort_value(first[key], key))
    return keys, order


@dataclass
class OnlineCRCPath:
    bounds: Array
    hit: Array
    running_risk: Array
    lambda_t: Array
    dates: list[str] = field(default_factory=list)


class OnlineCRC(JoblibMixin):
    """Expand a base VaR / drawdown bound with an online CRC threshold."""

    def __init__(self, alpha: float = 0.05, gamma: float = 0.05, B: float = 1.0) -> None:
        if not np.isfinite(B) or float(B) <= 0.0:
            raise ValueError("B must be a finite positive upper bound on the loss")
        if not 0.0 < float(alpha) < float(B):
            raise ValueError("alpha must be in (0, B)")
        if not np.isfinite(gamma) or float(gamma) <= 0.0:
            raise ValueError("gamma must be > 0")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.B = float(B)
        self.lambda_t = 0.0
        self.running_risk = 0.0
        self._hit_sum = 0.0
        self._n = 0

    def initialize(self, losses: Array, base_bounds: Array) -> OnlineCRC:
        crc = ConformalRiskControl(alpha=self.alpha, B=self.B).calibrate(losses, base_bounds)
        self.lambda_t = float(crc.lambda_hat)
        self.running_risk = 0.0
        self._hit_sum = 0.0
        self._n = 0
        return self

    def _observe(self, hits: Array) -> float:
        finite = np.asarray(hits, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return self.running_risk
        self._hit_sum += float(finite.sum())
        self._n += int(finite.size)
        self.running_risk = self._hit_sum / float(self._n)
        return self.running_risk

    def _adapt(self, risk: float) -> None:
        nxt = self.lambda_t + self.gamma * (float(risk) - self.alpha)
        self.lambda_t = float(max(0.0, nxt))

    def predict_bound(self, base_bound: Array | float) -> Array:
        return np.asarray(base_bound, dtype=float) + float(self.lambda_t)

    def update(self, loss: Array | float, base_bound: Array | float) -> Array:
        """Apply λ_t, then one Gibbs–Candès step from that timestamp's hit risk."""
        y, base = _align_pair(loss, base_bound)
        bound = self.predict_bound(base)
        mask = np.isfinite(y) & np.isfinite(base)
        hits = np.full(y.size, np.nan)
        if bool(np.any(mask)):
            hits[mask] = loss_hit(y[mask], bound[mask])
            risk = float(np.mean(hits[mask]))
            self._observe(hits[mask])
            self._adapt(risk)
        return bound

    def run(
        self,
        losses: Array,
        base_bounds: Array,
        dates: NDArray[Any] | list[object],
    ) -> OnlineCRCPath:
        """Walk dates in order. Names on one date share a single λ update."""
        y, base = _align_pair(losses, base_bounds)
        raw = list(dates)
        if len(raw) != y.size:
            raise ValueError("dates must have the same length as losses")
        keys, order = _ordered_date_keys(raw)
        out_bounds = np.full(y.size, np.nan)
        out_hit = np.full(y.size, np.nan)
        out_risk = np.full(y.size, np.nan)
        lambdas: list[float] = []
        kept: list[str] = []
        for key in order:
            idx = np.array([k == key for k in keys], dtype=bool)
            bound = self.update(y[idx], base[idx])
            out_bounds[idx] = bound
            finite = np.isfinite(y[idx]) & np.isfinite(base[idx])
            hits = np.full(int(idx.sum()), np.nan)
            if bool(np.any(finite)):
                hits[finite] = loss_hit(y[idx][finite], bound[finite])
            out_hit[idx] = hits
            out_risk[idx] = self.running_risk
            lambdas.append(self.lambda_t)
            kept.append(key)
        return OnlineCRCPath(
            bounds=out_bounds,
            hit=out_hit,
            running_risk=out_risk,
            lambda_t=np.asarray(lambdas, dtype=float),
            dates=kept,
        )

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="conformal",
            name="online_crc",
            version="v1",
            extra={
                "alpha": self.alpha,
                "gamma": self.gamma,
                "B": self.B,
                "lambda_t": self.lambda_t,
            },
        )


def _synthetic_loss_path(
    n_cal: int, n_test: int, seed: int
) -> tuple[Array, Array, NDArray[np.int64]]:
    rng = np.random.default_rng(int(seed))
    n = int(n_cal) + int(n_test)
    losses = rng.exponential(scale=0.04, size=n)
    base = np.full(n, 0.01)
    dates = np.arange(n, dtype=np.int64)
    return losses, base, dates


def bench_online_crc(
    alpha: float = 0.05,
    gamma: float = 0.05,
    B: float = 1.0,
    seed: int = 12,
    n_cal: int = 200,
    n_test: int = 800,
    *,
    losses: Array | None = None,
    base: Array | None = None,
    dates: NDArray[np.int64] | Array | None = None,
    n_warm: int | None = None,
    dgp: str | None = None,
) -> dict[str, float | str]:
    """Online CRC diagnostic. Panel loss path preferred; toy path is ``dgp=fixture``."""
    if losses is not None or base is not None or dates is not None:
        if losses is None or base is None or dates is None:
            raise ValueError("pass losses, base, and dates together for panel path")
        losses_a = np.asarray(losses, dtype=float).reshape(-1)
        base_a = np.asarray(base, dtype=float).reshape(-1)
        dates_a = np.asarray(dates).reshape(-1)
        if losses_a.size != base_a.size or losses_a.size != dates_a.size:
            raise ValueError("losses/base/dates length mismatch")
        warm = int(n_warm if n_warm is not None else max(losses_a.size // 5, 20))
        warm = min(warm, max(losses_a.size - 10, 1))
        dgp_label = dgp or "panel"
    else:
        losses_a, base_a, dates_a = _synthetic_loss_path(n_cal, n_test, seed)
        warm = int(n_cal)
        dgp_label = "fixture"
    oc = OnlineCRC(alpha=alpha, gamma=gamma, B=B)
    oc.initialize(losses_a[:warm], base_a[:warm])
    eval_dates = dates_a[warm:]
    path = oc.run(losses_a[warm:], base_a[warm:], eval_dates)
    finite_hits = np.isfinite(path.hit)
    hits = path.hit[finite_hits]
    n = int(hits.size)
    mean_risk = float(np.mean(hits)) if n else float("nan")
    from quant_fund.metrics.inference import grouped_mean_tstat
    from quant_fund.metrics.probability import kupiec_pof

    if n >= 10:
        rate, lr, kp = kupiec_pof(hits, alpha)
    else:
        rate, lr, kp = float("nan"), float("nan"), float("nan")
    out: dict[str, float | str] = {
        "mean_risk": mean_risk,
        "coverage": float(1.0 - mean_risk) if np.isfinite(mean_risk) else float("nan"),
        "nominal": float(alpha),
        "alpha": float(alpha),
        "n": float(n),
        "miss_rate": rate,
        "kupiec_lr": lr,
        "kupiec_p": kp,
        "dgp": dgp_label,
        "claim": "research_metric_only",
    }
    if dgp_label != "fixture" and n:
        date_rate, date_t, date_p, n_dates = grouped_mean_tstat(
            hits, np.asarray(eval_dates, dtype=str)[finite_hits], target=float(alpha)
        )
        out["date_clustered_miss_rate"] = date_rate
        out["date_clustered_t"] = date_t
        out["date_clustered_p"] = date_p
        out["date_clustered_n_dates"] = float(n_dates)
    if dgp_label == "fixture":
        out["seed"] = float(seed)
    return out
