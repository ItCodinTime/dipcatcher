"""Causal book-level risk overlay for the paper backtest.

Uses pyRisk VaR/ES and pyriskmgmt EWMA ES to scale (or flatten) target weights
from information available *before* the next fill: prior close NAV path only.
A hard drawdown budget targets peak-to-trough < ``dd_limit`` (default 5%).

Additional delay-1 caps: fractional Kelly (Thorp; negative μ → 0, ADR-032)
and CRC loss-bound sizing (Angelopoulos et al. 2022). Kelly / CRC lookbacks
default to 63 and do not fire on short unit-test paths.

This cannot mint alpha. It caps loss. Constant leverage leaves Sharpe unchanged
(rf = 0). Vol targeting is time-varying leverage; it cannot manufacture Sharpe
5 from IC ≈ 0. A remaining-drawdown halt can drag Sharpe toward zero if it
parks the book in cash.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.metrics.risk import historical_es
from quant_fund.models.crc import ConformalRiskControl
from quant_fund.risk.pyrisk import ExpectedShortfall
from quant_fund.risk.pyriskmgmt import ewma_var_es

_EPS = 1e-12


class BookRiskOverlay:
    """Stateful scaler. Call ``preview_scale()`` then ``observe(nav_close)``."""

    def __init__(
        self,
        *,
        vol_target: float = 0.025,
        dd_limit: float = 0.05,
        es_limit: float = 0.006,
        tail_p: float = 0.01,
        lookback: int = 63,
        periods_per_year: float = 252.0,
        kelly_fraction: float = 0.25,
        kelly_lookback: int = 63,
        crc_alpha: float = 0.05,
        crc_lookback: int = 63,
    ) -> None:
        if not np.isfinite(vol_target) or vol_target <= 0:
            raise ValueError("vol_target must be finite and positive")
        if not np.isfinite(dd_limit) or not 0.0 < dd_limit < 1.0:
            raise ValueError("dd_limit must be in (0, 1)")
        if not np.isfinite(es_limit) or es_limit <= 0:
            raise ValueError("es_limit must be finite and positive")
        if not np.isfinite(kelly_fraction) or kelly_fraction < 0.0:
            raise ValueError("kelly_fraction must be finite and non-negative")
        if not np.isfinite(crc_alpha) or crc_alpha < 0.0 or crc_alpha >= 1.0:
            raise ValueError("crc_alpha must be in [0, 1)")
        self.vol_target = float(vol_target)
        self.dd_limit = float(dd_limit)
        self.es_limit = float(es_limit)
        self.tail_p = float(tail_p)
        self.lookback = max(int(lookback), 8)
        self.periods_per_year = float(periods_per_year)
        self.kelly_fraction = float(kelly_fraction)
        self.kelly_lookback = max(int(kelly_lookback), 8)
        self.crc_alpha = float(crc_alpha)
        self.crc_lookback = max(int(crc_lookback), 8)
        self.navs: list[float] = []
        self.scales: list[float] = []
        self.n_halt = 0
        self.n_scaled = 0

    def preview_scale(self) -> float:
        scale = self._scale_from_history()
        self.scales.append(float(scale))
        if scale <= 1e-12:
            self.n_halt += 1
        elif scale < 1.0 - 1e-12:
            self.n_scaled += 1
        return float(scale)

    def observe(self, nav_close: float) -> None:
        if np.isfinite(nav_close) and nav_close > 0:
            self.navs.append(float(nav_close))

    def _scale_from_history(self) -> float:
        prior_scale = min(1.0, self.vol_target / 0.20)
        if len(self.navs) < 2:
            return float(prior_scale)
        nav = np.asarray(self.navs, dtype=float)
        peak = float(np.max(nav))
        last = float(nav[-1])
        if peak <= 0 or last <= 0:
            return 0.0
        dd = max(0.0, 1.0 - last / peak)
        remaining = self.dd_limit - dd
        # Halt with a 50 bp cushion so a next-open gap is less likely to
        # push peak-to-trough through dd_limit.
        if remaining <= 0.005:
            return 0.0
        rets = nav[1:] / nav[:-1] - 1.0
        window = rets[-self.lookback :]
        # Until a vol window exists, assume 20% annualized vol (not 100% scale).
        prior_scale = min(1.0, self.vol_target / 0.20)
        if window.size < 8:
            return float(prior_scale)
        vol = float(np.std(window, ddof=1) * np.sqrt(self.periods_per_year))
        scale = 1.0
        if np.isfinite(vol) and vol > 1e-8:
            scale = min(scale, self.vol_target / vol)
        elif not np.isfinite(vol):
            # Preserve the conservative prior only when the realized estimate
            # is unavailable. A finite zero-volatility window is valid evidence
            # and must not inherit this arbitrary warm-up cap.
            scale = min(scale, prior_scale)
        es_hist = float(ExpectedShortfall(window, alpha=self.tail_p).empirical_cvar())
        _var_ewma, es_ewma = ewma_var_es(window, alpha=1.0 - self.tail_p)
        del _var_ewma
        es = max(
            es_hist if np.isfinite(es_hist) else 0.0,
            float(es_ewma) if np.isfinite(es_ewma) else 0.0,
        )
        if es > 1e-8:
            scale = min(scale, self.es_limit / es)
            scale = min(scale, remaining / (3.0 * es))
        # Fractional Kelly (Thorp): f* = μ/σ². Negative μ → 0 (ADR-032).
        # Lookback is independent of the vol window so short-unit tests that
        # set overlay.lookback=8 do not trip Kelly on a 9-observation path.
        if self.kelly_fraction > 0.0 and window.size >= self.kelly_lookback:
            mu = float(np.mean(window[-self.kelly_lookback :]))
            var = float(np.var(window[-self.kelly_lookback :], ddof=1))
            if (not np.isfinite(mu)) or (not np.isfinite(var)) or var <= _EPS or mu <= 0.0:
                scale = 0.0
            else:
                scale = min(scale, min(1.0, self.kelly_fraction * (mu / var)))
        # CRC from a 0 bound: size so trailing ES stays under λ̂.
        if self.crc_alpha > 0.0 and window.size >= self.crc_lookback:
            crc_window = window[-self.crc_lookback :]
            losses = -crc_window
            crc = ConformalRiskControl(alpha=self.crc_alpha).calibrate(
                losses, np.zeros(crc_window.size, dtype=float)
            )
            allowed = max(float(crc.lambda_hat), _EPS)
            crc_es = float(historical_es(losses, 1.0 - self.crc_alpha))
            if np.isfinite(crc_es) and crc_es > _EPS:
                scale = min(scale, allowed / crc_es)
        return float(max(0.0, min(scale, 1.0)))

    def snapshot(self) -> dict[str, Any]:
        last = float(self.scales[-1]) if self.scales else 1.0
        return {
            "vol_target": self.vol_target,
            "dd_limit": self.dd_limit,
            "es_limit": self.es_limit,
            "tail_p": self.tail_p,
            "lookback": self.lookback,
            "kelly_fraction": self.kelly_fraction,
            "crc_alpha": self.crc_alpha,
            "n_observe": int(len(self.navs)),
            "n_halt": int(self.n_halt),
            "n_scaled": int(self.n_scaled),
            "last_scale": last,
            "mean_scale": float(np.mean(self.scales)) if self.scales else 1.0,
            "sources": ("lprtk/pyRisk", "GianMarcoOddo/pyriskmgmt"),
            "research_only": True,
            "execution_claim": "paper_backtest",
        }
