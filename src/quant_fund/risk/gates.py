"""Causal book-level risk gates. Delay 1. Cannot mint alpha.

Sharpe is invariant to *constant* leverage when rf = 0:

    SR(c r) = sign(c) SR(r)    for finite c ≠ 0.

Vol targeting uses time-varying leverage σ*/σ_t (Moreira–Muir 2017), so
Sharpe *can* change if expected return does not scale 1:1 with vol. It
cannot manufacture Sharpe 5 from IC ≈ 0: if E[r] ≈ 0 the scaled series is
still noise. A 2.5% vol target with Sharpe 5 is only ~12.5%/year (~3.4×
not 10×). 10× at 2.5% vol needs Sharpe ~10.

Every scale applied to bar t uses information through t-1 only.
Negative-mean Kelly is clipped to 0 (ADR-032: no sign-flip of a loser).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.risk import historical_es
from quant_fund.metrics.snooping import stepm
from quant_fund.models.crc import ConformalRiskControl

Array = NDArray[np.float64]
_EPS = 1e-12


@dataclass(frozen=True)
class GateSpec:
    """Paper-book overlay knobs. ``None`` disables that gate."""

    vol_target: float | None = 0.025
    vol_lookback: int = 60
    vol_cap: float = 8.0
    dd_limit: float | None = 0.05
    dd_mode: str = "halt"
    dd_cushion: float = 0.005
    es_limit: float | None = 0.006
    es_lookback: int = 63
    tail_p: float = 0.01
    kelly_fraction: float | None = 0.25
    kelly_lookback: int = 63
    crc_alpha: float | None = 0.05
    crc_lookback: int = 63
    crash_lookback: int = 10
    crash_return: float = -0.2
    stepm_enable: bool = False
    stepm_min_obs: int = 80
    stepm_step: int = 63
    stepm_n_boot: int = 128
    stepm_alpha: float = 0.05
    periods_per_year: float = 252.0


@dataclass
class GateResult:
    returns: Array
    scale: Array
    spec: GateSpec
    n_halt: int = 0
    n_scaled: int = 0
    notes: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "n_halt": int(self.n_halt),
            "n_scaled": int(self.n_scaled),
            "mean_scale": float(np.mean(self.scale)) if self.scale.size else 1.0,
            "last_scale": float(self.scale[-1]) if self.scale.size else 1.0,
            "sharpe_scale_invariant_for_constant_leverage": True,
            "vol_target_cannot_mint_sharpe5_from_ic0": True,
            "research_only": True,
            "live_pnl_claim": False,
            "notes": list(self.notes),
        }


def vol_target(
    returns: Array,
    *,
    target: float = 0.025,
    lookback: int = 60,
    cap: float = 8.0,
    periods_per_year: float = 252.0,
) -> Array:
    """Causal next-bar leverage: target / trailing realized vol. Delay 1."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    out = np.zeros_like(r)
    lb = max(int(lookback), 2)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("vol target must be finite and positive")
    for i in range(lb, len(r) - 1):
        window = r[i - lb + 1 : i + 1]
        sig = float(np.std(window, ddof=1)) * np.sqrt(float(periods_per_year))
        if not np.isfinite(sig) or sig < _EPS:
            continue
        out[i + 1] = r[i + 1] * float(np.clip(target / sig, 0.0, float(cap)))
    return out


def vol_target_leverage(
    returns: Array,
    *,
    target: float = 0.025,
    lookback: int = 60,
    cap: float = 8.0,
    periods_per_year: float = 252.0,
) -> Array:
    """Leverage known at t (data through t) for the *next* bar."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    lev = np.ones_like(r)
    lb = max(int(lookback), 2)
    prior = min(1.0, float(target) / 0.20)
    lev[:] = prior
    for i in range(lb - 1, len(r)):
        window = r[i - lb + 1 : i + 1]
        sig = float(np.std(window, ddof=1)) * np.sqrt(float(periods_per_year))
        if not np.isfinite(sig) or sig < _EPS:
            lev[i] = prior
            continue
        lev[i] = float(np.clip(target / sig, 0.0, float(cap)))
    return lev


def dd_halt(returns: Array, *, limit: float = 0.05) -> Array:
    """Causal flatten after equity hits -limit from peak. Stays cash."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    eq = 1.0
    peak = 1.0
    halted = False
    out = np.zeros_like(r)
    for i, x in enumerate(r):
        if halted:
            continue
        eq *= 1.0 + float(x)
        peak = max(peak, eq)
        dd = eq / peak - 1.0
        out[i] = float(x)
        if dd <= -abs(float(limit)):
            halted = True
    return out


def dd_remaining_leverage(
    returns: Array,
    *,
    limit: float = 0.05,
    cushion: float = 0.005,
) -> Array:
    """Remaining-drawdown scale known at t for the next bar."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    lev = np.ones_like(r)
    eq = 1.0
    peak = 1.0
    for i, x in enumerate(r):
        eq *= 1.0 + float(x) if np.isfinite(x) else 1.0
        peak = max(peak, eq)
        dd = max(0.0, 1.0 - eq / peak) if peak > 0 else 1.0
        remaining = float(limit) - dd
        lev[i] = (
            0.0 if remaining <= float(cushion) else min(1.0, remaining / max(float(limit), _EPS))
        )
    return lev


def es_leverage(
    returns: Array,
    *,
    es_limit: float = 0.006,
    lookback: int = 63,
    tail_p: float = 0.01,
) -> Array:
    """Scale known at t so trailing ES of the unlevered book stays ≤ es_limit."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    lev = np.ones_like(r)
    lb = max(int(lookback), 8)
    alpha = 1.0 - float(tail_p)
    for i in range(lb - 1, len(r)):
        window = r[i - lb + 1 : i + 1]
        es = float(historical_es(-window, alpha))
        if not np.isfinite(es) or es <= _EPS:
            continue
        lev[i] = float(min(1.0, es_limit / es))
    return lev


def kelly_leverage(
    returns: Array,
    *,
    fraction: float = 0.25,
    lookback: int = 63,
    cap: float = 1.0,
) -> Array:
    """Fractional Kelly f = κ μ/σ² (Thorp). Delay-ready leverage at t.

    Daily μ/σ² equals annualized μ/σ². Negative μ → 0 (no sign-flip).
    ``fraction`` is κ ∈ (0, 1]; ``cap`` is an extra wealth-fraction ceiling.
    """
    if not np.isfinite(fraction) or fraction < 0.0:
        raise ValueError("kelly fraction must be finite and non-negative")
    r = np.asarray(returns, dtype=float).reshape(-1)
    lev = np.ones_like(r)
    lb = max(int(lookback), 8)
    for i in range(lb - 1, len(r)):
        window = r[i - lb + 1 : i + 1]
        mu = float(np.mean(window))
        var = float(np.var(window, ddof=1))
        if (not np.isfinite(mu)) or (not np.isfinite(var)) or var <= _EPS:
            lev[i] = 0.0
            continue
        f_star = mu / var
        if f_star <= 0.0:
            lev[i] = 0.0
            continue
        lev[i] = float(min(float(cap), float(fraction) * f_star))
    return lev


def crc_leverage(
    returns: Array,
    *,
    alpha: float = 0.05,
    lookback: int = 63,
    step: int = 5,
) -> Array:
    """CRC loss bound from 0, then size so trailing ES stays under λ̂.

    Angelopoulos–Bates–Malik–Jordan (2022). Calibrated on losses through t;
    applied to bar t+1 by the caller. Cannot invert a losing book.
    Recomputed every ``step`` bars and held; still delay-1.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    lev = np.ones_like(r)
    lb = max(int(lookback), 8)
    step = max(int(step), 1)
    last = 1.0
    for i in range(lb - 1, len(r)):
        if (i - (lb - 1)) % step == 0:
            window = r[i - lb + 1 : i + 1]
            losses = -window
            crc = ConformalRiskControl(alpha=float(alpha)).calibrate(
                losses, np.zeros(window.size, dtype=float)
            )
            allowed = max(float(crc.lambda_hat), _EPS)
            es = float(historical_es(losses, 1.0 - float(alpha)))
            if np.isfinite(es) and es > _EPS:
                last = float(min(1.0, allowed / es))
            else:
                last = 1.0
        lev[i] = last
    return lev


def crash_leverage(
    returns: Array,
    *,
    lookback: int = 10,
    crash_return: float = -0.2,
) -> Array:
    """Nautica-style crash: trailing lookback total return ≤ crash_return → 0."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    lev = np.ones_like(r)
    lb = max(int(lookback), 1)
    wealth = np.cumprod(1.0 + np.where(np.isfinite(r), r, 0.0))
    for i in range(lb, len(r)):
        base = wealth[i - lb]
        last = wealth[i]
        if base <= _EPS:
            continue
        trail = last / base - 1.0
        if trail <= float(crash_return):
            lev[i] = 0.0
    return lev


def stepm_leverage(
    excess: Array,
    *,
    min_obs: int = 80,
    step: int = 21,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 7,
) -> Array:
    """Expanding-window Romano–Wolf StepM allow/deny. Delay 1.

    ``excess`` is T (one book vs 0) or T×K (challengers vs ridge). Column 0
    is the book being sized. Decision at t uses rows ``:t+1`` and is applied
    to bar t+1. If StepM does not reject the no-skill null, size is 0.
    """
    f = np.asarray(excess, dtype=float)
    if f.ndim == 1:
        f = f.reshape(-1, 1)
    t, _k = f.shape
    lev = np.zeros(t, dtype=float)
    min_obs = max(int(min_obs), 10)
    step = max(int(step), 1)
    last = 0.0
    for i in range(min_obs - 1, t, step):
        panel = f[: i + 1]
        result = stepm(panel, n_boot=int(n_boot), alpha=float(alpha), seed=int(seed))
        last = 1.0 if (result.rejected and bool(result.rejected[0])) else 0.0
        end = min(t, i + step)
        lev[i:end] = last
    if t > 0:
        # Shift to next bar: leverage known at t sizes t+1.
        shifted = np.zeros(t, dtype=float)
        shifted[1:] = lev[:-1]
        return shifted
    return lev


def _delay1(leverage_through_t: Array) -> Array:
    """Apply leverage known at t to return t+1."""
    lev = np.asarray(leverage_through_t, dtype=float).reshape(-1)
    out = np.zeros_like(lev)
    if lev.size > 1:
        out[1:] = lev[:-1]
    return out


def _min_lev(*levels: Array) -> Array:
    stacked = np.vstack([np.asarray(x, dtype=float).reshape(-1) for x in levels])
    return np.min(stacked, axis=0)


def apply_gate_stack(
    returns: Array,
    spec: GateSpec | None = None,
    *,
    stepm_excess: Array | None = None,
) -> GateResult:
    """Compose causal gates. Vol/Kelly/CRC/ES/crash/StepM on the raw path,
    then a drawdown halt (or remaining-budget scale) on the scaled path.
    """
    spec = spec or GateSpec()
    r = np.asarray(returns, dtype=float).reshape(-1)
    n = int(r.size)
    notes = [
        "Sharpe is invariant to constant leverage (rf=0).",
        "Vol targeting cannot create Sharpe 5 from IC≈0.",
        "Negative Kelly is clipped to 0 (no sign-flip).",
        "blend_weight stays 0. Not a live P&L claim.",
    ]
    lev = np.ones(n, dtype=float)
    if spec.vol_target is not None:
        lev = _min_lev(
            lev,
            vol_target_leverage(
                r,
                target=float(spec.vol_target),
                lookback=spec.vol_lookback,
                cap=spec.vol_cap,
                periods_per_year=spec.periods_per_year,
            ),
        )
        notes.append("vol_target=Moreira–Muir 2017")
    if spec.kelly_fraction is not None and spec.kelly_fraction > 0.0:
        lev = _min_lev(
            lev,
            kelly_leverage(
                r,
                fraction=float(spec.kelly_fraction),
                lookback=spec.kelly_lookback,
            ),
        )
        notes.append("kelly=Thorp fractional")
    if spec.crc_alpha is not None:
        lev = _min_lev(
            lev,
            crc_leverage(r, alpha=float(spec.crc_alpha), lookback=spec.crc_lookback),
        )
        notes.append("crc=Angelopoulos et al. 2022")
    if spec.es_limit is not None:
        lev = _min_lev(
            lev,
            es_leverage(
                r,
                es_limit=float(spec.es_limit),
                lookback=spec.es_lookback,
                tail_p=spec.tail_p,
            ),
        )
        notes.append("es_halt=Acerbi–Tasche ES")
    if spec.crash_lookback > 0:
        lev = _min_lev(
            lev,
            crash_leverage(
                r,
                lookback=spec.crash_lookback,
                crash_return=spec.crash_return,
            ),
        )
        notes.append("crash=nautica −20%/10d analog")
    if spec.stepm_enable:
        excess = r if stepm_excess is None else stepm_excess
        lev = _min_lev(
            lev,
            stepm_leverage(
                excess,
                min_obs=spec.stepm_min_obs,
                step=spec.stepm_step,
                n_boot=spec.stepm_n_boot,
                alpha=spec.stepm_alpha,
            ),
        )
        notes.append("stepm=Romano–Wolf 2005 allow/deny")
    scale = _delay1(lev)
    scaled = r * scale
    if spec.dd_limit is not None:
        if spec.dd_mode == "remaining":
            dd_lev = dd_remaining_leverage(
                scaled, limit=float(spec.dd_limit), cushion=spec.dd_cushion
            )
            dd_scale = _delay1(dd_lev)
            scale = scale * dd_scale
            scaled = r * scale
            notes.append("dd=remaining-budget")
        else:
            scaled = dd_halt(scaled, limit=float(spec.dd_limit))
            with np.errstate(divide="ignore", invalid="ignore"):
                rebuilt = np.where(
                    np.abs(r) > _EPS, scaled / r, np.where(scaled == 0.0, 0.0, scale)
                )
            scale = np.where(np.isfinite(rebuilt), rebuilt, scale)
            notes.append("dd=halt-and-stay-cash")
    n_halt = int(np.sum(scale <= 1e-12))
    n_scaled = int(np.sum((scale > 1e-12) & (scale < 1.0 - 1e-12)))
    return GateResult(
        returns=scaled,
        scale=scale,
        spec=spec,
        n_halt=n_halt,
        n_scaled=n_scaled,
        notes=notes,
    )
