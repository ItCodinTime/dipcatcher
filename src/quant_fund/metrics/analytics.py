"""Institutional book analytics for backtest + paper ledgers.

All outputs are execution / risk diagnostics. When ``data_source`` is SYNTHETIC
they are research-only — never live P&L evidence. See docs/MATH_SPEC.md.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.overfitting import min_trl_from_returns
from quant_fund.metrics.probability import (
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
)
from quant_fund.metrics.returns import annualized_vol, calmar_ratio, max_drawdown
from quant_fund.metrics.risk import historical_es, historical_var, losses_from_returns
from quant_fund.metrics.scoring import mean_fissler_ziegel
from quant_fund.portfolio.attribution import factor_selection_timing
from quant_fund.utils.hashing import hash_bytes

Array = NDArray[np.float64]


def net_gross_exposure(
    weights: Array | None = None,
    *,
    gross_series: Array | None = None,
    net_series: Array | None = None,
) -> dict[str, float]:
    """Net / gross from a weight vector or time series of book exposures."""
    out: dict[str, float] = {}
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w[np.isfinite(w)]
        out["gross"] = float(np.sum(np.abs(w)))
        out["net"] = float(np.sum(w))
        out["n_long"] = int(np.sum(w > 1e-12))
        out["n_short"] = int(np.sum(w < -1e-12))
    if gross_series is not None:
        g = np.asarray(gross_series, dtype=float)
        g = g[np.isfinite(g)]
        if g.size:
            out["gross_mean"] = float(np.mean(g))
            out["gross_max"] = float(np.max(g))
    if net_series is not None:
        n = np.asarray(net_series, dtype=float)
        n = n[np.isfinite(n)]
        if n.size:
            out["net_mean"] = float(np.mean(n))
            out["net_abs_mean"] = float(np.mean(np.abs(n)))
    return out


def mean_turnover(turnover_series: Array) -> float:
    t = np.asarray(turnover_series, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return float("nan")
    return float(np.mean(t))


def capacity_proxy(
    traded_notional: Array,
    adv_dollars: Array,
) -> dict[str, float]:
    """ADV participation capacity proxy: mean/max |trade|/ADV.

    High participation ⇒ capacity constrained. Research diagnostic only.
    """
    tn = np.asarray(traded_notional, dtype=float)
    adv = np.asarray(adv_dollars, dtype=float)
    mask = np.isfinite(tn) & np.isfinite(adv) & (adv > 0)
    if not mask.any():
        return {
            "mean_participation": float("nan"),
            "max_participation": float("nan"),
            "p95_participation": float("nan"),
            "n": 0,
        }
    part = np.abs(tn[mask]) / adv[mask]
    return {
        "mean_participation": float(np.mean(part)),
        "max_participation": float(np.max(part)),
        "p95_participation": float(np.quantile(part, 0.95)),
        "n": int(part.size),
    }


def execution_diagnostics(
    returns: Array,
    *,
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """Realized vol, max DD, Calmar as *execution diagnostics* — not research headlines."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    out = {
        "realized_vol_ann": annualized_vol(r, periods_per_year) if r.size else float("nan"),
        "max_drawdown": max_drawdown(r) if r.size else 0.0,
        "calmar_diagnostic": calmar_ratio(r, periods_per_year) if r.size >= 2 else float("nan"),
        "n_returns": int(r.size),
        "role": 1.0,  # 1 = execution diagnostic, not research headline
    }
    return out


def equity_curve_analytics(
    nav: Array,
    *,
    periods_per_year: float = 252.0,
) -> dict[str, float | int]:
    """NAV-level equity / drawdown analytics for a paper or sim ledger path.

    Complements return-based ``execution_diagnostics`` with peak NAV, ending NAV,
    and max drawdown from the cumulative equity curve. Execution diagnostic only
    (``role=1``) — never a live P&L claim.
    """
    n = np.asarray(nav, dtype=float)
    n = n[np.isfinite(n)]
    if n.size == 0:
        return {
            "n_points": 0,
            "nav_start": float("nan"),
            "nav_end": float("nan"),
            "nav_peak": float("nan"),
            "total_return": float("nan"),
            "max_drawdown_nav": 0.0,
            "max_underwater_duration": 0,
            "n_underwater_episodes": 0,
            "realized_vol_ann": float("nan"),
            "role": 1.0,
        }
    peak = np.maximum.accumulate(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, n / peak - 1.0, 0.0)
    max_dd = float(np.min(dd)) if dd.size else 0.0
    if n.size >= 2 and bool(np.all(n[:-1] != 0)):
        rets = np.diff(n) / n[:-1]
        rets = rets[np.isfinite(rets)]
        vol = annualized_vol(rets, periods_per_year) if rets.size else float("nan")
    else:
        vol = float("nan")
    nav0 = float(n[0])
    nav1 = float(n[-1])
    total_ret = float(nav1 / nav0 - 1.0) if nav0 != 0 else float("nan")
    uw = underwater_periods(dd)
    max_uw = int(max((int(e["duration"]) for e in uw), default=0))
    return {
        "n_points": int(n.size),
        "nav_start": nav0,
        "nav_end": nav1,
        "nav_peak": float(np.max(n)),
        "total_return": total_ret,
        "max_drawdown_nav": max_dd,
        "max_underwater_duration": max_uw,
        "n_underwater_episodes": int(len(uw)),
        "realized_vol_ann": float(vol),
        "role": 1.0,
    }


def underwater_periods(
    drawdowns: Array,
    *,
    threshold: float = 0.0,
) -> list[dict[str, float | int | bool]]:
    """Contiguous underwater episodes where drawdown < ``threshold`` (typically 0).

    Each episode: start/end indices (inclusive), duration in bars, trough depth
    and index, and whether the episode recovered (ended before series end with
    return to threshold). Execution / risk diagnostic only.
    """
    dd = np.asarray(drawdowns, dtype=float)
    if dd.size == 0:
        return []
    under = np.isfinite(dd) & (dd < float(threshold))
    episodes: list[dict[str, float | int | bool]] = []
    i = 0
    n = int(dd.size)
    while i < n:
        if not under[i]:
            i += 1
            continue
        start = i
        while i < n and under[i]:
            i += 1
        end = i - 1
        segment = dd[start : end + 1]
        trough_rel = int(np.argmin(segment))
        trough_idx = start + trough_rel
        recovered = bool(end < n - 1)  # left underwater before series end
        episodes.append(
            {
                "start_idx": int(start),
                "end_idx": int(end),
                "duration": int(end - start + 1),
                "trough": float(segment[trough_rel]),
                "trough_idx": int(trough_idx),
                "recovered": recovered,
            }
        )
    return episodes


def drawdown_duration_stats(
    returns: Array | None = None,
    *,
    nav: Array | None = None,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Drawdown duration / underwater summary from returns or a NAV path.

    Prefer ``nav`` when both are given (paper equity path). Returns max/mean
    underwater duration, time underwater fraction, and episode list. ``role=1``
    execution diagnostic — never a live P&L claim.
    """
    dd: Array
    if nav is not None:
        n = np.asarray(nav, dtype=float)
        n = n[np.isfinite(n)]
        if n.size == 0:
            dd = np.array([], dtype=float)
        else:
            peak = np.maximum.accumulate(n)
            with np.errstate(divide="ignore", invalid="ignore"):
                dd = np.where(peak > 0, n / peak - 1.0, 0.0)
    elif returns is not None:
        from quant_fund.metrics.returns import drawdown_series

        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        dd = drawdown_series(r) if r.size else np.array([], dtype=float)
    else:
        dd = np.array([], dtype=float)

    episodes = underwater_periods(dd, threshold=threshold)
    durations = [int(e["duration"]) for e in episodes]
    n_bars = int(dd.size)
    under_bars = int(sum(durations)) if durations else 0
    max_dur = int(max(durations)) if durations else 0
    mean_dur = float(np.mean(durations)) if durations else 0.0
    # Current underwater streak (open episode at end)
    current = 0
    if episodes and not bool(episodes[-1]["recovered"]):
        current = int(episodes[-1]["duration"])
    deepest = float(min((float(e["trough"]) for e in episodes), default=0.0))
    return {
        "n_bars": n_bars,
        "n_episodes": int(len(episodes)),
        "max_underwater_duration": max_dur,
        "mean_underwater_duration": mean_dur,
        "time_underwater_frac": float(under_bars / n_bars) if n_bars else 0.0,
        "current_underwater_duration": int(current),
        "deepest_trough": deepest,
        "episodes": episodes,
        "threshold": float(threshold),
        "role": 1.0,
        "live_pnl_claim": False,
        "research_only": True,
    }


def stress_report(
    weights: Array,
    cov: Array | None = None,
    *,
    shock_sigma: float = 1.0,
    liquidity_haircut: float = 0.10,
    corr_spike: float = 0.9,
    asset_vols: Array | None = None,
    liquidity_freeze_frac: float = 0.5,
    gap_open_sigma: float = 2.0,
) -> dict[str, Any]:
    """Aggregate stylized stress scenarios into a single ``stress_report`` dict.

    Includes per-scenario PnL, ranked adverse contributions (most negative first),
    and a sum of adverse (negative) scenario PnLs. Stylized / hypothetical only —
    not a named historical crisis replay and never a live P&L claim.
    """
    raw = stress_scenarios(
        weights,
        cov,
        shock_sigma=shock_sigma,
        liquidity_haircut=liquidity_haircut,
        corr_spike=corr_spike,
        asset_vols=asset_vols,
        liquidity_freeze_frac=liquidity_freeze_frac,
        gap_open_sigma=gap_open_sigma,
    )
    # PnL contribution keys (exclude metadata / param echoes)
    pnl_keys = (
        "shock_down_pnl",
        "shock_up_pnl",
        "liquidity_haircut_pnl",
        "liquidity_freeze_pnl",
        "gap_open_pnl",
        "corr_spike_1sigma_pnl",
    )
    contributions: list[dict[str, float | str]] = []
    adverse_sum = 0.0
    for key in pnl_keys:
        val = raw.get(key, float("nan"))
        try:
            fval = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            fval = float("nan")
        contributions.append({"scenario": key, "pnl": fval})
        if np.isfinite(fval) and fval < 0:
            adverse_sum += fval
    ranked = sorted(
        [c for c in contributions if np.isfinite(float(c["pnl"]))],
        key=lambda c: float(c["pnl"]),
    )
    worst = ranked[0] if ranked else None
    return {
        "scenarios": dict(raw),
        "contributions": contributions,
        "ranked_adverse": ranked,
        "worst_scenario": None if worst is None else str(worst["scenario"]),
        "worst_pnl": None if worst is None else float(worst["pnl"]),
        "adverse_pnl_sum": float(adverse_sum),
        "n_scenarios": int(len(pnl_keys)),
        "note": "stylized_hypothetical",
        "live_pnl_claim": False,
        "research_only": True,
        "role": "stress_report",
    }


def rolling_challenger_metrics(
    divergence_by_challenger: dict[str, list[float] | Array],
    *,
    window: int = 10,
) -> dict[str, Any]:
    """Rolling L1 divergence metrics for one or more paper shadow challengers.

    Research / infrastructure diagnostic only — never a live promotion signal.
    ``window`` is the trailing mean length (clamped to series length).
    """
    out: dict[str, Any] = {
        "window": int(window),
        "challengers": {},
        "role": "paper_shadow_research_diagnostic",
        "live_pnl_claim": False,
        "research_only": True,
    }
    w = max(int(window), 1)
    for name, series in divergence_by_challenger.items():
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out["challengers"][str(name)] = {
                "n": 0,
                "mean_l1": float("nan"),
                "max_l1": float("nan"),
                "rolling_mean_l1": float("nan"),
                "last_l1": float("nan"),
            }
            continue
        roll = arr[-min(w, arr.size) :]
        out["challengers"][str(name)] = {
            "n": int(arr.size),
            "mean_l1": float(np.mean(arr)),
            "max_l1": float(np.max(arr)),
            "rolling_mean_l1": float(np.mean(roll)),
            "last_l1": float(arr[-1]),
        }
    # Rank by mean L1 ascending (closest to champion first)
    ranked = sorted(
        (
            (k, v["mean_l1"])
            for k, v in out["challengers"].items()
            if v.get("n", 0) > 0 and v["mean_l1"] == v["mean_l1"]
        ),
        key=lambda kv: kv[1],
    )
    out["closest_challenger"] = ranked[0][0] if ranked else None
    out["farthest_challenger"] = ranked[-1][0] if ranked else None
    return out


def book_var_es(
    returns: Array,
    *,
    alpha: float = 0.95,
) -> dict[str, float]:
    """Historical VaR/ES on book returns. Loss L=-R. MATH_SPEC convention."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 5:
        return {
            "var": float("nan"),
            "es": float("nan"),
            "alpha": float(alpha),
            "n": int(r.size),
        }
    losses = losses_from_returns(r)
    return {
        "var": historical_var(losses, alpha),
        "es": historical_es(losses, alpha),
        "alpha": float(alpha),
        "n": int(r.size),
    }


def var_backtest_hooks(
    returns: Array,
    var_level: float | Array,
    *,
    alpha: float = 0.95,
    es_level: float | Array | None = None,
) -> dict[str, float]:
    """VaR hits plus optional Acerbi--Székely / Fissler--Ziegel ES diagnostics.

    Hit = 1 when loss > VaR (exception). ``var_level`` may be scalar or per-period.
    When ``es_level`` is supplied, it must be scalar or aligned with the VaR
    series; Z1/Z2 and mean FZ0 score are research diagnostics, never promotion gates.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return {
            "hit_rate": float("nan"),
            "kupiec_lr": float("nan"),
            "kupiec_p": float("nan"),
            "christoffersen_ind_lr": float("nan"),
            "christoffersen_ind_p": float("nan"),
            "christoffersen_cc_lr": float("nan"),
            "christoffersen_cc_p": float("nan"),
            "acerbi_szekely_z1": float("nan"),
            "acerbi_szekely_z2": float("nan"),
            "fissler_ziegel_mean": float("nan"),
            "es_hit_count": 0.0,
            "n": int(r.size),
        }
    losses = losses_from_returns(r)
    if np.ndim(var_level) == 0:
        va = np.full(losses.shape, float(var_level))
    else:
        va = np.asarray(var_level, dtype=float)
        if va.size != losses.size:
            # align by truncating to min length
            n = min(va.size, losses.size)
            losses, va = losses[-n:], va[-n:]
    hits = (losses > va).astype(float)
    rate, lr, kp = kupiec_pof(hits, 1.0 - alpha)
    ind_lr, ind_p, _ = christoffersen_independence(hits)
    cc_lr, cc_p, _ = christoffersen_cc(hits, 1.0 - alpha)
    z1 = z2 = fz_mean = float("nan")
    es_hits = 0
    if es_level is not None:
        if np.ndim(es_level) == 0:
            es = np.full(losses.shape, float(es_level))
        else:
            es = np.asarray(es_level, dtype=float).reshape(-1)
            if es.size != losses.size:
                n = min(es.size, losses.size)
                losses, va, es = losses[-n:], va[-n:], es[-n:]
                hits = (losses > va).astype(float)
        z1, es_hits = acerbi_szekely_z1(losses, va, es, 1.0 - alpha)
        z2, _ = acerbi_szekely_z2(losses, va, es)
        fz_mean = mean_fissler_ziegel(losses, va, es, float(alpha))
    return {
        "hit_rate": float(rate),
        "kupiec_lr": float(lr),
        "kupiec_p": float(kp),
        "christoffersen_ind_lr": float(ind_lr),
        "christoffersen_ind_p": float(ind_p),
        "christoffersen_cc_lr": float(cc_lr),
        "christoffersen_cc_p": float(cc_p),
        "acerbi_szekely_z1": float(z1),
        "acerbi_szekely_z2": float(z2),
        "fissler_ziegel_mean": float(fz_mean),
        "es_hit_count": float(es_hits),
        "n": int(hits.size),
    }


def stress_scenarios(
    weights: Array,
    cov: Array | None = None,
    *,
    shock_sigma: float = 1.0,
    liquidity_haircut: float = 0.10,
    corr_spike: float = 0.9,
    asset_vols: Array | None = None,
    liquidity_freeze_frac: float = 0.5,
    gap_open_sigma: float = 2.0,
) -> dict[str, float | str]:
    """Stylized book stress: ±σ shock, liquidity haircut/freeze, gap open, corr spike.

    Not a named historical crisis replay. Returns hypothetical P&L fractions.
    """
    w = np.asarray(weights, dtype=float)
    n = w.size
    if asset_vols is None:
        vols = np.full(n, 0.02)
    else:
        vols = np.asarray(asset_vols, dtype=float)
        if vols.size != n:
            vols = np.resize(vols, n)
    # ±σ shock on each name (independent stylized)
    shock = shock_sigma * vols
    pnl_down = float(-np.abs(w) @ shock)  # adverse move against absolute exposure
    pnl_up = float(w @ shock)
    # Liquidity haircut: force sell haircut on |w|
    liq = float(-liquidity_haircut * np.sum(np.abs(w)))
    # Liquidity freeze: cannot exit `freeze_frac` of gross; mark adverse vol shock
    # on the frozen sleeve (stylized — not a vendor halt tape).
    freeze = float(np.clip(liquidity_freeze_frac, 0.0, 1.0))
    frozen_gross = freeze * float(np.sum(np.abs(w)))
    liq_freeze = float(-frozen_gross * float(np.mean(vols)) * shock_sigma)
    # Gap open: overnight jump of gap_open_sigma * vol against the book
    gap_shock = gap_open_sigma * vols
    gap_pnl = float(-np.abs(w) @ gap_shock)
    # Correlation spike: if cov given, rebuild with corr≈corr_spike and recompute vol
    corr_pnl = float("nan")
    if cov is not None:
        s = np.asarray(cov, dtype=float)
        if s.shape == (n, n):
            d = np.sqrt(np.clip(np.diag(s), 1e-18, None))
            # replace correlation with constant corr_spike
            corr = np.full((n, n), float(corr_spike))
            np.fill_diagonal(corr, 1.0)
            stressed = np.outer(d, d) * corr
            port_var = float(w @ stressed @ w)
            corr_pnl = -float(np.sqrt(max(port_var, 0.0)))  # 1σ adverse under spiked corr
    return {
        "shock_down_pnl": pnl_down,
        "shock_up_pnl": pnl_up,
        "liquidity_haircut_pnl": liq,
        "liquidity_freeze_pnl": liq_freeze,
        "liquidity_freeze_frac": float(freeze),
        "gap_open_pnl": gap_pnl,
        "gap_open_sigma": float(gap_open_sigma),
        "corr_spike_1sigma_pnl": corr_pnl,
        "corr_spike": float(corr_spike),
        "liquidity_haircut": float(liquidity_haircut),
        "note": "stylized_hypothetical",
    }


def attribution_hook(
    portfolio_ret: Array,
    market_ret: Array,
    residual_ret: Array,
    cost_ret: Array,
) -> dict[str, Any]:
    """Factor/selection/timing hook via existing attribution.py.

    Fail-closed on empty or length-mismatched inputs: returns NaN components
    with ``ok=False`` and a ``status`` string rather than raising. Successful
    path sets ``ok=True`` / ``status="ok"``. Always ``live_pnl_claim=False``.
    """
    p = np.asarray(portfolio_ret, dtype=float).reshape(-1)
    m = np.asarray(market_ret, dtype=float).reshape(-1)
    r = np.asarray(residual_ret, dtype=float).reshape(-1)
    c = np.asarray(cost_ret, dtype=float).reshape(-1)
    base: dict[str, Any] = {
        "market": float("nan"),
        "selection_residual": float("nan"),
        "cost": float("nan"),
        "total": float("nan"),
        "convention": 1.0,
        "ok": False,
        "live_pnl_claim": False,
        "research_only": True,
    }
    if p.size == 0 or m.size == 0 or r.size == 0 or c.size == 0:
        base["status"] = "empty_input"
        return base
    if not (p.size == m.size == r.size == c.size):
        base["status"] = "length_mismatch"
        return base
    out: dict[str, Any] = dict(factor_selection_timing(p, m, r, c))
    out["ok"] = True
    out["status"] = "ok"
    out["live_pnl_claim"] = False
    out["research_only"] = True
    return out


def book_diagnostics(
    returns: Array,
    *,
    gross_exposure: Array | None = None,
    net_exposure: Array | None = None,
    turnover: Array | None = None,
    traded_notional: Array | None = None,
    adv_dollars: Array | None = None,
    weights: Array | None = None,
    cov: Array | None = None,
    var_alpha: float = 0.95,
    label: str = "RESEARCH_SIM",
    data_source: str = "SYNTHETIC",
) -> dict[str, Any]:
    """Aggregate HF-grade diagnostics for a sim/paper book. Explicitly labeled."""
    r = np.asarray(returns, dtype=float) if returns is not None else np.array([])
    expo = net_gross_exposure(weights, gross_series=gross_exposure, net_series=net_exposure)
    exec_d = execution_diagnostics(r)
    # Reconstruct a unit-start equity path from returns for NAV-level DD summary
    if r.size:
        eq_path = np.cumprod(1.0 + r)
        eq_path = np.concatenate([[1.0], eq_path])
        eq_an = equity_curve_analytics(eq_path)
    else:
        eq_an = equity_curve_analytics(np.array([]))
    ves = book_var_es(r, alpha=var_alpha)
    # Use historical VaR as the threshold series for backtest hooks when possible
    hooks: dict[str, float] = {}
    if r.size >= 10 and np.isfinite(ves.get("var", float("nan"))):
        hooks = var_backtest_hooks(r, float(ves["var"]), alpha=var_alpha, es_level=float(ves["es"]))
    turn = mean_turnover(turnover) if turnover is not None else float("nan")
    cap = (
        capacity_proxy(traded_notional, adv_dollars)
        if traded_notional is not None and adv_dollars is not None
        else {}
    )
    stress = stress_scenarios(weights, cov) if weights is not None else {}
    s_report = stress_report(weights, cov) if weights is not None else {}
    dd_dur = drawdown_duration_stats(r if r.size else None)
    synthetic = str(data_source).upper() == "SYNTHETIC"
    return {
        "label": label,
        "data_source": "SYNTHETIC" if synthetic else data_source,
        "research_only": True,
        "live_pnl_claim": False,
        "disclaimer": (
            "SYNTHETIC / simulated book diagnostics — not live broker P&L."
            if synthetic
            else "Simulated/paper book diagnostics — not a live P&L claim."
        ),
        "exposure": expo,
        "execution": exec_d,
        "equity": eq_an,
        "drawdown_duration": dd_dur,
        "var_es": ves,
        "var_backtest": hooks,
        "mean_turnover": turn,
        "capacity": cap,
        "stress": stress,
        "stress_report": s_report,
        # Research-only Bailey–LdP MinTRL smoke (forbidden-key hygiene).
        "min_trl": min_trl_from_returns(r),
    }


# Shared analytics export schema (paper ledger metrics ↔ backtest metrics JSON).
# Keys present on book_diagnostics output; exporters should intersect with this set.
#
# Dual-catalog note: this export may nest equity ``nav_*`` and stress ``*_pnl``
# diagnostic keys. That is intentional for paper/backtest diagnostics and is
# *not* governed by research ``FORBIDDEN_RESEARCH_METRIC_KEYS`` (scorecard hygiene).
# Honesty gate here is ``live_pnl_claim=False`` via ``validate_analytics_export``.
ANALYTICS_SCHEMA_KEYS: tuple[str, ...] = (
    "label",
    "data_source",
    "research_only",
    "live_pnl_claim",
    "disclaimer",
    "exposure",
    "execution",
    "equity",
    "drawdown_duration",
    "var_es",
    "var_backtest",
    "mean_turnover",
    "capacity",
    "stress",
    "stress_report",
)


def analytics_export_digest(blob: dict[str, Any]) -> str:
    """Return a canonical self-excluding digest for an analytics export."""
    normalized = dict(blob)
    normalized.pop("analytics_export_sha256", None)
    return hash_bytes(
        json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), allow_nan=True, default=str
        ).encode()
    )


def export_analytics_dict(
    diagnostics: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project diagnostics onto the shared paper/backtest analytics schema.

    Extra keys (e.g. backtest sharpe, risk_gate_rejects) may be merged under
    ``extra`` but are not part of the shared schema contract. Always forces
    ``live_pnl_claim=False`` and ``research_only=True``.
    """
    out: dict[str, Any] = {k: diagnostics[k] for k in ANALYTICS_SCHEMA_KEYS if k in diagnostics}
    out["research_only"] = True
    out["live_pnl_claim"] = False
    if "label" not in out:
        out["label"] = str(diagnostics.get("label", "RESEARCH_SIM"))
    if "data_source" not in out:
        out["data_source"] = str(diagnostics.get("data_source", "SYNTHETIC"))
    # Preserve the shared schema even when callers provide a thin/sanitized
    # diagnostics dict. Missing MinTRL is unavailable evidence, not zero.
    out.setdefault("min_trl", None)
    if extra:
        # Nest non-schema extras under "extra" to keep schema keys stable,
        # except allow a few well-known backtest scalars at top level.
        known_top = (
            "run_id",
            "total_return",
            "sharpe",
            "n",
            "max_drawdown",
            "commission",
            "spread",
            "impact",
            "risk_gate_rejects",
            "kill_switch_halts",
            "flag_high_sharpe",
        )
        extras_nest: dict[str, Any] = {}
        for k, v in extra.items():
            if k in known_top:
                out[k] = v
            elif k not in ANALYTICS_SCHEMA_KEYS:
                extras_nest[k] = v
        if extras_nest:
            out["extra"] = extras_nest
    out["schema_keys"] = list(ANALYTICS_SCHEMA_KEYS)
    out["schema_role"] = "paper_backtest_aligned"
    return out


def validate_analytics_export(
    blob: dict[str, Any] | None,
    *,
    require_all_schema_keys: bool = True,
) -> dict[str, Any]:
    """Validate a paper/backtest ``analytics_export`` dict (fail-closed).

    Returns ``ok`` / ``errors`` / ``warnings``. ``live_pnl_claim=True`` is always
    an error. Does not raise — callers decide policy. Research/infra only.

    Dual-catalog: nested equity/stress ``pnl``/``nav`` diagnostic keys are allowed
    here; research family blobs instead use ``family_blob_forbidden_metrics_absent``.
    Promotion/export paths must keep ``live_pnl_claim=False`` (never synthetic-as-live).
    """
    errors: list[str] = []
    warnings: list[str] = []
    if blob is None:
        return {
            "ok": False,
            "errors": ["analytics_export_missing"],
            "warnings": warnings,
            "role": "analytics_export_validation",
            "live_pnl_claim": False,
            "research_only": True,
        }
    if not isinstance(blob, dict):
        return {
            "ok": False,
            "errors": ["analytics_export_not_a_dict"],
            "warnings": warnings,
            "role": "analytics_export_validation",
            "live_pnl_claim": False,
            "research_only": True,
        }
    if blob.get("live_pnl_claim") is True:
        errors.append("live_pnl_claim_must_be_false")
    if blob.get("research_only") is False:
        errors.append("research_only_must_be_true_or_absent")
    digest = blob.get("analytics_export_sha256")
    if digest is not None:
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            errors.append("analytics_export_sha256_invalid")
        elif digest != analytics_export_digest(blob):
            errors.append("analytics_export_sha256_mismatch")
    if require_all_schema_keys:
        for key in ANALYTICS_SCHEMA_KEYS:
            if key not in blob:
                errors.append(f"missing_schema_key:{key}")
    else:
        for key in ("label", "data_source", "live_pnl_claim", "research_only"):
            if key not in blob:
                warnings.append(f"missing_soft_key:{key}")
    role = blob.get("schema_role")
    if role is not None and role != "paper_backtest_aligned":
        warnings.append(f"unexpected_schema_role:{role}")
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "present_schema_keys": [k for k in ANALYTICS_SCHEMA_KEYS if k in blob],
        "role": "analytics_export_validation",
        "live_pnl_claim": False,
        "research_only": True,
    }
