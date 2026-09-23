"""Ops snapshot: point-in-time health view over limits, staleness, switches.

Every check returns ``ok`` / ``warn`` / ``breach`` / ``insufficient_data`` —
never silently green. Utilization ≥ ``warn_fraction`` of a configured limit is
``warn``; over the limit is ``breach``. Missing inputs yield
``insufficient_data``, not a false pass. Research-only; no live-P&L claim.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from quant_fund.config.models import AppConfig

OK = "ok"
WARN = "warn"
BREACH = "breach"
INSUFFICIENT = "insufficient_data"


def _check(value: float, limit: float, warn_fraction: float = 0.8) -> dict[str, Any]:
    if not np.isfinite(value):
        return {"status": INSUFFICIENT, "value": None, "limit": limit}
    util = value / limit if limit > 0 else (1.0 if value > 0 else 0.0)
    if value > limit:
        status = BREACH
    elif util >= warn_fraction:
        status = WARN
    else:
        status = OK
    return {
        "status": status,
        "value": float(value),
        "limit": float(limit),
        "utilization": float(util),
    }


def ops_snapshot(
    *,
    nav: float,
    cash: float,
    positions: dict[str, float],
    marks: dict[str, float],
    config: AppConfig,
    asof: datetime | None = None,
    mark_age_bars: dict[str, int] | None = None,
    n_open_orders: int | None = None,
    kill_switch_state: str | None = None,
    peak_nav: float | None = None,
    recon_mismatches: int | None = None,
    drift_alert: bool | None = None,
    warn_fraction: float = 0.8,
) -> dict[str, Any]:
    """Point-in-time ops snapshot. ``positions``/``marks`` are unit/price maps.

    ``mark_age_bars`` maps held security -> bars since its last valid mark.
    ``peak_nav`` enables the drawdown check; ``recon_mismatches`` is the count
    from ``paper.recon.reconcile_*``; ``drift_alert`` is ``drift_report['alert']``.
    """
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("nav must be finite and positive")
    cfg = config.risk_gate
    pos_val = {s: q * marks.get(s, np.nan) for s, q in positions.items() if abs(q) > 1e-12}
    gross = sum(abs(v) for v in pos_val.values() if np.isfinite(v))
    net = sum(v for v in pos_val.values() if np.isfinite(v))
    max_name_val = max((abs(v) for v in pos_val.values()), default=0.0)
    unmarked = sorted(s for s, v in pos_val.items() if not np.isfinite(v))

    checks: dict[str, Any] = {
        "gross": _check(gross / nav, cfg.max_gross, warn_fraction),
        "net": _check(abs(net) / nav, cfg.max_net, warn_fraction),
        "largest_name": _check(max_name_val / nav, cfg.max_name, warn_fraction),
        "cash_buffer": {
            "status": WARN if cash < 0 else OK,
            "value": float(cash),
        },
        "unmarked_positions": {
            "status": BREACH if unmarked else OK,
            "securities": unmarked,
        },
        "kill_switch": {
            "status": OK
            if kill_switch_state == "ENABLED"
            else (INSUFFICIENT if kill_switch_state is None else BREACH),
            "state": kill_switch_state,
        },
    }

    if mark_age_bars is not None:
        stale = sorted(s for s, a in mark_age_bars.items() if a > cfg.stale_price_bars)
        checks["mark_staleness"] = {
            "status": BREACH if stale else OK,
            "stale_securities": stale,
            "stale_price_bars": cfg.stale_price_bars,
        }
    else:
        checks["mark_staleness"] = {"status": INSUFFICIENT}

    if peak_nav is not None and np.isfinite(peak_nav) and peak_nav > 0:
        dd = 1.0 - nav / peak_nav
        checks["drawdown"] = {
            "status": OK,
            "value": float(dd),
            "note": "no configured limit; informational",
        }
    else:
        checks["drawdown"] = {"status": INSUFFICIENT}

    checks["open_orders"] = {
        "status": OK if n_open_orders is not None else INSUFFICIENT,
        "count": n_open_orders,
    }
    checks["reconciliation"] = {
        "status": INSUFFICIENT
        if recon_mismatches is None
        else (OK if recon_mismatches == 0 else BREACH),
        "mismatches": recon_mismatches,
    }
    checks["feature_drift"] = {
        "status": INSUFFICIENT if drift_alert is None else (BREACH if drift_alert else OK),
        "alert": drift_alert,
    }

    statuses = [c["status"] for c in checks.values()]
    overall = (
        BREACH
        if BREACH in statuses
        else (
            WARN
            if WARN in statuses
            else (INSUFFICIENT if all(s == INSUFFICIENT for s in statuses) else OK)
        )
    )
    return {
        "asof": None if asof is None else asof.isoformat(),
        "nav": float(nav),
        "cash": float(cash),
        "gross": float(gross),
        "net": float(net),
        "n_positions": int(len(pos_val)),
        "checks": checks,
        "overall_status": overall,
        "research_only": True,
        "live_pnl_claim": False,
    }


def render_markdown(snapshot: dict[str, Any]) -> str:
    """Human-readable ops board — one line per check."""
    lines = [
        "# Ops snapshot",
        "",
        f"- asof: {snapshot.get('asof')}",
        f"- overall: **{snapshot['overall_status']}**",
        f"- nav: {snapshot['nav']:,.2f} | cash: {snapshot['cash']:,.2f} | "
        f"gross: {snapshot['gross']:,.0f} | net: {snapshot['net']:,.0f} | "
        f"positions: {snapshot['n_positions']}",
        "",
        "| check | status | value | limit | util |",
        "|---|---|---|---|---|",
    ]
    for name, c in snapshot["checks"].items():
        val = c.get("value")
        lim = c.get("limit")
        util = c.get("utilization")
        extra = ""
        if c.get("stale_securities"):
            extra = f" stale={','.join(c['stale_securities'])}"
        if c.get("securities"):
            extra = f" missing={','.join(c['securities'])}"
        if name == "kill_switch":
            val = c.get("state")
        if name == "open_orders":
            val = c.get("count")
        if name == "reconciliation":
            val = c.get("mismatches")
        lines.append(
            f"| {name} | {c['status']} | "
            f"{'' if val is None else (f'{val:,.4f}' if isinstance(val, float) else val)} | "
            f"{'' if lim is None else f'{lim:,.3f}'} | "
            f"{'' if util is None else f'{util:.1%}'}{extra} |"
        )
    lines += ["", "_research only — not live P&L_"]
    return "\n".join(lines) + "\n"
