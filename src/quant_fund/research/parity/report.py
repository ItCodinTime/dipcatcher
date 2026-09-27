"""Markdown + JSON diff-report writer.

The report is a research artifact: every section restates the SYNTHETIC
label, the engine versions, and the discrepancy taxonomy counts. It never
headlines P&L as evidence — parity is a correctness comparison.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.research.parity.canonical import EngineRun
from quant_fund.research.parity.reconcile import DiffClass, PairReport

TAXONOMY_DOC: dict[DiffClass, str] = {
    DiffClass.OK: "within tolerance",
    DiffClass.BAR_BOUNDARY: "fill/valuation shifted one bar (signal lag or bar-open vs bar-close boundary)",
    DiffClass.FILL_PRICE_BASIS: "fill price basis differs (open vs close vs averaged price)",
    DiffClass.SIZING_NAV_BASIS: "share count sized on a different NAV/price basis (signal-close vs execution-open)",
    DiffClass.ROUNDING: "integer-lot vs fractional-share sizing",
    DiffClass.COST_MODEL: "commission/fee model applied differently",
    DiffClass.CALENDAR: "bar exists in one engine's series only",
    DiffClass.MISSING_FILL: "fill present in one engine only (gate reject, sizing collapse)",
    DiffClass.MIN_NOTIONAL: "dust trade under one engine's minimum-notional gate",
    DiffClass.UNEXPLAINED: "FAILURE — no convention explains this diff",
}


def engine_versions() -> dict[str, str]:
    import importlib

    versions: dict[str, str] = {}
    for name, mod in (
        ("native", "quant_fund.backtest.engine"),
        ("vectorbt", "vectorbt"),
        ("backtrader", "backtrader"),
        ("nautilus", "nautilus_trader"),
    ):
        try:
            m = importlib.import_module(mod)
            versions[name] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - version probing must never fail the report
            versions[name] = "not installed"
    versions["python"] = platform.python_version()
    return versions


def render_markdown(
    *,
    runs: dict[str, EngineRun],
    reports: list[PairReport],
    scenario: str,
    panel_desc: str,
    shootout: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Framework parity report",
        "",
        f"- scenario: `{scenario}`",
        f"- data: {panel_desc} (**SYNTHETIC** — correctness fixture, not market evidence)",
        f"- generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "- engines:",
    ]
    for name, ver in engine_versions().items():
        lines.append(f"  - {name}: {ver}")
    lines += ["", "## Run status", ""]
    for key in sorted(runs):
        r = runs[key]
        lines.append(
            f"- {r.engine}/{r.strategy}: **{r.status}**"
            + (f" — {'; '.join(r.notes[:3])}" if r.notes else "")
        )
    lines += ["", "## Pairwise reconciliation (reference: native)", ""]
    lines.append(
        "| strategy | pair | verdict | bars | nav max | nav rms | final | matched fills | classes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for rep in reports:
        classes = ", ".join(f"{k}:{v}" for k, v in sorted(rep.class_counts.items())) or "—"
        s = rep.summary()
        lines.append(
            f"| {s['strategy']} | {s['pair']} | {s['verdict']} | {s['n_equity_bars']} "
            f"| {s['nav_max_abs']:.3e} | {s['nav_rms']:.3e} | {s['nav_final_abs']:.3e} "
            f"| {s['n_trades_matched']} | {classes} |"
        )
    lines += ["", "## Discrepancy detail (first rows per pair)", ""]
    for rep in reports:
        if not rep.discrepancies:
            continue
        lines.append(f"### {rep.strategy}: {rep.engine_a} vs {rep.engine_b}")
        lines.append("")
        for d in rep.discrepancies[:20]:
            lines.append(f"- `{d.kind}` {d.where}: **{d.cls.value}** — {d.detail}")
        lines.append("")
    lines += ["## Taxonomy", ""]
    for cls, doc in TAXONOMY_DOC.items():
        lines.append(f"- `{cls.value}`: {doc}")
    if shootout:
        lines += ["", "## Runtime shootout", ""]
        lines.append("| engine | strategy | run 1 (s) | min (s) | median (s) | status |")
        lines.append("|---|---|---|---|---|---|")
        for row in shootout.get("rows", []):
            lines.append(
                f"| {row['engine']} | {row['strategy']} | {row['first']:.4f} "
                f"| {row['min']:.4f} | {row['median']:.4f} | {row['status']} |"
            )
        lines.append(
            f"\ncommand: `{shootout.get('command', 'n/a')}` — reps={shootout.get('reps')}"
        )
    lines += [
        "",
        "---",
        "All data SYNTHETIC. Parity is an engine-correctness statement,",
        "not a strategy-performance claim.",
    ]
    return "\n".join(lines)


def write_report(
    dest: str | Path,
    *,
    runs: dict[str, EngineRun],
    reports: list[PairReport],
    scenario: str,
    panel_desc: str,
    shootout: dict[str, Any] | None = None,
) -> Path:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(
        runs=runs,
        reports=reports,
        scenario=scenario,
        panel_desc=panel_desc,
        shootout=shootout,
    )
    path.write_text(md, encoding="utf-8")
    summary = {
        "scenario": scenario,
        "panel": panel_desc,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "engines": engine_versions(),
        "runs": {
            k: {"status": r.status, "notes": r.notes} for k, r in sorted(runs.items())
        },
        "pairs": [r.summary() for r in reports],
        "label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
