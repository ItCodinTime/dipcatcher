"""Write a parity report: summary, divergence ledger, shortfall, code path.

The markdown file is the thing a person reads. The JSON files are the
same numbers for a machine. Nothing here is a research scorecard and
nothing here is a live P&L claim.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from quant_fund.parity.checker import check_parity
from quant_fund.parity.replay import ParityRun
from quant_fund.parity.shortfall import attribute_shortfall
from quant_fund.parity.strategy import resolve_decide
from quant_fund.parity.trace import code_path_guard
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _combined_marks(backtest: ParityRun, shadow: ParityRun) -> dict[str, tuple[float, float]]:
    names = set(backtest.end_marks) | set(shadow.end_marks)
    marks: dict[str, tuple[float, float]] = {}
    for sid in sorted(names):
        bt = backtest.end_marks.get(sid, shadow.end_marks.get(sid))
        sh = shadow.end_marks.get(sid, backtest.end_marks.get(sid))
        if bt is None or sh is None:
            continue
        marks[sid] = (float(bt), float(sh))
    return marks


def build_report(
    backtest: ParityRun,
    shadow: ParityRun,
    *,
    backtest_strategy: Any,
    shadow_strategy: Any,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Assemble the parity, shortfall, and code-path sections."""
    parity = check_parity(backtest, shadow, tol=tol)
    shortfall = attribute_shortfall(
        backtest.fills,
        shadow.fills,
        _combined_marks(backtest, shadow),
        backtest_terminal_delta=backtest.terminal_delta,
        paper_terminal_delta=shadow.terminal_delta,
    )
    guard = code_path_guard(
        backtest_fn=resolve_decide(backtest_strategy),
        shadow_fn=resolve_decide(shadow_strategy),
        backtest_traces=backtest.call_traces,
        shadow_traces=shadow.call_traces,
    )
    return {
        "summary": {
            "match": parity["match"] and guard["same_code_path"],
            "decisions_match": parity["match"],
            "same_code_path": guard["same_code_path"],
            "n_bars": parity["n_bars"],
            "n_divergent": parity["n_divergent"],
            "divergence_rate": parity["divergence_rate"],
            "by_cause": parity["by_cause"],
            "first_divergence": parity["first_divergence"],
            "max_abs_weight_delta": parity["max_abs_weight_delta"],
            "terminal_gap": shortfall["terminal_gap"],
            "shortfall_residual": shortfall["residual"],
            "synthetic": bool(backtest.synthetic or shadow.synthetic),
            "data_source": "SYNTHETIC"
            if (backtest.synthetic or shadow.synthetic)
            else "recorded_session",
            "backtest_pacing": backtest.pacing,
            "shadow_pacing": shadow.pacing,
            "backtest_tape_fingerprint": backtest.tape_fingerprint,
            "shadow_tape_fingerprint": shadow.tape_fingerprint,
            "live_pnl_claim": False,
            "research_only": True,
            "would_promote_live": False,
            "evidence": "simulated_broker_replay",
            "note": (
                "Simulated broker replay. Terminal gap is a marked-value "
                "difference inside the simulator, not a live P&L claim and "
                "not a research score."
            ),
        },
        "parity": parity,
        "shortfall": shortfall,
        "code_path": guard,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    body = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    sealed = {**body, "receipt_sha256": hash_bytes(canonical_json_bytes(body))}
    path.write_text(
        json.dumps(sealed, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return "—"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    """Human-readable report. See ``docs/BACKTEST_LIVE_PARITY.md``."""
    summary = report["summary"]
    parity = report["parity"]
    shortfall = report["shortfall"]
    code_path = report["code_path"]
    causes = parity["by_cause"]
    cause_lines = "\n".join(
        f"| `{name}` | {causes.get(name, 0)} |" for name in parity["cause_precedence"]
    )
    components = shortfall["components"]
    component_lines = "\n".join(
        f"| `{name}` | {_fmt(components[name])} |"
        for name in ("delay", "spread", "impact", "fees", "missed_fills", "opportunity")
    )
    ledger = parity["ledger"]
    if ledger:
        preview = ledger[:20]
        ledger_lines = "\n".join(
            "| {event} | `{sid}` | `{cause}` | {detail} |".format(
                event=_fmt(row["event_time"]),
                sid=row["security_id"],
                cause=row["cause"],
                detail=row["detail"],
            )
            for row in preview
        )
        if len(ledger) > len(preview):
            ledger_lines += (
                f"\n| … | | | {len(ledger) - len(preview)} more rows in the JSON ledger |"
            )
    else:
        ledger_lines = "| — | — | — | no divergent rows |"
    first = summary["first_divergence"]
    first_line = (
        "none"
        if not first
        else f"{_fmt(first['event_time'])} `{first['security_id']}` `{first['cause']}` ({first['detail']})"
    )
    return f"""# Backtest / shadow parity report

Simulated broker replay. `live_pnl_claim` is false. `would_promote_live` is false.
This file does not authorize live orders and it is not a research scorecard.

How to read it: `docs/BACKTEST_LIVE_PARITY.md`.

## Verdict

| Field | Value |
|---|---|
| Decisions match | {summary["decisions_match"]} |
| Same code path | {summary["same_code_path"]} |
| Bars compared | {summary["n_bars"]} |
| Divergent bars | {summary["n_divergent"]} |
| Divergence rate | {_fmt(summary["divergence_rate"])} |
| First divergence | {first_line} |
| Data source | {summary["data_source"]} |
| Terminal gap (backtest − paper, simulated marked value) | {_fmt(summary["terminal_gap"])} |
| Shortfall residual | {_fmt(summary["shortfall_residual"])} |

A zero divergence rate means these two runs agreed bar by bar. It does not
mean the strategy has economic value.

## Divergences by cause

Precedence is top to bottom: the first matching cause wins, so a fill
difference caused by a different tape is labeled `data`.

| Cause | Rows |
|---|---|
{cause_lines}

## Divergence ledger

| Bar | Security | Cause | Detail |
|---|---|---|---|
{ledger_lines}

## Implementation shortfall

Positive dollars mean the paper book underperformed the backtest book.
The six rows sum to the fill-implied gap (`component_sum` {_fmt(shortfall["component_sum"])},
fill gap {_fmt(shortfall["fill_gap"])}, algebraic residual {_fmt(shortfall["algebraic_residual"])}).

| Component | Dollars |
|---|---|
{component_lines}

`opportunity` includes paper-only quantity and any shared-quantity
difference in the terminal print (`shared_mark_gap`
{_fmt(shortfall["opportunity_detail"]["shared_mark_gap"])}).

The arrival shortfall versus the decision price is supplemental and is
**not** part of this identity (`paper_arrival_shortfall` in `shortfall.json`).

## Code path

| Check | Equal |
|---|---|
| Import graph | {code_path["import_graph_equal"]} |
| Static callees | {code_path["static_callees_equal"]} |
| Runtime call trace | {code_path["runtime_trace_equal"]} |
| Backtest-only runtime calls | {", ".join(code_path["runtime_only_backtest"]) or "—"} |
| Shadow-only runtime calls | {", ".join(code_path["runtime_only_shadow"]) or "—"} |
| Backtest-only static callees | {", ".join(code_path["callees_only_backtest"]) or "—"} |
| Shadow-only static callees | {", ".join(code_path["callees_only_shadow"]) or "—"} |
"""


def write_report(directory: Path, report: Mapping[str, Any]) -> dict[str, str]:
    """Write ``summary.json``, ``divergence_ledger.json``, ``shortfall.json``, ``code_path.json``, and ``report.md``."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": directory / "summary.json",
        "divergence_ledger": directory / "divergence_ledger.json",
        "shortfall": directory / "shortfall.json",
        "code_path": directory / "code_path.json",
        "markdown": directory / "report.md",
    }
    _dump(paths["summary"], report["summary"])
    _dump(
        paths["divergence_ledger"],
        {
            "ledger": report["parity"]["ledger"],
            "by_cause": report["parity"]["by_cause"],
            "live_pnl_claim": False,
            "research_only": True,
        },
    )
    _dump(paths["shortfall"], report["shortfall"])
    _dump(paths["code_path"], report["code_path"])
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def assert_clean_report(report: Mapping[str, Any], *, residual_tol: float = 1e-6) -> None:
    """Fail closed when a smoke run is not an honest zero-divergence replay."""
    summary = report["summary"]
    if summary.get("live_pnl_claim") is not False:
        raise RuntimeError("parity report made a live P&L claim")
    if summary.get("research_only") is not True:
        raise RuntimeError("parity report is not research-only")
    if summary.get("would_promote_live") is not False:
        raise RuntimeError("parity report would promote live")
    if summary.get("decisions_match") is not True:
        raise RuntimeError(
            "parity smoke diverged: "
            + json.dumps(summary.get("first_divergence"), default=_json_default)
        )
    if summary.get("same_code_path") is not True:
        raise RuntimeError("parity smoke code paths differ")
    residual = float(summary["shortfall_residual"])
    if abs(residual) > residual_tol:
        raise RuntimeError(f"shortfall residual {residual} exceeds {residual_tol}")
    algebraic = float(report["shortfall"]["algebraic_residual"])
    if abs(algebraic) > residual_tol:
        raise RuntimeError(f"shortfall algebraic residual {algebraic} exceeds {residual_tol}")
