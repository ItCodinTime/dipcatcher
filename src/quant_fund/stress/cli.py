"""Stress-report CLI. Research simulation only; this command does not trade."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quant_fund.stress.catalog import CRISIS_CATALOG
from quant_fund.stress.report import build_stress_report, render_html, render_markdown
from quant_fund.stress.strategy import load_return_panel, load_strategy

stress_app = typer.Typer(
    help="Research stress and scenario engine. Simulation only. Does not submit orders."
)


@stress_app.command("crises")
def crises_cmd() -> None:
    """List catalog episodes and whether each one is historical."""
    for crisis in CRISIS_CATALOG:
        kind = "historical" if crisis.historical else "hypothetical"
        print(f"{crisis.crisis_id}\t{kind}\t{crisis.name}")


@stress_app.command("report")
def report_cmd(
    strategy: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML research strategy."),
    out: Path = typer.Option(..., help="Output Markdown or HTML path."),
    fmt: str = typer.Option("markdown", "--format", help="markdown or html."),
    returns: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional decimal-return CSV. Overrides strategy.returns_csv.",
    ),
    level: float = typer.Option(0.95, help="VaR confidence level."),
    seed: int = typer.Option(0, help="Scenario seed."),
    n_scenarios: int = typer.Option(256, help="Synthetic scenario count."),
    n_boot: int = typer.Option(200, help="Bootstrap draws for VaR intervals."),
) -> None:
    """Write a stress report for one research strategy."""
    loaded = load_strategy(strategy)
    panel_path = returns
    if panel_path is None and loaded.returns_csv:
        panel_path = Path(loaded.returns_csv)
    names = None
    panel = None
    if panel_path is not None:
        names, panel = load_return_panel(panel_path)
    report = build_stress_report(
        loaded,
        names=names,
        panel=panel,
        level=level,
        seed=seed,
        n_scenarios=n_scenarios,
        n_boot=n_boot,
    )
    kind = fmt.strip().lower()
    if kind == "markdown":
        text = render_markdown(report)
    elif kind == "html":
        text = render_html(report)
    else:
        raise typer.BadParameter("format must be markdown or html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"research_only={report['research_only']} live_trading={report['live_trading']}")
    print(f"wrote {out}")
    print(f"wrote {sidecar}")
