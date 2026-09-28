"""Validate, forecast, optimize, and backtest commands.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer

from .app import app
from .support import _cfg


def _parse_asof(value: str | None) -> datetime | None:
    """Interpret naive CLI dates as UTC for the UTC-timestamped panels."""
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


@app.command()
def validate(
    model_id: str,
    config: Path = typer.Option(Path("configs/research.yaml")),
    metrics: Path | None = typer.Option(
        None, help="Optional JSON metrics blob (mean_ic, net_spread, turnover, ...)."
    ),
    claim_live: bool = typer.Option(
        False, help="Set when asserting a live/production claim (fails closed on SYNTHETIC)."
    ),
    # Fail closed: assert the leakage suite passed explicitly (--leakage-ok)
    # rather than defaulting the promotion gate to "passed".
    leakage_ok: bool = typer.Option(False, help="Whether the leakage suite passed."),
) -> None:
    """Fail-closed research / promotion gates (see docs/VALIDATION.md)."""
    import json

    from quant_fund.validation.gates import validate_candidate

    cfg = _cfg(config)
    result = validate_candidate(
        model_id,
        cfg,
        metrics_path=metrics,
        claim_live=claim_live,
        leakage_ok=leakage_ok,
    )
    typer.echo(json.dumps(result, indent=2, default=str))
    if result.get("data_label") == "SYNTHETIC":
        typer.echo("DATA_LABEL=SYNTHETIC")
    raise typer.Exit(code=0 if result.get("ok") else 1)


@app.command()
def forecast(
    config: Path = typer.Option(Path("configs/research.yaml")), date: str | None = None
) -> None:
    from quant_fund.pipeline.forecast import forecast_asof

    cfg = _cfg(config)
    asof = _parse_asof(date)
    state = forecast_asof(cfg, asof)
    if "SYNTHETIC" in state.notes:
        typer.echo("SYNTHETIC")
    for f in state.forecasts[:15]:
        hz = next(iter(f.interval_lo), "5d")
        lo = f.interval_lo.get(hz)
        hi = f.interval_hi.get(hz)
        extra = ""
        if lo is not None and hi is not None:
            extra = (
                f" interval[{hz}]=[{lo:+.4%},{hi:+.4%}] "
                f"interval_alpha={f.interval_alpha} {f.interval_method}"
            )
        typer.echo(
            f"{f.symbol:8} alpha={f.alpha.get('5d', 0):+.4%} rank={f.rank_percentile.get('5d', 0):.2f} "
            f"vol={f.volatility.get('5d', 0):.3f}{extra}"
        )


@app.command("kronos-forecast")
def kronos_forecast(
    config: Path = typer.Option(Path("configs/research.yaml")), date: str | None = None
) -> None:
    """Research-only Kronos candle-path forecasts (requires train.kronos.enabled).

    Loads strictly local, pre-downloaded artifacts — never the network or live
    execution paths. Quantile bands are the predicted candle envelope, not a
    calibrated predictive interval.
    """
    from quant_fund.pipeline.kronos import forecast_kronos_frame

    cfg = _cfg(config)
    asof = _parse_asof(date)
    try:
        state = forecast_kronos_frame(cfg, asof=asof)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if "SYNTHETIC" in state.notes:
        typer.echo("SYNTHETIC")
    typer.echo("kronos.adapter.v1 — research-only candle-path forecast")
    for f in state.forecasts[:15]:
        hz = next(iter(f.expected_returns), "5d")
        q = f.quantiles.get(hz, {})
        typer.echo(
            f"{f.symbol:8} expected={f.expected_returns.get(hz, 0):+.4%} "
            f"q05={q.get(0.05, 0):+.4%} q50={q.get(0.5, 0):+.4%} q95={q.get(0.95, 0):+.4%} "
            f"p_up={f.probability_positive.get(hz, 0):.2f} vol={f.volatility.get(hz, 0):.3f}"
        )


@app.command()
def optimize(
    config: Path = typer.Option(Path("configs/research.yaml")), date: str | None = None
) -> None:
    from quant_fund.pipeline.forecast import optimize_asof

    cfg = _cfg(config)
    asof = _parse_asof(date)
    w = optimize_asof(cfg, asof)
    typer.echo(w.head(20))


@app.command()
def backtest(
    config: Path = typer.Option(Path("configs/backtest.yaml")),
    engine: str = typer.Option(
        "ref", "--engine", help="ref (event loop) or fast (bit-identical vectorized replay)"
    ),
) -> None:
    import polars as pl

    from quant_fund.backtest.engine import run_backtest
    from quant_fund.backtest.fast_replay import run_backtest_fast
    from quant_fund.pipeline.dataset import ensure_silver
    from quant_fund.pipeline.forecast import build_causal_weight_panel, decision_dates

    if engine not in ("ref", "fast"):
        raise typer.BadParameter("--engine must be 'ref' or 'fast'")
    cfg = _cfg(config)
    bars = ensure_silver(cfg)
    # features for adv/vol
    from quant_fund.features.engine import build_features

    feat = build_features(bars, cfg)
    # Gold drops warmup bars (universe membership) and the label-horizon tail;
    # optimize_asof fails closed on a decision date with no panel row, so the
    # replay grid is the overlap only (same contract as execution-sensitivity).
    dates = decision_dates(cfg, feat["event_time"].unique().sort().to_list())
    if len(dates) < 2:
        raise typer.BadParameter(
            "need at least 2 decision dates on both the feature panel and the causal gold panel"
        )
    feat = feat.filter(pl.col("event_time").is_in(dates))
    # Causal: optimize_asof(asof=d) per date — no end-of-sample weight broadcast
    weights = build_causal_weight_panel(cfg, dates)
    run = run_backtest_fast if engine == "fast" else run_backtest
    result = run(feat, weights, cfg)
    if result.source_note == "SYNTHETIC":
        typer.echo("SYNTHETIC")
    typer.echo(result.metrics)


__all__ = [
    "backtest",
    "forecast",
    "kronos_forecast",
    "optimize",
    "validate",
]
