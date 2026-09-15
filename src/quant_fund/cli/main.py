"""quant CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from quant_fund.config import dump_resolved, load_config
from quant_fund.pipeline.doctor import doctor as run_doctor
from quant_fund.pipeline.train import train_family
from quant_fund.utils.logging import configure_logging, get_logger

app = typer.Typer(
    help="Artificial Hedge · Dipcatcher scientific hedge lab. Default mode is research, never live."
)
train_app = typer.Typer(help="Train a forecast family.")
app.add_typer(train_app, name="train")


def _cfg(config: Path):
    cfg = load_config(config)
    configure_logging()
    dump_resolved(cfg, Path(cfg.data.root) / "metadata" / "resolved_config.json")
    return cfg


@app.command()
def doctor(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    info = run_doctor(str(config))
    for k, v in info.items():
        typer.echo(f"{k}: {v}")


@app.command()
def ingest(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    from quant_fund.data.ingest import ingest as do_ingest

    cfg = _cfg(config)
    paths = do_ingest(cfg)
    log = get_logger(cmd="ingest", source=cfg.data.source)
    log.info("ingested", paths={k: str(v) for k, v in paths.items()})
    if cfg.data.source == "synthetic":
        typer.echo("SYNTHETIC ingest complete")
    for k, v in paths.items():
        typer.echo(f"{k}: {v}")


@app.command("build-features")
def build_features_cmd(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    from quant_fund.pipeline.dataset import build_gold

    cfg = _cfg(config)
    feats, _ = build_gold(cfg)
    typer.echo(f"features rows={feats.height} cols={len(feats.columns)}")


@app.command("build-labels")
def build_labels_cmd(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    from quant_fund.pipeline.dataset import build_gold

    cfg = _cfg(config)
    _, labs = build_gold(cfg)
    typer.echo(f"labels rows={labs.height}")


@train_app.callback(invoke_without_command=True)
def train_callback(
    ctx: typer.Context,
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str | None = typer.Option(None),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    typer.echo(
        "Specify a family: ranking, distribution, volatility, alpha, regime, tail, covariance, liquidity"
    )


def _train(family: str, config: Path, model: str | None) -> None:
    cfg = _cfg(config)
    result = train_family(cfg, family, model)
    typer.echo(result)


@train_app.command("ranking")
def train_ranking(
    config: Path = typer.Option(Path("configs/research.yaml")), model: str = "ridge"
) -> None:
    _train("ranking", config, model)


@train_app.command("distribution")
def train_distribution(
    config: Path = typer.Option(Path("configs/research.yaml")), model: str = "gaussian"
) -> None:
    _train("distribution", config, model)


@train_app.command("volatility")
def train_volatility(
    config: Path = typer.Option(Path("configs/research.yaml")), model: str = "ewma"
) -> None:
    _train("volatility", config, model)


@train_app.command("alpha")
def train_alpha(
    config: Path = typer.Option(Path("configs/research.yaml")), model: str = "ridge"
) -> None:
    _train("alpha", config, model)


@train_app.command("covariance")
def train_covariance(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    _train("covariance", config, None)


@train_app.command("regime")
def train_regime(
    config: Path = typer.Option(Path("configs/research.yaml")), model: str = "hmm"
) -> None:
    _train("regime", config, model)


@train_app.command("tail")
def train_tail(
    config: Path = typer.Option(Path("configs/research.yaml")), model: str = "historical"
) -> None:
    _train("tail", config, model)


@train_app.command("liquidity")
def train_liquidity(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    _train("liquidity", config, None)


@app.command()
def validate(model_id: str, config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    typer.echo(f"validate {model_id}: see walk-forward metrics in MLflow / reports")


@app.command()
def forecast(
    config: Path = typer.Option(Path("configs/research.yaml")), date: str | None = None
) -> None:
    from datetime import datetime

    from quant_fund.pipeline.forecast import forecast_asof

    cfg = _cfg(config)
    asof = datetime.fromisoformat(date) if date else None
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


@app.command()
def optimize(
    config: Path = typer.Option(Path("configs/research.yaml")), date: str | None = None
) -> None:
    from datetime import datetime

    from quant_fund.pipeline.forecast import optimize_asof

    cfg = _cfg(config)
    asof = datetime.fromisoformat(date) if date else None
    w = optimize_asof(cfg, asof)
    typer.echo(w.head(20))


@app.command()
def backtest(config: Path = typer.Option(Path("configs/backtest.yaml"))) -> None:
    import polars as pl

    from quant_fund.backtest.engine import run_backtest
    from quant_fund.pipeline.dataset import ensure_silver
    from quant_fund.pipeline.forecast import optimize_asof

    cfg = _cfg(config)
    bars = ensure_silver(cfg)
    # features for adv/vol
    from quant_fund.features.engine import build_features

    feat = build_features(bars, cfg)
    w = optimize_asof(cfg)
    dates = feat["event_time"].unique().sort().to_list()
    weights = pl.concat(
        [w.drop("event_time").with_columns(pl.lit(d).alias("event_time")) for d in dates]
    )
    result = run_backtest(feat, weights, cfg)
    if result.source_note == "SYNTHETIC":
        typer.echo("SYNTHETIC")
    typer.echo(result.metrics)


@app.command()
def research(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    """Run the scientific hedge lab benches and write a labeled notebook."""
    from quant_fund.research.agent import run_research

    cfg = _cfg(config)
    nb = run_research(cfg)
    if nb.synthetic:
        typer.echo("SYNTHETIC")
    typer.echo(f"{nb.firm} · {nb.product} scientific lab {nb.version}")
    typer.echo(nb.disclaimer)
    for h in nb.hypotheses:
        typer.echo(f"{h.id}: {h.decision} (p={h.p_value:.4g})")
    for r in nb.rankers:
        typer.echo(
            f"{r['name']}: IC={r['mean_ic']:.4f} RankIC={r.get('mean_rank_ic') or 0:.4f} "
            f"t={r['t_ic']:.2f} mono={r.get('decile_monotonicity') or 0:.3f}"
        )
    vol = nb.families.get("volatility", {})
    if vol:
        typer.echo(f"volatility QLIKE ewma={vol.get('qlike_ewma')} rolling={vol.get('qlike_rolling')}")
    rl = nb.families.get("reinforcement", {})
    if rl:
        typer.echo(
            f"linucb reward={rl.get('mean_policy_reward')} "
            f"vs_uniform={rl.get('mean_advantage_vs_uniform')} "
            f"regret={rl.get('mean_regret_vs_oracle')}"
        )
    conf = nb.families.get("conformal", {})
    if conf:
        aci = conf.get("aci", {})
        raw = conf.get("gaussian_raw", {})
        wrap = conf.get("wrappee", "scaled_gaussian")
        typer.echo(
            f"conformal wrappee={wrap} "
            f"raw_cov={raw.get('coverage')} "
            f"cqr_raw_cov={conf.get('cqr_raw', {}).get('coverage')} "
            f"scaled_cov={conf.get('scaled', conf.get('scaled_gaussian', {})).get('coverage')} "
            f"cqr_cov={conf.get('cqr', {}).get('coverage')} "
            f"aci_cov={aci.get('coverage')} "
            f"aci_width={aci.get('mean_width')} "
            f"mondrian_cov={conf.get('mondrian_aci', {}).get('coverage')} "
            f"high_x={conf.get('mondrian_aci', {}).get('high_x_coverage')} "
            f"worst_x={conf.get('mondrian_aci', {}).get('worst_x_coverage')}"
        )
    ev = nb.families.get("evalues", {})
    if ev:
        typer.echo(
            f"evalues cov={ev.get('coverage')} e_final={ev.get('e_final')} "
            f"ever_cross={ev.get('ever_cross')}"
        )
    jp = nb.families.get("jackknife_plus", {})
    if jp:
        typer.echo(
            f"jackknife_plus cov={jp.get('coverage')} width={jp.get('mean_width')} "
            f"floor={jp.get('coverage_floor')}"
        )
    crc = nb.families.get("crc", {})
    if crc:
        typer.echo(
            f"crc wrappee={crc.get('wrappee')} risk={crc.get('risk')} "
            f"crc_stat={crc.get('crc_stat')} lambda={crc.get('lambda_hat')} "
            f"high_vol_bound={crc.get('high_vol_mean_bound')} "
            f"low_vol_bound={crc.get('low_vol_mean_bound')}"
        )
    wcqr = nb.families.get("weighted_conformal", {})
    if wcqr:
        typer.echo(
            f"weighted_conformal cov={wcqr.get('coverage')} "
            f"width={wcqr.get('mean_width')} "
            f"unweighted_cov={wcqr.get('unweighted_coverage')}"
        )
    caps = nb.families.get("interval_risk", {})
    if caps:
        typer.echo(
            f"interval_risk mean_cap={caps.get('mean_cap')} "
            f"frac_binding={caps.get('frac_binding')} "
            f"mean_width={caps.get('mean_width')}"
        )
    qb = nb.families.get("quantile_bandit", {})
    if qb:
        typer.echo(
            f"quantile_bandit reward={qb.get('mean_policy_reward')} "
            f"vs_uniform={qb.get('mean_advantage_vs_uniform')} "
            f"regret={qb.get('mean_regret_vs_oracle')}"
        )
    typer.echo(f"json={nb.artifacts.get('json')}")
    typer.echo(f"markdown={nb.artifacts.get('markdown')}")


@app.command()
def lab(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    """Alias for `quant research`."""
    from quant_fund.research.agent import run_research

    cfg = _cfg(config)
    nb = run_research(cfg)
    if nb.synthetic:
        typer.echo("SYNTHETIC")
    typer.echo(nb.disclaimer)
    typer.echo(nb.artifacts.get("json"))


@app.command()
def report(
    latest: bool = typer.Option(False, "--latest"),
    config: Path = typer.Option(Path("configs/research.yaml")),
) -> None:
    from quant_fund.reporting.report import latest_report_dir, write_report

    cfg = _cfg(config)
    dest = latest_report_dir(Path(cfg.data.root)) / "latest.md"
    write_report(
        dest,
        "Research report",
        {"config": cfg.dump(), "note": "see MLflow for experiment metrics"},
        synthetic=cfg.data.source == "synthetic",
    )
    typer.echo(dest)


@app.command()
def api(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("quant_fund.api.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
