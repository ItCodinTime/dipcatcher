"""``train`` sub-app commands (one per forecast family)."""

from __future__ import annotations

from pathlib import Path

import typer

from quant_fund.pipeline.train import train_family

from ._app import (
    _cfg,
    train_app,
)


@train_app.callback(invoke_without_command=True)
def train_callback(
    ctx: typer.Context,
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str | None = typer.Option(None),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    typer.echo(
        "Specify a family: ranking, reinforcement, calibration, distribution, volatility, alpha, regime, tail, covariance, liquidity"
    )


def _train(family: str, config: Path, model: str | None) -> None:
    cfg = _cfg(config)
    result = train_family(cfg, family, model)
    typer.echo(result)


@train_app.command("ranking")
def train_ranking(
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str = typer.Option(
        "ridge",
        help=(
            "auto, composite, ridge, elasticnet, neural, ensemble, xgboost, "
            "lightgbm, lambdarank, or xendcg"
        ),
    ),
) -> None:
    _train("ranking", config, model)


@train_app.command("distribution")
def train_distribution(
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str = typer.Option(
        "gaussian",
        help="auto, empirical, gaussian, linear_qr, xgboost, or lightgbm",
    ),
) -> None:
    _train("distribution", config, model)


@train_app.command("calibration")
def train_calibration(
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str = typer.Option("isotonic", help="auto, isotonic, or platt"),
) -> None:
    _train("calibration", config, model)


@train_app.command("volatility")
def train_volatility(
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str = typer.Option(
        "ewma",
        help="auto, ewma, rolling, har, garch, or tree",
    ),
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


@train_app.command("reinforcement")
def train_reinforcement(
    config: Path = typer.Option(Path("configs/research.yaml")),
    model: str = typer.Option(
        "linucb", help="auto, linucb, thompson, quantile_thompson, or policy_gradient"
    ),
) -> None:
    """Train a research-only contextual RL policy on the causal gold panel."""
    _train("reinforcement", config, model)


@train_app.command("liquidity")
def train_liquidity(config: Path = typer.Option(Path("configs/research.yaml"))) -> None:
    _train("liquidity", config, None)
