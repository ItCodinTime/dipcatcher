"""CLI for frozen Lightspeed engines. Research only; no Alpaca."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

ls_app = typer.Typer(
    help="Lightspeed engines (TQQQ rotation, nautica momentum). Research only; no live broker."
)


@ls_app.command("specs")
def specs_cmd() -> None:
    from quant_fund.lightspeed.specs import frozen_families

    typer.echo(json.dumps(frozen_families(), indent=2, default=str))


@ls_app.command("demo")
def demo_cmd(seed: int = typer.Option(7, "--seed")) -> None:
    """Synthetic path through both frozen engines. Not a P&L claim."""
    from quant_fund.lightspeed.metalabel import meta_label_gate
    from quant_fund.lightspeed.momentum import momentum_target_weights
    from quant_fund.lightspeed.rotation import tqqq_target_weights
    from quant_fund.lightspeed.specs import MomentumSpec, MomentumUniverse, nautica_momentum_v1

    rng = np.random.default_rng(int(seed))
    n = 260
    t = np.arange(n, dtype=float)
    qqq = 100.0 * np.exp(0.0006 * t + 0.01 * np.cumsum(rng.normal(size=n)))
    tqqq = 100.0 * np.exp(0.0016 * t + 0.028 * np.cumsum(rng.normal(size=n)))
    rot = tqqq_target_weights(qqq, tqqq)
    last_rot = {k: float(v[-1]) for k, v in rot.items()}

    spec = nautica_momentum_v1()
    demo_spec = MomentumSpec(
        family=spec.family,
        name=spec.name,
        status=spec.status,
        universe=MomentumUniverse(risk=("AAA", "BBB"), defensive="SGOV"),
        params=spec.params,
        live_disabled=True,
    )
    closes = {
        "AAA": 50.0 * np.exp(0.0012 * t + 0.015 * np.cumsum(rng.normal(size=n))),
        "BBB": 50.0 * np.exp(0.0002 * t + 0.012 * np.cumsum(rng.normal(size=n))),
        "SGOV": 100.0 + 0.004 * t,
    }
    mom = momentum_target_weights(closes, demo_spec)
    last_mom = {k: float(v[-1]) for k, v in mom.items()}

    ret = np.diff(np.log(qqq), prepend=np.log(qqq[0]))
    gap = np.concatenate([[0.0], np.diff(qqq) / qqq[:-1]])
    gate = meta_label_gate(gap, ret)
    typer.echo(
        json.dumps(
            {
                "tqqq_last": last_rot,
                "nautica_last": last_mom,
                "metalabel_last_multiplier": float(gate.multiplier[-1]),
                "metalabel_in_unit_interval": bool(
                    np.all((gate.multiplier >= 0.0) & (gate.multiplier <= 1.0))
                ),
                "research_only": True,
                "live_pnl_claim": False,
                "broker": None,
                "champion": "ridge",
                "blend_weight": 0,
            },
            indent=2,
        )
    )


@ls_app.command("hunt")
def hunt_cmd(
    labels: str = typer.Option("data/file_us/gold/labels.parquet", "--labels"),
    cost_bps: float = typer.Option(10.0, "--cost-bps"),
    vol_tgt: float = typer.Option(0.025, "--vol-target"),
    dd: float = typer.Option(0.05, "--dd-limit"),
) -> None:
    """Causal overnight / Kalman-pairs / directional TSMOM hunt. Prints honest cards."""
    from quant_fund.hedge_lab.target_hunt import run_target_hunt

    receipt = run_target_hunt(
        labels, one_way_cost=float(cost_bps) / 1e4, vol_tgt=vol_tgt, dd_limit=dd
    )
    slim = [
        {
            "name": c["name"],
            "sharpe": c.get("sharpe"),
            "cagr": c.get("cagr"),
            "max_drawdown": c.get("max_drawdown"),
            "total_return": c.get("total_return"),
            "holdout_sharpe": c.get("holdout_sharpe"),
            "n": c.get("n_returns"),
        }
        for c in receipt["cards"][:25]
    ]
    typer.echo(json.dumps({"hit_sharpe5": receipt["hit_sharpe5"], "best": receipt["best_abs_sharpe"], "top": slim}, indent=2))


@ls_app.command("race")
def race_cmd(
    config: Path = typer.Option(Path("configs/hedge_lab.yaml")),
    label: str = typer.Option("future_idio_return_1", "--label"),
    n_boot: int = typer.Option(1000, "--n-boot"),
    cost_bps: float = typer.Option(10.0, "--cost-bps"),
    cpu_fraction: float = typer.Option(0.6, "--cpu-fraction"),
) -> None:
    """Public CS challengers vs ridge + risk gates. Not live P&L. Not a promotion."""
    from quant_fund.hedge_lab.gated_race import run_gated_race

    receipt = run_gated_race(
        str(config),
        label,
        one_way_cost=float(cost_bps) / 1e4,
        n_boot=int(n_boot),
        cpu_fraction=float(cpu_fraction),
    )
    slim_ic = receipt.get("cs_ic", [])
    typer.echo(
        json.dumps(
            {
                "promote": receipt["promote"],
                "champion": receipt["champion"],
                "blend_weight": receipt["blend_weight"],
                "gates": {
                    "reality_check_p": receipt["gates"].get("reality_check_p"),
                    "spa_p_consistent": receipt["gates"].get("spa_p_consistent"),
                    "stepm_rejected": receipt["gates"].get("stepm_rejected"),
                    "dm": {
                        k: {"p_value": v.get("p_value"), "preferred": v.get("preferred")}
                        for k, v in (receipt["gates"].get("dm") or {}).items()
                    },
                },
                "cs_ic": slim_ic,
                "best_dd_safe": receipt.get("best_dd_safe"),
                "artifact_path": receipt.get("artifact_path"),
                "research_only": True,
                "live_pnl_claim": False,
            },
            indent=2,
            default=str,
        )
    )


@ls_app.command("confirm")
def confirm_cmd(
    config: Path = typer.Option(Path("configs/hedge_lab.yaml")),
    label: str = typer.Option("future_idio_return_1", "--label"),
    n_boot: int = typer.Option(1000, "--n-boot"),
    cost_bps: float = typer.Option(10.0, "--cost-bps"),
    cpu_fraction: float = typer.Option(0.6, "--cpu-fraction"),
) -> None:
    """Pre-declared tsmom vs ridge on the frozen 2025-01-02 holdout. Not a promotion."""
    from quant_fund.hedge_lab.gated_race import run_holdout_confirm

    receipt = run_holdout_confirm(
        str(config),
        label,
        one_way_cost=float(cost_bps) / 1e4,
        n_boot=int(n_boot),
        cpu_fraction=float(cpu_fraction),
    )

    def _slim(window: dict) -> dict:
        return {
            "cs_ic": window.get("cs_ic"),
            "gates": {
                "reality_check_p": (window.get("gates") or {}).get("reality_check_p"),
                "spa_p_consistent": (window.get("gates") or {}).get("spa_p_consistent"),
                "stepm_rejected": (window.get("gates") or {}).get("stepm_rejected"),
                "dm": {
                    k: {"p_value": v.get("p_value"), "preferred": v.get("preferred")}
                    for k, v in ((window.get("gates") or {}).get("dm") or {}).items()
                },
            },
            "cs_ls": {
                name: {
                    "sharpe": card.get("sharpe"),
                    "max_drawdown": card.get("max_drawdown"),
                    "cagr": card.get("cagr"),
                    "n_returns": card.get("n_returns"),
                }
                for name, card in (window.get("cs_ls") or {}).items()
            },
        }

    typer.echo(
        json.dumps(
            {
                "promote": receipt["promote"],
                "champion": receipt["champion"],
                "blend_weight": receipt["blend_weight"],
                "selection_end": receipt["selection_end"],
                "holdout_start": receipt["holdout_start"],
                "selection": _slim(receipt["selection"]),
                "holdout": _slim(receipt["holdout"]),
                "artifact_path": receipt.get("artifact_path"),
                "research_only": True,
                "live_pnl_claim": False,
            },
            indent=2,
            default=str,
        )
    )


@ls_app.command("book")
def book_cmd(
    config: Path = typer.Option(Path("configs/hedge_lab.yaml")),
    label: str = typer.Option("future_idio_return_1", "--label"),
    n_boot: int = typer.Option(1000, "--n-boot"),
    cost_bps: float = typer.Option(10.0, "--cost-bps"),
) -> None:
    """Frozen Lightspeed books + nautica vs ridge on the file tape. Not live P&L."""
    from quant_fund.hedge_lab.lightspeed_book import run_lightspeed_file_book

    receipt = run_lightspeed_file_book(
        str(config),
        label,
        one_way_cost=float(cost_bps) / 1e4,
        n_boot=int(n_boot),
    )
    typer.echo(json.dumps(receipt, indent=2, default=str))
