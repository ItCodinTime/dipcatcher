"""Typer app, sub-CLI mounts, and shared helpers for the quant CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from quant_fund.config import dump_resolved, load_config
from quant_fund.hmm.cli import hmm_app
from quant_fund.leakage.cli import leakage_app
from quant_fund.lightspeed.cli import ls_app
from quant_fund.proof.cli import proof_app
from quant_fund.proofcore.cli import proofcore_app
from quant_fund.quant_models.cli import qm_app
from quant_fund.reality.cli import reality_app
from quant_fund.utils.logging import configure_logging


def format_data_label(*, synthetic: bool, data_source: str) -> str:
    """Always-printed DATA_LABEL line for research / paper CLI output."""
    return f"DATA_LABEL={'SYNTHETIC' if synthetic else data_source}"


def format_fdr_families(hypotheses: list) -> str:
    """BH-FDR family split summary — calibration/discovery/bound never pooled."""
    cal_n = sum(1 for h in hypotheses if getattr(h, "family", None) == "calibration")
    disc_n = sum(1 for h in hypotheses if getattr(h, "family", None) == "discovery")
    bound_n = sum(1 for h in hypotheses if getattr(h, "family", None) == "bound")
    cal = sum(
        1
        for h in hypotheses
        if getattr(h, "family", None) == "calibration" and getattr(h, "reject_fdr", False)
    )
    disc = sum(
        1
        for h in hypotheses
        if getattr(h, "family", None) == "discovery" and getattr(h, "reject_fdr", False)
    )
    return (
        "BH-FDR families (split, never pooled): "
        f"calibration n={cal_n} rejects={cal}; "
        f"discovery n={disc_n} rejects={disc}; "
        f"bound n={bound_n} (not FDR-adjusted)"
    )


app = typer.Typer(
    help="Dipcatcher — Artificial Hedge's proprietary research lab. Default mode is research, never live."
)


train_app = typer.Typer(help="Train a forecast family.")
app.add_typer(train_app, name="train")
app.add_typer(hmm_app, name="hmm")
app.add_typer(ls_app, name="ls")
app.add_typer(qm_app, name="qm")
# PROOFCORE sub-typers (DESIGN.md §9.2). `pit` is mounted by W1 in the same
# block; each PROOFCORE sub-typer module keeps its imports function-level
# (lazy) so importing the CLI never eagerly pulls the new stack.
app.add_typer(proof_app, name="proof")
app.add_typer(leakage_app, name="leakage")
app.add_typer(reality_app, name="reality")
app.add_typer(proofcore_app, name="proofcore")


def _cfg(config: Path):
    cfg = load_config(config)
    configure_logging()
    dump_resolved(cfg, Path(cfg.data.root) / "metadata" / "resolved_config.json")
    return cfg
