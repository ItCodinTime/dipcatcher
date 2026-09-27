"""Typer app, sub-CLI mounts, and shared helpers for the quant CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from quant_fund.config import dump_resolved, load_config
from quant_fund.hmm.cli import hmm_app
from quant_fund.lightspeed.cli import ls_app
from quant_fund.pit.cli import pit_app
from quant_fund.quant_models.cli import qm_app
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
app.add_typer(pit_app, name="pit")
app.add_typer(qm_app, name="qm")


def _cfg(config: Path):
    cfg = load_config(config)
    configure_logging()
    dump_resolved(cfg, Path(cfg.data.root) / "metadata" / "resolved_config.json")
    return cfg


def _collect_param_value(raw: str) -> object:
    """Coerce a --param value to int/float when it cleanly parses, else str."""
    text = raw.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
