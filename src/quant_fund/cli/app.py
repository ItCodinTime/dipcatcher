"""Typer application object shared by command modules.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

import typer

from quant_fund.hmm.cli import hmm_app
from quant_fund.lightspeed.cli import ls_app
from quant_fund.quant_models.cli import qm_app

app = typer.Typer(
    help="Dipcatcher — Artificial Hedge's proprietary research lab. Default mode is research, never live."
)
train_app = typer.Typer(help="Train a forecast family.")
app.add_typer(train_app, name="train")
app.add_typer(hmm_app, name="hmm")
app.add_typer(ls_app, name="ls")
app.add_typer(qm_app, name="qm")

__all__ = [
    "app",
    "train_app",
]
