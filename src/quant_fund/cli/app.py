"""Typer application object shared by command modules.

``app`` is the same object as ``quant_fund.cli._app.app``.
"""

from __future__ import annotations

from quant_fund.cli._app import app, train_app

__all__ = [
    "app",
    "train_app",
]
