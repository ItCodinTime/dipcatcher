"""quant CLI."""

# Command modules register Typer callbacks at import time. Keep this order.
# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path

import typer

from quant_fund.config import dump_resolved, load_config
from quant_fund.hmm.cli import hmm_app
from quant_fund.lightspeed.cli import ls_app
from quant_fund.pipeline.doctor import doctor as run_doctor
from quant_fund.pipeline.train import train_family
from quant_fund.quant_models.cli import qm_app
from quant_fund.utils.logging import configure_logging, get_logger

from quant_fund.cli.app import app, train_app
from quant_fund.cli.support import (
    _cfg as _cfg,
    _collect_param_value as _collect_param_value,
    format_data_label,
    format_fdr_families,
)
from quant_fund.cli.data_cmds import (
    doctor,
    ingest,
    collect,
    build_features_cmd,
    build_labels_cmd,
)
from quant_fund.cli.forecast_cmds import (
    validate,
    forecast,
    kronos_forecast,
    optimize,
    backtest,
)
from quant_fund.cli.micro_cmds import (
    candle_book,
    kyle_ofi,
)
from quant_fund.cli.research_cmds import research
from quant_fund.cli.book_cmds import (
    session_book_cmd,
    vendor_book_map_cmd,
    book_panel_cmd,
    northset,
)
from quant_fund.cli.report_cmds import (
    verify_research,
    lab,
    report,
    tearsheet_cmd,
)
from quant_fund.cli.ops_cmds import (
    api,
    paper,
    monitor,
    sim_live,
)
from quant_fund.cli.train_cmds import (
    _train as _train,
    train_callback,
    train_ranking,
    train_distribution,
    train_calibration,
    train_volatility,
    train_alpha,
    train_covariance,
    train_regime,
    train_tail,
    train_reinforcement,
    train_liquidity,
)

__all__ = [
    "format_data_label",
    "format_fdr_families",
    "app",
    "train_app",
    "doctor",
    "ingest",
    "collect",
    "build_features_cmd",
    "build_labels_cmd",
    "train_callback",
    "train_ranking",
    "train_distribution",
    "train_calibration",
    "train_volatility",
    "train_alpha",
    "train_covariance",
    "train_regime",
    "train_tail",
    "train_reinforcement",
    "train_liquidity",
    "validate",
    "forecast",
    "kronos_forecast",
    "optimize",
    "backtest",
    "candle_book",
    "kyle_ofi",
    "research",
    "session_book_cmd",
    "vendor_book_map_cmd",
    "book_panel_cmd",
    "northset",
    "verify_research",
    "lab",
    "report",
    "tearsheet_cmd",
    "api",
    "paper",
    "monitor",
    "sim_live",
    "Path",
    "typer",
    "dump_resolved",
    "load_config",
    "hmm_app",
    "ls_app",
    "run_doctor",
    "train_family",
    "qm_app",
    "configure_logging",
    "get_logger",
]


if __name__ == "__main__":
    app()
