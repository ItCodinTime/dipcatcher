"""quant CLI."""

# Command modules register Typer callbacks at import time. Keep this order.
# ruff: noqa: I001

from __future__ import annotations

import types
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
from quant_fund.cli import book_cmds as _book_cmds
from quant_fund.cli import data_cmds as _data_cmds
from quant_fund.cli import forecast_cmds as _forecast_cmds
from quant_fund.cli import micro_cmds as _micro_cmds
from quant_fund.cli import ops_cmds as _ops_cmds
from quant_fund.cli import report_cmds as _report_cmds
from quant_fund.cli import research_cmds as _research_cmds
from quant_fund.cli import support as _support
from quant_fund.cli import train_cmds as _train_cmds

_COMMAND_MODULES = (
    _support,
    _data_cmds,
    _forecast_cmds,
    _micro_cmds,
    _research_cmds,
    _book_cmds,
    _report_cmds,
    _ops_cmds,
    _train_cmds,
)
_MISSING = object()


def _publish_command_modules() -> None:
    """Copy command-module globals here so rebound commands resolve patches."""
    facade = globals()
    for module in _COMMAND_MODULES:
        for key, value in vars(module).items():
            if key.startswith("__"):
                continue
            current = facade.get(key, _MISSING)
            if current is not _MISSING and current is not value:
                raise RuntimeError(
                    f"facade global conflict on {key!r} while publishing {module.__name__}"
                )
            facade[key] = value


def _rebind_to_facade(fn: types.FunctionType) -> types.FunctionType:
    rebound = types.FunctionType(
        fn.__code__,
        globals(),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    rebound.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    rebound.__dict__.update(fn.__dict__)
    rebound.__module__ = __name__
    rebound.__qualname__ = fn.__qualname__
    return rebound


def _rebind_command_functions() -> None:
    facade = globals()
    for module in _COMMAND_MODULES:
        for key, value in list(vars(module).items()):
            if isinstance(value, types.FunctionType) and value.__module__ == module.__name__:
                facade[key] = _rebind_to_facade(value)
    for cmd in app.registered_commands:
        callback = cmd.callback
        if isinstance(callback, types.FunctionType) and callback.__name__ in facade:
            cmd.callback = facade[callback.__name__]
    for cmd in train_app.registered_commands:
        callback = cmd.callback
        if isinstance(callback, types.FunctionType) and callback.__name__ in facade:
            cmd.callback = facade[callback.__name__]
    group_callback = train_app.registered_callback
    callback = getattr(group_callback, "callback", None)
    if isinstance(callback, types.FunctionType) and callback.__name__ in facade:
        group_callback.callback = facade[callback.__name__]


_publish_command_modules()
_rebind_command_functions()

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
