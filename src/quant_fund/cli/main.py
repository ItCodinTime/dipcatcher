"""quant CLI."""

from __future__ import annotations

from quant_fund.cli._app import (
    _cfg as _cfg,
)
from quant_fund.cli._app import (
    _collect_param_value as _collect_param_value,
)
from quant_fund.cli._app import (
    app as app,
)
from quant_fund.cli._app import (
    format_data_label as format_data_label,
)
from quant_fund.cli._app import (
    format_fdr_families as format_fdr_families,
)
from quant_fund.cli._app import (
    train_app as train_app,
)
from quant_fund.cli.audit_cmds import (
    audit_record as audit_record,
)
from quant_fund.cli.audit_cmds import (
    audit_trace as audit_trace,
)
from quant_fund.cli.audit_cmds import (
    verify_ledger_cmd as verify_ledger_cmd,
)
from quant_fund.cli.book_cmds import (
    book_panel_cmd as book_panel_cmd,
)
from quant_fund.cli.book_cmds import (
    candle_book as candle_book,
)
from quant_fund.cli.book_cmds import (
    kyle_ofi as kyle_ofi,
)
from quant_fund.cli.book_cmds import (
    northset as northset,
)
from quant_fund.cli.book_cmds import (
    session_book_cmd as session_book_cmd,
)
from quant_fund.cli.book_cmds import (
    vendor_book_map_cmd as vendor_book_map_cmd,
)
from quant_fund.cli.data_cmds import (
    build_features_cmd as build_features_cmd,
)
from quant_fund.cli.data_cmds import (
    build_labels_cmd as build_labels_cmd,
)
from quant_fund.cli.data_cmds import (
    collect as collect,
)
from quant_fund.cli.data_cmds import (
    doctor as doctor,
)
from quant_fund.cli.data_cmds import (
    ingest as ingest,
)
from quant_fund.cli.lake_cmds import (
    lake_import as lake_import,
)
from quant_fund.cli.lake_cmds import (
    lake_quality as lake_quality,
)
from quant_fund.cli.lake_cmds import (
    lineage_show as lineage_show,
)
from quant_fund.cli.lake_cmds import (
    lineage_verify as lineage_verify,
)
from quant_fund.cli.ops_cmds import (
    api as api,
)
from quant_fund.cli.ops_cmds import (
    monitor as monitor,
)
from quant_fund.cli.ops_cmds import (
    paper as paper,
)
from quant_fund.cli.ops_cmds import (
    report as report,
)
from quant_fund.cli.ops_cmds import (
    sim_live as sim_live,
)
from quant_fund.cli.ops_cmds import (
    tearsheet_cmd as tearsheet_cmd,
)
from quant_fund.cli.research_cmds import (
    backtest as backtest,
)
from quant_fund.cli.research_cmds import (
    fleet as fleet,
)
from quant_fund.cli.research_cmds import (
    forecast as forecast,
)
from quant_fund.cli.research_cmds import (
    kronos_forecast as kronos_forecast,
)
from quant_fund.cli.research_cmds import (
    lab as lab,
)
from quant_fund.cli.research_cmds import (
    optimize as optimize,
)
from quant_fund.cli.research_cmds import (
    research as research,
)
from quant_fund.cli.research_cmds import (
    validate as validate,
)
from quant_fund.cli.research_cmds import (
    verify_identities as verify_identities,
)
from quant_fund.cli.research_cmds import (
    verify_research as verify_research,
)
from quant_fund.cli.train_cmds import (
    _train as _train,
)
from quant_fund.cli.train_cmds import (
    train_alpha as train_alpha,
)
from quant_fund.cli.train_cmds import (
    train_calibration as train_calibration,
)
from quant_fund.cli.train_cmds import (
    train_callback as train_callback,
)
from quant_fund.cli.train_cmds import (
    train_covariance as train_covariance,
)
from quant_fund.cli.train_cmds import (
    train_distribution as train_distribution,
)
from quant_fund.cli.train_cmds import (
    train_liquidity as train_liquidity,
)
from quant_fund.cli.train_cmds import (
    train_ranking as train_ranking,
)
from quant_fund.cli.train_cmds import (
    train_regime as train_regime,
)
from quant_fund.cli.train_cmds import (
    train_reinforcement as train_reinforcement,
)
from quant_fund.cli.train_cmds import (
    train_tail as train_tail,
)
from quant_fund.cli.train_cmds import (
    train_volatility as train_volatility,
)


def _maybe_install_observation_hooks() -> None:
    """Wrap pipeline entry points only when the operator opted in."""
    import os

    flag = os.environ.get("DIPCATCHER_OBSERVE", "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    from quant_fund.observe.install import install_passive_hooks

    install_passive_hooks()


_maybe_install_observation_hooks()

if __name__ == "__main__":
    app()
