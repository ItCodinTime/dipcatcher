"""Northset research exports with cycle-safe lazy package resolution."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "amihud_illiquidity",
    "attach_candle_book_features",
    "bench_candle_order_book",
    "bench_northset",
    "book_metrics_from_snapshot",
    "book_uncrossed_rate",
    "candle_features_from_bars",
    "candle_geometry",
    "corwin_schultz_spread",
    "gap_finite_rate",
    "garman_klass_vs_close_to_close",
    "kyle_lambda",
    "microprice",
    "ohlc_identity_rate",
    "order_flow_imbalance",
    "parkinson_vs_close_to_close",
    "rogers_satchell_vs_close_to_close",
    "roll_spread",
    "session_candles_from_daily",
    "session_reconstructs_daily_rate",
    "session_volume_conservation_rate",
    "synthesize_l2_from_bars",
    "synthesize_snapshots_from_bars",
    "yang_zhang_variance",
    "queue_imbalance",
    "vpin_proxy",
    "abdi_ranaldo_spread",
    "session_chain_rate",
    "session_vpin",
    "session_bipower_jump",
    "liquidity_sweep_frame",
    "sweep_rates",
    "sweep_evidence_battery",
    "sweep_forward_frame",
]

_EXPORT_MODULES = {
    "aggregate_session_book_to_daily": "quant_fund.microstructure.synthetic_lob",
    "synthesize_session_l2": "quant_fund.microstructure.synthetic_lob",
    "snapshots_to_panel": "quant_fund.microstructure.book_panel",
    "write_book_panel": "quant_fund.microstructure.book_panel",
    "load_book_panel": "quant_fund.microstructure.book_panel",
    "validate_book_panel": "quant_fund.microstructure.book_panel",
    "vpin_proxy": "quant_fund.northset.estimators",
    "queue_imbalance": "quant_fund.northset.estimators",
    "attach_candle_book_features": "quant_fund.microstructure.candle_book_features",
    "candle_features_from_bars": "quant_fund.microstructure.candle_book_features",
    "book_metrics_from_snapshot": "quant_fund.microstructure.book_metrics",
    "microprice": "quant_fund.microstructure.book_metrics",
    "bench_candle_order_book": "quant_fund.microstructure.bench",
    "synthesize_l2_from_bars": "quant_fund.microstructure.synthetic_lob",
    "synthesize_snapshots_from_bars": "quant_fund.microstructure.synthetic_lob",
    "bench_northset": "quant_fund.northset.benches",
    "candle_geometry": "quant_fund.northset.candles",
    "amihud_illiquidity": "quant_fund.northset.estimators",
    "corwin_schultz_spread": "quant_fund.northset.estimators",
    "garman_klass_vs_close_to_close": "quant_fund.northset.estimators",
    "kyle_lambda": "quant_fund.northset.estimators",
    "order_flow_imbalance": "quant_fund.northset.estimators",
    "parkinson_vs_close_to_close": "quant_fund.northset.estimators",
    "rogers_satchell_vs_close_to_close": "quant_fund.northset.estimators",
    "roll_spread": "quant_fund.northset.estimators",
    "yang_zhang_variance": "quant_fund.northset.estimators",
    "book_uncrossed_rate": "quant_fund.northset.identities",
    "gap_finite_rate": "quant_fund.northset.identities",
    "ohlc_identity_rate": "quant_fund.northset.identities",
    "session_candles_from_daily": "quant_fund.northset.identities",
    "session_reconstructs_daily_rate": "quant_fund.northset.identities",
    "session_volume_conservation_rate": "quant_fund.northset.identities",
    "session_chain_rate": "quant_fund.northset.identities",
    "abdi_ranaldo_spread": "quant_fund.northset.estimators",
    "session_vpin": "quant_fund.northset.estimators",
    "session_bipower_jump": "quant_fund.northset.estimators",
    "liquidity_sweep_frame": "quant_fund.northset.sweeps",
    "sweep_rates": "quant_fund.northset.sweeps",
    "sweep_evidence_battery": "quant_fund.northset.sweep_research",
    "sweep_forward_frame": "quant_fund.northset.sweep_research",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
