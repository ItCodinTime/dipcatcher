"""Candle + order-book microstructure helpers used by Northset.

Fuses OHLCV bars with L2 snapshots for research features and benches.
Always ``research_only`` / ``live_pnl_claim=false``. Not a live trading path.
The catalog family and CLI name is Northset (ADR-021).

``bench_candle_order_book`` is lazy — importing it eagerly cycles through
``research.agent`` → ``northset``.
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.book_metrics import (
    DEPTH_SHAPE_FIELDS,
    METRICS_OPTIONAL_NAN_OK_KEYS,
    METRICS_REQUIRED_FINITE_KEY_DOCS,
    METRICS_REQUIRED_FINITE_KEYS,
    QUEUE_STRUCTURE_FIELDS,
    SIDE_NOTIONAL_FIELDS,
    SIDE_STRUCTURE_FIELDS,
    TOB_SHARE_FIELDS,
    assert_metrics_key_partition,
    assert_metrics_optional_not_inf,
    assert_metrics_required_finite,
    book_metrics_from_snapshot,
    concentration_top_finite_rate,
    depth_shape_finite_rate,
    microprice,
    queue_priority_finite_rate,
    side_notional_finite_rate,
    tob_size_share_finite_rate,
)
from quant_fund.microstructure.candle_book_features import (
    attach_candle_book_features,
    candle_features_from_bars,
)
from quant_fund.microstructure.synthetic_lob import (
    ensure_book_panel_shape_columns,
    synthesize_l2_from_bars,
    synthesize_snapshots_from_bars,
)

__all__ = [
    "attach_candle_book_features",
    "bench_candle_order_book",
    "DEPTH_SHAPE_FIELDS",
    "QUEUE_STRUCTURE_FIELDS",
    "SIDE_NOTIONAL_FIELDS",
    "METRICS_OPTIONAL_NAN_OK_KEYS",
    "METRICS_REQUIRED_FINITE_KEY_DOCS",
    "METRICS_REQUIRED_FINITE_KEYS",
    "TOB_SHARE_FIELDS",
    "SIDE_STRUCTURE_FIELDS",
    "assert_metrics_key_partition",
    "assert_metrics_optional_not_inf",
    "assert_metrics_required_finite",
    "book_metrics_from_snapshot",
    "concentration_top_finite_rate",
    "queue_priority_finite_rate",
    "side_notional_finite_rate",
    "tob_size_share_finite_rate",
    "depth_shape_finite_rate",
    "candle_features_from_bars",
    "microprice",
    "ensure_book_panel_shape_columns",
    "synthesize_l2_from_bars",
    "synthesize_snapshots_from_bars",
    "validate_book_panel",
    "load_book_panel",
    "write_book_panel",
    "snapshots_to_panel",
    "BOOK_PANEL_REQUIRED",
    "VENDOR_PRESETS",
    "remap_vendor_quotes_to_panel",
    "dry_run_vendor_book_map",
    "remap_vendor_bars",
    "dry_run_vendor_bar_map",
    "synthesize_session_l2",
]


_LAZY = {
    "vendor_panel_from_bars": (
        "quant_fund.microstructure.vendor_book_map",
        "vendor_panel_from_bars",
    ),
    "disguise_panel_as_polygon": (
        "quant_fund.microstructure.vendor_book_map",
        "disguise_panel_as_polygon",
    ),
    "disguise_panel_as_alpaca": (
        "quant_fund.microstructure.vendor_book_map",
        "disguise_panel_as_alpaca",
    ),
    "VENDOR_PRESETS": ("quant_fund.microstructure.vendor_book_map", "VENDOR_PRESETS"),
    "remap_vendor_quotes_to_panel": (
        "quant_fund.microstructure.vendor_book_map",
        "remap_vendor_quotes_to_panel",
    ),
    "dry_run_vendor_book_map": (
        "quant_fund.microstructure.vendor_book_map",
        "dry_run_vendor_book_map",
    ),
    "remap_vendor_bars": (
        "quant_fund.microstructure.vendor_book_map",
        "remap_vendor_bars",
    ),
    "dry_run_vendor_bar_map": (
        "quant_fund.microstructure.vendor_book_map",
        "dry_run_vendor_bar_map",
    ),
    "aggregate_session_book_to_daily": (
        "quant_fund.microstructure.synthetic_lob",
        "aggregate_session_book_to_daily",
    ),
    "synthesize_session_l2": ("quant_fund.microstructure.synthetic_lob", "synthesize_session_l2"),
    "bench_candle_order_book": ("quant_fund.microstructure.bench", "bench_candle_order_book"),
    "validate_book_panel": ("quant_fund.microstructure.book_panel", "validate_book_panel"),
    "load_book_panel": ("quant_fund.microstructure.book_panel", "load_book_panel"),
    "write_book_panel": ("quant_fund.microstructure.book_panel", "write_book_panel"),
    "snapshots_to_panel": ("quant_fund.microstructure.book_panel", "snapshots_to_panel"),
    "BOOK_PANEL_REQUIRED": ("quant_fund.microstructure.book_panel", "BOOK_PANEL_REQUIRED"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    from importlib import import_module

    return getattr(import_module(mod_name), attr)
