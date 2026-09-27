"""Deterministic research panels over bar files.

The lagged return uses only the previous close of the same symbol. Bytes are
canonical JSON so a migration that preserves file bytes preserves this output.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from quant_fund.schemas.errors import DataContractError
from quant_fund.utils.hashing import canonical_json_bytes


def lagged_return_panel(frame: pl.DataFrame) -> pl.DataFrame:
    """Simple return against the prior close within each symbol."""
    missing = [name for name in ("symbol", "event_time", "close") if name not in frame.columns]
    if missing:
        raise DataContractError(f"research panel missing columns: {missing}")
    return (
        frame.sort(["symbol", "event_time"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias("ret_1")
        )
        .select(["symbol", "event_time", "close", "ret_1"])
    )


def lagged_return_panel_bytes(path: Path) -> bytes:
    """Canonical bytes of :func:`lagged_return_panel` for one Parquet file."""
    frame = pl.read_parquet(path)
    return canonical_json_bytes(lagged_return_panel(frame).to_dicts())
