"""CLEAN CONTROL: legitimate forward-return label builder.

Mirrors the `labels/engine.py` idiom (explicit `fwd_*` naming, horizons as
parameters). MUST trip ZERO error-severity findings (DESIGN.md §6.4).
"""

from __future__ import annotations

import polars as pl


def forward_returns(frame: pl.DataFrame, horizons: list[int]) -> pl.DataFrame:
    """Legitimate: forward returns are TARGETS with explicit fwd_ names."""
    exprs = [
        (pl.col("close").shift(-h).over("security_id") / pl.col("close") - 1.0).alias(
            f"fwd_ret_{h}"
        )
        for h in horizons
    ]
    return frame.with_columns(exprs)
