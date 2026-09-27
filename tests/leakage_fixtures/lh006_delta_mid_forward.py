"""SEEDED LEAK (LH006): forward difference under a contemporaneous name.

Mirrors audit F9 (`delta_mid = mid.shift(-1) - mid`). Deliberately leaky
strategy file for the leakage-hunter CI gate (DESIGN.md §6.4). MUST trip
LH006. Do not import from strategy code.
"""

from __future__ import annotations

import polars as pl


def add_microstructure(frame: pl.DataFrame) -> pl.DataFrame:
    """Leaky: `delta_mid` reads as contemporaneous but encodes next-bar mid."""
    return frame.with_columns(
        (pl.col("mid").shift(-1).over("security_id") - pl.col("mid")).alias("delta_mid")
    )
