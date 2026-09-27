"""SEEDED LEAK (LH010): backward fill on a time-series frame.

Deliberately leaky strategy file for the leakage-hunter CI gate
(DESIGN.md §6.4). MUST trip LH010 (warning). Do not import from strategy code.
"""

from __future__ import annotations

import polars as pl


def clean_panel(frame: pl.DataFrame) -> pl.DataFrame:
    """Leaky: bfill drags future observations into earlier rows."""
    ordered = frame.sort("event_time")
    return ordered.with_columns(pl.col("mid").bfill())
