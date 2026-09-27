"""SEEDED LEAK (LH004): as-of join keyed on event_time.

Deliberately leaky strategy file for the leakage-hunter CI gate
(DESIGN.md §6.4). MUST trip LH004. Do not import from strategy code.
"""

from __future__ import annotations

import polars as pl


def fuse_features(grid: pl.LazyFrame, features: pl.LazyFrame) -> pl.LazyFrame:
    """Leaky: joins on event_time, ignoring publish/known time."""
    return grid.join_asof(features, on="event_time", by="security_id", strategy="backward")
