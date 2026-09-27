"""SEEDED LEAK (LH002): centered rolling window.

Deliberately leaky strategy file for the leakage-hunter CI gate
(DESIGN.md §6.4). MUST trip LH002. Do not import from strategy code.
"""

from __future__ import annotations

import polars as pl


def smooth_trend(close: pl.Series) -> pl.Series:
    """Leaky: centered window averages in future bars."""
    return close.rolling_mean(window_size=21, center=True)
