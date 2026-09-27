"""SEEDED LEAK (LH001): backward shift on a price-like series.

Deliberately leaky strategy file for the leakage-hunter CI gate
(DESIGN.md §6.4). MUST trip LH001. Do not import from strategy code.
"""

from __future__ import annotations

import polars as pl


def momentum_signal(close: pl.Series) -> pl.Series:
    """Leaky: tomorrow's close becomes today's 'signal'."""
    return close.shift(-1) - close
