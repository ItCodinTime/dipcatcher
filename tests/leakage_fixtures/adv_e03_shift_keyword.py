"""ADVERSARIAL §1a-E3 (POSITIVE, LH001): keyword-form negative shift."""

from __future__ import annotations

import polars as pl


def momentum_signal() -> pl.Expr:
    return pl.col("close").shift(periods=-1)
