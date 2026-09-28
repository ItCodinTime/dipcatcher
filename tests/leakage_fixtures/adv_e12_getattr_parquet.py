"""ADVERSARIAL §1a-E12 (POSITIVE, LH009): dynamic-attribute parquet read."""

from __future__ import annotations

import polars as pl


def sneak(path):
    reader = getattr(pl, "read_" + "parquet")
    return reader(path)
