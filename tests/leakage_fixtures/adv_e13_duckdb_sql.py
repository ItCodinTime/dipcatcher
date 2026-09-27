"""ADVERSARIAL §1a-E13 (POSITIVE, LH009): parquet read inside a SQL string."""

from __future__ import annotations

import duckdb


def sneak():
    return duckdb.sql("select * from read_parquet('lake/raw.parquet')")
