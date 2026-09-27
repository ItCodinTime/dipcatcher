"""CLEAN CONTROL: explicitly-named `fwd_delta_mid` target.

Mirrors the `northset/benches.py` forward-target idiom: the forward
difference is legal because the name announces it (`fwd_` prefix) and the
file is on the LH001 target-builder allowlist. MUST trip ZERO error-severity
findings (DESIGN.md §6.4).
"""

from __future__ import annotations

import polars as pl


def add_targets(frame: pl.DataFrame) -> pl.DataFrame:
    """Legitimate: next-bar mid move carried under an explicit fwd_ name."""
    return frame.with_columns(
        (pl.col("mid").shift(-1).over("security_id") - pl.col("mid")).alias("fwd_delta_mid")
    )
