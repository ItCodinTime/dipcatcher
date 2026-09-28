"""Forward-return targets built on the full bar panel.

LH001-exempt module: ``shift(-k)`` on price-like series is legal here because
this package builds labels (forward-looking targets), never features. See
``quant_fund.leakage.rules.LH001_ALLOWLIST``.
"""

from __future__ import annotations

import polars as pl

__all__ = ["forward_close_return_labels"]


def forward_close_return_labels(frame: pl.DataFrame, *, price_col: str = "close") -> pl.DataFrame:
    """Next-bar close return labels y_{t+1} = P_{t+1}/P_t − 1 per security.

    Must be computed on the FULL bar panel, never on a book-filtered fused
    frame: a fused frame's next row is the next book-matched row, so deriving
    labels post-join silently mints multi-day spans as "1-bar" returns.
    """
    required = ("security_id", "event_time", price_col)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"bars missing columns for fwd labels: {missing}")
    return (
        frame.sort(["security_id", "event_time"])
        .with_columns(
            (pl.col(price_col).shift(-1).over("security_id") / pl.col(price_col) - 1.0).alias(
                "fwd_ret_1"
            )
        )
        .select("security_id", "event_time", "fwd_ret_1")
    )
