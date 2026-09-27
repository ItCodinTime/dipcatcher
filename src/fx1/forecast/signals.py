"""Placeholder maps from a forecast score to a bounded signal.

These are not strategies and not orders. They exist so a later fx-1 forecast
can be turned into a diagnostic position for the existing return metrics.
``mapping_role`` is always ``PLACEHOLDER_NOT_A_STRATEGY``.
"""

from __future__ import annotations

import polars as pl

from fx1.forecast.schema import SchemaError

PLACEHOLDER_NOT_A_STRATEGY = "PLACEHOLDER_NOT_A_STRATEGY"
SIGNAL_MAPPINGS = frozenset({"sign", "rank", "threshold"})


def map_signals(frame: pl.DataFrame, mapping: str, *, threshold: float = 0.0) -> pl.DataFrame:
    """Attach ``signal`` in ``[-1, 1]`` from the ``score`` column.

    ``sign`` is the sign of the score. ``threshold`` is the sign when
    ``abs(score) > threshold`` and zero otherwise. ``rank`` is a
    cross-sectional average rank at each ``event_time``, scaled to ``[-1, 1]``.
    A single name maps to zero because a rank has no cross-section.
    """
    if mapping not in SIGNAL_MAPPINGS:
        raise ValueError(f"unknown signal mapping {mapping!r}; expected {sorted(SIGNAL_MAPPINGS)}")
    if "score" not in frame.columns:
        raise SchemaError("signal mapping requires a score column")
    if not (threshold >= 0.0) or threshold != threshold:
        raise ValueError("threshold must be finite and >= 0")
    finite = pl.col("score").is_not_null() & pl.col("score").is_finite()
    if mapping == "sign":
        signal = pl.when(finite).then(pl.col("score").sign()).otherwise(None)
    elif mapping == "threshold":
        signal = (
            pl.when(finite & (pl.col("score").abs() > threshold))
            .then(pl.col("score").sign())
            .when(finite)
            .then(0.0)
            .otherwise(None)
        )
    else:
        count = pl.len().over("event_time")
        rank = pl.col("score").rank(method="average").over("event_time")
        scaled = (rank - 1.0) / (count - 1.0) * 2.0 - 1.0
        signal = pl.when(~finite).then(None).when(count <= 1).then(0.0).otherwise(scaled)
    return frame.with_columns(
        signal.alias("signal"),
        pl.lit(mapping).alias("mapping"),
        pl.lit(PLACEHOLDER_NOT_A_STRATEGY).alias("mapping_role"),
    )
