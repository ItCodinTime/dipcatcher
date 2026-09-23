"""Causal event timing for the spot/perpetual pair candidate."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
from scripts.eval_basis_pair import basis_pair_events


def _bars() -> tuple[pl.DataFrame, pl.DataFrame]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    basis = [0.001 + 0.0001 * np.sin(i) for i in range(30)] + [0.005, 0.0, 0.0]
    perp = []
    spot = []
    for i, b in enumerate(basis):
        date = start + timedelta(days=i)
        known = date + timedelta(hours=23, minutes=59)
        perp.append(
            {"security_id": "BTC", "event_time": date, "close": 100 * (1 + b),
             "available_time": known}
        )
        spot.append(
            {"security_id": "BTC", "event_time": date, "close": 100.0,
             "available_time": known}
        )
    return pl.DataFrame(perp), pl.DataFrame(spot)


def test_basis_pair_entry_then_explicit_exit() -> None:
    perp, spot = _bars()
    events = basis_pair_events(perp, spot)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    assert events.select("event_time", "target_weight").rows() == [
        (start + timedelta(days=30), 0.05),
        (start + timedelta(days=31), 0.0),
    ]


def test_basis_pair_rejects_bar_unavailable_by_next_open() -> None:
    perp, spot = _bars()
    start = datetime(2024, 1, 1, tzinfo=UTC)
    spot = spot.with_columns(
        pl.when(pl.col("event_time") == start + timedelta(days=30))
        .then(start + timedelta(days=31))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    assert basis_pair_events(perp, spot).is_empty()
