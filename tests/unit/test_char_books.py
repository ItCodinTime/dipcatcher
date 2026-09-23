"""Lagged characteristic books do not use today's score on today's return."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from quant_fund.hedge_lab.char_books import binary_timing, lagged_characteristic_book


def test_lagged_book_ignores_todays_score() -> None:
    rows = []
    day0 = datetime(2024, 1, 1)
    for d in range(6):
        for name, score, ret in (("AAA", 2.0, 0.01), ("BBB", -1.0, -0.02), ("CCC", 0.0, 0.0), ("DDD", 0.5, 0.005), ("EEE", -0.5, -0.001)):
            rows.append(
                {
                    "event_time": day0 + timedelta(days=d),
                    "security_id": name,
                    "score": score,
                    "ret": ret,
                }
            )
    frame = pl.DataFrame(rows)
    shocked = frame.with_columns(
        pl.when(
            (pl.col("security_id") == "BBB")
            & (pl.col("event_time") == day0 + timedelta(days=5))
        )
        .then(pl.lit(99.0))
        .otherwise(pl.col("score"))
        .alias("score")
    )
    _, base = lagged_characteristic_book(frame, "score", prefer_high=True, k_frac=0.4, one_way_cost=0.0)
    _, alt = lagged_characteristic_book(shocked, "score", prefer_high=True, k_frac=0.4, one_way_cost=0.0)
    assert np.allclose(base, alt)


def test_binary_timing_is_lagged() -> None:
    signal = np.array([False, False, True, True])
    asset = np.array([0.1, 0.2, 0.3, 0.4])
    pnl = binary_timing(signal, asset, one_way_cost=0.0)
    assert pnl.tolist() == [0.0, 0.0, 0.0, 0.4]
