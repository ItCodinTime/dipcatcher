# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Synthetic L2 book
#
# Data label: SYNTHETIC
#
# Not investment advice. No live-trading claim.
# Bars are a planted random walk. The book, VPIN proxy, queue imbalance, and
# candle/book as-of join are research diagnostics on that labeled book.

# %%
from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from quant_fund.microstructure.candle_book_features import attach_candle_book_features
from quant_fund.microstructure.synthetic_lob import synthesize_l2_from_bars
from quant_fund.northset.estimators import queue_imbalance, vpin_proxy
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

N_SESSIONS = 40
VPIN_WINDOW = 5


def _bars() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    start = datetime(2020, 1, 2, 21, 0, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for security_id, price0 in (("AAA", 50.0), ("BBB", 20.0), ("CCC", 100.0), ("DDD", 8.0)):
        price = price0
        for step in range(N_SESSIONS):
            event_time = start + timedelta(days=step)
            open_ = price
            close = max(price * (1.0 + float(rng.normal(0.0004, 0.01))), 0.5)
            high = max(open_, close) * (1.0 + abs(float(rng.normal(0.0, 0.002))))
            low = min(open_, close) * (1.0 - abs(float(rng.normal(0.0, 0.002))))
            rows.append(
                {
                    "security_id": security_id,
                    "symbol": security_id,
                    "event_time": event_time,
                    "available_time": event_time,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": float(rng.integers(1_000, 5_000)),
                }
            )
            price = close
    return pl.DataFrame(rows).with_columns(
        pl.col("event_time").cast(pl.Datetime("us", "UTC")),
        pl.col("available_time").cast(pl.Datetime("us", "UTC")),
    )


def _mean(frame: pl.DataFrame, column: str) -> float:
    values = np.asarray(frame[column].to_numpy(), dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise SystemExit(f"{column} has no finite values")
    return float(finite.mean())


def main() -> None:
    bars = _bars()
    book = queue_imbalance(synthesize_l2_from_bars(bars, depth=5, seed=7))
    scored = vpin_proxy(book, window=VPIN_WINDOW)
    fused = attach_candle_book_features(bars, book=book, depth=5, seed=7)
    book_source = {str(value) for value in fused["book_source"].unique().to_list()}
    book_dgp = {str(value) for value in fused["book_dgp"].unique().to_list()}
    if book_source != {"synthetic"} or book_dgp != {"synthetic_lob"}:
        raise SystemExit(f"unexpected book label {book_source=} {book_dgp=}")
    join_coverage = float(fused["join_coverage"][0])
    if not 0.0 <= join_coverage <= 1.0:
        raise SystemExit("join coverage is outside [0, 1]")
    scores: dict[str, object] = {
        "data_label": "SYNTHETIC",
        "claim": "research_only",
        "not_investment_advice": "true",
        "no_live_trading_claim": "true",
        "n_bars": bars.height,
        "n_book_rows": scored.height,
        "n_fused_rows": fused.height,
        "book_source": "synthetic",
        "book_dgp": "synthetic_lob",
        "vpin_mean": _mean(scored, "vpin"),
        "queue_imbalance_mean": _mean(scored, "queue_imbalance"),
        "join_coverage": join_coverage,
        "fused_queue_imbalance_mean": _mean(fused, "queue_imbalance"),
    }
    if not family_blob_forbidden_metrics_absent(scores):
        raise SystemExit("score dict contains a forbidden headline metric")
    for key in sorted(scores):
        value = scores[key]
        if isinstance(value, float):
            print(f"{key}={value:.6g}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"example_failed={type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
