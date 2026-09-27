"""Aggregate aligned, linked, scored records into a per-(symbol, bar) signal.

The signal frame is what the research harness consumes: one row per
``(symbol, event_time)`` where ``event_time`` is the bar the news became
usable at. No forward information participates — see ``alignment`` for the
no-lookahead contract.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Protocol

import polars as pl

from quant_fund.research.news_signals.alignment import BarGrid, attach_signal_bars
from quant_fund.research.news_signals.entities import EntityLinker, resolve_symbols
from quant_fund.research.news_signals.schema import NewsRecord
from quant_fund.research.news_signals.sentiment import SentimentScore, classify_event


class _Scorer(Protocol):
    def score(self, text: str, language: str) -> SentimentScore: ...


@dataclass(frozen=True)
class SignalBuildResult:
    """Signal frame plus build provenance (record/drop accounting)."""

    frame: pl.DataFrame
    n_records_in: int
    n_aligned: int
    n_dropped_past_end: int
    n_unlinked: int


SIGNAL_COLUMNS: tuple[str, ...] = (
    "symbol",
    "event_time",
    "news_sent_mean",
    "news_sent_net",
    "news_count",
    "n_pos",
    "n_neg",
    "n_en",
    "n_zh",
    "n_ja",
    "n_earnings",
    "n_guidance",
    "n_mna",
    "n_litigation",
    "n_capital_return",
    "n_product",
    "n_macro",
    "n_credit",
    "n_other_event",
)


def build_signal_frame(
    records: Iterable[NewsRecord],
    grid: BarGrid,
    *,
    scorer: _Scorer,
    linker: EntityLinker | None = None,
    strict: bool = True,
    include_adrs: bool = True,
) -> SignalBuildResult:
    """Link, score, align, and aggregate records into the signal frame.

    * Each record contributes to every ticker it resolves to (multi-entity
      headlines count once per linked symbol).
    * ``news_sent_mean`` is the mean record score at the (symbol, bar) cell;
      ``news_sent_net`` the signed hit balance ``n_pos - n_neg``.
    * ``include_adrs`` additionally emits cross-listed tickers so the signal
      can join whichever listing the return panel carries (``7203.T`` news
      also lands on ``TM``); default on — the join is what decides coverage.
    """
    linker = linker or EntityLinker()
    record_list = list(records)
    aligned, dropped = attach_signal_bars(record_list, grid, strict=strict)

    buckets: dict[tuple[str, object], dict[str, float]] = defaultdict(
        lambda: {c: 0.0 for c in SIGNAL_COLUMNS[2:]}
    )
    unlinked = 0
    for a in aligned:
        rec = a.record
        symbols = resolve_symbols(rec, linker=linker, include_adrs=include_adrs)
        if not symbols:
            unlinked += 1
            continue
        s = scorer.score(rec.text, rec.language)
        event = classify_event(rec.text, rec.language)
        event_col = {
            "earnings": "n_earnings",
            "guidance": "n_guidance",
            "mna": "n_mna",
            "litigation": "n_litigation",
            "capital_return": "n_capital_return",
            "product": "n_product",
            "macro": "n_macro",
            "credit": "n_credit",
        }.get(event, "n_other_event")
        for sym in symbols:
            cell = buckets[(sym, a.bar_time)]
            cell["news_sent_mean"] += s.score
            cell["news_sent_net"] += s.score
            cell["news_count"] += 1.0
            cell["n_pos"] += float(s.n_pos)
            cell["n_neg"] += float(s.n_neg)
            cell[f"n_{rec.language}"] += 1.0
            cell[event_col] += 1.0

    rows: list[dict[str, object]] = []
    for (sym, bar_time), cell in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n = cell["news_count"]
        row: dict[str, object] = {"symbol": sym, "event_time": bar_time}
        row["news_sent_mean"] = cell["news_sent_mean"] / n if n else 0.0
        row["news_sent_net"] = cell["news_sent_net"]
        for col in SIGNAL_COLUMNS[2:]:
            if col not in row:
                row[col] = cell[col]
        rows.append(row)

    schema = {
        "symbol": pl.String,
        "event_time": pl.Datetime(time_unit="us", time_zone="UTC"),
        **{c: pl.Float64 for c in SIGNAL_COLUMNS[2:]},
    }
    frame = pl.DataFrame(rows, schema=schema, orient="row") if rows else pl.DataFrame(schema=schema)
    return SignalBuildResult(
        frame=frame,
        n_records_in=len(record_list),
        n_aligned=len(aligned),
        n_dropped_past_end=dropped,
        n_unlinked=unlinked,
    )
