"""Multilingual financial-news signal research (en / zh / ja).

Research-only package: ingest normalized headline records, link multilingual
entity mentions to canonical tickers, score them with a lightweight
deterministic sentiment/event classifier, align them point-in-time to the
repo's daily bar grid (no lookahead), and evaluate predictive IC through the
existing ``quant_fund.metrics.cross_section`` harness.

Everything here operates on clearly-labeled SYNTHETIC fixture corpora or,
optionally, on real bar data for the return panel only. The live fetch path
(``ingest.LiveNewsFetcher``) is never exercised by the test suite.
"""

from __future__ import annotations

from quant_fund.research.news_signals.schema import (
    SUPPORTED_LANGUAGES,
    NewsRecord,
    normalize_utc,
    record_from_dict,
)
from quant_fund.research.news_signals.ingest import (
    LiveNewsFetcher,
    load_fixture_corpus,
    write_fixture_corpus,
)
from quant_fund.research.news_signals.entities import (
    ALIAS_TABLE,
    BARE_TICKER_STOPWORDS,
    AliasEntry,
    EntityLinker,
    resolve_symbols,
)
from quant_fund.research.news_signals.sentiment import (
    EVENT_TYPES,
    LexiconScorer,
    SentimentScore,
    TfidfSentiment,
    classify_event,
)
from quant_fund.research.news_signals.alignment import (
    AlignedNews,
    BarGrid,
    attach_signal_bars,
    lookahead_violations,
)
from quant_fund.research.news_signals.signals import (
    SIGNAL_COLUMNS,
    SignalBuildResult,
    build_signal_frame,
)
from quant_fund.research.news_signals.eval import (
    NEWS_EVAL_SCHEMA,
    SignalIC,
    evaluate_signal_ic,
    run_news_eval,
    write_news_receipt,
)

__all__ = [
    "ALIAS_TABLE",
    "BARE_TICKER_STOPWORDS",
    "EVENT_TYPES",
    "NEWS_EVAL_SCHEMA",
    "SIGNAL_COLUMNS",
    "SUPPORTED_LANGUAGES",
    "AliasEntry",
    "AlignedNews",
    "BarGrid",
    "EntityLinker",
    "LexiconScorer",
    "LiveNewsFetcher",
    "NewsRecord",
    "SentimentScore",
    "SignalBuildResult",
    "SignalIC",
    "TfidfSentiment",
    "attach_signal_bars",
    "build_signal_frame",
    "classify_event",
    "evaluate_signal_ic",
    "load_fixture_corpus",
    "lookahead_violations",
    "normalize_utc",
    "record_from_dict",
    "resolve_symbols",
    "run_news_eval",
    "write_fixture_corpus",
    "write_news_receipt",
]
