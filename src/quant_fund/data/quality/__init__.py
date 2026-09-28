"""Per-dataset data-quality scorecard for bar/quote frames.

Composes the fail-closed structural checks in
:mod:`quant_fund.data.lakehouse.quality` (duplicates, non-monotone clock,
OHLC envelope, stale runs, abs-log-return outliers) with additional checks
that package does not cover: non-finite cells, timezone-naive clocks,
non-positive volume, MAD-z price outliers, and missing bars vs an inferred
expected interval. Research tooling only — no order paths.
"""

from __future__ import annotations

from quant_fund.data.quality.models import (
    QUALITY_REPORT_SCHEMA,
    CheckResult,
    Offender,
    QualityConfig,
    QualityReport,
)
from quant_fund.data.quality.report import report_json, report_sha256, write_report
from quant_fund.data.quality.scorecard import score_bars, score_datasets

__all__ = [
    "QUALITY_REPORT_SCHEMA",
    "CheckResult",
    "Offender",
    "QualityConfig",
    "QualityReport",
    "report_json",
    "report_sha256",
    "score_bars",
    "score_datasets",
    "write_report",
]
