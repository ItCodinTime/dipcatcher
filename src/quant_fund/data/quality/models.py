"""Pydantic models for the bar/quote data-quality scorecard.

The report is a deterministic, JSON-serializable artifact: no wall-clock
timestamps, no process-dependent ordering. ``schema`` follows the same
``dipcatcher.<domain>.<artifact>.vN`` convention as
``dipcatcher.lake.quality.v1`` in :mod:`quant_fund.data.lakehouse.quality`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

QUALITY_REPORT_SCHEMA = "dipcatcher.data.quality.v1"


class Offender(BaseModel):
    """One worst-offending row for a check.

    ``timestamp`` is the row's resolved clock value rendered as a string
    (ISO-8601 for temporal clocks, the raw value otherwise). ``value`` carries
    the severity metric when the check has one (gap size in seconds, run
    length, |z| score); ``extra`` carries any other identifying fields.
    """

    timestamp: str | None = None
    symbol: str | None = None
    detail: str
    value: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class CheckResult(BaseModel):
    """Outcome of one quality check on one dataset.

    ``violations`` counts offending rows or events; ``checked`` counts the
    rows (or adjacent pairs, for gap checks) that were evaluated. A check is
    ``applicable=False`` when the frame lacks the columns it needs; such
    checks are excluded from the score.
    """

    name: str
    applicable: bool
    violations: int
    checked: int
    weight: float
    detail: str = ""
    worst: list[Offender] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Fraction of evaluated items with no violation, clamped to [0, 1]."""
        if not self.applicable or self.checked <= 0:
            return 1.0
        return max(0.0, 1.0 - self.violations / self.checked)


class QualityConfig(BaseModel):
    """Thresholds for :func:`quant_fund.data.quality.score_bars`.

    Defaults mirror :class:`quant_fund.data.lakehouse.quality.QualityThresholds`
    where the checks overlap (``stale_run_length``, ``max_abs_log_return``).
    """

    stale_run_length: int = 5
    max_abs_log_return: float = 0.5
    mad_z_threshold: float = 3.5
    expected_interval_seconds: float | None = None
    gap_factor: float = 1.5
    max_offenders: int = 8
    symbol_column: str | None = None
    clock_column: str | None = None


class QualityReport(BaseModel):
    """Per-dataset quality scorecard.

    ``score`` is the weight-normalized mean per-check pass rate in [0, 1];
    1.0 is clean. ``passed`` is True only when every applicable check found
    zero violations.
    """

    model_config = ConfigDict(populate_by_name=True)

    # Serializes as "schema" (repo receipt convention); the trailing
    # underscore avoids shadowing the pydantic v1-era BaseModel attribute.
    schema_: str = Field(default=QUALITY_REPORT_SCHEMA, alias="schema")
    dataset: str | None = None
    rows: int
    symbols: int | None = None
    symbol_column: str | None = None
    clock_column: str | None = None
    start: str | None = None
    end: str | None = None
    passed: bool
    score: float
    checks: list[CheckResult]
    thresholds: dict[str, Any] = Field(default_factory=dict)

    def check(self, name: str) -> CheckResult | None:
        """Return one check by name, or None."""
        for result in self.checks:
            if result.name == name:
                return result
        return None
