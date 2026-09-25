"""Hypothesis → config → bench → result trace generation.

The highest-value fx-1 training signal is the full research loop: a stated
hypothesis, the config diff that tests it, the harness bench that runs it,
and the gate decision that results. Traces are recorded as structured
objects; only gate-resolved traces (pass *or* fail) are admissible — a trace
with no verdict teaches nothing verifiable.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class GateVerdict(StrEnum):
    PASSED = "passed"
    REJECTED = "rejected"


class ResearchTrace(BaseModel):
    """One complete hypothesis-to-verdict research loop."""

    hypothesis: str
    config_diff: str = Field(description="Unified-diff or YAML fragment")
    bench_command: str = Field(description="Harness command that tests it")
    scores: dict[str, float] = Field(
        description="Proper scores only — headline metrics are rejected"
    )
    verdict: GateVerdict | None = None
    receipt_sha256: str = Field(min_length=64, max_length=64)
    recorded_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def admissible(self) -> bool:
        return self.verdict is not None

    @property
    def trace_id(self) -> str:
        canonical = self.model_dump_json()
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_sft_messages(self, system: str) -> list[dict[str, str]]:
        verdict_text = (
            f"Gate verdict: {self.verdict.value}." if self.verdict else "Gate verdict: pending."
        )
        scores_text = "\n".join(f"- {k}: {v}" for k, v in sorted(self.scores.items()))
        user = (
            f"Hypothesis: {self.hypothesis}\n\n"
            f"Config change under test:\n```\n{self.config_diff}\n```\n"
            f"What do the benches say?"
        )
        assistant = (
            f"Ran `{self.bench_command}` through the harness. Proper scores:\n"
            f"{scores_text}\n\n{verdict_text}\n"
            f"Evidence is bound to receipt {self.receipt_sha256[:16]}… and is "
            "verifiable with `dipcatcher verify-research`. This is research "
            "evidence — not live performance, and not a promotion unless the "
            "verdict is passed *and* promotion gates confirm."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]


# Proper-score tokens the lab recognizes; anything outside this set in a
# trace's scores is rejected fail-closed (headline metrics are not scores).
_ALLOWED_SCORE_TOKENS = frozenset(
    {
        "crps",
        "pinball",
        "pit",
        "qlike",
        "brier",
        "logloss",
        "log_loss",
        "ece",
        "kupiec",
        "hmm",
        "likelihood",
        "coverage",
        "sharpness",
        "rank_ic",
        "spearman",
        "auc",
    }
)


def validate_trace_scores(scores: dict[str, float]) -> None:
    """Fail-closed: every score key must contain an allowed proper-score token."""
    for key in scores:
        tokens = set(key.lower().replace("-", "_").split("_"))
        if not tokens & _ALLOWED_SCORE_TOKENS:
            raise ValueError(
                f"score key {key!r} is not a recognized proper score; "
                "fx-1 traces carry scientific scores only"
            )
