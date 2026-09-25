"""Leakage-proof evaluation: masked twins and the memory-gap metric.

KTD-Fin (2026) showed frontier trading agents' apparent alpha collapses once
tickers and calendar information are masked — the models were recalling, not
reasoning. fx-1 therefore evaluates every domain task twice: unmasked and
masked. The *memory gap* (unmasked minus masked pass rate) is a ship-gate
metric: a model whose competence evaporates under masking has memorized
market episodes instead of learning procedure.

Masking is deterministic and consistent within a task: the same surface form
always maps to the same placeholder, so reasoning over masked entities is
still possible while recall is impossible.
"""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel

from fx1.eval.suite import EvalTask

_TICKER_RE = re.compile(r"\b[A-Z]{2,6}(?:USDT|USD|EUR)?\b")
_DATE_RES = (
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
               r"[a-z]*\s+\d{1,2},?\s+\d{4}\b"),
    re.compile(r"\bQ[1-4]\s*\d{4}\b"),
    re.compile(r"\b(19|20)\d{2}\b"),
)
# Proper-score / method vocabulary must survive masking — it is procedural
# knowledge, the thing we want to keep measurable.
_VOCAB_ALLOWLIST = frozenset(
    {
        "CRPS", "PIT", "QLIKE", "Brier".upper(), "ECE", "HMM", "VaR".upper(),
        "ES", "OFI", "VPIN", "Kyle".upper(), "Roll".upper(), "CPCV", "HAC",
        "DM", "RC", "SPA", "StepM".upper(), "TWAP", "SYNTHETIC", "OHLC",
        "OHLCV", "LLOB", "TODO",
    }
)


def _placeholder(kind: str, surface: str) -> str:
    digest = hashlib.sha256(surface.encode()).hexdigest()[:6].upper()
    return f"<{kind}_{digest}>"


def mask_text(text: str) -> str:
    """Deterministically mask tickers, dates, and years in *text*."""

    def ticker_sub(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in _VOCAB_ALLOWLIST:
            return token
        return _placeholder("ASSET", token)

    masked = _TICKER_RE.sub(ticker_sub, text)
    for pattern in _DATE_RES:
        masked = pattern.sub(
            lambda m: _placeholder("DATE", m.group(0)), masked
        )
    return masked


def mask_task(task: EvalTask) -> EvalTask:
    """Produce the masked twin of an eval task.

    Required tokens are masked consistently so a genuinely reasoning model
    can still pass; forbidden patterns are kept (they target violations, and
    violations under masking are still violations).
    """
    masked_messages = [
        {"role": m["role"], "content": mask_text(m["content"])}
        for m in task.messages
    ]
    return EvalTask(
        name=f"{task.name}__masked",
        kind=task.kind,
        messages=masked_messages,
        forbidden_patterns=list(task.forbidden_patterns),
        required_tokens=[mask_text(t) for t in task.required_tokens],
        enforce_honesty=task.enforce_honesty,
    )


def masked_twins(tasks: list[EvalTask]) -> list[EvalTask]:
    """Interleaved unmasked + masked twin set for a task list."""
    out: list[EvalTask] = []
    for task in tasks:
        out.append(task)
        out.append(mask_task(task))
    return out


class MemoryGapReport(BaseModel):
    n_pairs: int
    unmasked_pass_rate: float
    masked_pass_rate: float
    memory_gap: float
    within_budget: bool


def memory_gap_report(
    unmasked_pass: list[bool], masked_pass: list[bool], *, budget: float = 0.25
) -> MemoryGapReport:
    """Paired memory-gap computation. Fail-closed on mismatched inputs."""
    if len(unmasked_pass) != len(masked_pass):
        raise ValueError("masked and unmasked outcomes must be paired")
    if not unmasked_pass:
        raise ValueError("empty task set")
    n = len(unmasked_pass)
    u = sum(unmasked_pass) / n
    m = sum(masked_pass) / n
    return MemoryGapReport(
        n_pairs=n,
        unmasked_pass_rate=u,
        masked_pass_rate=m,
        memory_gap=u - m,
        within_budget=(u - m) <= budget,
    )
