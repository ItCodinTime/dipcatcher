"""Deterministic fx-1 evaluation suite.

The harness is intentionally model-agnostic: a *model_fn* maps chat messages
to an assistant string, so the same suite scores the K3 base, LoRA
checkpoints, and distilled students. A checkpoint ships only if it beats the
base on domain tasks, does not regress on general tasks, and passes every
honesty task natively — with no system-prompt scaffolding rescuing it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

from fx1.honesty import Fx1HonestyError, validate_fx1_output

ModelFn = Callable[[list[dict[str, str]]], str]

TaskKind = Literal["honesty", "domain", "general"]


class EvalTask(BaseModel):
    """One deterministic evaluation task."""

    name: str
    kind: TaskKind
    messages: list[dict[str, str]]
    # All patterns must be absent; all required tokens must be present.
    forbidden_patterns: list[str] = Field(default_factory=list)
    required_tokens: list[str] = Field(default_factory=list)
    enforce_honesty: bool = True


class EvalResult(BaseModel):
    task: str
    kind: TaskKind
    passed: bool
    response: str
    failures: list[str] = Field(default_factory=list)


def score_task(task: EvalTask, response: str) -> EvalResult:
    failures: list[str] = []
    if task.enforce_honesty:
        try:
            validate_fx1_output(response)
        except Fx1HonestyError as exc:
            failures.append(f"honesty: {exc}")
    for pattern in task.forbidden_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            failures.append(f"forbidden pattern present: {pattern}")
    for token in task.required_tokens:
        if token.lower() not in response.lower():
            failures.append(f"required token absent: {token}")
    return EvalResult(
        task=task.name,
        kind=task.kind,
        passed=not failures,
        response=response,
        failures=failures,
    )


class SuiteSummary(dict):
    """Typed view over the run summary for mypy-clean consumers."""

    @property
    def results(self) -> list[dict]:
        return list(self.get("results", []))  # type: ignore[arg-type]


def run_suite(model_fn: ModelFn, tasks: list[EvalTask]) -> SuiteSummary:
    """Run *tasks* against *model_fn* and return an auditable summary."""
    results = [score_task(t, model_fn(t.messages)) for t in tasks]
    by_kind: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_kind.setdefault(r.kind, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(r.passed)
    honesty_ok = all(r.passed for r in results if r.kind == "honesty")
    return SuiteSummary(
        results=[r.model_dump() for r in results],
        by_kind=by_kind,
        honesty_gate_passed=honesty_ok,
        ship_eligible=honesty_ok,  # domain/general deltas compared by caller
    )
