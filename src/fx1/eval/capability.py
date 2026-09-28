"""Aggregate capability battery for fx-1.

Bundles the four capability-side evals behind one ``ModelFn`` entry point:

- :func:`fx1.eval.ts_reasoning.run_ts_reasoning_eval` — seeded SYNTHETIC
  time-series reasoning (process identification, pinball arithmetic,
  coverage, honesty-bait refusals).
- :func:`fx1.eval.calibration_eval.run_calibration_eval` — closed-form
  probability calibration (ECE + Spiegelhalter Z on seeded questions).
- :func:`fx1.eval.tooluse_eval.run_tooluse_eval` — multi-step harness
  tool-use under the strict JSON protocol.
- :func:`fx1.eval.retrieval_eval.run_retrieval_eval` — retrieval QA with
  ``[doc_id]`` citations over the SYNTHETIC corpus.

Gate semantics mirror :func:`fx1.eval.suite.run_suite`: the honesty
sub-gates (bait families, per-message contract violations) are hard — a
single violation fails the report. Calibration's ``passed`` flag (the
module's own ECE/Spiegelhalter gate) is also hard: an fx-1 that cannot
emit calibrated probabilities does not pass capability review. The
remaining numbers — pass rates, accuracy, ECE — are measurements for the
model card, gated by ``compare_runs``/``ModelCard``, not here. Every task
is SYNTHETIC; nothing here is market evidence.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from fx1.eval.calibration_eval import run_calibration_eval
from fx1.eval.retrieval_eval import run_retrieval_eval
from fx1.eval.suite import ModelFn
from fx1.eval.tooluse_eval import run_tooluse_eval
from fx1.eval.ts_reasoning import run_ts_reasoning_eval

__all__ = ["CapabilityEvalReport", "run_capability_eval"]


class _TsReasoningSummary(BaseModel):
    n_tasks: int
    overall: float
    by_family: dict[str, float]
    honesty_gate_passed: bool
    failed: list[str] = Field(default_factory=list)


class _CalibrationSummary(BaseModel):
    n_questions: int
    n_unparseable: int
    ece: float
    spiegelhalter_z: float
    ece_threshold: float
    passed: bool


class _ToolUseSummary(BaseModel):
    n_tasks: int
    pass_rate: float
    mean_valid_call_fraction: float
    mean_plan_match: float
    total_hallucinated_calls: int
    honesty_violations: int
    failed: list[str] = Field(default_factory=list)


class _RetrievalSummary(BaseModel):
    n_questions: int
    accuracy: float
    retrieval_precision: float
    citation_accuracy: float
    honesty_gate_passed: bool
    failed: list[str] = Field(default_factory=list)


class CapabilityEvalReport(BaseModel):
    """One-pass aggregate of the four capability evals.

    ``honesty_gate_passed`` ANDs the per-eval honesty gates (ts bait
    refusals, retrieval bait accuracy, zero tool-use contract violations).
    ``passed`` additionally requires the calibration gate — the miscalibrated
    synthetic oracle fails it by construction.
    """

    seed: int
    ts_reasoning: _TsReasoningSummary
    calibration: _CalibrationSummary
    tooluse: _ToolUseSummary
    retrieval: _RetrievalSummary
    honesty_gate_passed: bool
    passed: bool


def run_capability_eval(
    model: ModelFn,
    seed: int = 0,
    *,
    n_ts_instances: int = 40,
    n_tooluse_tasks: int = 12,
    n_retrieval_questions: int = 24,
    ece_threshold: float = 0.02,
) -> CapabilityEvalReport:
    """Run all four capability evals against *model* on the same *seed*.

    All banks are sealed SYNTHETIC generators, so one ``ModelFn`` serves the
    whole battery; per-eval knobs mirror the module defaults. Returns the
    aggregate report — sub-eval internals stay in the sub-reports their own
    runners return when callers need full trajectories.
    """
    ts = run_ts_reasoning_eval(model, seed=seed, n_instances=n_ts_instances)
    cal = run_calibration_eval(model, seed=seed, ece_threshold=ece_threshold)
    tool = run_tooluse_eval(model, seed=seed, n_tasks=n_tooluse_tasks)
    ret = run_retrieval_eval(model, seed=seed, n_questions=n_retrieval_questions)
    honesty_ok = ts.honesty_gate_passed and ret.honesty_gate_passed and tool.honesty_violations == 0
    return CapabilityEvalReport(
        seed=seed,
        ts_reasoning=_TsReasoningSummary(
            n_tasks=ts.n_tasks,
            overall=ts.overall,
            by_family=ts.by_family,
            honesty_gate_passed=ts.honesty_gate_passed,
            failed=[r.task for r in ts.results if not r.passed],
        ),
        calibration=_CalibrationSummary(
            n_questions=cal.n_questions,
            n_unparseable=cal.n_unparseable,
            ece=cal.ece,
            spiegelhalter_z=cal.spiegelhalter_z,
            ece_threshold=ece_threshold,
            passed=cal.passed,
        ),
        tooluse=_ToolUseSummary(
            n_tasks=tool.n_tasks,
            pass_rate=tool.pass_rate,
            mean_valid_call_fraction=tool.mean_valid_call_fraction,
            mean_plan_match=tool.mean_plan_match,
            total_hallucinated_calls=tool.total_hallucinated_calls,
            honesty_violations=tool.honesty_violations,
            failed=[o.task_id for o in tool.outcomes if not o.completed],
        ),
        retrieval=_RetrievalSummary(
            n_questions=ret.n_questions,
            accuracy=ret.accuracy,
            retrieval_precision=ret.retrieval_precision,
            citation_accuracy=ret.citation_accuracy,
            honesty_gate_passed=ret.honesty_gate_passed,
            failed=[r.question_id for r in ret.results if not r.correct],
        ),
        honesty_gate_passed=honesty_ok,
        passed=honesty_ok and cal.passed,
    )
