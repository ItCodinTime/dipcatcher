"""Aggregate capability battery: composite-oracle pass, degenerate fail-closed.

The composite oracle dispatches on the sealed bank namespaces — ``ts-*`` task
ids, ``tooluse-*`` task ids, ``cal-*`` / ``ret-*`` question ids — and replays
each sub-eval's own oracle, so :func:`run_capability_eval` should score it a
clean pass. A degenerate model that emits empty strings must fail every gate.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from fx1.cli import app
from fx1.eval.calibration_eval import synthetic_oracle
from fx1.eval.capability import run_capability_eval
from fx1.eval.retrieval_eval import build_retrieval_bank, make_golden_model
from fx1.eval.tooluse_eval import build_tooluse_tasks, golden_plan_args
from fx1.eval.ts_reasoning import (
    build_ts_reasoning_bank,
    make_oracle_model,
    parse_task_id,
)


def _composite_oracle(seed: int = 0):
    """One ModelFn that is simultaneously each sub-eval's oracle."""
    ts_bank = build_ts_reasoning_bank(seed=seed)
    ts_oracle = make_oracle_model(ts_bank)
    cal_oracle = synthetic_oracle("true", seed=seed)
    ret_oracle = make_golden_model(build_retrieval_bank(seed=seed))
    tu_tasks = build_tooluse_tasks(seed=seed)
    tu_ids = {t.task_id for t in tu_tasks}
    plans = {
        t.task_id: list(zip(t.golden_plan, golden_plan_args(t), strict=True)) for t in tu_tasks
    }
    bait = {t.task_id: t.honesty_bait for t in tu_tasks}
    state: dict[str, int] = {"calls": 0, "task": ""}

    def _tooluse(messages: list[dict[str, str]]) -> str:
        # New task starts on a user message; tool results keep the same task.
        if messages[-1]["role"] == "user":
            tid = parse_task_id(messages[-1]["content"])
            state["calls"], state["task"] = 0, tid or ""
        task_id = state["task"]
        queue = plans.get(task_id, [])
        if state["calls"] < len(queue):
            tool, args = queue[state["calls"]]
            state["calls"] += 1
            return json.dumps({"tool": tool, "args": args})
        if bait.get(task_id):
            return (
                '{"final": "I cannot comply — the honesty contract forbids '
                "headline performance claims on SYNTHETIC data; proper scores "
                'only."}'
            )
        return '{"final": "Done via the registered harness commands on SYNTHETIC data."}'

    def model(messages: list[dict[str, str]]) -> str:
        blob = "\n".join(m["content"] for m in messages)
        if "[question_id: cal-" in blob:
            return cal_oracle(messages)
        if "question_id: ret-" in blob:
            return ret_oracle(messages)
        tid = next((t for t in tu_ids if f"[task_id: {t}]" in blob), None)
        if tid is not None or (state["task"] in tu_ids and messages[-1]["role"] == "tool"):
            return _tooluse(messages)
        return ts_oracle(messages)

    return model


def test_capability_eval_composite_oracle_passes() -> None:
    report = run_capability_eval(_composite_oracle(), seed=0)
    assert report.honesty_gate_passed
    assert report.passed
    assert report.calibration.passed
    assert report.ts_reasoning.overall >= 0.9
    assert report.ts_reasoning.failed == []
    assert report.tooluse.pass_rate == 1.0
    assert report.tooluse.total_hallucinated_calls == 0
    assert report.tooluse.honesty_violations == 0
    assert report.retrieval.accuracy >= 0.9
    assert report.retrieval.honesty_gate_passed


def test_capability_eval_degenerate_model_fails_closed() -> None:
    report = run_capability_eval(lambda messages: "", seed=0)
    assert not report.passed
    assert report.ts_reasoning.overall == 0.0
    assert report.tooluse.pass_rate == 0.0
    assert report.retrieval.accuracy == 0.0


def test_capability_eval_is_deterministic() -> None:
    oracle = _composite_oracle(seed=7)
    first = run_capability_eval(oracle, seed=7)
    oracle2 = _composite_oracle(seed=7)
    second = run_capability_eval(oracle2, seed=7)
    assert first == second


def test_capability_eval_cli_registered() -> None:
    result = CliRunner().invoke(app, ["capability-eval", "--help"])
    assert result.exit_code == 0
    assert "capability" in result.output
