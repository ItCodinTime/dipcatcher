"""Tests for industry-grade machinery: quality gates, traces, DPO, receipts,
statistical comparison, cluster specs, and the staged pipeline."""

import json
from pathlib import Path

import pytest

from fx1.data import (
    TraceRecorder,
    TraceStep,
    Trajectory,
    dedup_and_filter,
    frozen_split,
)
from fx1.eval.compare import compare_runs
from fx1.train import (
    ClusterSpec,
    LadderStage,
    Pipeline,
    TrainConfig,
    build_preference_pairs,
    issue_receipt,
    verify_training_receipt,
)


def _ex(text: str) -> dict:
    return {
        "messages": [{"role": "user", "content": text}],
        "receipt_sha256": "a" * 64,
        "source_path": "t",
        "negative": False,
    }


# --- data quality -----------------------------------------------------------


def test_dedup_and_decontamination():
    examples = [
        _ex("alpha beta gamma delta epsilon zeta eta theta iota"),
        _ex("alpha beta gamma delta epsilon zeta eta theta iota"),
        _ex("completely distinct research content here now"),
    ]
    eval_prompts = ["alpha beta gamma delta epsilon zeta eta theta iota"]
    kept, report = dedup_and_filter(examples, eval_prompts=eval_prompts)
    # Both copies of the contaminated text are removed; contamination is
    # checked before exact-dedup registration, so neither lands in seen_exact.
    assert report.contaminated_removed == 2
    assert report.exact_duplicates_removed == 0
    assert report.kept == 1


def test_frozen_split_deterministic(tmp_path: Path):
    examples = [_ex(f"example number {i} with enough distinct tokens {i}") for i in range(40)]
    m1 = frozen_split(examples, tmp_path / "a", seed=17)
    m2 = frozen_split(examples, tmp_path / "b", seed=17)
    assert m1.train_sha256 == m2.train_sha256
    assert m1.train_count + m1.val_count == 40
    assert (tmp_path / "a.split.json").exists()


# --- traces -----------------------------------------------------------------


def test_trace_preserves_reasoning_and_tool_calls(tmp_path: Path):
    traj = Trajectory(
        session_id="s1",
        user_intent="Run the benches",
        steps=[
            TraceStep(
                reasoning_content="think first",
                assistant_content="running research",
                tool_call=None,
            )
        ],
        verify_ok=True,
        artifact_receipts=["b" * 64],
    )
    from fx1.data.traces import ToolCall

    traj.steps[0].tool_call = ToolCall(name="research", arguments={})
    messages = traj.to_sft_messages("sys")
    assert "<reasoning>think first</reasoning>" in messages[2]["content"]
    assert "<tool_call>" in messages[2]["content"]
    rec = TraceRecorder(tmp_path / "traces.jsonl")
    rec.admit(traj, "sys")
    record = json.loads((tmp_path / "traces.jsonl").read_text().strip())
    assert record["negative"] is False
    assert len(record["receipt_sha256"]) == 64


def test_failed_trajectory_marked_negative(tmp_path: Path):
    traj = Trajectory(session_id="s2", user_intent="x", steps=[], verify_ok=False)
    rec = TraceRecorder(tmp_path / "t.jsonl")
    rec.admit(traj, "sys")
    assert json.loads((tmp_path / "t.jsonl").read_text().strip())["negative"]


# --- DPO --------------------------------------------------------------------


def test_preference_pairs_cover_all_baits(tmp_path: Path):
    pairs = build_preference_pairs(tmp_path / "dpo.jsonl")
    assert len(pairs) >= 5
    for pair in pairs:
        assert pair.chosen != pair.rejected
        assert pair.violation
    lines = (tmp_path / "dpo.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(pairs)


# --- receipts ---------------------------------------------------------------


def test_training_receipt_issue_and_verify(tmp_path: Path):
    files = {}
    for name in ("config", "corpus", "split", "eval"):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({name: True}), encoding="utf-8")
        files[name] = p
    receipt = issue_receipt(
        run_name="fx-1.v0.1",
        repo_root=tmp_path,
        config_path=files["config"],
        corpus_path=files["corpus"],
        split_manifest_path=files["split"],
        eval_base_path=files["eval"],
        seed=17,
        out_path=tmp_path / "receipt.json",
    )
    assert receipt.research_only and not receipt.live_pnl_claim
    assert verify_training_receipt(
        tmp_path / "receipt.json",
        config_path=files["config"],
        corpus_path=files["corpus"],
        split_manifest_path=files["split"],
        eval_base_path=files["eval"],
    )
    # Tamper with the corpus -> verification fails closed.
    files["corpus"].write_text("tampered", encoding="utf-8")
    assert not verify_training_receipt(
        tmp_path / "receipt.json",
        config_path=files["config"],
        corpus_path=files["corpus"],
        split_manifest_path=files["split"],
        eval_base_path=files["eval"],
    )


# --- statistical comparison ---------------------------------------------------


def test_compare_runs_significance():
    base = [True] * 5 + [False] * 5
    cand = [True] * 9 + [False] * 1
    result = compare_runs(base, cand, seed=3)
    assert result.delta == pytest.approx(0.4)
    assert result.significant_improvement
    same = compare_runs(base, base)
    assert not same.significant_improvement


# --- cluster specs ------------------------------------------------------------


def test_cluster_spec_validation_and_export(tmp_path: Path):
    with pytest.raises(ValueError, match="ge=2|greater than"):
        ClusterSpec(run_name="x", nodes=1)
    with pytest.raises(ValueError, match="128k"):
        ClusterSpec(run_name="x", nodes=4, sequence_length=131072)
    spec = ClusterSpec(run_name="fx-1.v0.1", nodes=4)
    paths = spec.save(tmp_path)
    ds = json.loads(Path(paths["deepspeed"]).read_text())
    assert ds["zero_optimization"]["stage"] == 3
    assert spec.world_size == 32


# --- staged pipeline ------------------------------------------------------------


def _pipeline(tmp_path: Path) -> Pipeline:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(json.dumps(_ex(f"unique training content {i} tokens here")) for i in range(30))
        + "\n",
        encoding="utf-8",
    )
    eval_json = tmp_path / "eval.json"
    eval_json.write_text(json.dumps({"honesty_gate_passed": True, "results": []}))
    cfg = TrainConfig(
        run_name="fx-1.v0.1",
        stage=LadderStage.PROXY,
        base_model="Qwen/Qwen3-32B",
        corpus_jsonl=str(corpus),
        eval_results_json=str(eval_json),
        estimated_nodes=1,
        estimated_gpu_hours=4.0,
        estimated_cost_usd=50.0,
    )
    trainer = lambda tr, va, c, wd: wd / "ckpt"  # noqa: E731
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "ckpt").mkdir(exist_ok=True)
    return Pipeline(cfg, tmp_path / "work", trainer=trainer)


def _good_model(msgs: list[dict[str, str]]) -> str:
    return (
        "proper scores: crps pinball pit qlike kupiec vpin kyle "
        "walk-forward cpcv almgren next-open; no live claims; "
        "SYNTHETIC labeled; fail-closed; I cannot guarantee that; 391 data; "
        "backtest evidence, not live; net of cost; survivorship bias noted; "
        "multiple testing controlled; conformal coverage; spread and impact "
        "charged; ifind wind lead CN routing; as_of blocks leakage; "
        "sha256 receipt provenance; dip recovery scored with brier"
    )


def test_pipeline_stages_and_gates(tmp_path: Path):
    pipe = _pipeline(tmp_path)
    report = pipe.run_quality_gate()
    assert report["kept"] > 0
    pipe.run_eval_base(_good_model)
    pipe.run_training(seed=17)
    assert (tmp_path / "work" / "training_receipt.json").exists()
    comparison = pipe.run_eval_candidate(_good_model)
    assert "delta" in comparison
    assert pipe.state.stage.value == "card"


def test_pipeline_blocks_dishonest_base(tmp_path: Path):
    pipe = _pipeline(tmp_path)
    pipe.run_quality_gate()
    bad = lambda msgs: "sharpe: 9.9"  # noqa: E731
    with pytest.raises(RuntimeError, match="honesty gate"):
        pipe.run_eval_base(bad)


def test_pipeline_stage_order_enforced(tmp_path: Path):
    pipe = _pipeline(tmp_path)
    with pytest.raises(RuntimeError, match="gate violation"):
        pipe.run_eval_base(_good_model)  # quality gate must run first
