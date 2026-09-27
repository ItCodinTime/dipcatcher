"""Training-pipeline tests: eval-before-train, provenance, K3 constraints."""

import json
from pathlib import Path

import pytest

from fx1.train import LadderStage, TrainConfig, build_training_manifest


def _corpus(path: Path, with_provenance: bool = True) -> Path:
    record = {
        "messages": [{"role": "user", "content": "q"}],
        "source_path": "r.json",
        "negative": False,
    }
    if with_provenance:
        record["receipt_sha256"] = "a" * 64
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def _eval_summary(path: Path, honesty_ok: bool = True) -> Path:
    path.write_text(
        json.dumps({"honesty_gate_passed": honesty_ok, "results": []}),
        encoding="utf-8",
    )
    return path


def _config(tmp_path: Path, **kw) -> TrainConfig:
    corpus = _corpus(tmp_path / "corpus.jsonl")
    ev = _eval_summary(tmp_path / "eval.json")
    defaults = {
        "run_name": "fx-1.v0.1",
        "stage": LadderStage.PROXY,
        "base_model": "Qwen/Qwen3-32B",
        "corpus_jsonl": str(corpus),
        "eval_results_json": str(ev),
        "estimated_nodes": 1,
        "estimated_gpu_hours": 24.0,
        "estimated_cost_usd": 500.0,
    }
    defaults.update(kw)
    return TrainConfig(**defaults)


def test_final_k3_requires_multi_node_and_k3_base(tmp_path: Path):
    with pytest.raises(ValueError, match="multi-node"):
        _config(
            tmp_path,
            stage=LadderStage.FINAL_K3,
            base_model="moonshotai/Kimi-K3",
            estimated_nodes=1,
        )
    with pytest.raises(ValueError, match="Kimi-K3"):
        _config(tmp_path, stage=LadderStage.FINAL_K3, estimated_nodes=8)


def test_eval_before_train_gate(tmp_path: Path):
    cfg = _config(tmp_path)
    Path(cfg.eval_results_json).unlink()
    with pytest.raises(FileNotFoundError, match="eval"):
        build_training_manifest(cfg, tmp_path / "m.json")
    # Honesty gate failed in eval summary blocks training.
    _eval_summary(Path(cfg.eval_results_json), honesty_ok=False)
    with pytest.raises(ValueError, match="honesty"):
        build_training_manifest(cfg, tmp_path / "m.json")


def test_corpus_provenance_fail_closed(tmp_path: Path):
    cfg = _config(tmp_path)
    _corpus(Path(cfg.corpus_jsonl), with_provenance=False)
    with pytest.raises(ValueError, match="receipt_sha256"):
        build_training_manifest(cfg, tmp_path / "m.json")


def test_manifest_written_and_immutable_fields(tmp_path: Path):
    cfg = _config(tmp_path)
    out = tmp_path / "manifest.json"
    manifest = build_training_manifest(cfg, out)
    assert out.exists()
    assert manifest["live_pnl_claim"] is False
    assert manifest["research_only"] is True
    assert len(manifest["corpus_sha256"]) == 64
    assert manifest["corpus_stats"]["lines"] == 1
