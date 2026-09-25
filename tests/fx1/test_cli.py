"""CLI tests for the fx1 entry point."""

import json
from pathlib import Path

from typer.testing import CliRunner

from fx1.cli import app

runner = CliRunner()


def _receipt(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "test/v1",
                "research_only": True,
                "live_pnl_claim": False,
                "correctness": {"metric": 1.0},
            }
        ),
        encoding="utf-8",
    )


def test_harness_list_shows_registered_commands():
    result = runner.invoke(app, ["harness", "list"])
    assert result.exit_code == 0
    for name in ("doctor", "verify-research", "research", "northset", "collect"):
        assert name in result.stdout


def test_harness_run_rejects_unregistered():
    result = runner.invoke(app, ["harness", "run", "trade-live"])
    assert result.exit_code != 0


def test_corpus_build_cli(tmp_path: Path):
    _receipt(tmp_path / "r.json")
    out = tmp_path / "corpus.jsonl"
    result = runner.invoke(
        app, ["corpus", "build", "--receipts-dir", str(tmp_path), "--out", str(out)]
    )
    assert result.exit_code == 0
    stats = json.loads(result.stdout)
    assert stats["positive"] == 1
    assert out.exists()


def test_train_manifest_cli_blocks_without_eval(tmp_path: Path):
    _receipt(tmp_path / "r.json")
    corpus = tmp_path / "corpus.jsonl"
    runner.invoke(app, ["corpus", "build", "--receipts-dir", str(tmp_path), "--out", str(corpus)])
    cfg = {
        "run_name": "fx-1.v0.1",
        "stage": "proxy",
        "base_model": "Qwen/Qwen3-32B",
        "corpus_jsonl": str(corpus),
        "eval_results_json": str(tmp_path / "missing_eval.json"),
        "estimated_nodes": 1,
        "estimated_gpu_hours": 12.0,
        "estimated_cost_usd": 200.0,
    }
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = runner.invoke(app, ["train", "manifest", "--config", str(cfg_path)])
    assert result.exit_code != 0  # eval-before-train gate blocks
