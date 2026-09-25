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


def _built_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.jsonl"
    result = runner.invoke(
        app, ["corpus", "build", "--receipts-dir", str(tmp_path), "--out", str(corpus)]
    )
    assert result.exit_code == 0
    return corpus


def _ship_eligible_card(tmp_path: Path) -> Path:
    from fx1.modelcard import EvalDelta, ModelCard

    card = ModelCard(
        version="fx-1.v0.1",
        corpus_sha256="a" * 64,
        corpus_receipt_range="b5942241..f0e1d2c3",
        training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5,
            domain_pass_rate_candidate=0.7,
            general_pass_rate_base=0.9,
            general_pass_rate_candidate=0.9,
            honesty_gate_candidate=True,
        ),
    )
    path = tmp_path / "modelcard.json"
    card.save(path)
    return path


def test_corpus_build_cli_accepts_repeated_receipts_dir(tmp_path: Path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _receipt(dir_a / "r1.json")
    _receipt(dir_b / "r2.json")
    out = tmp_path / "corpus.jsonl"
    result = runner.invoke(
        app,
        [
            "corpus",
            "build",
            "--receipts-dir",
            str(dir_a),
            "--receipts-dir",
            str(dir_b),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["loaded"] == 2


def test_corpus_build_full_cli(tmp_path: Path):
    _receipt(tmp_path / "r.json")
    notebooks = tmp_path / "note.md"
    notebooks.write_text("# research note\n\nsome findings\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    out = tmp_path / "full.jsonl"
    result = runner.invoke(
        app,
        [
            "corpus",
            "build-full",
            "--receipts-dir",
            str(tmp_path),
            "--notebooks",
            str(notebooks),
            "--artifacts-dir",
            str(artifacts),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout)["positive"] >= 1


def test_dipbench_cli_is_labeled_synthetic():
    result = runner.invoke(app, ["dipbench"])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["label"] == "SYNTHETIC"
    assert "brier_overall" in report["metrics"]


def test_dpo_cli_writes_pairs(tmp_path: Path):
    out = tmp_path / "dpo.jsonl"
    result = runner.invoke(app, ["dpo", "--out", str(out)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["pairs"] > 0
    assert out.exists()


def test_curriculum_cli(tmp_path: Path):
    _receipt(tmp_path / "r.json")
    corpus = _built_corpus(tmp_path)
    out = tmp_path / "curriculum.jsonl"
    result = runner.invoke(app, ["curriculum", "--corpus", str(corpus), "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()


def test_masked_eval_cli_structural():
    result = runner.invoke(app, ["maskedaEval"])
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["twin_tasks"] > 0


def test_contamination_audit_cli(tmp_path: Path):
    _receipt(tmp_path / "r.json")
    corpus = _built_corpus(tmp_path)
    out = tmp_path / "contamination_report.json"
    result = runner.invoke(app, ["contamination-audit", "--corpus", str(corpus), "--out", str(out)])
    assert result.exit_code == 0  # small fresh corpus must not be flagged
    assert json.loads(result.stdout)["overall_flagged"] is False
    assert out.exists()


def test_modelcard_cli_reports_ship_gate(tmp_path: Path):
    card = _ship_eligible_card(tmp_path)
    result = runner.invoke(app, ["modelcard", str(card)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["ship_eligible"] is True


def test_sbom_cli_from_repo_lockfile(tmp_path: Path):
    lockfile = Path(__file__).resolve().parents[2] / "uv.lock"
    out = tmp_path / "sbom.json"
    result = runner.invoke(app, ["sbom", "--lockfile", str(lockfile), "--out", str(out)])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["entries"] > 100
    assert out.exists()


def test_sign_and_attestation_cli(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FX1_SIGNING_KEY", "test-key")
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    _ship_eligible_card(checkpoint)
    (checkpoint / "weights.bin").write_bytes(b"stub-weights")
    signed = runner.invoke(app, ["sign", str(checkpoint)])
    assert signed.exit_code == 0
    assert (checkpoint / "release.sig").exists()
    status = runner.invoke(app, ["attestation", str(checkpoint)])
    assert status.exit_code == 0


def test_sign_cli_fails_closed_without_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FX1_SIGNING_KEY", raising=False)
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"stub")
    result = runner.invoke(app, ["sign", str(checkpoint)])
    assert result.exit_code != 0


def test_mrm_cli_compiles_dossier(tmp_path: Path):
    card = _ship_eligible_card(tmp_path)
    contamination = tmp_path / "contamination_report.json"
    contamination.write_text(json.dumps({"overall_flagged": False}), encoding="utf-8")
    out = tmp_path / "dossier.json"
    result = runner.invoke(
        app,
        [
            "mrm",
            "--modelcard",
            str(card),
            "--validation-artifact",
            str(contamination),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["complete"] and report["contamination_flagged"] is False


def test_eval_cli_fails_closed_without_hosted_key(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    result = runner.invoke(
        app, ["eval", "--backend", "hosted_k3", "--out", str(tmp_path / "eval.json")]
    )
    assert result.exit_code != 0  # no key -> honest failure, never fabricated


def test_sources_scenarios_offline_is_honest(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FX1_PLUGIN_ROOTS", str(tmp_path))  # no bundled scripts
    result = runner.invoke(app, ["sources", "scenarios"])
    assert result.exit_code == 1
    assert "unavailable" in result.output
