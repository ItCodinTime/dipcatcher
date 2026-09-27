"""Receipt-extension backward compatibility against sealed-receipt verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from quant_fund.research.catalog import (
    BENCHMARK_CATALOG_VERSION,
    REQUIRED_BENCHMARK_FAMILIES,
    RESEARCH_RECEIPT_SCHEMA_VERSION,
)
from quant_fund.research.receipt_schema import unavailable_overfitting_block
from quant_fund.research.verify import _receipt_digest, verify_research_artifact
from quant_fund.utils.hashing import hash_bytes, hash_file

import quant_fund.research.explainability.receipt as receipt_module
from quant_fund.research.explainability import (
    attach_explainability_report,
    attach_explainability_sidecar,
    build_report,
    explainability_artifact_entry,
    explainability_sidecar_path,
    verify_explainability_sidecar,
    write_report,
)

from .conftest import FEATURES


def _sealed_receipt(tmp_path: Path) -> Path:
    """Minimal receipt that passes ``verify_research_artifact`` (same shape as the
    fixture in tests/unit/research/test_research_verify.py)."""
    run_id = "a" * 64
    immutable = tmp_path / "runs"
    immutable.mkdir()
    (immutable / f"{run_id}.md").write_text("# research")
    payload = {
        "schema_version": RESEARCH_RECEIPT_SCHEMA_VERSION,
        "firm": "Artificial Hedge",
        "product": "Dipcatcher",
        "version": "1.0.0",
        "generated_at": "2026-09-16T00:00:00+00:00",
        "data_source": "SYNTHETIC",
        "synthetic": True,
        "disclaimer": "research only",
        "ranking_target": "future_return_1",
        "claim": "research_only",
        "rankers": [],
        "hypotheses": [],
        "provenance": {
            "run_id": run_id,
            "git_revision": "HEAD",
            "git_worktree_sha256": "e" * 64,
            "config_sha256": "b" * 64,
            "dataset_sha256": "c" * 64,
            "dataset_content_sha256": "d" * 64,
            "northset_inputs_sha256": "e" * 64,
            "row_count": 10,
            "column_count": 3,
            "point_in_time": True,
            "execution_claim": "research_only",
            "benchmark_catalog_version": BENCHMARK_CATALOG_VERSION,
            "runtime": {
                "python": "3.12.0",
                "implementation": "CPython",
                "platform": "test",
                "machine": "test",
                "byteorder": "little",
                "packages": {
                    "numpy": "2.0.0",
                    "polars": "1.0.0",
                    "scipy": "1.0.0",
                    "scikit-learn": "1.0.0",
                },
            },
        },
        "scorecard": {
            name: {
                "executed": True,
                "nonempty": True,
                "finite_observation": True,
                "forbidden_metrics_absent": True,
                "claim": "research_metric_only",
            }
            for name in REQUIRED_BENCHMARK_FAMILIES
        },
        "families": {name: {"executed": True} for name in REQUIRED_BENCHMARK_FAMILIES},
        "backtest_overfitting": unavailable_overfitting_block(),
        "artifacts": {
            "immutable_json": str(immutable / f"{run_id}.json"),
            "immutable_markdown": str(immutable / f"{run_id}.md"),
            "immutable_markdown_sha256": hash_file(immutable / f"{run_id}.md"),
        },
    }
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(payload))
    (immutable / f"{run_id}.json").write_text(json.dumps(payload))
    return path


def _small_report(planted_model):
    model, x, y = planted_model
    return build_report(
        model,
        x[:160],
        y[:160],
        feature_names=FEATURES,
        n_blocks=2,
        n_repeats=2,
        seed=9,
        synthetic=True,
        generated_at="2026-09-16T00:00:00+00:00",
    )


def test_sidecar_preserves_sealed_receipt_verification(planted_model, tmp_path: Path) -> None:
    receipt = _sealed_receipt(tmp_path)
    receipt_bytes = receipt.read_bytes()
    baseline = verify_research_artifact(receipt)
    assert baseline["valid"] is True, baseline["errors"]

    report = _small_report(planted_model)
    paths = attach_explainability_report(receipt, report)
    assert paths["sidecar"].is_file()
    assert paths["sidecar"].name == "latest.explainability.json"

    # The sealed receipt bytes are untouched and the stock verifier is identical.
    assert receipt.read_bytes() == receipt_bytes
    after = verify_research_artifact(receipt)
    assert after == baseline
    # The immutable copy is untouched too.
    immutable = Path(json.loads(receipt.read_text())["artifacts"]["immutable_json"])
    assert json.loads(immutable.read_text()) == json.loads(receipt.read_text())

    binding = verify_explainability_sidecar(receipt)
    assert binding["valid"] is True
    assert binding["reports_checked"] == 3


def test_sidecar_detects_report_tampering(planted_model, tmp_path: Path) -> None:
    receipt = _sealed_receipt(tmp_path)
    report = _small_report(planted_model)
    paths = attach_explainability_report(receipt, report)
    paths["markdown"].write_text("# forged\n")
    binding = verify_explainability_sidecar(receipt)
    assert binding["valid"] is False
    assert any("sidecar_report_hash_mismatch" in e for e in binding["errors"])


def test_sidecar_detects_receipt_tampering(planted_model, tmp_path: Path) -> None:
    receipt = _sealed_receipt(tmp_path)
    report = _small_report(planted_model)
    attach_explainability_report(receipt, report)
    receipt.write_text(receipt.read_text() + "\n")
    binding = verify_explainability_sidecar(receipt)
    assert binding["valid"] is False
    assert "sidecar_receipt_hash_mismatch" in binding["errors"]


def test_sidecar_detects_its_own_payload_tampering(planted_model, tmp_path: Path) -> None:
    receipt = _sealed_receipt(tmp_path)
    paths = attach_explainability_report(receipt, _small_report(planted_model))
    payload = json.loads(paths["sidecar"].read_text())
    payload["generated_at"] = "rewritten"
    paths["sidecar"].write_text(json.dumps(payload))
    binding = verify_explainability_sidecar(receipt)
    assert binding["valid"] is False
    assert "sidecar_payload_hash_mismatch" in binding["errors"]


def test_attach_report_rejects_external_output_before_writing(
    planted_model, tmp_path: Path
) -> None:
    receipt_root = tmp_path / "receipt"
    receipt_root.mkdir()
    receipt = _sealed_receipt(receipt_root)
    outside = tmp_path / "outside"
    with pytest.raises(ValueError, match="report_outside_receipt_root"):
        attach_explainability_report(receipt, _small_report(planted_model), report_dir=outside)
    assert not outside.exists()


def test_sidecar_binds_run_id(planted_model, tmp_path: Path) -> None:
    receipt = _sealed_receipt(tmp_path)
    report = _small_report(planted_model)
    paths = attach_explainability_report(receipt, report)
    payload = json.loads(paths["sidecar"].read_text())
    assert payload["receipt"]["run_id"] == "a" * 64
    assert payload["receipt"]["sha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert payload["receipt"]["synthetic"] is True
    assert payload["schema"] == "explainability_attachment.v1"
    assert payload["claim"] == "research_only"
    kinds = {entry["kind"] for entry in payload["reports"]}
    assert kinds == {"markdown", "html", "json"}


def test_attach_sidecar_rejects_missing_inputs(planted_model, tmp_path: Path) -> None:
    receipt = _sealed_receipt(tmp_path)
    with pytest.raises(FileNotFoundError, match="receipt not found"):
        attach_explainability_sidecar(tmp_path / "nope.json", tmp_path)
    with pytest.raises(ValueError, match="no explainability report files"):
        (tmp_path / "empty").mkdir()
        attach_explainability_sidecar(receipt, tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="report_missing"):
        attach_explainability_sidecar(receipt, [tmp_path / "ghost.md"])


def test_verify_sidecar_missing_and_malformed(tmp_path: Path) -> None:
    receipt = _sealed_receipt(tmp_path)
    missing = verify_explainability_sidecar(receipt)
    assert missing["valid"] is False
    assert "sidecar_missing" in missing["errors"]
    explainability_sidecar_path(receipt).write_text("{not json")
    malformed = verify_explainability_sidecar(receipt)
    assert malformed["valid"] is False
    assert any("sidecar_unreadable" in e for e in malformed["errors"])
    explainability_sidecar_path(receipt).write_text('{"schema": "other"}')
    wrong = verify_explainability_sidecar(receipt)
    assert "sidecar_schema_mismatch" in wrong["errors"]


def test_explicit_sidecar_paths_outside_receipt_root(tmp_path: Path) -> None:
    """Reports outside the receipt root cannot be attached or read."""
    (tmp_path / "research_root").mkdir()
    receipt = _sealed_receipt(tmp_path / "research_root")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    report_file = outside / "explainability.md"
    report_file.write_text("# report\n")
    with pytest.raises(ValueError, match="report_outside_receipt_root"):
        attach_explainability_sidecar(receipt, [report_file])
    link = receipt.parent / "linked-report.md"
    link.symlink_to(report_file)
    with pytest.raises(ValueError, match="report_outside_receipt_root"):
        attach_explainability_sidecar(receipt, [link])
    with pytest.raises(ValueError, match="report outside receipt root"):
        explainability_artifact_entry(outside, receipt.parent)


def test_verify_rejects_outside_report_before_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_root = tmp_path / "receipt_root"
    receipt_root.mkdir()
    receipt = _sealed_receipt(receipt_root)
    inside = receipt_root / "explainability.md"
    inside.write_text("# inside\n")
    sidecar = attach_explainability_sidecar(receipt, [inside])
    outside = tmp_path / "outside.md"
    outside.write_text("# outside\n")
    payload = json.loads(sidecar.read_text())
    payload["reports"][0]["path"] = "../outside.md"
    payload["reports"][0]["sha256"] = hash_file(outside)
    unsigned = {key: value for key, value in payload.items() if key != "sidecar_sha256_payload"}
    payload["sidecar_sha256_payload"] = hash_bytes(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    )
    sidecar.write_text(json.dumps(payload))

    original_hash_file = receipt_module.hash_file

    def guarded_hash_file(path: Path) -> str:
        if Path(path).resolve() == outside.resolve():
            raise AssertionError("outside report was read")
        return original_hash_file(path)

    monkeypatch.setattr(receipt_module, "hash_file", guarded_hash_file)
    result = verify_explainability_sidecar(receipt)
    assert result["valid"] is False
    assert any("sidecar_report_outside_receipt_root" in error for error in result["errors"])


def test_optional_artifact_entry_for_new_receipts(planted_model, tmp_path: Path) -> None:
    """The embeddable entry is additive: the verifier ignores unknown artifact keys,
    but a sealed receipt's immutable copy must match, so this is a write-time
    option for receipts minted in the future — never a retrofit."""
    receipt_dir = tmp_path / "research_root"
    receipt_dir.mkdir()
    _sealed_receipt(receipt_dir)
    report = _small_report(planted_model)
    report_paths = write_report(report, receipt_dir / "explainability")
    entry = explainability_artifact_entry(receipt_dir / "explainability", receipt_dir)
    assert entry["schema"] == "explainability_artifacts.v1"
    for kind in ("markdown", "html", "json"):
        assert entry[kind].startswith("explainability/")
        assert len(entry[f"{kind}_sha256"]) == 64
        assert Path(receipt_dir / entry[kind]).is_file()
    assert len(report_paths) == 3


def test_embedded_entry_on_unsealed_receipt_still_verifies(tmp_path: Path) -> None:
    """A receipt minted WITH artifacts.explainability (both copies) verifies."""
    receipt = _sealed_receipt(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["artifacts"]["explainability"] = {
        "schema": "explainability_artifacts.v1",
        "markdown": "explainability/explainability.md",
    }
    # Re-seal: recompute digest over the extended payload and sync both copies.
    payload["artifacts"]["immutable_json_sha256"] = _receipt_digest(payload)
    receipt.write_text(json.dumps(payload))
    immutable = Path(payload["artifacts"]["immutable_json"])
    immutable.write_text(json.dumps(payload))
    result = verify_research_artifact(receipt)
    assert result["valid"] is True, result["errors"]


def test_reports_dir_collected_from_directory(planted_model, tmp_path: Path) -> None:
    receipt = _sealed_receipt(tmp_path)
    report = _small_report(planted_model)
    out_dir = receipt.parent / "explainability"
    write_report(report, out_dir)
    sidecar = attach_explainability_sidecar(receipt, out_dir)
    payload = json.loads(sidecar.read_text())
    assert {entry["kind"] for entry in payload["reports"]} == {"markdown", "html", "json"}


def test_numpy_scalar_y_accepted(planted_model) -> None:
    model, x, y = planted_model
    report = build_report(model, x[:64], np.asarray(y[:64]), n_blocks=2, n_repeats=2, seed=1)
    assert report.n_rows == 64
