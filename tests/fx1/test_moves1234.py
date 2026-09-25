"""Tests for the four uniqueness moves: leakage-proof eval, attested
inference, corpus ledger, and the MRM dossier."""

import json
from pathlib import Path

import pytest

from fx1.data.ledger import GENESIS, CorpusLedger
from fx1.eval import (
    DEFAULT_BANK,
    masked_twins,
    memory_gap_report,
    partition_tasks,
    post_cutoff_pass_rate,
    run_contamination_audit,
)
from fx1.eval.masking import mask_text
from fx1.modelcard import EvalDelta, ModelCard
from fx1.mrm import FIVE_ACTIVITIES, compile_dossier
from fx1.serve import (
    LocalFx1Backend,
    OperatorProofManifest,
    TEEQuote,
    attestation_ladder_status,
    sign_release,
    verify_quote,
    verify_release,
)

# --- Move 1: masking ----------------------------------------------------------

def test_masking_deterministic_and_consistent():
    text = "AAPL dipped on 2024-03-15; compare with MSFT in Q2 2024."
    m1, m2 = mask_text(text), mask_text(text)
    assert m1 == m2
    assert "AAPL" not in m1 and "MSFT" not in m1
    assert "2024" not in m1
    # same surface form -> same placeholder
    assert m1.count(m1.split("dipped")[0].strip()) >= 1


def test_masking_preserves_proper_score_vocab():
    text = "CRPS and QLIKE with VPIN features, Kupiec validation."
    assert mask_text(text) == text


def test_masked_twins_pairing():
    twins = masked_twins(list(DEFAULT_BANK)[:3])
    assert len(twins) == 6
    assert twins[1].name.endswith("__masked")


def test_memory_gap_metric():
    report = memory_gap_report(
        [True] * 8 + [False] * 2, [True] * 6 + [False] * 4, budget=0.25
    )
    assert report.memory_gap == pytest.approx(0.2)
    assert report.within_budget
    big = memory_gap_report([True] * 10, [False] * 10, budget=0.25)
    assert not big.within_budget


def test_time_partition_and_post_cutoff_rate():
    tasks = list(DEFAULT_BANK)[:3]
    names = [t.name for t in tasks]
    created = {names[0]: "2025-01-01", names[1]: "2026-08-01"}
    part = partition_tasks(tasks, created, "2026-01-01")
    assert part.post_cutoff == [names[1]]
    assert part.pre_cutoff == [names[0]]
    assert part.undated == [names[2]]
    rate = post_cutoff_pass_rate(part, {names[1]: True})
    assert rate == 1.0


def test_contamination_audit_reports_and_flags():
    corpus = [
        "unrelated research text with plenty of distinct tokens here",
        "Which scores does the lab use for probabilistic distributions and calibration?",
    ]
    prompts = [
        "Which scores does the lab use for probabilistic distributions and calibration?"
    ]
    report = run_contamination_audit(
        corpus, prompts, canonical_pass=[True, True, True],
        rephrased_pass=[True, False, False],
    )
    assert report.ngram_hits and report.ngram_hits[0].example_index == 1
    methods = [p.method for p in report.probes]
    assert methods == ["ngram_containment", "min_k_percent", "rephrased_gap"]
    assert report.overall_flagged
    assert len(report.corpus_sha256) == 64


# --- Move 2: signing + attestation ----------------------------------------------

def _checkpoint(tmp_path: Path) -> Path:
    root = tmp_path / "ckpt"
    root.mkdir()
    (root / "modelcard.json").write_text(
        ModelCard(
            version="fx-1.v0.1", corpus_sha256="a" * 64,
            corpus_receipt_range="b..c", training_manifest_sha256="b" * 64,
            eval_delta=EvalDelta(
                domain_pass_rate_base=0.5, domain_pass_rate_candidate=0.7,
                general_pass_rate_base=0.9, general_pass_rate_candidate=0.9,
                honesty_gate_candidate=True,
            ),
        ).model_dump_json(),
        encoding="utf-8",
    )
    (root / "adapter.bin").write_bytes(b"weights")
    return root


def test_sign_and_verify_release(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FX1_SIGNING_KEY", "test-key")
    root = _checkpoint(tmp_path)
    sign_release(root)
    assert verify_release(root)
    (root / "adapter.bin").write_bytes(b"tampered")
    assert not verify_release(root)


def test_signing_requires_env_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FX1_SIGNING_KEY", raising=False)
    root = _checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="FX1_SIGNING_KEY"):
        sign_release(root)


def test_backend_refuses_unsigned_when_keyed(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FX1_SIGNING_KEY", "k")
    root = _checkpoint(tmp_path)
    with pytest.raises(RuntimeError, match="signature"):
        LocalFx1Backend(root)
    sign_release(root)
    assert LocalFx1Backend(root).card.version == "fx-1.v0.1"


def test_tee_quote_verification():
    quote = TEEQuote(
        platform="sev-snp", checkpoint_sha256="a" * 64,
        measurement="ab", report_data=f"nonce123:{'a' * 64}",
        signature="sig",
    )
    assert verify_quote(quote, expected_checkpoint_sha256="a" * 64,
                        nonce="nonce123")
    assert not verify_quote(quote, expected_checkpoint_sha256="b" * 64,
                            nonce="nonce123")
    assert not verify_quote(quote, expected_checkpoint_sha256="a" * 64,
                            nonce="other")


def test_zkml_manifest_requires_proofs(tmp_path: Path):
    with pytest.raises(ValueError, match="without proof"):
        OperatorProofManifest(
            checkpoint_sha256="a" * 64,
            covered_operators=["calibration_head"],
            proof_artifacts={},
        )
    proof = tmp_path / "proof.bin"
    proof.write_bytes(b"proof")
    manifest = OperatorProofManifest(
        checkpoint_sha256="a" * 64, covered_operators=["calibration_head"],
        proof_artifacts={"calibration_head": str(proof)},
    )
    assert manifest.verify_artifacts_exist()


def test_ladder_status(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FX1_SIGNING_KEY", "k")
    root = _checkpoint(tmp_path)
    status = attestation_ladder_status(root)
    assert status == {"signed_release": False, "tee": False,
                      "selective_zkml": False}
    sign_release(root)
    assert attestation_ladder_status(root)["signed_release"]


# --- Move 3: corpus ledger ---------------------------------------------------------

def test_ledger_chain_and_tamper_evidence(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    ledger = CorpusLedger(path)
    ledger.record_example(source_sha256="a" * 64, transform_sha256="b" * 64,
                          example_sha256="c" * 64)
    ledger.record_exclusion(source_sha256="d" * 64,
                            transform_sha256="b" * 64, rule="dedup_exact")
    ledger.record_example(source_sha256="e" * 64, transform_sha256="b" * 64,
                          example_sha256="f" * 64)
    assert ledger.verify_chain()
    export = ledger.audit_export()
    assert export["examples"] == 2 and export["exclusions"] == 1
    assert export["exclusion_rules"] == {"dedup_exact": 1}
    assert export["chain_valid"]
    # Tamper: rewrite an entry hash.
    lines = path.read_text().splitlines()
    entry = json.loads(lines[1])
    entry["rule"] = "dedup_near"
    lines[1] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n")
    assert not CorpusLedger(path).verify_chain()


def test_ledger_genesis_and_reload(tmp_path: Path):
    path = tmp_path / "l.jsonl"
    ledger = CorpusLedger(path)
    entry = ledger.record_example(source_sha256="a" * 64,
                                  transform_sha256="b" * 64,
                                  example_sha256="c" * 64)
    assert entry.prev_hash == GENESIS
    reloaded = CorpusLedger(path)
    assert reloaded.verify_chain()
    assert len(reloaded._entries) == 1


# --- Move 4: MRM dossier --------------------------------------------------------------

def test_mrm_dossier_compiles_five_activities(tmp_path: Path):
    card_path = tmp_path / "modelcard.json"
    ModelCard(
        version="fx-1.v0.1", corpus_sha256="a" * 64,
        corpus_receipt_range="b..c", training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5, domain_pass_rate_candidate=0.7,
            general_pass_rate_base=0.9, general_pass_rate_candidate=0.9,
            honesty_gate_candidate=True,
        ),
    ).save(card_path)
    art = tmp_path / "contamination_report.json"
    art.write_text(json.dumps({"overall_flagged": False}), encoding="utf-8")
    dossier = compile_dossier(
        modelcard_path=card_path,
        artifacts={"validation": art},
        out_path=tmp_path / "dossier.json",
    )
    assert dossier.complete
    assert {s.activity for s in dossier.sections} == set(FIVE_ACTIVITIES)
    assert not dossier.contamination_flagged
    assert dossier.ship_eligible
    assert (tmp_path / "dossier.json").exists()


def test_mrm_dossier_fail_closed_on_missing(tmp_path: Path):
    card_path = tmp_path / "modelcard.json"
    ModelCard(
        version="fx-1.v0.1", corpus_sha256="a" * 64,
        corpus_receipt_range="b..c", training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.5, domain_pass_rate_candidate=0.7,
            general_pass_rate_base=0.9, general_pass_rate_candidate=0.9,
            honesty_gate_candidate=True,
        ),
    ).save(card_path)
    with pytest.raises(FileNotFoundError):
        compile_dossier(
            modelcard_path=card_path,
            artifacts={"validation": tmp_path / "missing.json"},
            out_path=tmp_path / "d.json",
        )
