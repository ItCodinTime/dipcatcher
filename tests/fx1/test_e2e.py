"""End-to-end integration: the full fx-1 lifecycle in one test.

receipts -> corpus -> quality gate -> frozen split -> ledger -> eval base ->
(signed) training receipt -> candidate eval -> statistical comparison ->
model card -> signed release -> MRM dossier. An external auditor's "does it
actually all connect?" test.
"""

import json
from pathlib import Path

from fx1.data import build_corpus, dedup_and_filter, frozen_split
from fx1.data.ledger import CorpusLedger
from fx1.eval import DEFAULT_BANK, run_contamination_audit, run_suite
from fx1.eval.compare import compare_runs
from fx1.eval.masking import masked_twins, memory_gap_report
from fx1.modelcard import EvalDelta, ModelCard
from fx1.mrm import compile_dossier
from fx1.serve import LocalFx1Backend, sign_release
from fx1.train.receipts import issue_receipt, verify_training_receipt


def _model(messages: list[dict[str, str]]) -> str:
    return (
        "proper scores: crps pinball pit qlike brier log-loss ece kupiec vpin "
        "kyle walk-forward cpcv almgren next-open; research evidence class; "
        "no live claims; SYNTHETIC labeled; fail-closed; I cannot guarantee; "
        "391 data; receipt ab12cd34; verify with dipcatcher verify-research; "
        "calibrated estimate with uncertainty intervals"
    )


def test_full_lifecycle(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FX1_SIGNING_KEY", "e2e-key")
    monkeypatch.chdir(tmp_path)

    # 1. receipts -> corpus
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "r1.json").write_text(json.dumps({
        "schema": "bench/v1", "research_only": True, "live_pnl_claim": False,
        "correctness": {"crps": 0.31}, "disclaimer": "research only"}))
    (receipts / "r2.json").write_text(json.dumps({
        "schema": "bench/v1", "research_only": False, "live_pnl_claim": True}))
    corpus_path = tmp_path / "corpus.jsonl"
    stats = build_corpus(receipts, corpus_path)
    assert stats["positive"] == 1 and stats["negative"] == 1

    # 2. quality gate + frozen split, recorded in the ledger
    examples = [json.loads(x) for x in corpus_path.read_text().splitlines()]
    prompts = [m["content"] for t in DEFAULT_BANK for m in t.messages
               if m["role"] == "user"]
    kept, report = dedup_and_filter(examples, eval_prompts=prompts)
    assert report.kept == 2
    ledger = CorpusLedger(tmp_path / "ledger.jsonl")
    for ex in kept:
        ledger.record_example(source_sha256=ex["receipt_sha256"],
                              transform_sha256="t" * 64,
                              example_sha256="e" * 64)
    assert ledger.verify_chain()
    split = frozen_split(kept, tmp_path / "corpus")
    assert split.train_count + split.val_count == 2

    # 3. eval base + masked twins + contamination audit
    base_summary = run_suite(_model, list(DEFAULT_BANK))
    assert base_summary["honesty_gate_passed"]
    base_path = tmp_path / "eval_base.json"
    base_path.write_text(json.dumps(base_summary))
    twins = masked_twins(list(DEFAULT_BANK))
    assert len(twins) == 2 * len(DEFAULT_BANK)
    gap = memory_gap_report([True] * 10, [True] * 9 + [False], budget=0.25)
    assert gap.within_budget
    audit = run_contamination_audit(
        [" ".join(m["content"] for m in e["messages"]) for e in kept], prompts)
    audit_path = tmp_path / "contamination_report.json"
    audit_path.write_text(audit.model_dump_json())

    # 4. training receipt (immutable, verifiable)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"run": "fx-1.v0.1"}))
    receipt_path = tmp_path / "training_receipt.json"
    issue_receipt(
        run_name="fx-1.v0.1", repo_root=tmp_path, config_path=cfg_path,
        corpus_path=corpus_path,
        split_manifest_path=tmp_path / "corpus.split.json",
        eval_base_path=base_path, seed=17, out_path=receipt_path,
    )
    assert verify_training_receipt(
        receipt_path, config_path=cfg_path, corpus_path=corpus_path,
        split_manifest_path=tmp_path / "corpus.split.json",
        eval_base_path=base_path,
    )

    # 5. candidate eval + statistical comparison
    cand_summary = run_suite(_model, list(DEFAULT_BANK))
    cmp_result = compare_runs(
        [r["passed"] for r in base_summary.results if r["kind"] == "domain"],
        [r["passed"] for r in cand_summary.results if r["kind"] == "domain"],
    )
    assert cmp_result.n_tasks > 0

    # 6. model card + signed release + serving enforcement
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    card = ModelCard(
        version="fx-1.v0.1", corpus_sha256="a" * 64,
        corpus_receipt_range="b..c", training_manifest_sha256="b" * 64,
        eval_delta=EvalDelta(
            domain_pass_rate_base=0.8, domain_pass_rate_candidate=0.9,
            general_pass_rate_base=0.9, general_pass_rate_candidate=0.9,
            honesty_gate_candidate=True,
        ),
    )
    card.save(ckpt / "modelcard.json")
    (ckpt / "adapter.bin").write_bytes(b"weights")
    sign_release(ckpt)
    backend = LocalFx1Backend(ckpt)  # must serve: signed + ship-eligible
    assert backend.card.version == "fx-1.v0.1"

    # 7. MRM dossier
    dossier = compile_dossier(
        modelcard_path=ckpt / "modelcard.json",
        artifacts={"validation": audit_path},
        out_path=tmp_path / "dossier.json",
    )
    assert dossier.complete and dossier.ship_eligible
