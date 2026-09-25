"""Tests for notebook and ledger corpus sources."""

import json
from pathlib import Path

from fx1.data import build_full_corpus, ledger_examples, notebook_examples

SYSTEM = "You are fx-1."


def test_notebook_examples_chunked_and_hashed(tmp_path: Path):
    doc = tmp_path / "nb.md"
    doc.write_text("# A\nfirst\n# B\nsecond\n# C\nthird", encoding="utf-8")
    examples = notebook_examples(doc, SYSTEM)
    assert len(examples) == 3
    for ex in examples:
        assert len(ex.receipt_sha256) == 64
        assert not ex.negative
        assert "verify-research" in ex.messages[-1]["content"]


def test_ledger_live_claim_becomes_negative(tmp_path: Path):
    live = tmp_path / "live.json"
    live.write_text(json.dumps({"live_pnl_claim": True}), encoding="utf-8")
    examples = ledger_examples(live, SYSTEM)
    assert len(examples) == 1
    assert examples[0].negative
    assert examples[0].messages[-1]["content"].startswith("No.")


def test_ledger_nested_live_claim_detected(tmp_path: Path):
    nested = tmp_path / "nested.json"
    nested.write_text(
        json.dumps({"outer": {"inner": {"live_pnl_claim": True}}}), encoding="utf-8"
    )
    assert ledger_examples(nested, SYSTEM)[0].negative


def test_clean_ledger_positive(tmp_path: Path):
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps({"strategy": "x", "live_pnl_claim": False}),
                     encoding="utf-8")
    examples = ledger_examples(clean, SYSTEM)
    assert len(examples) == 1
    assert not examples[0].negative
    assert "simulated/backtest evidence" in examples[0].messages[-1]["content"]


def test_unparseable_ledger_skipped(tmp_path: Path):
    broken = tmp_path / "broken.json"
    broken.write_text("{oops", encoding="utf-8")
    assert ledger_examples(broken, SYSTEM) == []


def test_build_full_corpus_merges_sources(tmp_path: Path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "r.json").write_text(
        json.dumps({"research_only": True, "live_pnl_claim": False,
                    "correctness": {"m": 1}}),
        encoding="utf-8",
    )
    doc = tmp_path / "doc.md"
    doc.write_text("# Only\nsection", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "ledger.json").write_text(
        json.dumps({"live_pnl_claim": True}), encoding="utf-8"
    )
    out = tmp_path / "full.jsonl"
    stats = build_full_corpus(receipts, out, notebooks=[doc], artifacts_dir=artifacts)
    assert stats["positive"] == 2  # receipt + notebook section
    assert stats["negative"] == 1  # live-claiming ledger
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        assert len(json.loads(line)["receipt_sha256"]) == 64
