"""Corpus-builder tests: provenance, eligibility, fail-closed behavior."""

import json
from pathlib import Path

from fx1.data import build_corpus, load_receipts


def _write_receipt(path: Path, *, research_only: bool, live_pnl_claim: bool) -> None:
    payload = {
        "schema": "test/v1",
        "research_only": research_only,
        "live_pnl_claim": live_pnl_claim,
        "disclaimer": "test receipt",
        "correctness": {"metric": 1.0},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_receipts_and_eligibility(tmp_path: Path):
    _write_receipt(tmp_path / "good.json", research_only=True, live_pnl_claim=False)
    _write_receipt(tmp_path / "bad.json", research_only=False, live_pnl_claim=True)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    records = load_receipts(tmp_path)
    assert len(records) == 2  # unparseable file skipped
    by_path = {Path(r.path).name: r for r in records}
    assert by_path["good.json"].eligible
    assert not by_path["bad.json"].eligible
    assert len(by_path["good.json"].sha256) == 64


def test_build_corpus_provenance_and_negatives(tmp_path: Path):
    _write_receipt(tmp_path / "good.json", research_only=True, live_pnl_claim=False)
    _write_receipt(tmp_path / "bad.json", research_only=True, live_pnl_claim=True)
    out = tmp_path / "corpus.jsonl"
    stats = build_corpus(tmp_path, out)
    assert stats == {"loaded": 2, "positive": 1, "negative": 1, "skipped": 0}
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert len(lines) == 2
    for line in lines:
        assert len(line["receipt_sha256"]) == 64
        roles = [m["role"] for m in line["messages"]]
        assert roles == ["system", "user", "assistant"]
    pos = next(x for x in lines if not x["negative"])
    neg = next(x for x in lines if x["negative"])
    assert "verify-research" in pos["messages"][-1]["content"]
    assert "No." in neg["messages"][-1]["content"]  # refusal taught
