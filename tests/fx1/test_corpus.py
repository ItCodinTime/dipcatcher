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


def _write_run_manifest(
    path: Path, *, claim: str | None, synthetic: bool = True,
    live_pnl_claim: bool | None = None,
) -> None:
    """Lab research-run manifest schema (data/metadata/research/runs)."""
    payload: dict = {
        "schema_version": 1,
        "synthetic": synthetic,
        "disclaimer": "SYNTHETIC research. Not a live-P&L claim.",
        "families": {},
        "correctness": {"metric": 1.0},
    }
    if claim is not None:
        payload["claim"] = claim
    if live_pnl_claim is not None:
        payload["live_pnl_claim"] = live_pnl_claim
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_manifest_schema_is_eligible_and_labeled_synthetic(tmp_path: Path):
    _write_run_manifest(tmp_path / "run.json", claim="research_only")
    records = load_receipts(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record.eligible
    assert record.evidence_class == "synthetic"
    out = tmp_path / "corpus.jsonl"
    stats = build_corpus(tmp_path, out)
    assert stats["positive"] == 1 and stats["negative"] == 0
    line = json.loads(out.read_text().splitlines()[0])
    assistant = line["messages"][-1]["content"]
    assert "SYNTHETIC" in assistant and "simulated data" in assistant


def test_run_manifest_without_claim_fails_closed(tmp_path: Path):
    _write_run_manifest(tmp_path / "run.json", claim=None)
    record = load_receipts(tmp_path)[0]
    assert not record.eligible
    assert record.live_pnl_claim  # live claim assumed, fail-closed


def test_explicit_live_claim_overrides_research_only_claim(tmp_path: Path):
    _write_run_manifest(
        tmp_path / "run.json", claim="research_only", live_pnl_claim=True
    )
    record = load_receipts(tmp_path)[0]
    assert record.research_only and record.live_pnl_claim
    assert not record.eligible
    out = tmp_path / "corpus.jsonl"
    stats = build_corpus(tmp_path, out)
    assert stats["negative"] == 1


def test_build_corpus_accepts_multiple_receipt_dirs(tmp_path: Path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_receipt(dir_a / "good.json", research_only=True, live_pnl_claim=False)
    _write_run_manifest(dir_b / "run.json", claim="research_only")
    _write_run_manifest(dir_b / "claimless.json", claim=None)
    out = tmp_path / "corpus.jsonl"
    stats = build_corpus([dir_a, dir_b], out)
    assert stats == {"loaded": 3, "positive": 2, "negative": 1, "skipped": 0}
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert len({x["receipt_sha256"] for x in lines}) == 3  # distinct provenance
