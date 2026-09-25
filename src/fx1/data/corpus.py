"""Build the fx-1 supervised fine-tuning corpus.

Every example carries the SHA-256 of the receipt it was derived from, so the
training set is auditable end-to-end: fx-1 learns only from gate-passed,
verifiable research behavior. Gate-rejected / live-claiming receipts become
*negative* examples teaching refusal and honest reporting — never positives.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from fx1.data.receipts import ReceiptRecord, load_receipts

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "fx1_system.md"


class SFTExample(BaseModel):
    """One chat-formatted training example with provenance."""

    messages: list[dict[str, str]]
    receipt_sha256: str = Field(description="Provenance: source receipt hash")
    source_path: str
    negative: bool = Field(
        default=False,
        description="True for refusal/honesty examples from ineligible receipts",
    )


def _system_prompt() -> str:
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return "You are fx-1, a quant research model powered by the dipcatcher evidence engine."


def _positive_example(record: ReceiptRecord, system: str) -> SFTExample:
    correctness = record.payload.get("correctness", {})
    summary = json.dumps(correctness, indent=2, sort_keys=True) if correctness else "{}"
    synthetic_note = (
        "This is SYNTHETIC evidence — produced on simulated data, not market "
        "data — and must always be presented as such.\n"
        if record.evidence_class == "synthetic"
        else ""
    )
    user = (
        f"Summarize the verified result of this dipcatcher receipt "
        f"(schema {record.schema_name}) and state what it does and does not establish."
    )
    assistant = (
        f"Receipt {record.sha256[:16]}… (schema `{record.schema_name}`) is research-scoped "
        f"evidence (class: {record.evidence_class}). Key correctness metrics:\n"
        f"```json\n{summary}\n```\n"
        f"{synthetic_note}"
        f"Disclaimer: {record.disclaimer or 'research/backtest/simulated evidence only'}.\n"
        "This establishes correctness within the recorded workload only; it is not "
        "live performance evidence and cannot authorize promotion or live execution. "
        f"Verify independently with `uv run dipcatcher verify-research`."
    )
    return SFTExample(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        receipt_sha256=record.sha256,
        source_path=record.path,
    )


def _negative_example(record: ReceiptRecord, system: str) -> SFTExample:
    reasons = []
    if record.live_pnl_claim:
        reasons.append("it asserts a live P&L claim")
    if not record.research_only:
        reasons.append("it is not marked research_only")
    reason = " and ".join(reasons) or "it fails eligibility"
    user = (
        f"Can we use this receipt (schema {record.schema_name}) as evidence of trading performance?"
    )
    assistant = (
        f"No. This receipt is ineligible as fx-1 evidence because {reason}. "
        "The lab's honesty contract forbids treating such artifacts as performance "
        "evidence; only gate-passed, research-scoped receipts with "
        "`live_pnl_claim=false` qualify. I will not summarize it as a result."
    )
    return SFTExample(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        receipt_sha256=record.sha256,
        source_path=record.path,
        negative=True,
    )


def build_corpus(
    receipts_dir: str | Path,
    out_jsonl: str | Path,
) -> dict[str, int]:
    """Build the SFT corpus and write it as JSONL.

    Returns counts for auditing. Every emitted line is an :class:`SFTExample`
    with a ``receipt_sha256`` provenance field.
    """
    system = _system_prompt()
    records = load_receipts(receipts_dir)
    stats = {"loaded": len(records), "positive": 0, "negative": 0, "skipped": 0}
    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            example = (
                _positive_example(record, system)
                if record.eligible
                else _negative_example(record, system)
            )
            if record.eligible:
                stats["positive"] += 1
            else:
                stats["negative"] += 1
            fh.write(example.model_dump_json() + "\n")
    return stats


def build_full_corpus(
    receipts_dir: str | Path,
    out_jsonl: str | Path,
    *,
    notebooks: list[str | Path] | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, int]:
    """Full corpus: receipts + research notebooks/docs + ledger artifacts.

    All sources are provenance-hashed; live-claiming ledgers become negative
    examples. Deferred imports keep the receipts-only path dependency-light.
    """
    from fx1.data.ledgers import ledger_corpus
    from fx1.data.notebooks import notebook_examples

    stats = build_corpus(receipts_dir, out_jsonl)
    system = _system_prompt()
    extra: list[SFTExample] = []
    for doc in notebooks or []:
        extra.extend(notebook_examples(doc, system))
    if artifacts_dir is not None:
        extra.extend(ledger_corpus(artifacts_dir, system))
    out_path = Path(out_jsonl)
    with out_path.open("a", encoding="utf-8") as fh:
        for example in extra:
            fh.write(example.model_dump_json() + "\n")
    stats["positive"] += sum(1 for e in extra if not e.negative)
    stats["negative"] += sum(1 for e in extra if e.negative)
    stats["loaded"] += len(extra)
    return stats
