"""Ingest datasource payloads into the fx-1 corpus — gated and chained.

A fetched payload becomes training evidence only if it passes every gate:

1. The fetch itself succeeded (``FetchResult.ok``).
2. Time-stamped sources (``requires_as_of``) must carry an explicit
   observation date — undated market data is refused as leakage-prone.
3. Payloads asserting live trading performance become *negative* examples
   (refusal teaching), never positives — same rule as lab receipts.

Passed examples are recorded in the hash-chained :class:`CorpusLedger` with
the payload hash as source hash and this module's file hash as the
transformation hash, so every ingested datapoint is auditable end-to-end.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, Field

from fx1.data.corpus import SFTExample
from fx1.data.ledger import CorpusLedger
from fx1.data.sources.base import FetchResult
from fx1.data.sources.registry import get_spec

_TRANSFORM_PATH = Path(__file__).resolve()

# Free-text live-performance claims. Tight on purpose: the structured
# ``live_pnl_claim: true`` token or an explicit live-trading profit phrase.
# Commentary about the *rule* (e.g. "not live performance evidence") does not
# match — the claim must be affirmative.
_LIVE_TEXT = re.compile(
    r"(live[_\s-]?pnl[_\s-]?claim[\"'\s:]+true"
    r"|live trading (?:profit|returns?|pnl|p&l)"
    r"|real[- ]money (?:returns?|profits?)"
    r"|实盘(?:收益|盈利|回报))",
    re.IGNORECASE,
)


def _text_claims_live(text: str) -> bool:
    return bool(_LIVE_TEXT.search(text))


def transform_sha256() -> str:
    """Hash of the ingest code — the 'transformation' leg of lineage."""
    return hashlib.sha256(_TRANSFORM_PATH.read_bytes()).hexdigest()


class IngestDecision(BaseModel):
    """Outcome of the ingest gate: example plus the audit trail."""

    accepted: bool
    negative: bool = False
    reason: str = ""
    example: SFTExample | None = None
    payload_sha256: str = Field(default="")


def fetch_to_example(result: FetchResult, system: str) -> IngestDecision:
    """Apply the ingest gate to one fetch result."""
    if not result.ok:
        return IngestDecision(
            accepted=False,
            reason=f"fetch failed: {result.error}",
            payload_sha256=result.payload_sha256,
        )
    spec = get_spec(result.source)
    if spec.requires_as_of and not result.as_of:
        return IngestDecision(
            accepted=False,
            reason=(
                "leakage guard: time-stamped data from "
                f"{result.source} requires an explicit as_of observation date "
                "before it may enter the training corpus"
            ),
            payload_sha256=result.payload_sha256,
        )

    provenance = (
        f"source={result.source} api={result.api} "
        f"as_of={result.as_of or 'n/a'} fetched_at={result.fetched_at} "
        f"payload_sha256={result.payload_sha256}"
    )
    if _text_claims_live(result.text):
        example = SFTExample(
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": "Can I train on this datasource payload as "
                    "evidence of live trading performance?\n\n"
                    f"[provenance] {provenance}",
                },
                {
                    "role": "assistant",
                    "content": "No. The payload asserts live performance, which "
                    "the lab's honesty contract forbids as training evidence. "
                    "Datasource payloads are research-scoped observations; "
                    "they may inform analysis but never substantiate live P&L "
                    "claims.",
                },
            ],
            receipt_sha256=result.payload_sha256,
            source_path=f"datasource://{result.source}/{result.api}",
            negative=True,
        )
        return IngestDecision(
            accepted=True,
            negative=True,
            reason="payload asserts live performance → negative example",
            example=example,
            payload_sha256=result.payload_sha256,
        )

    excerpt = result.text[:4000]
    example = SFTExample(
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Summarize this {spec.display} payload "
                f"(as_of={result.as_of or 'n/a'}) and state exactly what it "
                "does and does not establish.",
            },
            {
                "role": "assistant",
                "content": f"Payload from {spec.display} (`{result.api}`), "
                f"observation date {result.as_of or 'n/a'}:\n```\n{excerpt}\n```\n"
                f"Provenance: `{provenance}`.\nThis establishes only what the "
                "source returned for that observation window; it is "
                "research/backtest evidence, not live performance, and any "
                "downstream claim must remain traceable to this payload hash.",
            },
        ],
        receipt_sha256=result.payload_sha256,
        source_path=f"datasource://{result.source}/{result.api}",
    )
    return IngestDecision(
        accepted=True,
        reason="accepted",
        example=example,
        payload_sha256=result.payload_sha256,
    )


def record_fetch_in_ledger(ledger: CorpusLedger, decision: IngestDecision) -> None:
    """Chain an ingest decision into the corpus ledger (tamper-evident)."""
    if decision.example is not None:
        example_sha = hashlib.sha256(decision.example.model_dump_json().encode()).hexdigest()
        ledger.record_example(
            source_sha256=decision.payload_sha256 or ("0" * 64),
            transform_sha256=transform_sha256(),
            example_sha256=example_sha,
        )
    else:
        ledger.record_exclusion(
            source_sha256=decision.payload_sha256 or ("0" * 64),
            transform_sha256=transform_sha256(),
            rule="datasource_ingest_gate",
        )
