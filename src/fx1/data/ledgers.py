"""Tournament / backtest / carry artifacts as fx-1 corpus sources.

Ledgers under ``artifacts/`` teach fx-1 to interpret strategy evidence
honestly. Any artifact asserting a live claim becomes a negative example —
the model learns to refuse it, never to repeat it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fx1.data.corpus import SFTExample


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _claims_live(payload: object) -> bool:
    """Fail-closed scan: any mapping key token implying a live claim."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            token = str(key).lower().replace("-", "_")
            if token == "live_pnl_claim" and value is True:
                return True
            if _claims_live(value):
                return True
    elif isinstance(payload, list):
        return any(_claims_live(item) for item in payload)
    return False


def ledger_examples(path: str | Path, system: str) -> list[SFTExample]:
    """Convert one JSON ledger artifact into an SFT example."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    digest = _sha256(source)
    keys = sorted(payload)[:12] if isinstance(payload, dict) else []
    if _claims_live(payload):
        return [
            SFTExample(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": "Is this artifact evidence of live trading performance?"},
                    {"role": "assistant", "content": (
                        f"No. Artifact {digest[:16]}… asserts a live P&L claim, "
                        "which the honesty contract forbids as fx-1 evidence. "
                        "I will not summarize it as a result."
                    )},
                ],
                receipt_sha256=digest,
                source_path=str(source),
                negative=True,
            )
        ]
    return [
        SFTExample(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    f"Summarize this harness ledger artifact and its evidence "
                    f"class. Top-level keys: {keys}"
                )},
                {"role": "assistant", "content": (
                    f"Artifact {digest[:16]}… is simulated/backtest evidence "
                    "from the dipcatcher harness. It may inform research "
                    "conclusions but is not live performance, and any strategy "
                    "reading must note costs, impact assumptions, and the "
                    "frozen slate it ran under."
                )},
            ],
            receipt_sha256=digest,
            source_path=str(source),
        )
    ]


def ledger_corpus(artifacts_dir: str | Path, system: str) -> list[SFTExample]:
    """All JSON ledger artifacts under a directory."""
    examples: list[SFTExample] = []
    for path in sorted(Path(artifacts_dir).rglob("*.json")):
        examples.extend(ledger_examples(path, system))
    return examples
