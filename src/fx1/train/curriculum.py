"""Curriculum builder: order the fx-1 corpus from foundations to frontier.

Training order matters. fx-1's curriculum walks competence levels: contracts
and vocabulary first, then interpretation, then full research loops, and
finally refusal-under-pressure. Within each level, order is deterministic
(seeded shuffle) so curricula are reproducible from the corpus hash.
"""

from __future__ import annotations

import hashlib
import json
import random
from enum import IntEnum
from pathlib import Path


class Level(IntEnum):
    CONTRACTS = 0       # system rules, evidence classes, honesty vocabulary
    INTERPRETATION = 1  # reading receipts, scorecards, ledgers
    RESEARCH_LOOP = 2   # hypothesis → bench → verdict traces
    REFUSAL = 3         # negative examples under pressure


def classify(example: dict) -> Level:
    if example.get("negative"):
        return Level.REFUSAL
    messages = example.get("messages", [])
    assistant = " ".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "assistant"
    )
    if "<tool_call>" in assistant or "Gate verdict" in assistant:
        return Level.RESEARCH_LOOP
    if "verify-research" in assistant or "evidence class" in assistant:
        return Level.INTERPRETATION
    return Level.CONTRACTS


def build_curriculum(
    corpus_jsonl: str | Path, out_jsonl: str | Path, *, seed: int = 17
) -> dict[str, int]:
    """Reorder the corpus into curriculum order; returns per-level counts."""
    lines = [
        json.loads(x)
        for x in Path(corpus_jsonl).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    rng = random.Random(seed)
    buckets: dict[Level, list[dict]] = {level: [] for level in Level}
    for line in lines:
        buckets[classify(line)].append(line)
    ordered: list[dict] = []
    counts: dict[str, int] = {}
    for level in Level:
        bucket = buckets[level]
        # Deterministic within-level shuffle keyed on content hash + seed.
        bucket.sort(key=lambda e: hashlib.sha256(
            (str(seed) + json.dumps(e, sort_keys=True)).encode()
        ).hexdigest())
        rng.shuffle(bucket)  # seeded, reproducible
        ordered.extend(bucket)
        counts[level.name.lower()] = len(bucket)
    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for line in ordered:
            fh.write(json.dumps(line) + "\n")
    return counts
