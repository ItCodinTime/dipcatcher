"""Training-run manifest builder for fx-1.

This module does not launch GPUs. It enforces the pipeline contract —
eval-before-train, corpus provenance, cost disclosure — and emits an
immutable run manifest that a cluster launcher consumes. A run that cannot
produce a valid manifest does not start.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from fx1.train.config import TrainConfig


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_corpus(corpus_path: Path) -> dict[str, int]:
    """Every corpus line must carry receipt provenance. Fail-closed."""
    stats = {"lines": 0, "positive": 0, "negative": 0}
    with corpus_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("receipt_sha256"):
                raise ValueError(f"corpus line {lineno} lacks receipt_sha256 provenance")
            stats["lines"] += 1
            key = "negative" if record.get("negative") else "positive"
            stats[key] += 1
    if stats["lines"] == 0:
        raise ValueError("corpus is empty")
    return stats


def _validate_eval_gate(eval_path: Path) -> None:
    summary = json.loads(eval_path.read_text(encoding="utf-8"))
    if not summary.get("honesty_gate_passed", False):
        raise ValueError(
            "eval harness honesty gate not passed — training is blocked "
            "until the base model's eval results are on record"
        )


def build_training_manifest(config: TrainConfig, out_path: str | Path) -> dict:
    """Validate the run contract and write an immutable manifest."""
    corpus_path = Path(config.corpus_jsonl)
    eval_path = Path(config.eval_results_json)
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus not found: {corpus_path}")
    if not eval_path.exists():
        raise FileNotFoundError(
            f"eval results not found: {eval_path} — the eval harness runs before any training"
        )
    corpus_stats = _validate_corpus(corpus_path)
    _validate_eval_gate(eval_path)
    manifest = {
        "run_name": config.run_name,
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "corpus_sha256": _file_sha256(corpus_path),
        "corpus_stats": corpus_stats,
        "eval_results_sha256": _file_sha256(eval_path),
        "live_pnl_claim": False,
        "research_only": True,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
