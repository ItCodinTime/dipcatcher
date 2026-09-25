"""fx-1 environment doctor — readiness signals for the model lane.

Mirrors the harness-side ``dipcatcher doctor``: presence flags only,
never secret values. Informational — nothing here hard-fails; missing
pieces are reported so an operator can fix the environment.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import fx1


def _count_jsonl(path: Path) -> int | None:
    if not path.is_file():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _count_receipts(path: Path) -> int | None:
    if not path.is_dir():
        return None
    return sum(1 for _ in path.rglob("*.json"))


def collect_status(root: Path | None = None) -> dict[str, object]:
    """Return fx-1 readiness signals rooted at *root* (default: cwd)."""
    root = root or Path.cwd()
    status: dict[str, object] = {
        "fx1_version": fx1.__version__,
        "base_model": fx1.BASE_MODEL,
        "harness_package": "ok" if importlib.util.find_spec("quant_fund") else "missing",
        # Presence flags only — values never leave the process.
        "moonshot_key": "set" if os.environ.get("MOONSHOT_API_KEY") else "unset",
        "signing_key": "set" if os.environ.get("FX1_SIGNING_KEY") else "unset",
    }
    receipts = _count_receipts(root / "receipts")
    status["receipts"] = receipts if receipts is not None else "missing"
    for name in ("corpus", "corpus_runs", "eval", "manifest"):
        path = root / "data" / "fx1" / f"{name}.{'json' if name == 'manifest' else 'jsonl'}"
        n = _count_jsonl(path)
        status[f"data_fx1_{name}"] = n if n is not None else "missing"
    try:
        from fx1.eval import DEFAULT_BANK

        status["eval_bank_tasks"] = len(DEFAULT_BANK)
    except Exception:  # noqa: BLE001 — doctor reports, never hard-fails
        status["eval_bank_tasks"] = "unavailable"
    return status
