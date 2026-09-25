"""Immutable training receipts for fx-1 — the lab's philosophy, applied to ML.

Every training run emits a receipt binding: git revision, dirty-worktree
state, config hash, corpus + split-manifest hashes, eval hashes (base and
candidate), seed, and environment. ``verify_training_receipt`` re-checks the
hashes against the files on disk. fx-1 inherits the harness's evidence
culture: a run without a verifiable receipt never happened.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(repo_root: Path) -> tuple[str, bool]:
    """(HEAD sha, dirty?) — unknown/True when git is unavailable (fail-closed)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo_root,
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


class TrainingReceipt(BaseModel):
    run_name: str
    created_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    git_revision: str
    dirty_worktree: bool
    config_sha256: str = Field(min_length=64, max_length=64)
    corpus_sha256: str = Field(min_length=64, max_length=64)
    split_manifest_sha256: str = Field(min_length=64, max_length=64)
    eval_base_sha256: str = Field(min_length=64, max_length=64)
    seed: int
    python: str
    platform: str
    env_fingerprint: str = Field(
        description="SHA-256 of sorted relevant environment variable names"
    )
    live_pnl_claim: bool = False
    research_only: bool = True


def issue_receipt(
    *,
    run_name: str,
    repo_root: str | Path,
    config_path: str | Path,
    corpus_path: str | Path,
    split_manifest_path: str | Path,
    eval_base_path: str | Path,
    seed: int,
    out_path: str | Path,
) -> TrainingReceipt:
    root = Path(repo_root)
    sha, dirty = _git_revision(root)
    env_names = sorted(
        k for k in os.environ if k.startswith(("CUDA", "NCCL", "TORCH", "FX1"))
    )
    receipt = TrainingReceipt(
        run_name=run_name,
        git_revision=sha,
        dirty_worktree=dirty,
        config_sha256=_sha256_file(Path(config_path)),
        corpus_sha256=_sha256_file(Path(corpus_path)),
        split_manifest_sha256=_sha256_file(Path(split_manifest_path)),
        eval_base_sha256=_sha256_file(Path(eval_base_path)),
        seed=seed,
        python=sys.version.split()[0],
        platform=platform.platform(),
        env_fingerprint=hashlib.sha256(
            "\n".join(env_names).encode()
        ).hexdigest(),
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    return receipt


def verify_training_receipt(
    receipt_path: str | Path,
    *,
    config_path: str | Path,
    corpus_path: str | Path,
    split_manifest_path: str | Path,
    eval_base_path: str | Path,
) -> bool:
    """Re-verify a receipt against the files on disk. Fail-closed."""
    receipt = TrainingReceipt.model_validate_json(
        Path(receipt_path).read_text(encoding="utf-8")
    )
    if receipt.live_pnl_claim or not receipt.research_only:
        return False
    checks = {
        "config_sha256": config_path,
        "corpus_sha256": corpus_path,
        "split_manifest_sha256": split_manifest_path,
        "eval_base_sha256": eval_base_path,
    }
    for field, path in checks.items():
        p = Path(path)
        if not p.exists():
            return False
        if _sha256_file(p) != getattr(receipt, field):
            return False
    return True


def loads_receipt(path: str | Path) -> TrainingReceipt:
    return TrainingReceipt.model_validate_json(Path(path).read_text(encoding="utf-8"))
