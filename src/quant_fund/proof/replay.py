"""Deterministic replay entry point, closed until the proven runner is causal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["replay_bundle"]


def replay_bundle(
    bundle_path: Path,
    *,
    bundle_dir: Path,
    pit_root: Path | None,
    vault: Any = None,
) -> tuple[bool, str]:
    """Return a failed replay verdict until decision-time reads are implemented."""
    return False, "runner_unavailable:per_decision_asof_not_implemented"
