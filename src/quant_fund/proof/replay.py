"""Deterministic replay behind the ``--replay`` flag (DESIGN.md §5.5 check 9).

Re-runs ``run_backtest_proven`` in a fresh tmp bundle dir with the recorded
seed + config and asserts bundle identity modulo ``created_utc`` (§8.1). The
replay bundle dir is seeded with the chain lines *preceding* the target bundle
so ``prev_bundle_hash`` (and therefore ``bundle_id``) reproduces for non-head
bundles too.

Replay needs two things the bundle commits to by hash only: the resolved
config (restored from the ``<id>.config.json`` sidecar written at mint time)
and the PIT vault root (supplied via ``pit_root`` — the bundle deliberately
commits to content hashes, not local paths).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from quant_fund.proof.bundle import (
    BUNDLES_JSONL,
    canonical_bundle_bytes,
    load_chain,
    sidecar_paths,
)
from quant_fund.proofcore.contracts import ProofBundleV1

__all__ = ["replay_bundle"]


def _identity_differs(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Field names whose values differ, ignoring created_utc (§8.1)."""
    ignored = {"created_utc"}
    keys = (set(a) | set(b)) - ignored
    return sorted(key for key in keys if a.get(key) != b.get(key))


def replay_bundle(
    bundle_path: Path,
    *,
    bundle_dir: Path,
    pit_root: Path | None,
    vault: Any = None,
) -> tuple[bool, str]:
    """Re-run the recorded run; return (matches_identity, detail).

    ``vault`` is a test seam (in-memory §4.3 fake) while W1 lands in parallel;
    production replay leaves it None and opens the real vault at ``pit_root``.
    """
    if pit_root is None:
        return False, "no_pit_root"
    bundle_path = Path(bundle_path)
    bundle_dir = Path(bundle_dir)
    try:
        bundle = ProofBundleV1.model_validate(json.loads(bundle_path.read_bytes()))
    except Exception as exc:
        return False, f"unreadable:{exc.__class__.__name__}"

    config_path = sidecar_paths(bundle_dir, bundle.bundle_id)["config"]
    if not config_path.exists():
        return False, "config_missing"

    from quant_fund.config.models import AppConfig  # lazy: layer-0 config package

    try:
        config = AppConfig.model_validate(json.loads(config_path.read_bytes()))
    except Exception as exc:
        return False, f"config_invalid:{exc.__class__.__name__}"

    from quant_fund.proof.runner import run_backtest_proven  # lazy: engine stack
    from quant_fund.proof.sign import HmacSha256Signer, Signer

    signer: Signer | None = None
    if bundle.signature.scheme == "hmac-sha256":
        try:
            signer = HmacSha256Signer.from_env()
        except Exception:
            return False, "no_signing_key"

    with tempfile.TemporaryDirectory(prefix="proofcore-replay-") as tmp:
        replay_dir = Path(tmp)
        # Seed the replay chain with every line preceding the target bundle so
        # prev_bundle_hash (inside the self-hash preimage) reproduces.
        try:
            chain = load_chain(bundle_dir)
        except Exception:
            chain = []
        preceding: list[ProofBundleV1] = []
        for entry in chain:
            if entry.bundle_id == bundle.bundle_id:
                break
            preceding.append(entry)
        if preceding:
            chain_path = replay_dir / BUNDLES_JSONL
            with chain_path.open("wb") as handle:
                for entry in preceding:
                    handle.write(canonical_bundle_bytes(entry) + b"\n")
        try:
            rerun = run_backtest_proven(
                config,
                seed=bundle.seed,
                pit_root=Path(pit_root),
                bundle_dir=replay_dir,
                signer=signer,
                vault=vault,
            )
        except Exception as exc:
            return False, f"rerun_error:{exc.__class__.__name__}"

    differs = _identity_differs(bundle.model_dump(mode="json"), rerun.model_dump(mode="json"))
    if differs:
        return False, "mismatch:" + ",".join(differs)
    return True, "identical"
