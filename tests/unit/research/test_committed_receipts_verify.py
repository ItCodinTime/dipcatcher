"""Every committed receipt in ``receipts/`` must pass ``verify-receipt``.

Append-only evidence is only evidence when it verifies. ``receipts/legacy-
unsealed/`` holds the pre-seal-era artifacts quarantined as *unverifiable*;
anything in ``receipts/`` root is claimed live and must seal-verify.
``KNOWN_UNSEALED`` is a shrink-only ratchet for the legacy receipts that have
not migrated yet — a new unsealed receipt in ``receipts/`` fails this test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from quant_fund.research.receipt_v2 import verify_receipt_file

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPTS = REPO_ROOT / "receipts"

# Pre-seal-era receipts pending the legacy-unsealed/ migration (#231).
# Shrink-only: removing an entry is always safe; adding one is not.
# Values pin the file bytes — an unsealed receipt can't verify, so without
# the digest the exemption would also mask edits to its claims.
KNOWN_UNSEALED = {
    "adaptive_mix_20asset_1d_20260922.json": "b59422415e556f674ea03fb8e9a3b867408bf6d163cf7c11c64feeb7e33b4313",
    "adaptive_mix_band_search_20asset_1d_20260922.json": "5e02918850944a68d1d45815b5928f1eed462d456eac4723ac9290514e7b97f4",
    "basis_pair_candidate_20asset_1d_20260922.json": "d021801c94e33d8720f9084762138893978a61e36ef670b4c0140edeaf37c1ec",
    "basis_reversion_screen_20asset_1d_20260922.json": "61c9c8f5b3e502f0c5218dd743ad030ecbb962d6822bb7b81a2b962445aa378f",
    "dip_bench_crypto_1d_20260925.json": "584eb681dcd18fc65835c1573360b4b5fc9ebb72aa662bcbb9e7c101b06e2f03",
    "fast_replay_p42_conformance_20260927.json": "8236a26489e9253dfcc1f3d879a2fd276c0023fb4d167c764c5330406ce950d8",
    "incumbent_bench_qlib.json": "f455123351b44151d68876d1a93fa3a2dc449c51d280ee83926f22cd9861984f",
}


def test_committed_receipts_verify_or_are_known_legacy() -> None:
    unsealed: list[str] = []
    failures: list[str] = []
    for path in sorted(RECEIPTS.glob("*.json")):
        result = verify_receipt_file(path)
        if result["valid"]:
            continue
        if (
            result["errors"] == ["receipt_sha256_missing_or_invalid"]
            and path.name in KNOWN_UNSEALED
            and hashlib.sha256(path.read_bytes()).hexdigest() == KNOWN_UNSEALED[path.name]
        ):
            unsealed.append(path.name)
            continue
        failures.append(f"{path.name}: {result['errors']}")
    assert failures == [], "committed receipts fail verification:\n" + "\n".join(failures)
    assert not set(KNOWN_UNSEALED) - set(unsealed), (
        "KNOWN_UNSEALED has entries no longer unsealed in receipts/ — shrink the set"
    )
