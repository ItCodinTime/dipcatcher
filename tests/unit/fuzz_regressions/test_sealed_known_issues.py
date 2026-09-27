"""Fuzz findings in files covered by the Phase-1 seal — documented, not fixed.

``quant_fund.research.verify.verify_research_artifact`` backs the
``verify-research`` receipt-verification path, which is sealed (see
K3_SWARM_BRIEF / AGENTS.md honesty contract). The bug below is real and
reproducible; it is recorded here as ``xfail`` evidence rather than patched,
because verifier behavior must not silently drift under the seal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.verify import verify_research_artifact


@pytest.mark.xfail(
    strict=False,
    reason=(
        "sealed verify-research path: verify.py:206 read_text() catches "
        "(OSError, JSONDecodeError) but not UnicodeDecodeError — an "
        "undecodable receipt file crashes instead of returning "
        "{valid: False}. Documented in the fuzz PR; fix needs owner sign-off."
    ),
)
def test_undecodable_receipt_returns_invalid_report(tmp_path: Path) -> None:
    """Corrupt (non-UTF8) receipt bytes should yield a report, not a crash."""
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"\xff\xfe\x00 not utf8")
    report = verify_research_artifact(receipt)
    assert report["valid"] is False


def test_valid_json_still_verified(tmp_path: Path) -> None:
    # Sanity: the sealed path is untouched — a normal non-conforming JSON
    # blob produces the usual structured verdict.
    p = tmp_path / "receipt.json"
    p.write_text(json.dumps({"firm": "wrong"}))
    report = verify_research_artifact(p)
    assert report["valid"] is False
    assert any("notebook" in e for e in report["errors"])
