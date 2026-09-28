"""Pin the published Phase-1 evidence bytes after the code re-seal."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SEALED_SHA256 = {
    "data/metadata/cost_aware_tournament/us_wide_20260925/manifest.json": "b9ab870606f56c185c0deeeca4e5a0ff09dfec5a21aa05249d4ffb0d78db92fd",
    "data/metadata/cost_aware_tournament/us_wide_20260925/validation.attempt.json": "bf7d470fbc16697b0360235d2480a5066ec452938fd98c0e323bd8f07bf2857d",
    "data/metadata/cost_aware_tournament/us_wide_20260925/validation.json.gz": "cd6da69e1c3bc5dd3b67afe88c89347c3c73e573703fffbeaba7ebc3b62e891f",
    "data/metadata/net_tournament/us_wide_20260925/manifest.json": "f77cb76264d8619b0e1de311a663cee6de1d68a90cd655d2723e10e541fba87a",
    "data/metadata/net_tournament/us_wide_20260925/test.attempt.json": "9925b88456afd3226b5bbaf098c48cefd1746f871316e3695ff01775d60fffa7",
    "data/metadata/net_tournament/us_wide_20260925/test.json.gz": "dbf479bdb1aacfcc44956dad0add0eb71ae50c32753d6a0fd857585496c5e9d5",
    "data/metadata/net_tournament/us_wide_20260925/validation.attempt.json": "6bf1f9192c51d9e4a3e6fe8349a975c5015533eca78b98e56e53edda4dad87bc",
    "data/metadata/net_tournament/us_wide_20260925/validation.json.gz": "6ac2fc159ddb39ec22b0e457cc7c52b288eec181a04cad291c91fff9df0c1a11",
    "data/metadata/research/phase1_evidence_index.json": "70ac1092b9470113a4bae41945ffbf08d078a529ce053d741624ac340deb35a9",
}


@pytest.mark.parametrize(("relative", "expected"), sorted(_SEALED_SHA256.items()))
def test_original_phase1_receipt_bytes_are_unchanged(relative: str, expected: str) -> None:
    digest = hashlib.sha256((_ROOT / relative).read_bytes()).hexdigest()
    assert digest == expected, f"sealed receipt bytes changed: {relative}"


def test_forward_shadow_pins_the_published_index_receipt() -> None:
    expected = "bfce88b1efdccc1b15a4e925de79ad0ef10c0fd33102744c08f72374b08fb8dd"
    source = (_ROOT / "src/quant_fund/paper/forward_shadow.py").read_text(encoding="utf-8")
    pinned = [
        statement.value.value
        for statement in ast.parse(source).body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_PUBLISHED_INDEX_SHA256"
            for target in statement.targets
        )
        and isinstance(statement.value, ast.Constant)
    ]
    index = json.loads(
        (_ROOT / "data/metadata/research/phase1_evidence_index.json").read_text(encoding="utf-8")
    )
    assert pinned == [expected]
    assert index["receipt_sha256"] == expected
