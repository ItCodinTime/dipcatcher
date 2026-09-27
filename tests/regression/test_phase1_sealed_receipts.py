"""Pin the original Phase-1 evidence bytes; later code must not re-seal them."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SEALED_SHA256 = {
    "data/metadata/cost_aware_tournament/us_wide_20260925/manifest.json": "ea26eaf805da7963eea40d81ffb69078199d3695bed6669b281a38947160a763",
    "data/metadata/cost_aware_tournament/us_wide_20260925/validation.attempt.json": "1fe173b736824f8372d5e94c7c09cb7f8d30c71b4020796db3802babfca3efa5",
    "data/metadata/cost_aware_tournament/us_wide_20260925/validation.json.gz": "67c1a4738e479fb3c54a95d422e1e32e9a276c7dc63d1c876f554e0fc50f4273",
    "data/metadata/net_tournament/us_wide_20260925/manifest.json": "a4a8f384c14b4a7805fbab5f52158a11a38aa78d87368a65c933acd061054eaf",
    "data/metadata/net_tournament/us_wide_20260925/test.attempt.json": "1f87f6ca24f8787a812ab6f757ec1d6f5e2f96c805433b9c9def538bdbda9c9c",
    "data/metadata/net_tournament/us_wide_20260925/test.json.gz": "61f2a1cd147f5d418e190c685e2b7845e09e8a44746e64a3726797bea48b3efe",
    "data/metadata/net_tournament/us_wide_20260925/validation.attempt.json": "d4ffa0198f2bab8da516a96bb4829726d1e3ef900b015cd5a010117b28f98ff2",
    "data/metadata/net_tournament/us_wide_20260925/validation.json.gz": "abf8037e85987750fcc6cff42b53137464612e0409e97642e64a43ec961ece36",
    "data/metadata/research/phase1_evidence_index.json": "3e8d28e2ad08137d5e0ea7b8703600d710cdda37ebe83e384afc50fbc94d1f7a",
}


@pytest.mark.parametrize(("relative", "expected"), sorted(_SEALED_SHA256.items()))
def test_original_phase1_receipt_bytes_are_unchanged(relative: str, expected: str) -> None:
    digest = hashlib.sha256((_ROOT / relative).read_bytes()).hexdigest()
    assert digest == expected, f"sealed receipt bytes changed: {relative}"


def test_forward_shadow_still_pins_the_original_index_receipt() -> None:
    expected = "0ce794b56249952fce5b2ff1046eea9e50b2f4e6d691539b8019959131873204"
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
