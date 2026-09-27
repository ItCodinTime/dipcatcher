"""Evidence hashes must not depend on set ordering or ndarray repr."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from quant_fund.metrics.analytics import analytics_export_digest
from quant_fund.paper.ledger import _promotion_receipt_digest
from quant_fund.research.real_benchmark import _read_receipt
from quant_fund.utils.hashing import canonical_json_bytes, receipt_tree


def test_sets_canonicalize_to_sorted_lists() -> None:
    payload = {"ids": {"b", "a", "c"}}
    encoded = json.loads(canonical_json_bytes(payload))
    assert encoded == {"ids": ["a", "b", "c"]}
    assert canonical_json_bytes({"ids": {"c", "b", "a"}}) == canonical_json_bytes(payload)


def test_ndarray_canonicalizes_as_nested_lists() -> None:
    array = np.array([[1.0, np.nan], [2.0, 3.0]])
    assert canonical_json_bytes(array) == canonical_json_bytes([[1.0, None], [2.0, 3.0]])


def test_receipt_digests_agree_on_sets_and_ignore_self_hash() -> None:
    blob = {"labels": {"zeta", "alpha"}, "n": 2}
    as_list = {"labels": ["alpha", "zeta"], "n": 2}
    assert receipt_tree(blob)["labels"] == ["alpha", "zeta"]
    assert analytics_export_digest(blob) == analytics_export_digest(as_list)
    sealed = {**as_list, "analytics_export_sha256": "deadbeef"}
    assert analytics_export_digest(sealed) == analytics_export_digest(as_list)
    assert _promotion_receipt_digest({"names": {"b", "a"}, "receipt_sha256": "nope"}) == (
        _promotion_receipt_digest({"names": ["a", "b"]})
    )


def test_phase1_code_hashes_remain_bound_to_the_tracked_seals(tmp_path: Path) -> None:
    """Historical code hashes remain in their sealed receipts as code evolves."""
    root = Path("data/metadata")
    benchmark = _read_receipt(root / "real_benchmark/us_wide_20260925/manifest.json")
    assert re.fullmatch(r"[0-9a-f]{64}", benchmark["code_sha256"])
    sealed_code_hashes = None
    for name in ("net_tournament", "cost_aware_tournament"):
        manifest = _read_receipt(root / name / "us_wide_20260925/manifest.json")
        code_hashes = manifest["code_sha256"]
        assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in code_hashes.values())
        if sealed_code_hashes is None:
            sealed_code_hashes = code_hashes
        else:
            assert code_hashes == sealed_code_hashes

    tampered = dict(benchmark)
    tampered["code_sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        _read_receipt(path)
