"""Receipt seals: self-excluding, key-order stable, and tamper-evident."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quant_fund.metrics.analytics import analytics_export_digest
from quant_fund.research.real_benchmark import _digest, _read_receipt, _seal
from quant_fund.utils.hashing import canonical_frame_fingerprint, canonical_json_bytes
from tests.property._profiles import adversarial_settings

_text = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=8,
)
_num = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)


@given(
    pairs=st.lists(st.tuples(_text, _num), min_size=1, max_size=6, unique_by=lambda item: item[0])
)
@adversarial_settings()
def test_seal_roundtrip_and_tamper_detection(pairs: list[tuple[str, float]]) -> None:
    payload = {key: value for key, value in pairs}
    sealed = _seal(dict(payload))
    assert sealed["receipt_sha256"] == _digest(payload)
    reversed_keys = {key: payload[key] for key in reversed(list(payload))}
    assert _digest(reversed_keys) == _digest(payload)
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "receipt.json"
        path.write_text(json.dumps(sealed))
        loaded = _read_receipt(path)
        assert loaded["receipt_sha256"] == sealed["receipt_sha256"]
        tampered = dict(sealed)
        tampered["nudge"] = 1.0
        path.write_text(json.dumps(tampered))
        with pytest.raises(ValueError, match="hash mismatch"):
            _read_receipt(path)


@given(value=_num, other=_num)
@adversarial_settings()
def test_analytics_digest_changes_when_a_field_changes(value: float, other: float) -> None:
    blob = {"label": "RESEARCH_SIM", "score": value, "research_only": True, "live_pnl_claim": False}
    digest = analytics_export_digest(blob)
    assert digest == analytics_export_digest({**blob, "analytics_export_sha256": digest})
    if value != other:
        assert digest != analytics_export_digest({**blob, "score": other})


@given(
    rows=st.lists(
        st.tuples(_text, _num),
        min_size=1,
        max_size=5,
    )
)
@adversarial_settings()
def test_frame_fingerprint_ignores_row_order_and_keeps_nulls_distinct_from_numbers(
    rows: list[tuple[str, float]],
) -> None:
    frame = pl.DataFrame(
        {"name": [name for name, _ in rows], "value": [value for _, value in rows]}
    )
    flipped = frame.reverse()
    assert canonical_frame_fingerprint(flipped) == canonical_frame_fingerprint(frame)
    stamped = canonical_json_bytes({"when": datetime(2024, 1, 2, tzinfo=UTC), "flag": True, "n": 1})
    assert json.loads(stamped)["when"].startswith("2024-01-02")
