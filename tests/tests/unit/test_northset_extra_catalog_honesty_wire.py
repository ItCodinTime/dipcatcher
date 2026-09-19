"""verify-research wires northset_receipt_honesty_errors once (no northset dupes)."""

from __future__ import annotations

from pathlib import Path

from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    gap_finite_rate_honesty_errors,
    northset_receipt_honesty_errors,
    ohlc_identity_rate_honesty_errors,
    vpin_mean_honesty_errors,
)


def test_receipt_helpers_include_priority_trio() -> None:
    assert vpin_mean_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert gap_finite_rate_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert ohlc_identity_rate_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert len(NORTHSET_RECEIPT_HONESTY_HELPERS) >= 39


def test_receipt_dispatcher_empty_blob_ok() -> None:
    assert northset_receipt_honesty_errors({}) == []
    assert northset_receipt_honesty_errors(None) == []


def test_verify_wires_receipt_once_not_gap_vpin_individually() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "northset_receipt_honesty_errors" in src
    assert "for receipt_err in northset_receipt_honesty_errors" in src
    # Northset-family individual calls for receipt members must stay out (dispatcher owns them).
    assert 'gap_finite_rate_honesty_errors(families.get("northset"))' not in src
    assert 'vpin_mean_honesty_errors(families.get("northset"))' not in src
