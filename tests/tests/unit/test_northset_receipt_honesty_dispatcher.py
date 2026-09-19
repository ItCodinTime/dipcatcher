"""northset_receipt_honesty_errors dispatcher rewires orphaned helpers."""

from __future__ import annotations

from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_receipt_honesty_errors,
)


def test_dispatcher_nonempty() -> None:
    assert len(NORTHSET_RECEIPT_HONESTY_HELPERS) >= 30


def test_dispatcher_fans_out_amihud() -> None:
    assert "amihud_mean_negative" in northset_receipt_honesty_errors({"amihud_mean": -0.1})


def test_dispatcher_empty_blob() -> None:
    assert northset_receipt_honesty_errors({}) == []
