"""Northset receipt stamps mean_tob_notional_share."""

from __future__ import annotations

import inspect
from pathlib import Path

from quant_fund.northset import benches
from quant_fund.research.catalog import mean_tob_notional_share_honesty_errors


def test_bench_source_stamps_mean_tob_notional_share() -> None:
    src = inspect.getsource(benches.bench_northset)
    assert '"mean_tob_notional_share"' in src
    assert "tob_notional_share" in src


def test_tob_notional_share_honesty_bounds() -> None:
    assert mean_tob_notional_share_honesty_errors({"mean_tob_notional_share": 1.0}) == []
    assert "mean_tob_notional_share_out_of_open_unit_interval" in (
        mean_tob_notional_share_honesty_errors({"mean_tob_notional_share": 0.0})
    )


def test_verify_research_wires_mean_tob_notional_share_honesty() -> None:
    """verify-research must call mean_tob_notional_share_honesty_errors (≠ size-share)."""
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "mean_tob_notional_share_honesty_errors" in src
    assert "for mtn_err in mean_tob_notional_share_honesty_errors" in src
