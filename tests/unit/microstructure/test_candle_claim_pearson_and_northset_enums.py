"""Candle claim + Pearson IC∈[-1,1]; northset string enum honesty."""

from __future__ import annotations

from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    candle_all_ic_pearson_unit_honesty_errors,
    candle_order_book_claim_honesty_errors,
    northset_receipt_string_enum_honesty_errors,
)


def test_candle_claim_and_pearson_on_synth() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_order_book_claim_honesty_errors(receipt) == []
    assert candle_all_ic_pearson_unit_honesty_errors(receipt) == []


def test_candle_claim_and_pearson_flags() -> None:
    assert candle_order_book_claim_honesty_errors(
        {"family": "candle_order_book", "research_only": False}
    ) == ["candle_research_only_missing_or_false"]
    assert "ic_ofi_pearson_out_of_unit_interval" in (
        candle_all_ic_pearson_unit_honesty_errors(
            {"family": "candle_order_book", "ic_ofi_pearson": 1.2}
        )
    )


def test_northset_string_enums_on_synth() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_receipt_string_enum_honesty_errors(receipt) == []
    assert northset_receipt_string_enum_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert northset_receipt_string_enum_honesty_errors(
        {"family": "northset", "session_l2_identity_gate": "maybe"}
    ) == ["session_l2_identity_gate_invalid"]


def test_verify_wires_candle_claim_pearson() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_order_book_claim_honesty_errors" in src
    assert "candle_all_ic_pearson_unit_honesty_errors" in src
