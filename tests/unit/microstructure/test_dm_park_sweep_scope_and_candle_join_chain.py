"""DM vs Park honesty + sweep scope enums + candle join/chain."""

from __future__ import annotations

from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    candle_join_coverage_and_chain_honesty_errors,
    northset_dm_park_honesty_errors,
    northset_sweep_evidence_scope_honesty_errors,
)


def _northset():
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)


def test_dm_park_and_sweep_scope_on_synth() -> None:
    receipt = _northset()
    assert northset_dm_park_honesty_errors(receipt) == []
    assert northset_sweep_evidence_scope_honesty_errors(receipt) == []
    assert northset_dm_park_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert northset_sweep_evidence_scope_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS


def test_dm_park_flags_bad_p_and_preferred() -> None:
    assert "dm_gk_vs_park_p_out_of_unit_interval" in (
        northset_dm_park_honesty_errors({"dm_gk_vs_park_p": 1.5})
    )
    assert "dm_rs_vs_park_preferred_invalid" in (
        northset_dm_park_honesty_errors({"dm_rs_vs_park_preferred": "gk"})
    )


def test_candle_join_and_chain_on_synth() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_join_coverage_and_chain_honesty_errors(receipt) == []
    assert "n_scored_gt_n_fused" in candle_join_coverage_and_chain_honesty_errors(
        {
            "family": "candle_order_book",
            "n_bars": 10,
            "n_fused": 10,
            "n_scored": 11,
        }
    )


def test_verify_wires_candle_join_chain() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_join_coverage_and_chain_honesty_errors" in src
