"""Soft-verify: no silent northset stamps outside REQUIRED ∪ EXTRA ∪ BLOB_ONLY."""

from __future__ import annotations

from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import (
    bench_northset,
    northset_receipt_key_classification_honesty_errors,
)


def test_synth_receipt_fully_classified() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(bars, cfg)
    assert northset_receipt_key_classification_honesty_errors(receipt) == []


def test_unknown_stamp_fail_closed() -> None:
    errs = northset_receipt_key_classification_honesty_errors(
        {"structure_finite_rate": 1.0, "brand_new_silent_stamp": 0.5}
    )
    assert any("northset_receipt_unclassified_keys:" in e for e in errs)
    assert any("brand_new_silent_stamp" in e for e in errs)


def test_verify_wires_classification_and_candle_join() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "northset_receipt_key_classification_honesty_errors" in src
    assert 'join_coverage_honesty_errors(families.get("candle_order_book"))' in src
