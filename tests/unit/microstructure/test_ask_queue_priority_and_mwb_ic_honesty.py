"""Honesty: ask queue priority separate from bid; mwb FEATURE_COLS IC when scored."""

from __future__ import annotations

from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import FEATURE_COLS, bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    candle_feature_cols_ic_completeness_honesty_errors,
    candle_feature_cols_ic_honesty_errors,
    candle_microprice_weight_balance_ic_honesty_errors,
    northset_queue_priority_bid_ask_pair_honesty_errors,
    northset_shape_and_session_l2_floors_honesty_errors,
    northset_top_level_claim_honesty_errors,
)


def test_feature_cols_scores_ask_queue_and_mwb() -> None:
    assert "ask_queue_priority_proxy" in FEATURE_COLS
    assert "microprice_weight_balance" in FEATURE_COLS
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert "ic_ask_queue_priority_proxy" in receipt
    assert "ic_microprice_weight_balance" in receipt
    assert candle_feature_cols_ic_honesty_errors(receipt) == []
    assert candle_feature_cols_ic_completeness_honesty_errors(receipt) == []
    assert candle_microprice_weight_balance_ic_honesty_errors(receipt) == []


def test_ask_queue_priority_required_when_bid_proxy_finite() -> None:
    assert northset_queue_priority_bid_ask_pair_honesty_errors(
        {"mean_queue_priority_proxy": 0.3}
    ) == ["mean_ask_queue_priority_proxy_missing_while_bid_proxy_finite"]
    assert (
        northset_queue_priority_bid_ask_pair_honesty_errors(
            {
                "mean_queue_priority_proxy": 0.3,
                "mean_ask_queue_priority_proxy": 0.4,
            }
        )
        == []
    )
    # Never force equality
    assert (
        northset_queue_priority_bid_ask_pair_honesty_errors(
            {
                "mean_queue_priority_proxy": 0.2,
                "mean_ask_queue_priority_proxy": 0.8,
            }
        )
        == []
    )


def test_northset_synth_bid_ask_pair_and_claim_floors() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_queue_priority_bid_ask_pair_honesty_errors(receipt) == []
    assert northset_top_level_claim_honesty_errors(receipt) == []
    assert northset_shape_and_session_l2_floors_honesty_errors(receipt) == []
    assert northset_queue_priority_bid_ask_pair_honesty_errors in (NORTHSET_RECEIPT_HONESTY_HELPERS)


def test_verify_wires_mwb_ic_honesty() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_microprice_weight_balance_ic_honesty_errors" in src
