"""queue priority means ≤ same-side size concentration means."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_queue_priority_le_size_concentration_honesty_errors,
)


def test_violation_detected() -> None:
    assert northset_queue_priority_le_size_concentration_honesty_errors(
        {
            "mean_queue_priority_proxy": 0.9,
            "mean_bid_size_concentration_top": 0.5,
        }
    ) == ["mean_queue_priority_proxy_gt_mean_bid_size_concentration_top"]


def test_synth_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(),
        cfg,
    )
    assert northset_queue_priority_le_size_concentration_honesty_errors(receipt) == []


def test_absent_skipped() -> None:
    assert northset_queue_priority_le_size_concentration_honesty_errors({}) == []


def test_helper_registered() -> None:
    assert (
        northset_queue_priority_le_size_concentration_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
