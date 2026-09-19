"""Soft-verify northset n_bars / n_fused / n_scored sizing chain."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import northset_n_bars_scored_honesty_errors


def test_n_bars_scored_ok() -> None:
    assert northset_n_bars_scored_honesty_errors({"n_bars": 100, "n_scored": 80}) == []
    assert (
        northset_n_bars_scored_honesty_errors({"n_bars": 100, "n_fused": 90, "n_scored": 80}) == []
    )
    assert northset_n_bars_scored_honesty_errors({}) == []


def test_n_scored_gt_n_bars() -> None:
    assert "n_scored_gt_n_bars" in northset_n_bars_scored_honesty_errors(
        {"n_bars": 10, "n_scored": 11}
    )


def test_n_scored_gt_n_fused_fail_closed() -> None:
    assert "n_scored_gt_n_fused" in northset_n_bars_scored_honesty_errors(
        {"n_fused": 10, "n_scored": 11}
    )


def test_n_fused_gt_n_bars_fail_closed() -> None:
    assert "n_fused_gt_n_bars" in northset_n_bars_scored_honesty_errors(
        {"n_bars": 10, "n_fused": 11}
    )


def test_n_fused_not_nonneg_int_fail_closed() -> None:
    assert "n_fused_not_nonneg_int" in northset_n_bars_scored_honesty_errors({"n_fused": -1})
    assert "n_fused_not_nonneg_int" in northset_n_bars_scored_honesty_errors({"n_fused": 1.5})


def test_synth_northset_sizing_chain_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(),
        cfg,
    )
    assert "n_fused" in receipt
    assert int(receipt["n_scored"]) <= int(receipt["n_fused"]) <= int(receipt["n_bars"])
    assert northset_n_bars_scored_honesty_errors(receipt) == []
