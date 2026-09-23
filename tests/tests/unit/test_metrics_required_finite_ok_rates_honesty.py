"""metrics_required_finite_ok True ⇒ companion structure finite rates present/finite."""

from __future__ import annotations

from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    northset_metrics_required_finite_ok_rates_honesty_errors,
)


def test_synth_metrics_ok_implies_rates() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert receipt.get("metrics_required_finite_ok") is True
    assert northset_metrics_required_finite_ok_rates_honesty_errors(receipt) == []


def test_metrics_ok_missing_rate_fail_closed() -> None:
    blob = {
        "metrics_required_finite_ok": True,
        "depth_shape_finite_rate": 1.0,
        "concentration_top_finite_rate": 1.0,
        "queue_priority_finite_rate": 1.0,
        "side_notional_finite_rate": 1.0,
        "tob_size_share_finite_rate": 1.0,
        # structure_finite_rate missing
    }
    errs = northset_metrics_required_finite_ok_rates_honesty_errors(blob)
    assert "structure_finite_rate_missing_while_metrics_required_finite_ok" in errs


def test_metrics_ok_false_skips() -> None:
    assert (
        northset_metrics_required_finite_ok_rates_honesty_errors(
            {"metrics_required_finite_ok": False}
        )
        == []
    )


def test_verify_wires_metrics_required_ok_rates() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "northset_metrics_required_finite_ok_rates_honesty_errors" in src
