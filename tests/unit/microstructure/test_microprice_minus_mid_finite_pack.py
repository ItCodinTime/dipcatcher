"""microprice_minus_mid(+bps) means finite; candle IC⇒mean when scored."""

from __future__ import annotations

from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    candle_microprice_minus_mid_finite_pack_honesty_errors,
    candle_ofi_qp_slope_ic_implies_mean_honesty_errors,
    mean_microprice_minus_mid_honesty_errors,
)


def test_means_fail_closed() -> None:
    assert "mean_microprice_minus_mid_non_finite_fail_closed" in (
        mean_microprice_minus_mid_honesty_errors({"mean_microprice_minus_mid": float("inf")})
    )
    assert "mean_microprice_minus_mid_bps_non_finite_fail_closed" in (
        candle_microprice_minus_mid_finite_pack_honesty_errors(
            {"mean_microprice_minus_mid_bps": float("-inf")}
        )
    )


def test_synth_candle_and_northset() -> None:
    candle = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_microprice_minus_mid_finite_pack_honesty_errors(candle) == []
    assert candle_ofi_qp_slope_ic_implies_mean_honesty_errors(candle) == []
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    north = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert mean_microprice_minus_mid_honesty_errors(north) == []


def test_ic_implies_mean_microprice() -> None:
    bad = {
        "family": "candle_order_book",
        "ic_microprice_minus_mid": 0.1,
        # mean missing
    }
    assert "mean_microprice_minus_mid_missing_while_ic_microprice_minus_mid_scored" in (
        candle_ofi_qp_slope_ic_implies_mean_honesty_errors(bad)
    )


def test_verify_wires_both_families() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert 'mean_microprice_minus_mid_honesty_errors(families.get("candle_order_book"))' in src
    assert 'mean_microprice_minus_mid_honesty_errors(families.get("northset"))' in src
