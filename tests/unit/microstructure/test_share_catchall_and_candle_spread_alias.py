"""*_share ∈[0,1] catch-all + candle quoted/effective/half/bps identities."""

from __future__ import annotations

from pathlib import Path

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.microstructure.bench import bench_candle_order_book
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    candle_spread_alias_honesty_errors,
    northset_all_share_unit_honesty_errors,
)


def test_share_catchall_on_synth() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_all_share_unit_honesty_errors(receipt) == []
    assert northset_all_share_unit_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert "sweep_low_reclaim_share_out_of_unit_interval" in (
        northset_all_share_unit_honesty_errors({"sweep_low_reclaim_share": 1.5})
    )


def test_candle_spread_alias_identities() -> None:
    receipt = bench_candle_order_book(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=3).get_bars()
    )
    assert candle_spread_alias_honesty_errors(receipt) == []
    bad = dict(receipt)
    bad["mean_half_spread"] = float(receipt["mean_quoted_spread"])
    assert "mean_half_spread_not_half_of_mean_quoted_spread" in (
        candle_spread_alias_honesty_errors(bad)
    )


def test_verify_wires_candle_spread_alias() -> None:
    src = Path("src/quant_fund/research/verify.py").read_text(encoding="utf-8")
    assert "candle_spread_alias_honesty_errors" in src
