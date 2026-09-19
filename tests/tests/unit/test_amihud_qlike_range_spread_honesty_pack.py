"""amihud_mean ≥0; qlike means ≥0; corwin/abdi ∈[0,1]; roll ≥0."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    amihud_mean_honesty_errors,
    northset_qlike_means_honesty_errors,
    northset_range_spread_honesty_errors,
)


def test_amihud_qlike_range_fail_closed() -> None:
    assert amihud_mean_honesty_errors({"amihud_mean": -1e-9}) == ["amihud_mean_negative"]
    assert "parkinson_qlike_vs_cc_negative" in northset_qlike_means_honesty_errors(
        {"parkinson_qlike_vs_cc": -0.1}
    )
    assert "corwin_schultz_spread_out_of_unit_interval" in (
        northset_range_spread_honesty_errors({"corwin_schultz_spread": 1.5})
    )
    assert "abdi_ranaldo_spread_out_of_unit_interval" in (
        northset_range_spread_honesty_errors({"abdi_ranaldo_spread": -0.01})
    )
    assert "roll_spread_negative" in northset_range_spread_honesty_errors({"roll_spread": -0.01})
    # Roll may exceed 1
    assert northset_range_spread_honesty_errors({"roll_spread": 1.3}) == []


def test_synth_pack_and_tuple_wire() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert amihud_mean_honesty_errors(receipt) == []
    assert northset_qlike_means_honesty_errors(receipt) == []
    assert northset_range_spread_honesty_errors(receipt) == []
    assert amihud_mean_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert northset_qlike_means_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert northset_range_spread_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
