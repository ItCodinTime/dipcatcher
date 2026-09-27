"""YZ variance ≥0; park/gk/rs QLIKE ≥0; overnight share + RV/BV/semi ≥0 pack."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_overnight_rv_semi_honesty_errors,
    northset_qlike_means_honesty_errors,
    northset_range_spread_honesty_errors,
)


def test_negatives_fail_closed() -> None:
    assert "yang_zhang_variance_negative" in northset_range_spread_honesty_errors(
        {"yang_zhang_variance": -1e-9}
    )
    assert "parkinson_qlike_vs_cc_negative" in northset_qlike_means_honesty_errors(
        {"parkinson_qlike_vs_cc": -0.01}
    )
    assert "garman_klass_qlike_vs_cc_negative" in northset_qlike_means_honesty_errors(
        {"garman_klass_qlike_vs_cc": -0.01}
    )
    assert "rogers_satchell_qlike_vs_cc_negative" in (
        northset_qlike_means_honesty_errors({"rogers_satchell_qlike_vs_cc": -0.01})
    )
    assert "session_mean_rv_negative" in northset_overnight_rv_semi_honesty_errors(
        {"session_mean_rv": -0.01}
    )
    assert "session_mean_bv_negative" in northset_overnight_rv_semi_honesty_errors(
        {"session_mean_bv": -0.01}
    )
    assert "semi_up_negative" in northset_overnight_rv_semi_honesty_errors({"semi_up": -0.01})


def test_synth_pack_and_tuple() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert "yang_zhang_variance" in receipt
    assert northset_range_spread_honesty_errors(receipt) == []
    assert northset_qlike_means_honesty_errors(receipt) == []
    assert northset_overnight_rv_semi_honesty_errors(receipt) == []
    assert northset_overnight_rv_semi_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert northset_qlike_means_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert northset_range_spread_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
