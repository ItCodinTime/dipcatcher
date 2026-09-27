"""Catch-all *_p_ic∈[0,1] / *_n_dates≥0 + session_close / sweep / VoR IC packs."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_all_n_dates_nonneg_honesty_errors,
    northset_all_p_ic_unit_interval_honesty_errors,
    northset_session_close_ic_packs_honesty_errors,
    northset_sweep_signed_ic_packs_honesty_errors,
    northset_volume_over_range_ic_packs_honesty_errors,
)


def _synth_receipt():
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    return bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)


def test_catchall_flags_bad_p_ic_and_n_dates() -> None:
    assert "wick_skew_p_ic_out_of_unit_interval" in (
        northset_all_p_ic_unit_interval_honesty_errors({"wick_skew_p_ic": 1.2})
    )
    assert "spread_bps_n_dates_negative" in (
        northset_all_n_dates_nonneg_honesty_errors({"spread_bps_n_dates": -1})
    )
    # best_feature lane skipped
    assert northset_all_p_ic_unit_interval_honesty_errors({"best_feature_p_ic": 1.5}) == []


def test_catchall_and_packs_clean_on_synth() -> None:
    receipt = _synth_receipt()
    assert northset_all_p_ic_unit_interval_honesty_errors(receipt) == []
    assert northset_all_n_dates_nonneg_honesty_errors(receipt) == []
    assert northset_session_close_ic_packs_honesty_errors(receipt) == []
    assert northset_sweep_signed_ic_packs_honesty_errors(receipt) == []
    assert northset_volume_over_range_ic_packs_honesty_errors(receipt) == []
    for fn in (
        northset_all_p_ic_unit_interval_honesty_errors,
        northset_all_n_dates_nonneg_honesty_errors,
        northset_session_close_ic_packs_honesty_errors,
        northset_sweep_signed_ic_packs_honesty_errors,
        northset_volume_over_range_ic_packs_honesty_errors,
    ):
        assert fn in NORTHSET_RECEIPT_HONESTY_HELPERS


def test_session_close_pack_flags_bad_p() -> None:
    assert "session_close_mid_p_ic_out_of_unit_interval" in (
        northset_session_close_ic_packs_honesty_errors({"session_close_mid_p_ic": 1.1})
    )
