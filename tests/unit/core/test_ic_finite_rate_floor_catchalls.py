"""Catch-alls: *_t_ic / *_mean_ic finite; *_mean_rank_ic∈[-1,1]; *_finite_rate/_floor∈[0,1]."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_all_finite_rate_unit_honesty_errors,
    northset_all_floor_unit_honesty_errors,
    northset_all_mean_ic_finite_honesty_errors,
    northset_all_mean_rank_ic_unit_honesty_errors,
    northset_all_t_ic_finite_honesty_errors,
)


def _synth():
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    return bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)


def test_catchalls_flag_bad_values() -> None:
    assert "ofi_t_ic_non_finite_fail_closed" in (
        northset_all_t_ic_finite_honesty_errors({"ofi_t_ic": float("inf")})
    )
    assert "amihud_mean_ic_non_finite_fail_closed" in (
        northset_all_mean_ic_finite_honesty_errors({"amihud_mean_ic": float("-inf")})
    )
    assert "ofi_mean_rank_ic_out_of_unit_interval" in (
        northset_all_mean_rank_ic_unit_honesty_errors({"ofi_mean_rank_ic": 1.5})
    )
    assert "structure_finite_rate_out_of_unit_interval" in (
        northset_all_finite_rate_unit_honesty_errors({"structure_finite_rate": 1.2})
    )
    assert "session_l2_identity_floor_out_of_unit_interval" in (
        northset_all_floor_unit_honesty_errors({"session_l2_identity_floor": -0.1})
    )


def test_catchalls_clean_on_synth_and_wired() -> None:
    receipt = _synth()
    assert northset_all_t_ic_finite_honesty_errors(receipt) == []
    assert northset_all_mean_ic_finite_honesty_errors(receipt) == []
    assert northset_all_mean_rank_ic_unit_honesty_errors(receipt) == []
    assert northset_all_finite_rate_unit_honesty_errors(receipt) == []
    assert northset_all_floor_unit_honesty_errors(receipt) == []
    for fn in (
        northset_all_t_ic_finite_honesty_errors,
        northset_all_mean_ic_finite_honesty_errors,
        northset_all_mean_rank_ic_unit_honesty_errors,
        northset_all_finite_rate_unit_honesty_errors,
        northset_all_floor_unit_honesty_errors,
    ):
        assert fn in NORTHSET_RECEIPT_HONESTY_HELPERS
