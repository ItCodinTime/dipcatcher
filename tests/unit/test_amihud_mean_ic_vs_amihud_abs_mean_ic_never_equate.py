"""amihud_mean_ic ≠ amihud_abs_mean_ic — never-equate."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors,
)


def test_keys_and_companions_distinct() -> None:
    assert "amihud_mean_ic" != "amihud_abs_mean_ic"
    assert "amihud_p_ic" != "amihud_abs_p_ic"
    assert "amihud_t_ic" != "amihud_abs_t_ic"
    assert "amihud_n_dates" != "amihud_abs_n_dates"


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "amihud_mean_ic" in receipt
    assert "amihud_abs_mean_ic" in receipt
    assert amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors({"amihud_mean_ic": 0.1})
        == []
    )


def test_helper_registered() -> None:
    assert (
        amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
