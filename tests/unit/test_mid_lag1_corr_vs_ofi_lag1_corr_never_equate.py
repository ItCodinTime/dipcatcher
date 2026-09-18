"""mid_lag1_corr ≠ ofi_lag1_corr — never-equate (always-on panel lag1)."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors,
)


def test_keys_distinct() -> None:
    assert "mid_lag1_corr" != "ofi_lag1_corr"
    assert "mid_lag1_n_securities" != "ofi_lag1_n_securities"


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "mid_lag1_corr" in receipt
    assert "ofi_lag1_corr" in receipt
    assert mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors({"mid_lag1_corr": 0.5}) == []


def test_helper_registered() -> None:
    assert (
        mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
