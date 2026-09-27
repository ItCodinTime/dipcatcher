"""close_location_value_* IC aliases must match clv_* when both stamped."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    close_location_value_clv_alias_identity_honesty_errors,
)


def test_keys_distinct() -> None:
    assert "close_location_value_p_ic" != "clv_p_ic"
    assert "close_location_value_t_ic" != "clv_t_ic"


def test_synth_aliases_match() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "close_location_value_p_ic" in receipt
    assert "clv_p_ic" in receipt
    assert "close_location_value_t_ic" in receipt
    assert "clv_t_ic" in receipt
    assert close_location_value_clv_alias_identity_honesty_errors(receipt) == []


def test_mismatch_fail_closed() -> None:
    errs = close_location_value_clv_alias_identity_honesty_errors(
        {
            "close_location_value_p_ic": 0.1,
            "clv_p_ic": 0.2,
            "close_location_value_t_ic": 1.0,
            "clv_t_ic": 1.0,
        }
    )
    assert "close_location_value_p_ic_clv_p_ic_mismatch" in errs
    assert "close_location_value_t_ic_clv_t_ic_mismatch" not in errs


def test_one_side_absent_skipped() -> None:
    assert (
        close_location_value_clv_alias_identity_honesty_errors({"close_location_value_p_ic": 0.1})
        == []
    )
    assert close_location_value_clv_alias_identity_honesty_errors({"clv_p_ic": 0.1}) == []


def test_helper_registered() -> None:
    assert (
        close_location_value_clv_alias_identity_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
