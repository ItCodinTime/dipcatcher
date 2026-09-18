"""include_kyle_ofi ↔ kyle_ofi nest presence soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_include_kyle_ofi_nest_presence_honesty_errors,
)


def _cfg(*, include_kyle: bool) -> AppConfig:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.include_kyle_ofi = include_kyle
    return cfg


def test_synth_include_kyle_false_nest_absent() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        _cfg(include_kyle=False),
    )
    assert receipt.get("include_kyle_ofi") is False
    assert "kyle_ofi" not in receipt
    assert northset_include_kyle_ofi_nest_presence_honesty_errors(receipt) == []


def test_synth_include_kyle_true_nest_present() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(),
        _cfg(include_kyle=True),
    )
    assert receipt.get("include_kyle_ofi") is True
    assert isinstance(receipt.get("kyle_ofi"), dict)
    assert receipt["kyle_ofi"].get("research_only") is True
    assert northset_include_kyle_ofi_nest_presence_honesty_errors(receipt) == []


def test_true_missing_nest_fail_closed() -> None:
    assert "kyle_ofi_nest_missing_while_include_kyle_ofi_true" in (
        northset_include_kyle_ofi_nest_presence_honesty_errors(
            {"family": "northset", "include_kyle_ofi": True}
        )
    )


def test_false_with_nest_fail_closed() -> None:
    assert "kyle_ofi_nest_present_while_include_kyle_ofi_false" in (
        northset_include_kyle_ofi_nest_presence_honesty_errors(
            {
                "family": "northset",
                "include_kyle_ofi": False,
                "kyle_ofi": {"research_only": True},
            }
        )
    )


def test_helper_registered() -> None:
    assert (
        northset_include_kyle_ofi_nest_presence_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
