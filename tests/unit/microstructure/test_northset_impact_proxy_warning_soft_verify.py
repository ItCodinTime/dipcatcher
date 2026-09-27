"""northset impact_proxy_warning stamp-contract soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_impact_proxy_warning_honesty_errors,
)

_EXPECTED = "depth_or_ofi_proxy_not_signed_trade_flow"


def test_synth_stamps_expected_token() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert receipt.get("impact_proxy_warning") == _EXPECTED
    assert northset_impact_proxy_warning_honesty_errors(receipt) == []


def test_unexpected_token_fail_closed() -> None:
    assert northset_impact_proxy_warning_honesty_errors(
        {"family": "northset", "impact_proxy_warning": "signed_trade_flow"}
    ) == ["impact_proxy_warning_unexpected_token"]


def test_non_string_fail_closed() -> None:
    assert northset_impact_proxy_warning_honesty_errors(
        {"family": "northset", "impact_proxy_warning": 1}
    ) == ["impact_proxy_warning_not_nonempty_str"]
    assert northset_impact_proxy_warning_honesty_errors(
        {"family": "northset", "impact_proxy_warning": ""}
    ) == ["impact_proxy_warning_not_nonempty_str"]


def test_absent_or_other_family_skipped() -> None:
    assert northset_impact_proxy_warning_honesty_errors({"family": "northset"}) == []
    assert (
        northset_impact_proxy_warning_honesty_errors(
            {"family": "candle_order_book", "impact_proxy_warning": "nope"}
        )
        == []
    )


def test_helper_registered() -> None:
    assert northset_impact_proxy_warning_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
