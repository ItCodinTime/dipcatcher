"""Stamped northset receipt bool flags soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_receipt_bool_flags_honesty_errors,
    northset_receipt_honesty_errors,
)


def test_synth_receipt_bool_flags_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=20, seed=7).get_bars(),
        cfg,
    )
    for key in (
        "metrics_required_finite_ok",
        "shape_columns_ensured",
        "use_session_l2",
        "research_only",
        "include_kyle_ofi",
    ):
        assert key in receipt
        assert type(receipt[key]) is bool
    assert northset_receipt_bool_flags_honesty_errors(receipt) == []
    assert not any(e.endswith("_not_bool") for e in northset_receipt_honesty_errors(receipt))


def test_metrics_required_finite_ok_not_bool_fail_closed() -> None:
    assert "metrics_required_finite_ok_not_bool" in northset_receipt_bool_flags_honesty_errors(
        {"metrics_required_finite_ok": 1}
    )


def test_shape_columns_ensured_str_fail_closed() -> None:
    assert "shape_columns_ensured_not_bool" in northset_receipt_bool_flags_honesty_errors(
        {"shape_columns_ensured": "true"}
    )


def test_use_session_l2_none_fail_closed() -> None:
    assert "use_session_l2_not_bool" in northset_receipt_bool_flags_honesty_errors(
        {"use_session_l2": None}
    )


def test_absent_keys_skipped() -> None:
    assert northset_receipt_bool_flags_honesty_errors({}) == []


def test_helper_registered_in_receipt_dispatcher() -> None:
    assert northset_receipt_bool_flags_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
