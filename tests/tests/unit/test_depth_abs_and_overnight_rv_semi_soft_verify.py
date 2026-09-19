"""Stamp/soft-verify mean_depth_imbalance_abs + overnight/rv/semi companions."""

from __future__ import annotations

import inspect

from quant_fund.northset import benches
from quant_fund.research.catalog import (
    mean_depth_imbalance_abs_honesty_errors,
    northset_overnight_rv_semi_honesty_errors,
)


def test_bench_stamps_mean_depth_imbalance_abs() -> None:
    src = inspect.getsource(benches.bench_northset)
    assert '"mean_depth_imbalance_abs"' in src


def test_depth_imbalance_abs_honesty() -> None:
    assert mean_depth_imbalance_abs_honesty_errors({"mean_depth_imbalance_abs": 0.0}) == []
    assert "mean_depth_imbalance_abs_out_of_unit_interval" in (
        mean_depth_imbalance_abs_honesty_errors({"mean_depth_imbalance_abs": -0.01})
    )


def test_overnight_rv_semi_honesty() -> None:
    assert (
        northset_overnight_rv_semi_honesty_errors(
            {
                "overnight_share": 0.5,
                "session_mean_rv": 0.0,
                "session_mean_bv": 1.0,
                "semi_up": 0.1,
                "semi_down": 0.2,
            }
        )
        == []
    )
    assert "semi_up_negative" in northset_overnight_rv_semi_honesty_errors({"semi_up": -0.1})
