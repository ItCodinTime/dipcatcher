"""mean_imbalance_top stamp + mean_fwd_ret_after_* soft-verify."""

from __future__ import annotations

import inspect

from quant_fund.northset import benches
from quant_fund.research.catalog import (
    mean_imbalance_top_honesty_errors,
    northset_fwd_ret_after_sweep_honesty_errors,
)


def test_bench_stamps_mean_imbalance_top() -> None:
    src = inspect.getsource(benches.bench_northset)
    assert '"mean_imbalance_top"' in src


def test_imbalance_top_honesty() -> None:
    assert mean_imbalance_top_honesty_errors({"mean_imbalance_top": -1.0}) == []
    assert "mean_imbalance_top_out_of_unit_interval" in mean_imbalance_top_honesty_errors(
        {"mean_imbalance_top": 1.01}
    )


def test_fwd_ret_after_sweep_honesty() -> None:
    assert (
        northset_fwd_ret_after_sweep_honesty_errors(
            {
                "mean_fwd_ret_after_high_reclaim": -0.002,
                "mean_fwd_ret_after_low_follow": 0.001,
            }
        )
        == []
    )
    assert "mean_fwd_ret_after_high_follow_non_finite" in (
        northset_fwd_ret_after_sweep_honesty_errors(
            {"mean_fwd_ret_after_high_follow": float("inf")}
        )
    )
