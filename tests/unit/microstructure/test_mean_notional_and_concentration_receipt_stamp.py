"""Northset stamps mean_notional_imbalance + size concentration means."""

from __future__ import annotations

import inspect

from quant_fund.northset import benches
from quant_fund.research.catalog import (
    mean_notional_imbalance_honesty_errors,
    size_concentration_top_honesty_errors,
)


def test_bench_stamps_notional_and_concentration_means() -> None:
    src = inspect.getsource(benches.bench_northset)
    assert '"mean_notional_imbalance"' in src
    assert '"mean_bid_size_concentration_top"' in src
    assert '"mean_ask_size_concentration_top"' in src


def test_notional_imbalance_honesty() -> None:
    assert mean_notional_imbalance_honesty_errors({"mean_notional_imbalance": -0.5}) == []
    assert "mean_notional_imbalance_out_of_unit_interval" in (
        mean_notional_imbalance_honesty_errors({"mean_notional_imbalance": 1.5})
    )


def test_size_concentration_honesty_on_northset_keys() -> None:
    assert (
        size_concentration_top_honesty_errors(
            {
                "mean_bid_size_concentration_top": 0.4,
                "mean_ask_size_concentration_top": 0.6,
            }
        )
        == []
    )
