"""sweep_*_control_sample_adequate True ⇒ control n_dates/p/t companions honest."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_sweep_control_sample_adequate_honesty_errors,
)


def test_adequate_false_skips() -> None:
    assert (
        northset_sweep_control_sample_adequate_honesty_errors(
            {
                "sweep_reject_control_sample_adequate": False,
                "sweep_reject_control_diff_p": float("nan"),
            }
        )
        == []
    )


def test_adequate_true_requires_valid_p_and_n() -> None:
    bad = {
        "sweep_follow_control_sample_adequate": True,
        "sweep_follow_control_n_dates": 0,
        "sweep_follow_control_diff_p": 1.5,
        "sweep_follow_control_diff_t": float("inf"),
    }
    errs = northset_sweep_control_sample_adequate_honesty_errors(bad)
    assert "sweep_follow_control_n_dates_lt_one_while_sample_adequate" in errs
    assert "sweep_follow_control_diff_p_invalid_while_sample_adequate" in errs
    assert "sweep_follow_control_diff_t_non_finite_while_sample_adequate" in errs


def test_synth_receipt_and_tuple_wire() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    assert northset_sweep_control_sample_adequate_honesty_errors(receipt) == []
    assert northset_sweep_control_sample_adequate_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
