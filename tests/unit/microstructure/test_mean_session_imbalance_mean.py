"""mean_session_imbalance_mean receipt — session L2 path mean ≠ daily imbalance."""

from __future__ import annotations

import math

from quant_fund.config import load_config
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset


def test_northset_stamps_mean_session_imbalance_mean_when_session_l2() -> None:
    cfg = load_config("configs/research.yaml")
    cfg.data.source = "synthetic"
    cfg.northset.use_session_l2 = True
    cfg.northset.require_adjusted_ohlc = False
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=21).get_bars()
    receipt = bench_northset(bars, cfg)
    assert "mean_session_imbalance_mean" in receipt
    # When session L2 joins, path mean should be finite; never treat as daily imb.
    val = float(receipt["mean_session_imbalance_mean"])
    assert math.isfinite(val)
    assert -1.0 <= val <= 1.0


def test_mean_session_imbalance_mean_nan_without_column_key() -> None:
    """Honesty: key always present on receipt; NaN if session col absent."""
    cfg = load_config("configs/research.yaml")
    cfg.data.source = "synthetic"
    cfg.northset.use_session_l2 = False
    cfg.northset.require_adjusted_ohlc = False
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=22).get_bars()
    receipt = bench_northset(bars, cfg)
    assert "mean_session_imbalance_mean" in receipt


def test_session_imbalance_mean_honesty_helper() -> None:
    from quant_fund.research.catalog import northset_session_imbalance_mean_honesty_errors

    assert northset_session_imbalance_mean_honesty_errors({}) == []
    assert (
        northset_session_imbalance_mean_honesty_errors(
            {"mean_session_imbalance_mean": float("nan")}
        )
        == []
    )
    assert (
        northset_session_imbalance_mean_honesty_errors({"mean_session_imbalance_mean": 0.25}) == []
    )
    errs = northset_session_imbalance_mean_honesty_errors({"mean_session_imbalance_mean": 1.5})
    assert "mean_session_imbalance_mean_out_of_unit_interval" in errs
