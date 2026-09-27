"""Receipt stamps: mean_microprice_minus_mid(+_bps), mean_notional_imbalance, mean_tob_notional_share."""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    mean_microprice_minus_mid_honesty_errors,
    mean_notional_imbalance_honesty_errors,
    mean_tob_notional_share_honesty_errors,
)

_KEYS = (
    "mean_microprice_minus_mid",
    "mean_microprice_minus_mid_bps",
    "mean_notional_imbalance",
    "mean_tob_notional_share",
)


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return cfg


def test_bench_source_stamps_microprice_notional_tob() -> None:
    src = inspect.getsource(bench_northset)
    for key in _KEYS:
        assert f'"{key}"' in src


def test_synth_receipt_stamps_microprice_notional_tob_honest() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), _cfg()
    )
    mid = float(receipt["mean_microprice_minus_mid"])
    bps = float(receipt["mean_microprice_minus_mid_bps"])
    ni = float(receipt["mean_notional_imbalance"])
    tob = float(receipt["mean_tob_notional_share"])
    assert math.isfinite(mid)
    assert math.isfinite(bps)
    assert math.isfinite(ni) and -1.0 <= ni <= 1.0
    assert math.isfinite(tob) and 0.0 < tob <= 1.0
    assert mean_microprice_minus_mid_honesty_errors(receipt) == []
    assert mean_notional_imbalance_honesty_errors(receipt) == []
    assert mean_tob_notional_share_honesty_errors(receipt) == []


def test_honesty_rejects_bad_microprice_notional_tob() -> None:
    assert "mean_microprice_minus_mid_non_finite_fail_closed" in (
        mean_microprice_minus_mid_honesty_errors({"mean_microprice_minus_mid": float("inf")})
    )
    assert mean_notional_imbalance_honesty_errors({"mean_notional_imbalance": 1.5}) == [
        "mean_notional_imbalance_out_of_unit_interval"
    ]
    assert mean_tob_notional_share_honesty_errors({"mean_tob_notional_share": 0.0}) == [
        "mean_tob_notional_share_out_of_open_unit_interval"
    ]
