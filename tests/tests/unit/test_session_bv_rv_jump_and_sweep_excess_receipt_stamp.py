"""Receipt stamps: session_mean_bv/rv/jump + sweep event excess (mean_excess_bps path).

Lt already owns half/quoted/effective via test_northset_spread_means_receipt_stamp.py.
Bare mean_bv / mean_jump_ratio / mean_rv / mean_diff_bps / mean_excess_bps are
intermediate-only — receipt stamps are session_mean_* and sweep_*_event_mean_bps /
sweep_*_control_diff_mean_bps.
"""

from __future__ import annotations

import inspect
import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    northset_overnight_rv_semi_honesty_errors,
    session_mean_jump_ratio_honesty_errors,
    sweep_follow_event_mean_bps_honesty_errors,
    sweep_reject_event_mean_bps_honesty_errors,
)

_SESSION_KEYS = (
    "session_mean_bv",
    "session_mean_rv",
    "session_mean_jump_ratio",
)
_SWEEP_EXCESS_KEYS = (
    "sweep_reject_event_mean_bps",
    "sweep_follow_event_mean_bps",
)
_SWEEP_DIFF_KEYS = (
    "sweep_reject_control_diff_mean_bps",
    "sweep_follow_control_diff_mean_bps",
)


def _synth_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return cfg


def test_bench_source_stamps_session_bv_rv_jump_not_bare_mean_bv() -> None:
    src = inspect.getsource(bench_northset)
    for key in _SESSION_KEYS:
        assert f'"{key}"' in src, key
    # intermediate nan_jump uses bare names; receipt must use session_mean_*
    assert '"session_mean_bv"' in src
    assert '"session_mean_jump_ratio"' in src


def test_bench_source_stamps_sweep_excess_and_diff_from_nested_means() -> None:
    src = inspect.getsource(bench_northset)
    assert "mean_excess_bps" in src
    assert "mean_diff_bps" in src
    assert "event_mean_bps" in src
    assert "control_diff_mean_bps" in src


def test_synth_receipt_session_bv_rv_jump_honest() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()
    receipt = bench_northset(bars, _synth_cfg())
    bv = float(receipt["session_mean_bv"])
    rv = float(receipt["session_mean_rv"])
    jr = float(receipt["session_mean_jump_ratio"])
    assert math.isfinite(bv) and bv >= 0.0
    assert math.isfinite(rv) and rv >= 0.0
    assert math.isfinite(jr) and 0.0 <= jr <= 1.0
    # bare intermediates must not leak onto receipt
    assert "mean_bv" not in receipt
    assert "mean_jump_ratio" not in receipt
    assert northset_overnight_rv_semi_honesty_errors(receipt) == []
    assert session_mean_jump_ratio_honesty_errors(receipt) == []


def test_synth_receipt_stamps_sweep_excess_and_diff_keys() -> None:
    bars = SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars()
    receipt = bench_northset(bars, _synth_cfg())
    for key in _SWEEP_EXCESS_KEYS + _SWEEP_DIFF_KEYS:
        assert key in receipt, key
        val = float(receipt[key])
        # NaN OK (honesty skips); finite must not be ±inf
        if math.isfinite(val):
            assert abs(val) != float("inf")
    assert sweep_reject_event_mean_bps_honesty_errors(receipt) == []
    assert sweep_follow_event_mean_bps_honesty_errors(receipt) == []


def test_honesty_rejects_bad_session_and_sweep_excess() -> None:
    assert "session_mean_bv_negative" in northset_overnight_rv_semi_honesty_errors(
        {"session_mean_bv": -0.01}
    )
    assert session_mean_jump_ratio_honesty_errors({"session_mean_jump_ratio": 1.5}) == [
        "session_mean_jump_ratio_out_of_unit_interval"
    ]
    assert sweep_follow_event_mean_bps_honesty_errors(
        {"sweep_follow_event_mean_bps": float("inf")}
    ) == ["sweep_follow_event_mean_bps_non_finite"]
