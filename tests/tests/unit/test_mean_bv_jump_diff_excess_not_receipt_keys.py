"""Contract: bare mean_bv / mean_jump_ratio / mean_diff_bps / mean_excess_bps are NOT receipt keys.

They appear only as session_bipower_jump intermediates or nested sweep fields.
Receipt companions are session_mean_* and sweep_*_event_mean_bps /
sweep_*_control_diff_mean_bps (covered in
test_session_bv_rv_jump_and_sweep_excess_receipt_stamp.py).
"""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset

_BARE = (
    "mean_bv",
    "mean_jump_ratio",
    "mean_rv",
    "mean_diff_bps",
    "mean_excess_bps",
)


def test_synth_receipt_omits_bare_bv_jump_diff_excess_keys() -> None:
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    receipt = bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)
    for key in _BARE:
        assert key not in receipt, key
    # Real stamped companions exist
    assert "session_mean_bv" in receipt
    assert "session_mean_jump_ratio" in receipt
    assert "sweep_follow_event_mean_bps" in receipt
    assert "sweep_follow_control_diff_mean_bps" in receipt
