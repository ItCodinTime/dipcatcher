"""H24 volume conservation ≠ H23 reconstructs — never-equate polish."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H23_HYPOTHESIS_ID,
    H24_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    session_volume_conservation_vs_reconstructs_never_equate_honesty_errors,
)


def test_keys_and_hypothesis_ids_distinct() -> None:
    assert "session_reconstructs_daily_rate" != "session_volume_conservation_rate"
    assert H23_HYPOTHESIS_ID != H24_HYPOTHESIS_ID
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["session_reconstructs_daily_rate"] == H23_HYPOTHESIS_ID
    assert by_key["session_volume_conservation_rate"] == H24_HYPOTHESIS_ID


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "session_reconstructs_daily_rate" in receipt
    assert "session_volume_conservation_rate" in receipt
    # Values may both be 1.0 on synth — never-equate is key/hypothesis identity
    assert session_volume_conservation_vs_reconstructs_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        session_volume_conservation_vs_reconstructs_never_equate_honesty_errors(
            {"session_reconstructs_daily_rate": 1.0}
        )
        == []
    )


def test_helper_registered() -> None:
    assert (
        session_volume_conservation_vs_reconstructs_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
