"""H22 imbalance_top_p_ic ≠ session_reconstructs_daily_rate — never-equate."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H22_HYPOTHESIS_ID,
    H23_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors,
    northset_has_finite_imbalance_p_ic,
)


def test_keys_gates_and_h23_bind() -> None:
    assert "imbalance_top_p_ic" != "session_reconstructs_daily_rate"
    assert H22_HYPOTHESIS_ID != H23_HYPOTHESIS_ID
    by_key = {s[0]: s[1] for s in NORTHSET_H23_H28_SPECS}
    assert by_key["session_reconstructs_daily_rate"] == H23_HYPOTHESIS_ID
    assert by_key.get("imbalance_top_p_ic") != H23_HYPOTHESIS_ID
    elig = {"book_hypothesis_eligible": True}
    assert northset_has_finite_imbalance_p_ic({"imbalance_top_p_ic": 0.05, **elig})
    assert not northset_has_finite_imbalance_p_ic({"session_reconstructs_daily_rate": 1.0, **elig})


def test_synth_both_stamped_never_equate_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(),
        cfg,
    )
    assert "imbalance_top_p_ic" in receipt
    assert "session_reconstructs_daily_rate" in receipt
    assert imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors(receipt) == []


def test_one_key_absent_skipped() -> None:
    assert (
        imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors(
            {"imbalance_top_p_ic": 0.05}
        )
        == []
    )
    assert (
        imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors(
            {"session_reconstructs_daily_rate": 1.0}
        )
        == []
    )


def test_helper_registered() -> None:
    assert (
        imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
