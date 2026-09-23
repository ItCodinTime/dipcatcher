"""session_bulk_vpin ≠ vpin_mean / session_book_vpin_mean — VPIN triad never-equate."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    H32_HYPOTHESIS_ID,
    H43_HYPOTHESIS_ID,
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_has_finite_session_book_vpin_p_ic,
    session_bulk_vpin_honesty_errors,
    session_bulk_vpin_vs_siblings_never_equate_honesty_errors,
)


def test_keys_and_hyp_ids_distinct() -> None:
    assert "session_bulk_vpin" != "vpin_mean"
    assert "session_bulk_vpin" != "session_book_vpin_mean"
    assert "vpin_mean" != "session_book_vpin_mean"
    assert H32_HYPOTHESIS_ID != H43_HYPOTHESIS_ID
    assert not northset_has_finite_session_book_vpin_p_ic({"session_bulk_vpin": 0.5})
    assert northset_has_finite_session_book_vpin_p_ic({"session_book_vpin_p_ic": 0.05})


def test_synth_triad_stamped_never_equate_and_unit_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert "session_bulk_vpin" in receipt
    assert "vpin_mean" in receipt
    assert "session_book_vpin_mean" in receipt
    assert session_bulk_vpin_vs_siblings_never_equate_honesty_errors(receipt) == []
    assert session_bulk_vpin_honesty_errors(receipt) == []


def test_bulk_out_of_unit_fail_closed() -> None:
    assert session_bulk_vpin_honesty_errors({"session_bulk_vpin": 1.2}) == [
        "session_bulk_vpin_out_of_unit_interval"
    ]
    assert session_bulk_vpin_honesty_errors({"session_bulk_vpin": float("nan")}) == []


def test_bulk_absent_or_no_siblings_skipped() -> None:
    assert session_bulk_vpin_vs_siblings_never_equate_honesty_errors({}) == []
    assert (
        session_bulk_vpin_vs_siblings_never_equate_honesty_errors({"session_bulk_vpin": 0.3}) == []
    )


def test_helpers_registered() -> None:
    assert session_bulk_vpin_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert (
        session_bulk_vpin_vs_siblings_never_equate_honesty_errors
        in NORTHSET_RECEIPT_HONESTY_HELPERS
    )
