"""sweep_evidence_scope stamp-contract soft-verify (align allowed enums + companions)."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import northset_sweep_evidence_scope_honesty_errors


def test_synth_sweep_evidence_scope_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert receipt.get("sweep_evidence_scope") == "synthetic"
    assert receipt.get("data_source") == "SYNTHETIC"
    assert receipt.get("yang_zhang_qlike_scope") == "per_security_expanding_oos"
    assert receipt.get("vpin_method") == "count_window_bulk_ofi_proxy"
    assert receipt.get("sweep_inference_index") == "calendar_including_idle_zeros"
    assert northset_sweep_evidence_scope_honesty_errors(receipt) == []


def test_fixture_raw_unadjusted_allowed() -> None:
    assert (
        northset_sweep_evidence_scope_honesty_errors(
            {
                "sweep_evidence_scope": "fixture_raw_unadjusted",
                "price_basis": "raw_fixture_opt_out",
                "data_source": "alpaca",
            }
        )
        == []
    )


def test_legacy_vendor_scope_fail_closed() -> None:
    assert "sweep_evidence_scope_invalid" in northset_sweep_evidence_scope_honesty_errors(
        {"sweep_evidence_scope": "vendor"}
    )


def test_synthetic_data_source_requires_synthetic_scope() -> None:
    assert "sweep_evidence_scope_not_synthetic_despite_SYNTHETIC_data_source" in (
        northset_sweep_evidence_scope_honesty_errors(
            {
                "sweep_evidence_scope": "fixture_raw_unadjusted",
                "data_source": "SYNTHETIC",
            }
        )
    )


def test_empirical_adjusted_ok() -> None:
    assert (
        northset_sweep_evidence_scope_honesty_errors(
            {
                "sweep_evidence_scope": "empirical_adjusted",
                "price_basis": "split_adjusted",
                "data_source": "alpaca",
            }
        )
        == []
    )


def test_northset_requires_yang_zhang_and_vpin_honesty_stamps() -> None:
    errs = northset_sweep_evidence_scope_honesty_errors(
        {
            "family": "northset",
            "yang_zhang_qlike_vs_cc": 0.1,
            "vpin_mean": 0.2,
        }
    )
    assert "yang_zhang_qlike_scope_invalid" in errs
    assert "vpin_method_invalid" in errs


def test_northset_requires_cs_pair_overnight_gap_and_two_way_index_stamps() -> None:
    errs = northset_sweep_evidence_scope_honesty_errors(
        {
            "family": "northset",
            "corwin_schultz_spread": 0.01,
            "sweep_follow_overnight_gap_p": 0.04,
            "sweep_follow_two_way_cluster_p": 0.03,
        }
    )
    assert "corwin_schultz_pair_scope_invalid" in errs
    assert "sweep_overnight_gap_method_invalid" in errs
    assert "sweep_two_way_inference_index_invalid" in errs


def test_dm_park_suite_still_imports() -> None:
    # registry smoke — helper still exported for CoS dm_park suite
    from quant_fund.research.catalog import NORTHSET_RECEIPT_HONESTY_HELPERS

    assert northset_sweep_evidence_scope_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
