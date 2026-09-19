"""*_rate ∈[0,1] catch-all + nested sweep_evidence honesty."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_all_rate_unit_honesty_errors,
    northset_sweep_evidence_blob_honesty_errors,
)


def _synth():
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    return bench_northset(SyntheticMarketProvider(n_assets=4, n_days=24, seed=5).get_bars(), cfg)


def test_rate_catchall_and_sweep_evidence_on_synth() -> None:
    receipt = _synth()
    assert northset_all_rate_unit_honesty_errors(receipt) == []
    assert northset_sweep_evidence_blob_honesty_errors(receipt) == []
    assert northset_all_rate_unit_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
    assert northset_sweep_evidence_blob_honesty_errors in NORTHSET_RECEIPT_HONESTY_HELPERS


def test_rate_catchall_flags_oob() -> None:
    assert "doji_rate_out_of_unit_interval" in (
        northset_all_rate_unit_honesty_errors({"doji_rate": 1.2})
    )


def test_sweep_evidence_rejects_fdr_without_adequate_sample() -> None:
    blob = {
        "sweep_evidence": {
            "research_only": True,
            "horizons": [1, 5],
            "event_studies": [
                {
                    "sample_adequate": False,
                    "reject_fdr": True,
                    "n_events": 0,
                    "n_dates": 0,
                }
            ],
        }
    }
    errs = northset_sweep_evidence_blob_honesty_errors(blob)
    assert "sweep_evidence_event_studies_0_reject_fdr_with_inadequate_sample" in errs
