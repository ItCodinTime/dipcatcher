"""impact_estimator_scope stamp-contract soft-verify."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import northset_sweep_evidence_scope_honesty_errors


def test_synth_impact_estimator_scope_clean() -> None:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.require_adjusted_ohlc = False
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        cfg,
    )
    assert receipt.get("impact_estimator_scope") == "per_security_equal_weight"
    assert northset_sweep_evidence_scope_honesty_errors(receipt) == []


def test_impact_estimator_scope_invalid_fail_closed() -> None:
    assert "impact_estimator_scope_invalid" in northset_sweep_evidence_scope_honesty_errors(
        {"impact_estimator_scope": "portfolio_value_weight"}
    )


def test_empty_impact_estimator_scope_fail_closed() -> None:
    assert "impact_estimator_scope_invalid" in northset_sweep_evidence_scope_honesty_errors(
        {"impact_estimator_scope": ""}
    )


def test_valid_scope_alone_clean() -> None:
    assert (
        northset_sweep_evidence_scope_honesty_errors(
            {"impact_estimator_scope": "per_security_equal_weight"}
        )
        == []
    )
