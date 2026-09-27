"""use_session_l2 ↔ session_l2_identity_gate stamp consistency."""

from __future__ import annotations

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import bench_northset
from quant_fund.research.catalog import (
    NORTHSET_RECEIPT_HONESTY_HELPERS,
    northset_use_session_l2_gate_consistency_errors,
)


def _cfg(*, session_l2: bool) -> AppConfig:
    cfg = AppConfig()
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = session_l2
    return cfg


def test_synth_session_l2_on_gate_enforced() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        _cfg(session_l2=True),
    )
    assert receipt["use_session_l2"] is True
    assert receipt["session_l2_identity_gate"] == "enforced"
    assert northset_use_session_l2_gate_consistency_errors(receipt) == []


def test_synth_session_l2_off_gate_skipped() -> None:
    receipt = bench_northset(
        SyntheticMarketProvider(n_assets=4, n_days=16, seed=2).get_bars(),
        _cfg(session_l2=False),
    )
    assert receipt["use_session_l2"] is False
    assert receipt["session_l2_identity_gate"] == "skipped"
    assert northset_use_session_l2_gate_consistency_errors(receipt) == []


def test_true_gate_not_enforced_fail_closed() -> None:
    assert "use_session_l2_true_gate_not_enforced" in (
        northset_use_session_l2_gate_consistency_errors(
            {"use_session_l2": True, "session_l2_identity_gate": "skipped"}
        )
    )


def test_false_gate_not_skipped_fail_closed() -> None:
    assert "use_session_l2_false_gate_not_skipped" in (
        northset_use_session_l2_gate_consistency_errors(
            {"use_session_l2": False, "session_l2_identity_gate": "enforced"}
        )
    )


def test_incomplete_pair_skipped() -> None:
    assert northset_use_session_l2_gate_consistency_errors({"use_session_l2": True}) == []
    assert (
        northset_use_session_l2_gate_consistency_errors({"session_l2_identity_gate": "enforced"})
        == []
    )


def test_helper_registered() -> None:
    assert northset_use_session_l2_gate_consistency_errors in NORTHSET_RECEIPT_HONESTY_HELPERS
