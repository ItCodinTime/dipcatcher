"""TrialLedger series recording + data-snooping battery.

Research-diagnostic only — never a live Sharpe / P&L claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.validation.multiple_testing import TrialLedger

T = 260


def _series(seed: int, bump: float = 0.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, size=T) + bump


def _ledger() -> TrialLedger:
    ledger = TrialLedger()
    ledger.record_series("ridge", _series(1, 0.001))
    ledger.record_series("lgbm", _series(2, 0.0005))
    ledger.record_series("oracle", _series(3, 0.008))
    return ledger


def test_record_series_validation() -> None:
    ledger = TrialLedger()
    ledger.record_series("a", _series(1))
    assert ledger.n_series == 1 and ledger.n_trials == 0
    with pytest.raises(ValueError, match="non-empty"):
        ledger.record_series("  ", _series(2))
    with pytest.raises(ValueError, match="duplicate"):
        ledger.record_series("a", _series(3))
    with pytest.raises(ValueError, match="at least 2"):
        ledger.record_series("b", np.array([0.1]))
    with pytest.raises(ValueError, match="finite"):
        ledger.record_series("c", np.array([0.1, np.nan, 0.2]))


def test_snooping_requires_a_universe() -> None:
    ledger = TrialLedger()
    assert ledger.snooping() == {}
    ledger.record_series("only", _series(1))
    assert ledger.snooping() == {}


def test_snooping_rejects_unaligned_series() -> None:
    ledger = TrialLedger()
    ledger.record_series("a", _series(1))
    ledger.record_series("b", _series(2)[:-5])
    with pytest.raises(ValueError, match="align in length"):
        ledger.snooping()


def test_snooping_battery_structure_and_honesty() -> None:
    blob = _ledger().snooping(n_boot=400, seed=9)
    assert blob["research_only"] is True
    assert blob["claim"] == "research_diagnostic_only"
    assert blob["n_trials"] == 3
    assert blob["n_obs"] == T
    assert blob["best_trial"] == "oracle"
    assert blob["best_mean"] > 0.005
    for key in ("reality_check", "spa", "stepm", "mcs"):
        assert key in blob
    assert blob["spa"]["p_consistent"] < 0.05
    assert blob["reality_check"]["p_value"] < 0.05
    assert blob["stepm"]["n_rejected"] >= 1
    assert "oracle" in blob["stepm"]["rejected"]
    assert blob["mcs"]["n_included"] >= 1
    assert blob["mcs"]["included"]
    # Forbidden research keys must never appear (no Sharpe / P&L claims).
    flat = str(blob).lower()
    for token in ("sharpe", "pnl", "live_pnl_claim"):
        assert token not in flat
    # Per-trial tables are keyed by trial name and ordered like the input.
    assert set(blob["stepm"]["adjusted_p"]) == {"ridge", "lgbm", "oracle"}
    assert set(blob["mcs"]["p_values"]) == {"ridge", "lgbm", "oracle"}


def test_snooping_deterministic() -> None:
    a = _ledger().snooping(n_boot=300, seed=4)
    b = _ledger().snooping(n_boot=300, seed=4)
    assert a == b


def test_snooping_null_universe_does_not_reject() -> None:
    ledger = TrialLedger()
    for i in range(5):
        ledger.record_series(f"t{i}", _series(100 + i))
    blob = ledger.snooping(n_boot=400, seed=5)
    assert blob["spa"]["p_consistent"] > 0.05
    assert blob["stepm"]["n_rejected"] == 0
