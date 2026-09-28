"""Path threats, toy sensitivities, and the receipt extension."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from tests.unit.research.test_research_verify import _receipt

from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.verify import _receipt_digest, verify_research_artifact
from quant_fund.robustness.certify import certify
from quant_fund.robustness.schema import (
    ROBUSTNESS_EXTENSION_SCHEMA_VERSION,
    assert_stamp_target_allowed,
    migrate_robustness_view,
    robustness_extension_errors,
    stamp_robustness,
    write_stamped_notebook,
)
from quant_fund.robustness.strategies import FixedSchedule, LinearMargin
from quant_fund.robustness.threats import (
    apply_jitter,
    apply_vol_scaled_prices,
    causal_increment_vol,
    net_excess,
)


def test_vol_scaled_price_shock_edits_log_increments() -> None:
    prices = np.array([100.0, 101.0, 99.0, 102.0])
    eta = np.array([0.1, -0.2, 0.05])
    vol = np.array([0.01, 0.02, 0.01])
    shocked = apply_vol_scaled_prices(prices, eta, vol)
    increments = np.diff(np.log(shocked))
    assert increments == pytest.approx(np.diff(np.log(prices)) + vol * eta)
    assert shocked[0] == pytest.approx(prices[0])
    scale = causal_increment_vol(prices, floor=1e-4)
    assert scale[0] == pytest.approx(1e-4)
    assert np.all(scale[1:] > 0.0)


def test_one_hot_linear_toy_has_known_discrete_radii() -> None:
    strategy = LinearMargin(np.array([1.0, 0.0, 0.0]), name="one_hot")
    sample = np.array([0.5, -0.2, 0.3])
    card = certify(
        strategy,
        sample,
        sigma=0.2,
        draws=32,
        seed=5,
        run_attack=False,
        gradient="none",
        outcome_mean=0.5,
        outcome_scale=1.0,
        reference="gaussian",
        wasserstein_radius=0.25,
        evidence_class="SYNTHETIC",
    )
    assert card["certified_radius"]["status"] == "proven"
    assert card["certified_radius"]["value"] == pytest.approx(0.5)
    assert card["analytic_radius"]["value"] == pytest.approx(0.5)
    assert card["sensitivity"]["spikes"]["status"] == "proven"
    assert card["sensitivity"]["spikes"]["value"] == pytest.approx(0.5)
    assert card["sensitivity"]["missing_bars"]["value"] == pytest.approx(1.0)
    assert card["sensitivity"]["missing_bars"]["status"] == "exact_on_path"
    # Positions are +1 on every bar, turnover from flat is 1, gross is the sum.
    assert card["sensitivity"]["cost_shock"]["value"] == pytest.approx(float(np.sum(sample)))
    assert card["distributional_robustness"]["worst_case_mean"]["value"] == pytest.approx(0.25)
    assert card["distributional_robustness"]["worst_case_ratio"]["status"] == "proven_tight"
    assert card["distributional_robustness"]["radius_to_nonpositive_mean"][
        "value"
    ] == pytest.approx(0.5)
    positions = strategy.positions(sample)
    assert net_excess(positions, sample, 0.0) == pytest.approx(float(np.sum(sample)))


def test_fixed_schedule_flips_after_a_one_bar_roll() -> None:
    returns = np.array([0.0, 0.0, 1.0, 0.0])
    schedule = FixedSchedule(np.array([0.0, 0.0, 1.0, 0.0]))
    assert schedule.decision(returns) == 1
    assert schedule.decision(apply_jitter(returns, 1)) == 0
    card = certify(
        schedule,
        returns,
        draws=16,
        seed=1,
        run_attack=False,
        gradient="none",
        outcome_mean=1.0,
        outcome_scale=1.0,
        reference="gaussian",
        evidence_class="SYNTHETIC",
    )
    assert card["sensitivity"]["timing_jitter"]["value"] == pytest.approx(1.0)
    assert card["sensitivity"]["timing_jitter"]["status"] == "exact_on_path"
    assert card["gradient_attack"]["status"] == "unavailable"


def test_migration_does_not_mutate_or_recompute() -> None:
    original = {"schema_version": 1, "nested": {"kept": 1}}
    view = migrate_robustness_view(original)
    assert "robustness" not in original
    assert original["nested"]["kept"] == 1
    assert view["schema_version"] == 1
    assert view["extensions_schema_version"] == ROBUSTNESS_EXTENSION_SCHEMA_VERSION
    assert view["robustness"]["metrics_status"] == "legacy_uncomputed"
    assert view["robustness"]["strategies"] == []
    again = migrate_robustness_view(view)
    assert again["robustness"]["metrics_status"] == "legacy_uncomputed"
    schema2 = {"schema_version": 2, "backtest_overfitting": {"kept": True}}
    migrated = migrate_robustness_view(schema2)
    assert migrated["schema_version"] == 2
    assert migrated["backtest_overfitting"] == {"kept": True}
    assert migrated["robustness"]["metrics_status"] == "legacy_uncomputed"
    assert "robustness" not in schema2


def test_stamp_refuses_sealed_receipts_and_forbidden_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed"):
        assert_stamp_target_allowed(Path("/tmp/receipts/notebook.json"))
    card = certify(
        LinearMargin(np.array([1.0, 0.0]), name="stamp_linear"),
        np.array([0.4, 0.1]),
        draws=8,
        seed=0,
        run_attack=False,
        gradient="none",
        outcome_mean=0.2,
        outcome_scale=1.0,
        reference="gaussian",
        evidence_class="SYNTHETIC",
    )
    notebook = {"schema_version": 1, "claim": "research_only"}
    stamped = stamp_robustness(notebook, [card])
    assert "robustness" not in notebook
    assert family_blob_forbidden_metrics_absent(stamped["robustness"])
    assert robustness_extension_errors(stamped) == []
    destination = tmp_path / "notebook.json"
    write_stamped_notebook(destination, notebook, [card])
    assert destination.is_file()
    dirty = json.loads(destination.read_text())
    dirty["robustness"]["strategies"][0]["headline"] = {"sharpe": 1.0}
    assert "robustness_strategy:0:forbidden_metrics" in robustness_extension_errors(dirty)


def test_verify_accepts_a_stamped_notebook_and_rejects_a_bad_block(tmp_path: Path) -> None:
    path = _receipt(tmp_path)
    untouched = verify_research_artifact(path)
    assert untouched["valid"] is True
    payload = json.loads(path.read_text())
    card = certify(
        LinearMargin(np.array([1.0]), name="verified_threshold"),
        np.array([0.2]),
        draws=8,
        seed=0,
        run_attack=False,
        gradient="none",
        outcome_mean=0.2,
        outcome_scale=1.0,
        reference="gaussian",
        evidence_class="SYNTHETIC",
    )
    stamped = stamp_robustness(payload, [card])
    stamped["artifacts"]["immutable_json_sha256"] = _receipt_digest(stamped)
    text = json.dumps(stamped)
    path.write_text(text)
    Path(stamped["artifacts"]["immutable_json"]).write_text(text)
    checked = verify_research_artifact(path)
    assert checked["valid"] is True, checked["errors"]
    broken = json.loads(text)
    broken["robustness"]["research_only"] = False
    broken["artifacts"]["immutable_json_sha256"] = _receipt_digest(broken)
    broken_text = json.dumps(broken)
    path.write_text(broken_text)
    Path(broken["artifacts"]["immutable_json"]).write_text(broken_text)
    failed = verify_research_artifact(path)
    assert failed["valid"] is False
    assert "robustness_research_only" in failed["errors"]
