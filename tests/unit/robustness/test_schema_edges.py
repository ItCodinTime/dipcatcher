"""Fail-closed schema validation and stamp/migration edge cases."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from quant_fund.robustness.schema import (
    ROBUSTNESS_EXTENSION_SCHEMA_VERSION,
    assert_stamp_target_allowed,
    migrate_robustness_view,
    robustness_extension_errors,
    stamp_robustness,
    unavailable_robustness_block,
    write_stamped_notebook,
)


def _valid_scorecard() -> dict:
    radius_block = {
        "value": 0.5,
        "norm": "l2",
        "status": "proven",
        "proven": True,
    }
    sensitivity = {
        name: {
            "value": 1.0,
            "unit": "bars",
            "status": "exact_on_path",
            "method": "enumerated",
            "flipped": True,
        }
        for name in (
            "vol_scaled_path",
            "missing_bars",
            "stale_prints",
            "spikes",
            "timing_jitter",
            "cost_shock",
            "wasserstein_regime",
        )
    }
    return {
        "strategy": "toy",
        "evidence_class": "SYNTHETIC",
        "research_only": True,
        "live_trading_claim": False,
        "certified_radius": dict(radius_block),
        "empirical_attack_radius": {**radius_block, "status": "empirical", "value": 1.0},
        "distributional_robustness": {
            "reference": "gaussian",
            "worst_case_mean": {"value": 0.1, "status": "proven"},
            "worst_case_ratio": {"value": 0.5, "status": "proven_tight"},
        },
        "sensitivity": sensitivity,
        "radii_comparable": True,
        "limitations": ["toy only"],
    }


def _notebook() -> dict:
    return {"schema_version": 1, "claim": "research_only"}


class TestUnavailableBlock:
    def test_roundtrips_validation_clean(self) -> None:
        notebook = dict(_notebook())
        notebook["extensions_schema_version"] = ROBUSTNESS_EXTENSION_SCHEMA_VERSION
        notebook["robustness"] = unavailable_robustness_block()
        assert robustness_extension_errors(notebook) == []
        block = notebook["robustness"]
        assert block["metrics_status"] == "unavailable"
        assert block["strategies"] == []


class TestMigration:
    def test_absent_block_gets_legacy_view(self) -> None:
        original = _notebook()
        migrated = migrate_robustness_view(original)
        assert migrated is not original
        assert "robustness" not in original
        assert migrated["robustness"]["metrics_status"] == "legacy_uncomputed"
        assert migrated["robustness"]["migrated_from_absent"] is True
        assert migrated["extensions_schema_version"] == 1
        assert robustness_extension_errors(migrated) == []

    def test_existing_block_is_kept_verbatim(self) -> None:
        notebook = _notebook()
        notebook["extensions_schema_version"] = 1
        notebook["robustness"] = unavailable_robustness_block()
        out = migrate_robustness_view(notebook)
        assert out["robustness"]["metrics_status"] == "unavailable"
        assert "migrated_from_absent" not in out["robustness"]

    def test_schema2_accepted(self) -> None:
        migrated = migrate_robustness_view({"schema_version": 2})
        assert migrated["extensions_schema_version"] == 1

    def test_rejects_non_object_and_bad_version(self) -> None:
        with pytest.raises(TypeError):
            migrate_robustness_view([1, 2])
        with pytest.raises(ValueError, match="schema_version"):
            migrate_robustness_view({"schema_version": 3})
        with pytest.raises(ValueError, match="schema_version"):
            migrate_robustness_view({"schema_version": True})
        with pytest.raises(ValueError, match="schema_version"):
            migrate_robustness_view({})


class TestStampAndWrite:
    def test_stamp_returns_copy_and_validates(self) -> None:
        notebook = _notebook()
        stamped = stamp_robustness(notebook, [_valid_scorecard()])
        assert "robustness" not in notebook
        assert stamped["robustness"]["metrics_status"] == "computed"
        assert len(stamped["robustness"]["strategies"]) == 1
        assert robustness_extension_errors(stamped) == []

    def test_stamp_rejects_invalid_scorecard(self) -> None:
        bad = _valid_scorecard()
        bad["radii_comparable"] = True
        # certified (0.5) > empirical (use smaller value) -> incomparable claim
        bad["empirical_attack_radius"] = {
            "value": 0.1,
            "norm": "l2",
            "status": "empirical",
            "proven": False,
        }
        with pytest.raises(ValueError, match="certificate_exceeds_attack"):
            stamp_robustness(_notebook(), [bad])

    def test_stamp_type_gates(self) -> None:
        with pytest.raises(TypeError):
            stamp_robustness([1], [])
        with pytest.raises(TypeError):
            stamp_robustness(_notebook(), "notalist")
        with pytest.raises(TypeError):
            stamp_robustness(_notebook(), [[1]])

    def test_write_refuses_receipts_path(self, tmp_path: Path) -> None:
        target = tmp_path / "receipts" / "notebook.json"
        with pytest.raises(ValueError, match="receipts"):
            write_stamped_notebook(target, _notebook(), [])
        assert not target.exists()

    def test_assert_stamp_target_resolves(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="receipts"):
            assert_stamp_target_allowed(Path("deep/receipts/x.json"))
        assert_stamp_target_allowed(tmp_path / "plain.json")

    def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "nb.json"
        stamped = write_stamped_notebook(target, _notebook(), [_valid_scorecard()])
        assert json.loads(target.read_text()) == stamped


class TestExtensionErrors:
    def test_absent_is_valid(self) -> None:
        assert robustness_extension_errors({}) == []
        assert robustness_extension_errors(_notebook()) == []

    def test_non_object(self) -> None:
        assert robustness_extension_errors([1]) == ["robustness_notebook_not_object"]

    def test_version_mismatch(self) -> None:
        nb = _notebook()
        nb["extensions_schema_version"] = 99
        nb["robustness"] = unavailable_robustness_block()
        assert "robustness_extensions_schema_version" in robustness_extension_errors(nb)
        nb["extensions_schema_version"] = True
        assert "robustness_extensions_schema_version" in robustness_extension_errors(nb)

    def test_version_without_block(self) -> None:
        nb = _notebook()
        nb["extensions_schema_version"] = 1
        assert "robustness_missing" in robustness_extension_errors(nb)

    def test_block_field_errors(self) -> None:
        nb = _notebook()
        nb["extensions_schema_version"] = 1
        block = unavailable_robustness_block()
        block["schema_version"] = 2
        nb["robustness"] = block
        errors = robustness_extension_errors(nb)
        assert "robustness_schema_version" in errors

        block["schema_version"] = 1
        block["claim"] = "alpha"
        errors = robustness_extension_errors(nb)
        assert "robustness_claim" in errors

        block["claim"] = "robustness_diagnostic_only"
        block["research_only"] = False
        errors = robustness_extension_errors(nb)
        assert "robustness_research_only" in errors

        block["research_only"] = True
        block["live_trading_claim"] = True
        errors = robustness_extension_errors(nb)
        assert "robustness_live_trading_claim" in errors

    def test_forbidden_metric_tokens_fail_closed(self) -> None:
        nb = _notebook()
        nb["extensions_schema_version"] = 1
        block = unavailable_robustness_block()
        block["headline_sharpe"] = 2.0
        nb["robustness"] = block
        assert "robustness_forbidden_metrics" in robustness_extension_errors(nb)

    def test_bad_metrics_status_short_circuits(self) -> None:
        nb = _notebook()
        nb["extensions_schema_version"] = 1
        block = unavailable_robustness_block()
        block["metrics_status"] = "partial"
        nb["robustness"] = block
        errors = robustness_extension_errors(nb)
        assert errors == ["robustness_metrics_status"]

    def test_strategies_must_be_list(self) -> None:
        nb = _notebook()
        nb["extensions_schema_version"] = 1
        block = unavailable_robustness_block()
        block["metrics_status"] = "computed"
        block["strategies"] = {}
        nb["robustness"] = block
        errors = robustness_extension_errors(nb)
        assert errors == ["robustness_strategies"]

    def test_uncomputed_status_must_be_empty(self) -> None:
        nb = _notebook()
        nb["extensions_schema_version"] = 1
        block = unavailable_robustness_block()
        block["strategies"] = [_valid_scorecard()]
        nb["robustness"] = block
        errors = robustness_extension_errors(nb)
        assert "robustness_strategies_present_when_uncomputed" in errors

    def test_legacy_flag_required_for_legacy_status(self) -> None:
        nb = _notebook()
        nb["extensions_schema_version"] = 1
        block = unavailable_robustness_block()
        block["metrics_status"] = "legacy_uncomputed"
        nb["robustness"] = block
        errors = robustness_extension_errors(nb)
        assert "robustness_legacy_flag" in errors

    def test_scorecard_field_errors(self) -> None:
        nb = _notebook()
        nb["extensions_schema_version"] = 1
        block = unavailable_robustness_block()
        block["metrics_status"] = "computed"
        block["strategies"] = ["notadict", _valid_scorecard()]
        nb["robustness"] = block
        errors = robustness_extension_errors(nb)
        assert "robustness_strategy:0:not_object" in errors
        assert not any(e.startswith("robustness_strategy:1") for e in errors)

    def test_scorecard_detail_errors(self) -> None:
        card = _valid_scorecard()
        card["strategy"] = "  "
        card["evidence_class"] = "REAL"
        card["limitations"] = []
        errors = _collect_card_errors(card)
        assert "robustness_strategy:0:name" in errors
        assert "robustness_strategy:0:evidence_class" in errors
        assert "robustness_strategy:0:limitations" in errors

    def test_sensitivity_coverage_enforced(self) -> None:
        card = _valid_scorecard()
        del card["sensitivity"]["spikes"]
        card["sensitivity"]["mystery"] = card["sensitivity"]["cost_shock"]
        errors = _collect_card_errors(card)
        assert any(
            e.startswith("robustness_strategy:0:sensitivity_missing:") and "spikes" in e
            for e in errors
        )
        assert "robustness_strategy:0:sensitivity_unknown:mystery" in errors

    def test_radius_block_errors(self) -> None:
        card = _valid_scorecard()
        card["certified_radius"] = "missing"
        errors = _collect_card_errors(card)
        assert "robustness_strategy:0:certified_radius:missing" in errors

        card = _valid_scorecard()
        card["certified_radius"]["status"] = "guessed"
        card["certified_radius"]["norm"] = "l1"
        card["certified_radius"]["proven"] = "yes"
        card["certified_radius"]["value"] = -1.0
        errors = _collect_card_errors(card)
        for suffix in ("status", "norm", "proven", "value"):
            assert f"robustness_strategy:0:certified_radius:{suffix}" in errors

    def test_sensitivity_record_errors(self) -> None:
        card = _valid_scorecard()
        record = card["sensitivity"]["spikes"]
        record["status"] = "maybe"
        record["unit"] = ""
        record["method"] = " "
        record["flipped"] = "yes"
        record["value"] = math.inf
        errors = _collect_card_errors(card)
        prefix = "robustness_strategy:0:sensitivity:spikes"
        for suffix in ("status", "unit", "method", "flipped", "value"):
            assert f"{prefix}:{suffix}" in errors

        card = _valid_scorecard()
        card["sensitivity"]["spikes"] = 7
        errors = _collect_card_errors(card)
        assert f"{prefix}:missing" in errors

    def test_distribution_errors(self) -> None:
        card = _valid_scorecard()
        card["distributional_robustness"] = None
        errors = _collect_card_errors(card)
        assert "robustness_strategy:0:distribution" in errors

        card = _valid_scorecard()
        dist = card["distributional_robustness"]
        dist["reference"] = "laplace"
        dist["worst_case_mean"] = {"value": math.nan, "status": "proven"}
        dist["worst_case_ratio"] = {"value": 0.5, "status": "unknown"}
        errors = _collect_card_errors(card)
        prefix = "robustness_strategy:0"
        assert f"{prefix}:reference" in errors
        assert f"{prefix}:worst_case_mean" in errors
        assert f"{prefix}:worst_case_ratio" in errors

        # proven status needs a finite value; non-proven must carry None
        card = _valid_scorecard()
        dist = card["distributional_robustness"]
        dist["worst_case_ratio"] = {"value": None, "status": "proven_tight"}
        errors = _collect_card_errors(card)
        assert f"{prefix}:worst_case_ratio_value" in errors

        card = _valid_scorecard()
        dist = card["distributional_robustness"]
        dist["worst_case_ratio"] = {"value": 0.3, "status": "undefined"}
        errors = _collect_card_errors(card)
        assert f"{prefix}:worst_case_ratio_value" in errors

    def test_radii_comparable_checks(self) -> None:
        card = _valid_scorecard()
        card["radii_comparable"] = "yes"
        errors = _collect_card_errors(card)
        assert "robustness_strategy:0:radii_comparable" in errors

        # comparable with missing empirical value fails
        card = _valid_scorecard()
        card["empirical_attack_radius"]["value"] = None
        errors = _collect_card_errors(card)
        assert "robustness_strategy:0:certificate_exceeds_attack" in errors


def _collect_card_errors(card: dict) -> list[str]:
    nb = _notebook()
    nb["extensions_schema_version"] = 1
    block = unavailable_robustness_block()
    block["metrics_status"] = "computed"
    block["strategies"] = [card]
    nb["robustness"] = block
    return robustness_extension_errors(nb)
