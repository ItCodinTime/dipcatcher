"""Registry alias / promotion receipt extremes — champion fail-closed."""

from __future__ import annotations

import pytest

from quant_fund.config.models import PromotionConfig
from quant_fund.registry.mlflow_store import (
    attach_artifact_identity,
    promotion_decision,
    promotion_is_approved,
    set_alias,
)


def test_attach_artifact_identity_writes_verified_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str]] = []

    class FakeClient:
        def set_tag(self, run_id: str, key: str, value: str) -> None:
            calls.append((run_id, key, value))

    monkeypatch.setattr("quant_fund.registry.mlflow_store.MlflowClient", FakeClient)
    attach_artifact_identity(
        "run-id",
        {
            "artifact_sha256": "a" * 64,
            "manifest_valid": True,
            "artifact_class": "RidgeRanker",
            "manifest_schema": "model_artifact.v1",
        },
    )
    assert dict((key, value) for _, key, value in calls) == {
        "artifact_sha256": "a" * 64,
        "artifact_manifest_valid": "true",
        "artifact_class": "RidgeRanker",
        "artifact_manifest_schema": "model_artifact.v1",
    }
    assert {run_id for run_id, _, _ in calls} == {"run-id"}


def test_attach_artifact_identity_rejects_invalid_manifest_before_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_fund.registry.mlflow_store.MlflowClient",
        lambda: (_ for _ in ()).throw(AssertionError("client should not be called")),
    )
    with pytest.raises(ValueError, match="64-character sha256"):
        attach_artifact_identity("run-id", {"artifact_sha256": "bad", "manifest_valid": True})


def test_champion_requires_approved_receipt() -> None:
    with pytest.raises(PermissionError, match="approved promotion receipt"):
        set_alias("run-id", "champion")


def test_promotion_receipt_requires_identity_and_evidence() -> None:
    assert promotion_is_approved({"promote": True}) is False
    assert (
        promotion_is_approved(
            {
                "receipt_schema": "promotion.v1",
                "promote": True,
                "evidence_complete": True,
                "research_receipt_valid": True,
                "leakage_ok": True,
                "data_source": "file",
                "reasons": [],
                "run_id": "run-id",
            }
        )
        is True
    )
    assert (
        promotion_is_approved(
            {"promote": True, "evidence_complete": True, "run_id": "other-run"},
            run_id="run-id",
        )
        is False
    )


def test_promotion_receipt_rejects_unverified_artifact_identity() -> None:
    receipt = {
        "receipt_schema": "promotion.v1",
        "promote": True,
        "evidence_complete": True,
        "research_receipt_valid": True,
        "leakage_ok": True,
        "data_source": "vendor",
        "reasons": [],
        "run_id": "run-id",
        "artifact_sha256": "a" * 64,
        "manifest_valid": False,
    }
    assert promotion_is_approved(receipt, run_id="run-id") is False


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        {},
        {"promote": True, "evidence_complete": True},  # missing run_id
        {"promote": True, "run_id": "run-id"},  # missing evidence_complete
        {"evidence_complete": True, "run_id": "run-id"},  # missing promote
        {"promote": False, "evidence_complete": True, "run_id": "run-id"},
        {"promote": "true", "evidence_complete": True, "run_id": "run-id"},  # not bool True
        {"promote": True, "evidence_complete": "yes", "run_id": "run-id"},
        {"promote": True, "evidence_complete": True, "run_id": ""},
        {"promote": True, "evidence_complete": True, "run_id": 123},
        {"promote": True, "evidence_complete": True, "run_id": None},
    ],
)
def test_missing_identity_or_evidence_fields_fail_closed(receipt: object) -> None:
    assert promotion_is_approved(receipt) is False  # type: ignore[arg-type]
    with pytest.raises(PermissionError, match="approved promotion receipt"):
        set_alias("run-id", "champion", promotion=receipt)  # type: ignore[arg-type]


def test_tampered_champion_receipt_mismatched_run_id() -> None:
    tampered = {
        "promote": True,
        "evidence_complete": True,
        "run_id": "victim-run",
    }
    assert promotion_is_approved(tampered, run_id="attacker-run") is False
    with pytest.raises(PermissionError, match="approved promotion receipt"):
        set_alias("attacker-run", "champion", promotion=tampered)


def test_champion_requires_validated_research_receipt() -> None:
    receipt = {
        "receipt_schema": "promotion.v1",
        "promote": True,
        "evidence_complete": True,
        "leakage_ok": True,
        "data_source": "file",
        "reasons": [],
        "run_id": "run-id",
    }
    assert promotion_is_approved(receipt, run_id="run-id") is False


def test_tampered_promote_flag_and_reasons_fail_closed() -> None:
    # Decision said no, but someone flipped promote — still need full identity.
    flipped = {
        "promote": True,
        "evidence_complete": False,
        "run_id": "run-id",
        "reasons": ["evidence_incomplete"],
    }
    assert promotion_is_approved(flipped, run_id="run-id") is False
    with pytest.raises(PermissionError):
        set_alias("run-id", "champion", promotion=flipped)


def test_synthetic_receipt_never_champion() -> None:
    metrics = {
        "evidence_complete": True,
        "data_source": "SYNTHETIC",
        "synthetic": True,
        "mean_ic": 0.99,
        "net_spread": 0.99,
        "turnover": 0.01,
        "n_folds": 10,
        "fold_ic_stability": 1.0,
        "run_id": "synth-run",
    }
    decision = promotion_decision(metrics, PromotionConfig(), leakage_ok=True)
    assert decision["promote"] is False
    assert "synthetic_evidence_not_promotable" in decision["reasons"]
    assert promotion_is_approved(decision, run_id="synth-run") is False
    with pytest.raises(PermissionError):
        set_alias("synth-run", "champion", promotion=decision)

    # Even a hand-crafted promote=True synthetic receipt must fail closed.
    forged = {
        "promote": True,
        "evidence_complete": True,
        "run_id": "synth-run",
        "synthetic": True,
        "data_source": "SYNTHETIC",
    }
    assert promotion_is_approved(forged, run_id="synth-run") is False
    with pytest.raises(PermissionError):
        set_alias("synth-run", "champion", promotion=forged)

    forged_reasons = {
        "promote": True,
        "evidence_complete": True,
        "run_id": "synth-run",
        "reasons": ["synthetic_evidence_not_promotable"],
    }
    assert promotion_is_approved(forged_reasons, run_id="synth-run") is False


def test_promotion_decision_rejects_missing_data_source() -> None:
    metrics = {
        "evidence_complete": True,
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
    }
    decision = promotion_decision(metrics, PromotionConfig(), leakage_ok=True)
    assert decision["promote"] is False
    assert "data_source_missing" in decision["reasons"]


def test_promotion_decision_rejects_invalid_artifact_manifest_identity() -> None:
    metrics = {
        "data_source": "vendor",
        "evidence_complete": True,
        "mean_ic": 0.1,
        "net_spread": 0.1,
        "turnover": 0.1,
        "n_folds": 3,
        "artifact_sha256": "not-a-hash",
        "manifest_valid": False,
    }
    decision = promotion_decision(metrics, PromotionConfig(), leakage_ok=True)
    assert decision["promote"] is False
    assert "artifact_hash_invalid" in decision["reasons"]
    assert "artifact_manifest_invalid" in decision["reasons"]


def test_promotion_decision_normalizes_data_source_before_synthetic_gate() -> None:
    metrics = {
        "data_source": " synthetic ",
        "evidence_complete": True,
        "mean_ic": 0.5,
        "net_spread": 0.5,
        "turnover": 0.1,
        "n_folds": 5,
    }
    decision = promotion_decision(metrics, PromotionConfig(), leakage_ok=True)
    assert decision["promote"] is False
    assert decision["data_source"] == "synthetic"
    assert "synthetic_evidence_not_promotable" in decision["reasons"]


def test_champion_approval_rejects_whitespace_padded_synthetic_source() -> None:
    receipt = {
        "receipt_schema": "promotion.v1",
        "promote": True,
        "evidence_complete": True,
        "research_receipt_valid": True,
        "leakage_ok": True,
        "data_source": " SYNTHETIC ",
        "reasons": [],
        "run_id": "run-id",
    }
    assert promotion_is_approved(receipt, run_id="run-id") is False


def test_invalid_alias_rejected() -> None:
    with pytest.raises(ValueError, match="alias must be one of"):
        set_alias("run-id", "production")
