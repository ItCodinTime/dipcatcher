"""Typed artifact manifest and legacy compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest

from quant_fund.models.base import artifact_identity, load_joblib_artifact, save_joblib_artifact


def test_manifest_round_trip_binds_payload_class_and_features(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.joblib"
    payload = {"features": ["momentum", "volatility"], "weights": [0.2, 0.8]}

    save_joblib_artifact(payload, artifact)

    manifest = json.loads((tmp_path / "policy.joblib.manifest.json").read_text())
    assert manifest["schema"] == "model_artifact.v1"
    assert manifest["class"] == "dict"
    assert manifest["features"] == payload["features"]
    assert load_joblib_artifact(artifact) == payload
    identity = artifact_identity(artifact)
    assert identity["manifest_valid"] is True
    assert len(identity["artifact_sha256"]) == 64


def test_legacy_artifact_without_manifest_remains_readable(tmp_path: Path) -> None:
    artifact = tmp_path / "legacy.joblib"
    payload = {"features": ["close"], "value": 1}
    joblib.dump(payload, artifact)

    assert load_joblib_artifact(artifact) == payload


def test_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.joblib"
    save_joblib_artifact({"features": ["close"], "value": 1}, artifact)
    manifest_path = tmp_path / "policy.joblib.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["features"] = ["future_close"]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="feature mismatch"):
        load_joblib_artifact(artifact)


def test_manifest_binds_payload_provenance(tmp_path: Path) -> None:
    artifact = tmp_path / "provenance.joblib"
    payload = {"features": ["close"], "provenance": {"dataset_sha256": "a" * 64}}
    save_joblib_artifact(payload, artifact)
    manifest_path = tmp_path / "provenance.joblib.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["provenance"] == payload["provenance"]
    manifest["provenance"]["dataset_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_joblib_artifact(artifact)


def test_malformed_manifest_fails_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.joblib"
    save_joblib_artifact({"value": 1}, artifact)
    (tmp_path / "policy.joblib.manifest.json").write_text("not-json")

    with pytest.raises(ValueError, match="manifest is malformed"):
        load_joblib_artifact(artifact)
