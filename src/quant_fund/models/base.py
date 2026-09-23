"""Forecast model protocol."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

import joblib
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


class ModelMeta(BaseModel):
    family: str
    name: str
    version: str
    features: list[str] = Field(default_factory=list)
    horizon: str = "5d"
    seed: int = 42
    extra: dict[str, Any] = Field(default_factory=dict)


class ForecastModel(Protocol):
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> Any: ...

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def save(self, path: Path) -> None: ...

    def metadata(self) -> ModelMeta: ...


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: Path) -> Path:
    return Path(path).with_name(f"{Path(path).name}.manifest.json")


def _manifest_payload(path: Path, payload: Any) -> dict[str, Any]:
    metadata: Any = None
    if hasattr(payload, "metadata") and callable(payload.metadata):
        raw = payload.metadata()
        metadata = raw.model_dump(mode="json") if isinstance(raw, BaseModel) else raw
    features = getattr(payload, "features", None)
    if features is None and isinstance(payload, dict):
        features = payload.get("features")
    return {
        "schema": "model_artifact.v1",
        "artifact": Path(path).name,
        "sha256": _sha256_file(path),
        "class": type(payload).__name__,
        "metadata": metadata,
        "features": [str(value) for value in features] if isinstance(features, list) else None,
        "provenance": _manifest_provenance(payload),
    }


def _write_manifest(path: Path, payload: Any) -> None:
    manifest = _manifest_path(path)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=manifest.parent,
            prefix=f".{manifest.name}.",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(
                json.dumps(_manifest_payload(path, payload), sort_keys=True, indent=2) + "\n"
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, manifest)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _manifest_features(payload: Any) -> list[str] | None:
    features = getattr(payload, "features", None)
    if features is None and isinstance(payload, dict):
        features = payload.get("features")
    return [str(value) for value in features] if isinstance(features, list) else None


def _manifest_provenance(payload: Any) -> Any:
    provenance = getattr(payload, "provenance", None)
    if provenance is None and isinstance(payload, dict):
        provenance = payload.get("provenance")
    if isinstance(provenance, BaseModel):
        return provenance.model_dump(mode="json")
    return provenance


def _manifest_metadata(payload: Any) -> Any:
    if hasattr(payload, "metadata") and callable(payload.metadata):
        raw = payload.metadata()
        return raw.model_dump(mode="json") if isinstance(raw, BaseModel) else raw
    return None


def _verify_manifest(path: Path, payload: Any) -> None:
    manifest = _manifest_path(path)
    if not manifest.is_file():
        return
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"model artifact manifest is malformed: {manifest}") from exc
    if not isinstance(record, dict) or record.get("schema") != "model_artifact.v1":
        raise ValueError(f"model artifact manifest is malformed: {manifest}")
    if record.get("artifact") != path.name or record.get("sha256") != _sha256_file(path):
        raise ValueError(f"model artifact manifest mismatch: {path}")
    if record.get("class") != type(payload).__name__:
        raise ValueError(f"model artifact manifest class mismatch: {path}")
    if record.get("metadata") != _manifest_metadata(payload):
        raise ValueError(f"model artifact manifest metadata mismatch: {path}")
    if record.get("features") != _manifest_features(payload):
        raise ValueError(f"model artifact manifest feature mismatch: {path}")
    if record.get("provenance") != _manifest_provenance(payload):
        raise ValueError(f"model artifact manifest provenance mismatch: {path}")


def artifact_identity(path: Path) -> dict[str, Any]:
    """Return the verified identity fields for a persisted artifact."""
    artifact = Path(path)
    if not artifact.is_file():
        raise ValueError(f"model artifact is missing: {artifact}")
    manifest = _manifest_path(artifact)
    if not manifest.is_file():
        raise ValueError(f"model artifact manifest is missing: {manifest}")
    payload = load_joblib_artifact(artifact)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "artifact_sha256": _sha256_file(artifact),
        "manifest_valid": True,
        "artifact_class": type(payload).__name__,
        "manifest_schema": record["schema"],
    }


def save_joblib_artifact(payload: Any, path: Path) -> None:
    """Atomically persist a non-model payload with a checksum sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_name(f"{path.name}.sha256")
    temporary_path: Path | None = None
    temporary_sidecar: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            joblib.dump(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        checksum = _sha256_file(path)
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            mode="w",
            encoding="ascii",
            delete=False,
        ) as checksum_file:
            temporary_sidecar = Path(checksum_file.name)
            checksum_file.write(checksum + "\n")
            checksum_file.flush()
            os.fsync(checksum_file.fileno())
        os.replace(temporary_sidecar, sidecar)
        _write_manifest(path, payload)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if temporary_sidecar is not None:
            temporary_sidecar.unlink(missing_ok=True)


def load_joblib_artifact(path: Path) -> Any:
    """Load a joblib payload, verifying its optional checksum sidecar."""
    path = Path(path)
    sidecar = path.with_name(f"{path.name}.sha256")
    if sidecar.is_file():
        expected = sidecar.read_text(encoding="ascii").strip()
        if len(expected) != hashlib.sha256().digest_size * 2:
            raise ValueError(f"artifact checksum sidecar is malformed: {sidecar}")
        try:
            int(expected, 16)
        except ValueError as exc:
            raise ValueError(f"artifact checksum sidecar is malformed: {sidecar}") from exc
        actual = _sha256_file(path)
        if not hmac.compare_digest(expected, actual):
            raise ValueError(f"artifact checksum mismatch: {path}")
    payload = joblib.load(path)
    _verify_manifest(path, payload)
    return payload


class JoblibMixin:
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = path.with_name(f"{path.name}.sha256")
        temporary_path: Path | None = None
        temporary_sidecar: Path | None = None
        try:
            # Keep the temporary artifact beside the destination so os.replace
            # is atomic even when the model directory is on a separate mount.
            with NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                joblib.dump(self, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)

            checksum = _sha256_file(path)
            with NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{sidecar.name}.",
                suffix=".tmp",
                mode="w",
                encoding="ascii",
                delete=False,
            ) as checksum_file:
                temporary_sidecar = Path(checksum_file.name)
                checksum_file.write(checksum + "\n")
                checksum_file.flush()
                os.fsync(checksum_file.fileno())
            os.replace(temporary_sidecar, sidecar)
            _write_manifest(path, self)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if temporary_sidecar is not None:
                temporary_sidecar.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> Any:
        path = Path(path)
        sidecar = path.with_name(f"{path.name}.sha256")
        if sidecar.is_file():
            expected = sidecar.read_text(encoding="ascii").strip()
            if len(expected) != hashlib.sha256().digest_size * 2:
                raise ValueError(f"model artifact checksum sidecar is malformed: {sidecar}")
            try:
                int(expected, 16)
            except ValueError as exc:
                raise ValueError(
                    f"model artifact checksum sidecar is malformed: {sidecar}"
                ) from exc
            actual = _sha256_file(path)
            if not hmac.compare_digest(expected, actual):
                raise ValueError(f"model artifact checksum mismatch: {path}")
        model = joblib.load(path)
        _verify_manifest(path, model)
        if not isinstance(model, cls):
            raise TypeError(
                f"model artifact has type {type(model).__name__}; expected {cls.__name__}"
            )
        return model
