"""Format-agnostic local checkpoint loader.

Backends are imported only when requested. JSON, ONNX, and weights-only torch
loading are the defaults. Pickle, joblib, and full-module torch loading can
execute code; they require an explicit opt-in and a trusted SHA-256 digest.
The unsafe loaders consume the exact bytes whose digest was checked.
"""

from __future__ import annotations

import importlib
import io
import json
import re
from dataclasses import dataclass
from hmac import compare_digest
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import hash_bytes, hash_file

_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".pkl": "pickle",
    ".pickle": "pickle",
    ".joblib": "joblib",
    ".pt": "torch",
    ".pth": "torch",
    ".onnx": "onnx",
    ".json": "json",
}


class ArtifactBackendUnavailable(ImportError):
    """The requested checkpoint backend is not installed."""

    def __init__(self, backend: str) -> None:
        super().__init__(
            f"checkpoint backend {backend!r} is not installed; "
            "the harness stays importable without this optional dependency"
        )
        self.backend = backend


class UntrustedArtifactError(ValueError):
    """A checkpoint did not satisfy the explicit trusted-input policy."""


@dataclass(frozen=True)
class LoadedArtifact:
    """A local checkpoint plus the stamp recorded in run metadata."""

    path: str
    format: str
    sha256: str
    version: str | None
    payload: object


def import_optional(module: str) -> Any:
    """Import ``module`` or raise :class:`ArtifactBackendUnavailable`."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ArtifactBackendUnavailable(module) from exc


def resolve_format(path: Path, fmt: str | None) -> str:
    if fmt and fmt != "auto":
        return fmt
    resolved = _FORMAT_BY_SUFFIX.get(path.suffix.lower())
    if resolved is None:
        raise ValueError(
            f"cannot infer checkpoint format from {path.name!r}; "
            f"set model.checkpoint_format to one of {sorted(set(_FORMAT_BY_SUFFIX.values()))}"
        )
    return resolved


def _sidecar_version(path: Path) -> str | None:
    sidecar = Path(str(path) + ".version")
    if not sidecar.is_file():
        return None
    text = sidecar.read_text(encoding="utf-8").strip()
    return text or None


def _payload_version(payload: object) -> str | None:
    if isinstance(payload, dict) and payload.get("version") is not None:
        return str(payload["version"])
    version = getattr(payload, "version", None)
    if isinstance(version, str) and version:
        return version
    return None


def probe_artifact(path: str | Path, fmt: str | None = None) -> dict[str, str | None]:
    """Hash and version-stamp a checkpoint without deserializing it.

    Version comes from a sibling ``<path>.version`` file when present.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {file_path}")
    return {
        "path": str(file_path),
        "format": resolve_format(file_path, fmt),
        "sha256": hash_file(file_path),
        "version": _sidecar_version(file_path),
    }


def _trusted_bytes(path: Path, expected_sha256: str | None, allowed: bool) -> tuple[bytes, str]:
    if not allowed or expected_sha256 is None:
        raise UntrustedArtifactError(
            "pickle, joblib, and full-module torch require "
            "allow_unsafe_deserialization=true and trusted_checkpoint_sha256"
        )
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise UntrustedArtifactError("trusted_checkpoint_sha256 must be 64 lowercase hex digits")
    raw = path.read_bytes()
    actual = hash_bytes(raw)
    if not compare_digest(actual, expected_sha256):
        raise UntrustedArtifactError("checkpoint SHA-256 does not match the trusted digest")
    return raw, actual


def _load_pickle(raw: bytes) -> object:
    pickle = import_optional("pickle")
    # The caller explicitly opted in and pinned these exact bytes. Pickle is
    # intentionally unavailable through the default loading path.
    return pickle.loads(raw)  # noqa: S301  # nosec B301


def _load_joblib(raw: bytes) -> object:
    joblib = import_optional("joblib")
    return joblib.load(io.BytesIO(raw))


def _load_torch(path: Path | io.BytesIO, *, weights_only: bool) -> object:
    torch = import_optional("torch")
    return torch.load(path, map_location="cpu", weights_only=weights_only)


def _load_onnx(path: Path) -> object:
    onnx = import_optional("onnx")
    return onnx.load(str(path))


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


_BACKENDS = {
    "onnx": _load_onnx,
    "json": _load_json,
}


def load_artifact(
    path: str | Path,
    fmt: str | None = None,
    *,
    allow_unsafe_deserialization: bool = False,
    trusted_checkpoint_sha256: str | None = None,
) -> LoadedArtifact:
    """Load a checkpoint and stamp the bytes used.

    Pickle and joblib are disabled by default. Torch defaults to
    ``weights_only=True``. Unsafe formats require both an explicit opt-in and
    a trusted SHA-256 digest; the checked bytes are the bytes deserialized.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {file_path}")
    resolved = resolve_format(file_path, fmt)
    if resolved in {"pickle", "joblib"} or (resolved == "torch" and allow_unsafe_deserialization):
        raw, digest = _trusted_bytes(
            file_path, trusted_checkpoint_sha256, allow_unsafe_deserialization
        )
        if resolved == "pickle":
            payload = _load_pickle(raw)
        elif resolved == "joblib":
            payload = _load_joblib(raw)
        else:
            payload = _load_torch(io.BytesIO(raw), weights_only=False)
    elif resolved == "torch":
        payload = _load_torch(file_path, weights_only=True)
        digest = hash_file(file_path)
    else:
        backend = _BACKENDS.get(resolved)
        if backend is None:
            raise ValueError(f"unsupported checkpoint format {resolved!r}")
        payload = backend(file_path)
        digest = hash_file(file_path)
    if trusted_checkpoint_sha256 is not None and not compare_digest(
        digest, trusted_checkpoint_sha256
    ):
        raise UntrustedArtifactError("checkpoint SHA-256 does not match the trusted digest")
    version = _sidecar_version(file_path) or _payload_version(payload)
    return LoadedArtifact(
        path=str(file_path),
        format=resolved,
        sha256=digest,
        version=version,
        payload=payload,
    )
