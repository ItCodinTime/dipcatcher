"""Forecast model protocol."""

from __future__ import annotations

import hashlib
import hmac
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
        if not isinstance(model, cls):
            raise TypeError(
                f"model artifact has type {type(model).__name__}; "
                f"expected {cls.__name__}"
            )
        return model
