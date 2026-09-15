"""Forecast model protocol."""

from __future__ import annotations

from pathlib import Path
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


class JoblibMixin:
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> Any:
        return joblib.load(path)
