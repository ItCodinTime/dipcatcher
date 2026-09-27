"""Typed contract an external fx-1 forecaster must implement.

The class :class:`Fx1Model` is an interface. This repository does not ship
weights, a training loop, or a ``predict`` implementation for fx-1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import polars as pl


class ForecastModel(ABC):
    """Generic forecasting model the harness can load and call.

    ``load`` receives an optional local checkpoint path and a plain config
    mapping (horizon, version, format). ``predict`` receives a point-in-time
    feature frame and returns one forecast row per
    ``(event_time, security_id, horizon_bars)``.
    """

    name: str
    version: str

    @abstractmethod
    def load(
        self,
        checkpoint_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Bind a local checkpoint and/or config. Must not touch the network."""

    @abstractmethod
    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        """Forecast from features known at each row's ``event_time``.

        Implementations must not read rows with a later ``event_time`` when
        the harness is in walk-forward mode: the frame is already truncated.
        In batch mode the frame can contain later rows; row-local models are
        required. Do not expect label or forward-return columns.
        """


class Fx1Model(ForecastModel):
    """Contract for the external model named ``fx-1``.

    Subclass this *outside* the repo and point ``model.entrypoint`` at it.
    Resolving the name ``fx-1`` without an entrypoint fails closed.
    """

    name = "fx-1"
    version = "external"

    def load(
        self,
        checkpoint_path: str | Path | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError(
            "fx-1 is an external forecaster. Implement Fx1Model.load in the "
            "plugin named by model.entrypoint; this repository does not ship the model."
        )

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError(
            "fx-1 is an external forecaster. Implement Fx1Model.predict in the "
            "plugin named by model.entrypoint; this repository does not ship the model."
        )


class FeaturePipeline(Protocol):
    """Builds model inputs from adapter bars. Features at t use data available at t."""

    def feature_columns(self) -> list[str]:
        """Numeric columns the model is allowed to see, excluding identifiers."""

    def build(
        self,
        bars: pl.DataFrame,
        *,
        decision_time: datetime | None = None,
    ) -> pl.DataFrame:
        """Return a feature frame. ``decision_time`` drops anything not yet available."""
