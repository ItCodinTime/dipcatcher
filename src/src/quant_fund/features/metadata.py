"""Feature metadata. Every feature carries version and PIT flag."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class FeatureMetadata(BaseModel):
    name: str
    version: str
    lookback: int
    required_frequency: str
    source_columns: list[str]
    point_in_time_safe: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    family: str = "generic"


FEATURE_SET_VERSION = "features.v1"
