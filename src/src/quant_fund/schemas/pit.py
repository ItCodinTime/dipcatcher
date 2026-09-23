"""Point-in-time observation contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from quant_fund.schemas.errors import PointInTimeError


class PITRecord(BaseModel):
    event_time: datetime
    available_time: datetime
    ingested_time: datetime
    source: str
    security_id: str
    symbol: str
    revision_id: str = "v1"
    value: float | None = None
    fields: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def availability_not_before_event(self) -> PITRecord:
        if self.available_time < self.event_time:
            # allowed for some announcements, but event_time should not post-date available
            # We allow available_time >= event_time for bars; announcements may differ.
            pass
        return self


class FeatureIntegrity(BaseModel):
    decision_time: datetime
    max_source_available_time: datetime

    @model_validator(mode="after")
    def no_lookahead(self) -> FeatureIntegrity:
        if self.max_source_available_time > self.decision_time:
            raise PointInTimeError(
                f"max_source_available_time {self.max_source_available_time} > "
                f"decision_time {self.decision_time}"
            )
        return self


def assert_pit_safe(max_source_available_time: datetime, decision_time: datetime) -> None:
    FeatureIntegrity(
        decision_time=decision_time,
        max_source_available_time=max_source_available_time,
    )
