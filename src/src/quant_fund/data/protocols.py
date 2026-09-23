"""Market data provider protocol. Tests must not require the network."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import polars as pl


class MarketDataProvider(Protocol):
    def get_bars(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        security_ids: list[str] | None = None,
    ) -> pl.DataFrame: ...

    def get_corporate_actions(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame: ...

    def get_security_master(self) -> pl.DataFrame: ...


class SecurityMaster(Protocol):
    def asof(self, when: datetime, ticker: str) -> str | None:
        """Return security_id for ticker at `when`, or None."""
        ...

    def record(self, security_id: str, when: datetime) -> dict[str, str] | None: ...
