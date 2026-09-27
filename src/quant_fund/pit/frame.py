"""PitFrame: polars DataFrame + provenance, fail-closed (DESIGN.md §4.3).

The ONLY object ``PitVault.asof()`` returns. Rows are guaranteed
``known_at <= asof`` — the guarantee is re-checked by ``validate()`` so a
PitFrame can never be smuggled past a decision-time boundary unchecked.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import polars as pl

from quant_fund.proofcore.contracts import (
    VaultError,
    sha256_hex_bytes,
)
from quant_fund.proofcore.contracts import (
    VaultUnavailableError as _ContractsVaultUnavailableError,
)
from quant_fund.schemas.errors import PointInTimeError


class VaultUnavailableError(_ContractsVaultUnavailableError, PointInTimeError):
    """asof(t) requested before any version with known_at <= t exists.

    Boundary mapping (DESIGN.md §4.3/§8.3): subclasses BOTH
    ``contracts.VaultError`` and ``schemas.errors.PointInTimeError`` so
    existing ``except PointInTimeError`` callers catch vault failures
    unchanged. Neither base carries state; multiple inheritance is safe.
    """


def frame_content_sha256(frame: pl.DataFrame) -> str:
    """sha256 over canonical parquet bytes of the frame (in-memory write)."""
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return sha256_hex_bytes(buffer.getvalue())


@dataclass(frozen=True)
class PitFrame:
    """polars DataFrame + provenance. The ONLY object asof() returns."""

    frame: pl.DataFrame  # rows guaranteed known_at <= asof
    dataset: str
    asof: datetime  # tz-aware UTC
    max_known_at: datetime  # == frame["known_at"].max(), or asof if empty
    rows: int
    content_sha256: str  # sha256 over canonical parquet bytes of frame

    @classmethod
    def build(cls, frame: pl.DataFrame, *, dataset: str, asof: datetime) -> PitFrame:
        rows = frame.height
        max_known_at = cast(datetime, frame["known_at"].max()) if rows else asof
        return cls(
            frame=frame,
            dataset=dataset,
            asof=asof,
            max_known_at=max_known_at,
            rows=rows,
            content_sha256=frame_content_sha256(frame),
        )

    def validate(self, decision_time: datetime) -> None:
        """Fail-closed re-check: no row may carry known_at > decision_time."""
        if decision_time.tzinfo is None or decision_time.tzinfo.utcoffset(decision_time) is None:
            raise VaultError("decision_time must be timezone-aware (UTC)")
        if self.rows and self.max_known_at > decision_time:
            raise VaultUnavailableError(
                f"{self.dataset}: frame max known_at {self.max_known_at.isoformat()} exceeds "
                f"decision_time {decision_time.isoformat()} — refusing unobservable data"
            )
