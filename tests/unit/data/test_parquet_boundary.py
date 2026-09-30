"""Role-aware Parquet boundary tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import polars as pl
import pytest

from quant_fund.data.parquet import (
    ParquetContractError,
    parquet_row_count,
    parquet_schema,
    read_derived_parquet,
    read_evidence_parquet,
    read_market_parquet,
    read_vendor_parquet,
)


def test_market_read_requires_event_time_contract(tmp_path) -> None:
    path = tmp_path / "market.parquet"
    pl.DataFrame({"event_time": [datetime(2026, 1, 1, tzinfo=UTC)], "close": [1.0]}).write_parquet(
        path
    )
    out = read_market_parquet(path, dataset="bars", required_columns=("event_time", "close"))
    assert out.height == 1
    with pytest.raises(ParquetContractError, match="must require event_time"):
        read_market_parquet(path, dataset="bars", required_columns=("close",))


def test_vendor_read_is_explicitly_untrusted(tmp_path) -> None:
    path = tmp_path / "vendor.parquet"
    pl.DataFrame({"t": [1], "bp": [2.0]}).write_parquet(path)
    assert read_vendor_parquet(path, dataset="raw quotes").columns == ["t", "bp"]


def test_derived_read_enforces_schema_and_nonempty(tmp_path) -> None:
    path = tmp_path / "derived.parquet"
    pl.DataFrame(schema={"event_time": pl.Datetime("us", "UTC")}).write_parquet(path)
    with pytest.raises(ParquetContractError, match="artifact is empty"):
        read_derived_parquet(path, dataset="equity", required_columns=("event_time",))
    with pytest.raises(ParquetContractError, match="missing required columns"):
        read_derived_parquet(
            path,
            dataset="equity",
            required_columns=("event_time", "nav"),
            allow_empty=True,
        )


def test_evidence_read_checks_hash_before_parsing(tmp_path) -> None:
    path = tmp_path / "evidence.parquet"
    pl.DataFrame({"return": [0.01]}).write_parquet(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert (
        read_evidence_parquet(
            payload,
            dataset="trade log",
            expected_sha256=digest,
            required_columns=("return",),
        ).height
        == 1
    )
    with pytest.raises(ParquetContractError, match="sha256 mismatch"):
        read_evidence_parquet(
            payload,
            dataset="trade log",
            expected_sha256="0" * 64,
            required_columns=("return",),
        )


def test_metadata_helpers(tmp_path) -> None:
    path = tmp_path / "frame.parquet"
    pl.DataFrame({"x": [1, 2]}).write_parquet(path)
    assert parquet_schema(path, dataset="doctor").names() == ["x"]
    assert parquet_row_count(path, dataset="doctor") == 2
