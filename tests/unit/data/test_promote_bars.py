"""Contract tests for ``data.promote.promote_bars`` and the CLI command."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.data.adapters.parquet import ParquetMarketProvider
from quant_fund.data.promote import PROMOTE_RECEIPT_SCHEMA, promote_bars
from quant_fund.data.sources.base import SourceError
from quant_fund.data.sources.storage import write_source_frame
from quant_fund.schemas.errors import PointInTimeError

runner = CliRunner()

_TS = datetime(2024, 1, 3, 21, 0, tzinfo=UTC)


def _frame(source: str = "yahoo", rows: int = 3) -> pl.DataFrame:
    base = _TS
    return pl.DataFrame(
        {
            "security_id": ["AAPL"] * rows,
            "event_time": [base.replace(day=3 + i) for i in range(rows)],
            "available_time": [base.replace(day=3 + i) for i in range(rows)],
            "ingested_time": [datetime.now(UTC)] * rows,
            "source": [source] * rows,
            "revision_id": ["v0"] * rows,
            "open": [100.0 + i for i in range(rows)],
            "high": [101.0 + i for i in range(rows)],
            "low": [99.0 + i for i in range(rows)],
            "close": [100.5 + i for i in range(rows)],
            "volume": [1e6] * rows,
        }
    )


def test_promote_round_trips_through_provider(tmp_path: Path) -> None:
    paths = write_source_frame(_frame(), tmp_path, "yahoo")
    dest = tmp_path / "lake" / "raw"
    result = promote_bars(paths["data"], dest)
    assert result["rows"] == 3
    out = ParquetMarketProvider(dest).get_bars()
    assert out.height == 3
    assert out.get_column("security_id").to_list() == ["AAPL"] * 3


def test_promote_receipt_chains_source_hashes(tmp_path: Path) -> None:
    paths = write_source_frame(_frame(), tmp_path, "yahoo")
    dest = tmp_path / "lake" / "raw"
    result = promote_bars(paths["data"], dest)
    receipt = json.loads(result["receipt"].read_text())
    assert receipt["schema"] == PROMOTE_RECEIPT_SCHEMA
    assert (
        receipt["source_parquet_sha256"] == hashlib.sha256(paths["data"].read_bytes()).hexdigest()
    )
    assert (
        receipt["source_receipt_sha256"]
        == hashlib.sha256(paths["receipt"].read_bytes()).hexdigest()
    )
    assert receipt["source_label"] == "yahoo"
    assert receipt["dest_sha256"] == hashlib.sha256(result["data"].read_bytes()).hexdigest()


def test_promote_receipt_seal_verifies(tmp_path: Path) -> None:
    from quant_fund.research.receipt_v2 import verify_receipt_file

    paths = write_source_frame(_frame(), tmp_path, "yahoo")
    result = promote_bars(paths["data"], tmp_path / "raw")
    verdict = verify_receipt_file(result["receipt"])
    assert verdict["valid"], verdict["errors"]


def test_promote_missing_source_fails(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="not found"):
        promote_bars(tmp_path / "nope.parquet", tmp_path / "dest")


def test_promote_refuses_overwrite_without_force(tmp_path: Path) -> None:
    paths = write_source_frame(_frame(), tmp_path, "yahoo")
    dest = tmp_path / "raw"
    promote_bars(paths["data"], dest)
    with pytest.raises(SourceError, match="already exists"):
        promote_bars(paths["data"], dest)
    # --force replaces and re-receipts
    promote_bars(paths["data"], dest, force=True)


def test_promote_rejects_contract_violations(tmp_path: Path) -> None:
    bad = _frame().with_columns(pl.lit(-1.0).alias("volume"))
    paths = write_source_frame(bad, tmp_path, "yahoo")
    dest = tmp_path / "raw"
    with pytest.raises(PointInTimeError, match="invalid OHLCV"):
        promote_bars(paths["data"], dest)
    assert not (dest / "bars.parquet").exists()


def test_promote_rejects_mixed_source_labels(tmp_path: Path) -> None:
    frame = pl.concat([_frame("yahoo"), _frame("stooq")]).with_columns(
        pl.arange(0, 6).cast(pl.Datetime).alias("_idx")
    )
    # distinct event_times so the dedupe check is not what fires
    frame = frame.with_columns(
        pl.Series(
            "event_time",
            [_TS.replace(day=3 + i) for i in range(6)],
        )
    ).drop("_idx")
    path = tmp_path / "mixed.parquet"
    frame.write_parquet(path)
    with pytest.raises(SourceError, match="exactly one source label"):
        promote_bars(path, tmp_path / "raw")


def test_promote_rejects_bad_filename(tmp_path: Path) -> None:
    paths = write_source_frame(_frame(), tmp_path, "yahoo")
    with pytest.raises(SourceError, match="filename"):
        promote_bars(paths["data"], tmp_path / "raw", filename="../evil.parquet")
    with pytest.raises(SourceError, match="parquet"):
        promote_bars(paths["data"], tmp_path / "raw", filename="bars.csv")


def test_promote_empty_frame_fails(tmp_path: Path) -> None:
    empty = tmp_path / "empty.parquet"
    _frame(rows=0).write_parquet(empty)
    with pytest.raises(SourceError, match="no rows"):
        promote_bars(empty, tmp_path / "raw")


def test_promote_bars_cli_source_resolution(tmp_path: Path) -> None:
    write_source_frame(_frame(), tmp_path, "yahoo")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"data:\n  root: {tmp_path}\n  source: parquet\n")
    result = runner.invoke(
        app,
        [
            "promote-bars",
            "--config",
            str(cfg),
            "--source",
            "yahoo",
            "--dest-dir",
            str(tmp_path / "lake" / "raw"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "lake" / "raw" / "bars.parquet").is_file()


def test_promote_bars_cli_requires_a_source(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"data:\n  root: {tmp_path}\n")
    result = runner.invoke(app, ["promote-bars", "--config", str(cfg)])
    assert result.exit_code != 0
