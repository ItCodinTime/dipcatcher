"""End-to-end smoke for operator CLI commands without prior coverage.

Covers kyle-ofi (synthetic + λ dump), vendor-book-map (alias table + remap),
build-labels, validate fail-closed flags, and paper flag validation. These are
documented operator surfaces; the tests pin exit codes and honesty tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app

_CONFIG = Path("configs/research.yaml")


def _require_config() -> None:
    if not _CONFIG.exists():
        pytest.skip("configs/research.yaml missing")


def test_kyle_ofi_cli_synthetic_and_lambda_dump(tmp_path: Path) -> None:
    _require_config()
    dest = tmp_path / "kyle_lambda_series.parquet"
    result = CliRunner().invoke(
        app,
        [
            "kyle-ofi",
            "--config",
            str(_CONFIG),
            "--seed",
            "7",
            "--dump-lambda-series",
            str(dest),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "kyle_λ_depth=" in result.output
    assert "research_only=" in result.output
    assert f"dumped_lambda_series={dest}" in result.output
    assert dest.is_file()
    frame = pl.read_parquet(dest)
    assert frame.height > 0
    assert {"flow", "event_time"}.issubset(set(frame.columns))
    # Both Kyle flows are dumped (signed depth + OFI).
    assert set(frame["flow"].unique().to_list()) == {"ofi", "signed_depth"}


def test_vendor_book_map_alias_table_only() -> None:
    result = CliRunner().invoke(app, ["vendor-book-map", "--vendor", "alpaca"])
    assert result.exit_code == 0, result.output
    assert "vendor=alpaca" in result.output
    assert "best_bid ←" in result.output


def test_vendor_book_map_remaps_parquet(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    raw = pl.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": [
                datetime(2026, 1, 2, 15, 0, tzinfo=UTC),
                datetime(2026, 1, 2, 15, 1, tzinfo=UTC),
            ],
            "bid_price": [100.0, 100.1],
            "ask_price": [100.2, 100.3],
            "bid_size": [500.0, 600.0],
            "ask_size": [400.0, 300.0],
        }
    )
    source = tmp_path / "raw_quotes.parquet"
    raw.write_parquet(source)
    dest = tmp_path / "remapped.parquet"
    result = CliRunner().invoke(
        app,
        [
            "vendor-book-map",
            "--vendor",
            "alpaca",
            "--parquet",
            str(source),
            "--out",
            str(dest),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "remapped rows=2" in result.output
    assert dest.is_file()
    panel = pl.read_parquet(dest)
    assert panel.height == 2
    assert {"best_bid", "best_ask", "security_id", "event_time"}.issubset(set(panel.columns))


def test_vendor_book_map_rejects_unknown_vendor() -> None:
    result = CliRunner().invoke(app, ["vendor-book-map", "--vendor", "nyse"])
    assert result.exit_code != 0
    assert "vendor must be one of" in result.output


def test_build_labels_cli_emits_row_count() -> None:
    _require_config()
    result = CliRunner().invoke(app, ["build-labels", "--config", str(_CONFIG)])
    assert result.exit_code == 0, result.output
    assert "labels rows=" in result.output


def test_validate_cli_claim_live_on_synthetic_fails_closed() -> None:
    _require_config()
    result = CliRunner().invoke(
        app,
        ["validate", "synth-model", "--config", str(_CONFIG), "--claim-live"],
    )
    assert result.exit_code == 1
    assert "synthetic_claimed_as_live" in result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output


def test_validate_cli_emits_json_without_live_promotion() -> None:
    _require_config()
    result = CliRunner().invoke(
        app,
        ["validate", "synth-model", "--config", str(_CONFIG)],
    )
    # Exit code reflects research-correctness only; promotion must stay false.
    assert result.exit_code in {0, 1}
    payload = json.loads(result.output.split("DATA_LABEL=")[0])
    assert payload["promote"] is False


def test_paper_cli_rejects_conflicting_halt_flags() -> None:
    result = CliRunner().invoke(app, ["paper", "--halt", "--clear-halt", "--max-steps", "1"])
    assert result.exit_code != 0
    assert "pass only one of" in result.output
