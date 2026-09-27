"""Tests for the Dip Quality Score bench runner (real-data path)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from fx1.bench.run import BENCH_DISCLAIMER, run_dip_bench
from fx1.cli import app

runner = CliRunner()


def _write_series(
    root: Path, symbol: str, closes: list[float], start: date = date(2025, 1, 1)
) -> None:
    days = [start + timedelta(days=i) for i in range(len(closes))]
    pl.DataFrame(
        {
            "symbol": [symbol] * len(closes),
            "close": closes,
            "event_time": days,
        }
    ).write_parquet(root / f"{symbol.lower()}_1d.parquet")


def test_run_dip_bench_receipt_contract(tmp_path: Path):
    # Two assets: one dips 20% and recovers within the horizon, one does not.
    recoverer = [100.0] * 10 + [80.0] * 5 + [100.0] * 10
    sinker = [100.0] * 10 + [75.0] * 15
    _write_series(tmp_path, "RECOV", recoverer)
    _write_series(tmp_path, "SINK", sinker)

    receipt = run_dip_bench(tmp_path, threshold=0.10, horizons_bars={"1m": 10})
    assert receipt["research_only"] is True
    assert receipt["live_pnl_claim"] is False
    assert receipt["disclaimer"] == BENCH_DISCLAIMER
    assert receipt["n_events"] == 2
    assert receipt["baseline_recovery"]["1m"] == pytest.approx(0.5)
    assert receipt["per_asset"] == {
        "RECOV": {"bars": 25, "events": 1},
        "SINK": {"bars": 25, "events": 1},
    }
    # provenance: every input parquet hashed
    assert set(receipt["inputs_sha256"]) == {"recov_1d.parquet", "sink_1d.parquet"}
    assert all(len(h) == 64 for h in receipt["inputs_sha256"].values())
    # proper scores present; honesty gate on metric keys enforced
    metrics = receipt["climatology_forecast"]["metrics"]
    assert "brier_1m" in metrics and "brier_overall" in metrics


def test_run_dip_bench_empty_dir_fails_closed(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no .* series"):
        run_dip_bench(tmp_path)


def test_dipbench_cli_real_data_run(tmp_path: Path):
    _write_series(tmp_path, "RECOV", [100.0] * 5 + [80.0] * 3 + [100.0] * 5)
    out = tmp_path / "receipt.json"
    result = runner.invoke(
        app,
        ["dipbench", "--data-dir", str(tmp_path), "--out", str(out)],
    )
    assert result.exit_code == 0
    report = json.loads(result.stdout)
    assert report["label"] == "research/backtest evidence, not live performance"
    assert report["events"] == 1
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["n_events"] == 1
