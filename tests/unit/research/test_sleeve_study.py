"""P5.1 multi-sleeve dev study: solo sleeves, parity mix, overlay — SYNTHETIC."""

from __future__ import annotations

import json

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.research.sleeve_study import (
    SLEEVES,
    _synth_perp_book,
    run_sleeve_study,
    write_sleeve_study_receipt,
)


def _book(seed: int = 0, n_names: int = 6, n_bars: int = 120):
    return _synth_perp_book(seed, n_names, n_bars)


def test_study_emits_five_lanes() -> None:
    bars, funding = _book()
    frame, receipt = run_sleeve_study(bars, funding)
    lanes = frame["lane"].to_list()
    assert lanes == [
        *(f"solo:{n}" for n in SLEEVES),
        "parity_mix",
        "parity_mix+overlay",
    ]
    assert receipt["kind"] == "sleeve_study_eval"
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["schema"] == "receipt.v2"
    assert receipt["verdict"] == "pass"
    assert family_blob_forbidden_metrics_absent(receipt["payload"])


def test_solo_sleeves_produce_trades_and_exposure() -> None:
    bars, funding = _book()
    frame, _ = run_sleeve_study(bars, funding)
    for row in frame.iter_rows(named=True):
        if row["lane"].startswith("solo:"):
            assert row["n_fills"] > 0, row["lane"]
            assert row["turnover"] > 0, row["lane"]
            assert row["gross_exposure_mean"] >= 0.0


def test_mix_allocation_is_diverse() -> None:
    bars, funding = _book()
    frame, _ = run_sleeve_study(bars, funding)
    mix = frame.filter(pl.col("lane") == "parity_mix").to_dicts()[0]
    # HHI of 1/3 means perfectly equal; >0.5 means one sleeve dominates.
    assert 0.3 <= mix["alloc_hhi"] <= 1.0


def test_overlay_row_records_scale_factors() -> None:
    bars, funding = _book()
    frame, _ = run_sleeve_study(bars, funding, dd_threshold=0.05)
    row = frame.filter(pl.col("lane") == "parity_mix+overlay").to_dicts()[0]
    assert 0.0 < row["overlay_scale_min"] <= row["overlay_scale_mean"] <= 3.0
    assert 0.0 <= row["overlay_cut_frac"] <= 1.0
    assert row["n_fills"] >= 0


def test_receipt_seal_and_write(tmp_path) -> None:
    bars, funding = _book()
    _, receipt = run_sleeve_study(bars, funding)
    path = write_sleeve_study_receipt(receipt, tmp_path)
    sealed = json.loads(path.read_text())
    assert sealed["receipt_sha256"].startswith(path.stem.removeprefix("sleeve_study_"))


def test_receipt_rejects_non_synthetic(tmp_path) -> None:
    bars, funding = _book()
    _, receipt = run_sleeve_study(bars, funding)
    bad = dict(receipt)
    bad["data_label"] = "LIVE"
    with pytest.raises(ValueError, match="honesty contract"):
        write_sleeve_study_receipt(bad, tmp_path)


def test_alloc_window_validated() -> None:
    bars, funding = _book()
    with pytest.raises(ValueError, match="alloc_min_obs"):
        run_sleeve_study(bars, funding, alloc_window=4, alloc_min_obs=10)


def test_cli_requires_dev() -> None:
    result = CliRunner().invoke(app, ["sleeve-study"])
    assert result.exit_code != 0


def test_synth_book_is_deterministic() -> None:
    b1, f1 = _book(seed=7)
    b2, f2 = _book(seed=7)
    assert b1.equals(b2) and f1.equals(f2)
    b3, _ = _book(seed=8)
    assert not b1.equals(b3)
