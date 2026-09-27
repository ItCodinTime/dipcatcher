"""Cross-vendor reconciliation: pairwise diff, report shape, CLI exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from quant_fund.data.vendors import get_vendor_adapter
from quant_fund.data.vendors.reconcile import (
    ReconcileTolerances,
    main,
    reconcile_frames,
    reconcile_pair,
    render_console,
)

from .helpers import fixture_body

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "vendors"


def _frame(vendor: str, rel: str, symbol: str = "SPY") -> pl.DataFrame:
    return get_vendor_adapter(vendor).parse_body(fixture_body(rel), symbol=symbol)


@pytest.fixture()
def spy_frames() -> dict[str, pl.DataFrame]:
    return {
        "tiingo": _frame("tiingo", "tiingo/spy_daily.json"),
        "polygon": _frame("polygon", "polygon/spy_aggs.json"),
        "alpha_vantage": _frame("alpha_vantage", "alphavantage/spy_daily.json"),
        "stooq": _frame("stooq", "stooq/spy_daily.csv"),
    }


def test_clean_pair_within_tolerance(spy_frames: dict[str, pl.DataFrame]) -> None:
    report = reconcile_frames(
        {"tiingo": spy_frames["tiingo"], "alpha_vantage": spy_frames["alpha_vantage"]}
    )
    assert report["status"] == "clean"
    assert report["mismatch_count"] == 0
    pair = report["pairs"][0]
    assert pair["shared_bars"] == 4
    assert pair["matched_bars"] == 4
    assert pair["mismatches"] == []


def test_divergent_bar_is_flagged(spy_frames: dict[str, pl.DataFrame]) -> None:
    # The polygon fixture intentionally diverges on 2024-01-04.
    pair = reconcile_pair("tiingo", spy_frames["tiingo"], "polygon", spy_frames["polygon"])
    flagged_fields = {(m["event_time"], m["field"]) for m in pair.mismatches}
    assert pair.shared_bars == 4
    assert pair.divergent_bars == 1
    assert ("2024-01-04T21:00:00+00:00", "close") in flagged_fields
    assert ("2024-01-04T21:00:00+00:00", "high") in flagged_fields
    assert ("2024-01-04T21:00:00+00:00", "volume") in flagged_fields
    close_diff = next(m for m in pair.mismatches if m["field"] == "close")
    assert close_diff["left"] == pytest.approx(471.09)
    assert close_diff["right"] == pytest.approx(467.00)
    assert close_diff["rel_diff"] == pytest.approx(0.00868, abs=1e-4)


def test_full_four_vendor_report_all_pairs(spy_frames: dict[str, pl.DataFrame]) -> None:
    report = reconcile_frames(spy_frames)
    assert report["status"] == "divergent"
    # C(4,2)=6 pairs; the divergent bar appears in the 3 pairs involving polygon.
    assert len(report["pairs"]) == 6
    flagged = [p for p in report["pairs"] if p["mismatches"]]
    assert len(flagged) == 3
    assert all("polygon" in (p["left"], p["right"]) for p in flagged)
    assert report["mismatch_count"] == 9
    # Report is JSON-serializable end to end.
    assert json.loads(json.dumps(report))["status"] == "divergent"


def test_unshared_bars_are_reported(spy_frames: dict[str, pl.DataFrame]) -> None:
    truncated = spy_frames["tiingo"].head(3)
    report = reconcile_frames(
        {"tiingo": truncated, "stooq": spy_frames["stooq"]}
    )
    pair = report["pairs"][0]
    assert pair["shared_bars"] == 3
    assert pair["right_only"] == ["SPY@2024-01-05T21:00:00+00:00"]
    assert report["status"] == "divergent"  # unshared bars count as divergence
    assert report["unshared_bar_count"] == 1


def test_tolerance_override_can_legitimize_small_diffs(
    spy_frames: dict[str, pl.DataFrame],
) -> None:
    # tiingo vs stooq volumes differ by ~1e-5 rel — a pathological volume
    # tolerance would flag it; a sane one does not.
    strict = reconcile_frames(
        {"tiingo": spy_frames["tiingo"], "stooq": spy_frames["stooq"]},
        tolerances=ReconcileTolerances(rel=1e-9, abs=0.0, volume_rel=1e-9),
    )
    assert strict["status"] == "divergent"  # volume 61,234,567 vs 61,234,000
    relaxed = reconcile_frames(
        {"tiingo": spy_frames["tiingo"], "stooq": spy_frames["stooq"]},
        tolerances=ReconcileTolerances(rel=1e-4, abs=1e-9, volume_rel=0.05),
    )
    assert relaxed["status"] == "clean"


def test_reconcile_requires_two_frames(spy_frames: dict[str, pl.DataFrame]) -> None:
    with pytest.raises(ValueError, match="at least two"):
        reconcile_frames({"tiingo": spy_frames["tiingo"]})
    with pytest.raises(ValueError, match="join keys"):
        reconcile_pair("a", pl.DataFrame({"x": [1]}), "b", spy_frames["tiingo"])


def test_render_console_mentions_status(spy_frames: dict[str, pl.DataFrame]) -> None:
    report = reconcile_frames(
        {"tiingo": spy_frames["tiingo"], "polygon": spy_frames["polygon"]}
    )
    text = render_console(report)
    assert "tiingo vs polygon" in text
    assert "divergent" in text
    assert "467" in text


def test_cli_clean_and_divergent_exit_codes(tmp_path: Path) -> None:
    fixture_dir = str(FIXTURES)
    clean = main(
        [
            "--vendor",
            f"tiingo={fixture_dir}/tiingo/spy_daily.json",
            "--vendor",
            f"alpha_vantage={fixture_dir}/alphavantage/spy_daily.json",
            "--symbol",
            "SPY",
        ]
    )
    assert clean == 0
    out = tmp_path / "report.json"
    divergent = main(
        [
            "--vendor",
            f"tiingo={fixture_dir}/tiingo/spy_daily.json",
            "--vendor",
            f"polygon={fixture_dir}/polygon/spy_aggs.json",
            "--symbol",
            "SPY",
            "--json",
            str(out),
        ]
    )
    assert divergent == 1
    report = json.loads(out.read_text())
    assert report["status"] == "divergent"
    assert report["mismatch_count"] == 3


def test_cli_usage_errors() -> None:
    with pytest.raises(SystemExit):  # argparse usage error (exit 2)
        main(["--vendor", "tiingo=missing-file.json", "--vendor", "stooq=x.csv", "--symbol", "SPY"])
    with pytest.raises(SystemExit):
        main(
            [
                "--vendor",
                f"nope={FIXTURES}/tiingo/spy_daily.json",
                "--vendor",
                f"stooq={FIXTURES}/stooq/spy_daily.csv",
                "--symbol",
                "SPY",
            ]
        )
    with pytest.raises(SystemExit):
        main(
            [
                "--vendor",
                f"tiingo={FIXTURES}/tiingo/spy_daily.json",
                "--symbol",
                "SPY",
            ]
        )


def test_reconcile_tolerances_validate() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ReconcileTolerances(rel=-1.0)
