"""SYNTHETIC closed-form tests for regime-split performance reporting.

Every numeric assertion is hand-computed from the constructed path. Not market
evidence; research diagnostic only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.reporting.regime_performance import (
    benchmark_drawdown_labels,
    build_regime_performance_from_equity,
    build_regime_performance_report,
    bundled_h15_rate_map,
    equity_returns,
    h15_rate_regime_labels,
    regime_performance_markdown,
    summarize_by_regime,
    volatility_tercile_labels,
)

T0 = datetime(2020, 1, 2, tzinfo=UTC)  # Thursday; aligns with H.15 business days


def test_summarize_by_regime_closed_form() -> None:
    """Hand labels → exact mean / compound / hit rate."""
    rets = np.asarray([0.10, -0.05, 0.20, 0.00], dtype=float)
    labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    stats = summarize_by_regime(rets, labels)
    assert stats["a"]["n"] == 2
    # mean (0.10 + -0.05) / 2
    assert stats["a"]["mean_ret"] == pytest.approx(0.025)
    # (1.10 * 0.95) - 1
    assert stats["a"]["total_return"] == pytest.approx(0.045)
    assert stats["a"]["hit_rate"] == pytest.approx(0.5)
    assert stats["b"]["n"] == 2
    assert stats["b"]["mean_ret"] == pytest.approx(0.10)
    # (1.20 * 1.00) - 1
    assert stats["b"]["total_return"] == pytest.approx(0.20)
    assert stats["b"]["hit_rate"] == pytest.approx(0.5)


def test_benchmark_drawdown_state_closed_form() -> None:
    """Known underwater path: out, in, in, out with exact strategy means."""
    # wealth: 1.10, 1.045, 0.99275, 1.1913
    # dd vs peak: 0.0, -0.05, -0.0975..., 0.0
    bench = np.asarray([0.10, -0.05, -0.05, 0.20], dtype=float)
    labels = benchmark_drawdown_labels(bench)
    assert list(labels) == [
        "out_of_drawdown",
        "in_drawdown",
        "in_drawdown",
        "out_of_drawdown",
    ]
    strat = np.asarray([0.10, -0.01, -0.02, 0.05], dtype=float)
    stats = summarize_by_regime(strat, labels)
    assert stats["in_drawdown"]["n"] == 2
    assert stats["in_drawdown"]["mean_ret"] == pytest.approx(-0.015)
    assert stats["out_of_drawdown"]["n"] == 2
    assert stats["out_of_drawdown"]["mean_ret"] == pytest.approx(0.075)
    # compound in: 0.99 * 0.98 - 1 = -0.0298
    assert stats["in_drawdown"]["total_return"] == pytest.approx(-0.0298)


def test_volatility_terciles_with_explicit_cuts() -> None:
    """Lagged vol + fixed cuts → deterministic high_vol membership and mean.

    Window=3 on the alternating block yields lagged population std ≈ 0.0424,
    which sits above the high cut of 0.04 and below is labeled low at 0.0.
    """
    source = np.asarray(
        [0.01, 0.01, 0.01, 0.01, 0.10, 0.01, 0.10, 0.01, 0.10, 0.01, 0.10, 0.01],
        dtype=float,
    )
    strat = np.asarray(
        [0.0, 0.0, 0.0, 0.001, 0.001, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010, 0.010],
        dtype=float,
    )
    cuts = np.asarray([0.02, 0.04], dtype=float)
    labels, used_cuts = volatility_tercile_labels(source, window=3, cuts=cuts)
    assert used_cuts[0] == pytest.approx(0.02)
    assert used_cuts[1] == pytest.approx(0.04)
    high_mask = labels == "high_vol"
    assert int(np.sum(high_mask)) >= 3
    # Every high_vol bar in this construction carries strategy return 0.010.
    stats = summarize_by_regime(strat, labels)
    assert stats["high_vol"]["mean_ret"] == pytest.approx(0.010)
    assert stats["high_vol"]["n"] == int(np.sum(high_mask))
    assert stats["low_vol"]["mean_ret"] == pytest.approx(0.001)


def test_volatility_terciles_closed_form_means() -> None:
    """Hand labels for the three vol buckets → exact means and compounds."""
    labels = np.asarray(
        ["unknown", "unknown"] + ["low_vol"] * 4 + ["mid_vol"] * 4 + ["high_vol"] * 4,
        dtype=object,
    )
    strat = np.asarray(
        [0.0, 0.0] + [0.01] * 4 + [0.02] * 4 + [0.03] * 4,
        dtype=float,
    )
    stats = summarize_by_regime(strat, labels)
    assert stats["low_vol"]["n"] == 4
    assert stats["low_vol"]["mean_ret"] == pytest.approx(0.01)
    assert stats["mid_vol"]["mean_ret"] == pytest.approx(0.02)
    assert stats["high_vol"]["mean_ret"] == pytest.approx(0.03)
    assert stats["low_vol"]["total_return"] == pytest.approx(1.01**4 - 1.0)


def test_h15_overlap_only_closed_form() -> None:
    """Rate terciles scored only on overlapping dates; non-overlap excluded."""
    dates = [
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
        "2020-01-05",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",  # no rate print
    ]
    # Six overlapping prints → proper tercile cuts at 1/3 and 2/3 quantiles.
    # levels: 1,1, 2,2, 3,3 → cuts = (1.333..., 2.666...) with nanquantile
    rate_map = {
        "2020-01-02": 1.0,
        "2020-01-03": 1.0,
        "2020-01-04": 2.0,
        "2020-01-05": 2.0,
        "2020-01-06": 3.0,
        "2020-01-07": 3.0,
    }
    labels, cuts, n_overlap = h15_rate_regime_labels(
        dates, rate_map=rate_map, cuts=np.asarray([1.5, 2.5])
    )
    assert n_overlap == 6
    assert list(labels) == [
        "low_rate",
        "low_rate",
        "mid_rate",
        "mid_rate",
        "high_rate",
        "high_rate",
        "no_overlap",
    ]
    assert cuts[0] == pytest.approx(1.5)
    strat = np.asarray([0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.99], dtype=float)
    stats = summarize_by_regime(strat, labels)
    assert "no_overlap" not in stats  # excluded by contract
    assert stats["low_rate"]["mean_ret"] == pytest.approx(0.01)
    assert stats["mid_rate"]["mean_ret"] == pytest.approx(0.02)
    assert stats["high_rate"]["mean_ret"] == pytest.approx(0.03)
    # The 0.99 on the non-overlapping date must not leak into any regime.
    assert all(s["mean_ret"] < 0.5 for s in stats.values())


def test_h15_bundled_map_has_dgs10_and_overlap_filter() -> None:
    """Bundled DGS10 loads; dates outside the extract stay no_overlap."""
    rates = bundled_h15_rate_map("DGS10")
    assert "2020-03-16" in rates or len(rates) > 100  # covid window is bundled
    # A date far from every crisis window.
    labels, _cuts, n_overlap = h15_rate_regime_labels(
        ["1990-01-02", "1990-01-03"],
        rate_map=rates,
    )
    assert n_overlap == 0
    assert list(labels) == ["no_overlap", "no_overlap"]


def test_full_report_closed_form_integration() -> None:
    """End-to-end report: DD + H.15 overlap with exact bucket means."""
    bench = np.asarray([0.10, -0.05, -0.05, 0.20, 0.0, 0.0], dtype=float)
    strat = np.asarray([0.10, -0.01, -0.02, 0.05, 0.01, 0.02], dtype=float)
    dates = [
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
        "2020-01-05",
        "2020-01-06",
        "2020-01-07",
    ]
    rate_map = {
        "2020-01-02": 1.0,
        "2020-01-03": 1.0,
        "2020-01-04": 2.0,
        "2020-01-05": 2.0,
        # last two dates: no overlap
    }
    report = build_regime_performance_report(
        strat,
        dates=dates,
        benchmark_returns=bench,
        vol_window=3,
        h15_rate_map=rate_map,
        synthetic=True,
        label="SYNTHETIC_REGIME",
    )
    assert report["live_pnl_claim"] is False
    assert report["synthetic"] is True
    assert report["research_only"] is True

    dd = report["benchmark_drawdown"]["regimes"]
    # First four bars match the earlier closed-form DD path; bars 4,5 at peak.
    # dd path: out, in, in, out, out, out
    assert dd["in_drawdown"]["n"] == 2
    assert dd["in_drawdown"]["mean_ret"] == pytest.approx(-0.015)
    assert dd["out_of_drawdown"]["n"] == 4
    assert dd["out_of_drawdown"]["mean_ret"] == pytest.approx(
        float(np.mean([0.10, 0.05, 0.01, 0.02]))
    )

    h15 = report["h15_rate_regime"]
    assert h15["n_overlap"] == 4
    assert h15["n_no_overlap"] == 2
    # Four overlap points → assign_terciles uses the median fallback (n<6):
    # cuts = (1.5, 1.5). Levels 1.0 → low_rate; 2.0 → high_rate.
    assert h15["regimes"]["low_rate"]["mean_ret"] == pytest.approx(0.045)
    assert h15["regimes"]["high_rate"]["mean_ret"] == pytest.approx(float(np.mean([-0.02, 0.05])))
    assert "no_overlap" not in h15["regimes"]
    assert "mid_rate" not in h15["regimes"]

    md = regime_performance_markdown(report)
    assert "SYNTHETIC" in md
    assert "Volatility terciles" in md
    assert "H.15" in md


def test_equity_wrapper_and_cli(tmp_path: Path) -> None:
    """Equity frames round-trip; CLI writes JSON."""
    times = [T0 + timedelta(days=i) for i in range(5)]
    # NAV 100,110,104.5,99.275,119.13  ↔ returns 0.10,-0.05,-0.05,0.20
    nav = [100.0, 110.0, 104.5, 99.275, 119.13]
    eq = pl.DataFrame({"event_time": times, "nav": nav})
    bench = eq
    rets, dates = equity_returns(eq)
    assert rets == pytest.approx([0.10, -0.05, -0.05, 0.20])
    assert dates[0] == "2020-01-02" or dates[0].startswith("2020-01-")

    report = build_regime_performance_from_equity(eq, benchmark=bench, vol_window=3, synthetic=True)
    assert report["n_bars"] == 4
    assert report["benchmark_drawdown"]["n_in_drawdown"] == 2

    eq_path = tmp_path / "eq.parquet"
    out_json = tmp_path / "regime.json"
    eq.write_parquet(eq_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "regime-performance",
            "--equity",
            str(eq_path),
            "--benchmark",
            str(eq_path),
            "--out-json",
            str(out_json),
            "--vol-window",
            "3",
            "--synthetic",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_json.is_file()
    payload = out_json.read_text()
    assert "live_pnl_claim" in payload
    assert "volatility_terciles" in payload


def test_empty_and_mismatch_fail_closed() -> None:
    empty = build_regime_performance_report(np.asarray([], dtype=float), synthetic=True)
    assert empty["status"] == "empty_or_short"
    with pytest.raises(ValueError, match="identical length"):
        summarize_by_regime(np.asarray([0.1]), np.asarray(["a", "b"], dtype=object))
    with pytest.raises(ValueError, match="match strategy"):
        build_regime_performance_report(
            np.asarray([0.1, 0.2]),
            benchmark_returns=np.asarray([0.1]),
        )
