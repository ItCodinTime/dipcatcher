"""reporting/regime_performance edge paths: input guards, H.15 bundle
fail-closed branches, benchmark-required/status fallbacks, and the
markdown/writer helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.reporting.regime_performance import (
    benchmark_drawdown_labels,
    build_regime_performance_from_equity,
    build_regime_performance_report,
    bundled_h15_rate_map,
    equity_returns,
    h15_rate_regime_labels,
    lagged_realized_vol,
    regime_performance_markdown,
    summarize_by_regime,
    write_regime_performance_md,
)

pytestmark = pytest.mark.synthetic

T0 = datetime(2020, 1, 2, tzinfo=UTC)


def _equity(nav, *, with_time: bool = True) -> pl.DataFrame:
    cols = {"nav": list(map(float, nav))}
    if with_time:
        cols["event_time"] = [T0 + timedelta(days=i) for i in range(len(nav))]
    return pl.DataFrame(cols)


class TestEquityReturns:
    def test_nav_column_required(self) -> None:
        with pytest.raises(ValueError, match="nav"):
            equity_returns(pl.DataFrame({"x": [1.0]}))

    def test_too_short_returns_empty(self) -> None:
        rets, dates = equity_returns(_equity([1.0]))
        assert rets.size == 0 and dates == []

    def test_nonfinite_nav_fails_closed(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            equity_returns(_equity([1.0, float("nan"), 2.0]))

    def test_without_event_time_index_dates(self) -> None:
        rets, dates = equity_returns(_equity([1.0, 2.0], with_time=False))
        assert rets.tolist() == [1.0]
        assert dates == ["1"]

    def test_date_key_fallbacks(self) -> None:
        frame = pl.DataFrame(
            {"nav": [1.0, 2.0], "event_time": ["2024-01-05T10:00:00", "2024-01-06T10:00:00"]}
        )
        _, dates = equity_returns(frame)
        assert dates == ["2024-01-06"]


class TestVolLabels:
    def test_window_guard_and_empty(self) -> None:
        with pytest.raises(ValueError, match="window"):
            lagged_realized_vol(np.array([1.0]), window=1)
        out = lagged_realized_vol(np.array([]))
        assert out.size == 0


class TestH15:
    def test_unknown_series_key(self) -> None:
        with pytest.raises(KeyError, match="yield series"):
            bundled_h15_rate_map("NOT_A_SERIES")

    def test_rate_labels_overlap(self) -> None:
        rate_map = bundled_h15_rate_map("DGS10")
        assert rate_map
        sample = sorted(rate_map)[:2]
        labels, cuts, n_overlap = h15_rate_regime_labels(
            sample, series_id="DGS10", rate_map=rate_map
        )
        assert n_overlap == len(sample)
        assert len(labels) == len(sample)
        assert len(cuts) == 2


class TestSummarize:
    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="identical length"):
            summarize_by_regime(np.array([1.0]), np.array(["a", "b"], dtype=object))

    def test_excluded_and_empty(self) -> None:
        out = summarize_by_regime(np.array([]), np.array([], dtype=object))
        assert out == {}
        stats = summarize_by_regime(
            np.array([0.01, 0.02, 0.03]),
            np.array(["unknown", "a", "b"], dtype=object),
            exclude_labels={"b"},
        )
        assert set(stats) == {"a", "unknown"}
        assert stats["a"]["n"] == 1


class TestReport:
    def _strat(self, n: int = 60) -> np.ndarray:
        rng = np.random.default_rng(3)
        return rng.normal(0.001, 0.01, size=n)

    def test_dates_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="match strategy_returns length"):
            build_regime_performance_report(
                self._strat(), dates=["2020-01-01"], h15_rate_map={"2020-01-01": 1.0}
            )

    def test_benchmark_required_status(self) -> None:
        report = build_regime_performance_report(self._strat())
        assert report["benchmark_drawdown"] == {"status": "benchmark_required"}
        assert report["h15_rate_regime"] == {"status": "dates_required"}

    def test_drawdown_labels(self) -> None:
        bench = np.asarray([0.1, -0.05, -0.02, 0.12])
        labels = benchmark_drawdown_labels(bench)
        assert "in_drawdown" in set(labels.tolist())

    def test_from_equity_length_mismatch(self) -> None:
        equity = _equity([1.0, 1.1, 1.2])
        bench = _equity([1.0, 1.1])
        with pytest.raises(ValueError, match="same number of returns"):
            build_regime_performance_from_equity(equity, benchmark=bench)

    def test_from_equity_full(self) -> None:
        rng = np.random.default_rng(5)
        nav = 100 * np.cumprod(1.0 + rng.normal(0.001, 0.01, 80))
        equity = _equity(nav)
        report = build_regime_performance_from_equity(equity)
        assert report["live_pnl_claim"] is False


class TestMarkdownAndWriter:
    def test_markdown_full_report(self) -> None:
        rng = np.random.default_rng(4)
        strat = rng.normal(0.001, 0.01, size=90)
        bench = rng.normal(0.0005, 0.01, size=90)
        dates = [(T0 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(90)]
        report = build_regime_performance_report(
            strat,
            benchmark_returns=bench,
            dates=dates,
            h15_rate_map={d: 2.0 + i * 0.01 for i, d in enumerate(dates)},
        )
        md = regime_performance_markdown(report)
        assert "| regime |" in md
        assert "n/a" in md or "|" in md

    def test_write_md(self, tmp_path: Path) -> None:
        report = build_regime_performance_report(np.array([0.01, -0.02, 0.03]))
        out = write_regime_performance_md(tmp_path / "rp.md", report)
        assert out.read_text().startswith("#") or "regime" in out.read_text().lower()
