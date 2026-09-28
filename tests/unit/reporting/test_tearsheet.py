"""reporting/tearsheet coverage: fail-closed stubs, all render blocks,
attribution path, markdown + writer. SYNTHETIC equity frames only."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.reporting import tearsheet as T


def _equity(n: int = 60, seed: int = 0, temporal: bool = True) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    nav = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
    if temporal:
        t0 = dt.datetime(2024, 1, 1)
        times: list = [t0 + dt.timedelta(days=i) for i in range(n)]
    else:
        times = list(np.arange(n))
    return pl.DataFrame({"event_time": times, "nav": nav})


class TestNavAndReturns:
    def test_missing_nav_raises(self) -> None:
        with pytest.raises(ValueError, match="nav"):
            T._nav_and_returns(pl.DataFrame({"event_time": [1, 2]}))

    def test_nan_filtered_short(self) -> None:
        eq = pl.DataFrame({"event_time": [1, 2, 3], "nav": [1.0, float("nan"), 1.1]})
        nav, rets = T._nav_and_returns(eq)
        assert nav.size == 2 and rets.size == 1


class TestPeriodReturns:
    def test_monthly_buckets(self) -> None:
        eq = _equity()
        out = T.period_returns_table(eq)
        assert out
        assert all(isinstance(k, str) and len(k) == 7 for k in out)
        # ordered by key
        assert list(out) == sorted(out)

    def test_non_temporal_returns_empty(self) -> None:
        eq = _equity(temporal=False)
        assert T.period_returns_table(eq) == {}

    def test_missing_columns(self) -> None:
        assert T.period_returns_table(pl.DataFrame({"x": [1]})) == {}


class TestBuild:
    def test_empty_equity_fails_closed(self) -> None:
        out = T.build_tearsheet(pl.DataFrame({"nav": [1.0]}))
        assert out["summary"]["status"] == "empty_or_short"
        assert out["live_pnl_claim"] is False
        assert out["research_only"] is True

    def test_full_sheet_blocks(self) -> None:
        eq = _equity().with_columns(
            gross=pl.Series(np.full(60, 1.5)),
            net=pl.Series(np.full(60, 0.2)),
            turnover=pl.Series(np.full(60, 0.1)),
        )
        fills = pl.DataFrame(
            {
                "fill_time": [dt.datetime(2024, 1, 5)] * 3,
                "security_id": ["A", "B", "A"],
                "fee": [1.0, 2.0, 0.5],
                "spread_cost": [0.1, 0.2, 0.1],
                "impact_cost": [0.05, 0.0, 0.02],
            }
        )
        out = T.build_tearsheet(eq, fills=fills, synthetic=True, label="TEST")
        assert out["synthetic"] is True
        assert out["summary"]["n_bars"] == 60
        assert out["summary"]["total_return"] == pytest.approx(eq["nav"][-1] / eq["nav"][0] - 1)
        assert out["execution"]["n_fills"] == 3
        assert out["execution"]["cost_totals"]["fee"] == pytest.approx(3.5)
        assert out["execution"]["mean_turnover"] == pytest.approx(0.1)
        assert out["exposure"]["max_gross"] == pytest.approx(1.5)
        assert out["exposure"]["mean_net"] == pytest.approx(0.2)
        assert out["risk"]["var_95"] >= 0 or np.isfinite(out["risk"]["var_95"])
        assert out["period_returns"]
        assert "equity_curve" in out

    def test_drawdown_episodes_have_times(self) -> None:
        # craft a nav with a deep drawdown + recovery
        nav = np.concatenate(
            [np.linspace(100, 120, 20), np.linspace(120, 80, 15), np.linspace(80, 110, 25)]
        )
        eq = _equity().with_columns(nav=pl.Series(nav))
        out = T.build_tearsheet(eq)
        assert out["drawdown"]["n_episodes"] >= 1
        worst = out["drawdown"]["worst_episodes"][0]
        assert "start" in worst and "end" in worst
        assert worst["trough"] < 0

    def test_attribution_path(self) -> None:
        n = 10
        t0 = dt.datetime(2024, 1, 1)
        days = [t0 + dt.timedelta(days=i) for i in range(n)]
        weights = pl.DataFrame(
            {
                "event_time": [days[i] for i in range(n - 1) for _ in range(2)],
                "security_id": ["A", "B"] * (n - 1),
                "target_weight": [0.5, 0.5] * (n - 1),
            }
        )
        rng = np.random.default_rng(1)
        bars = pl.DataFrame(
            {
                "event_time": [d for d in days for _ in range(2)],
                "security_id": ["A", "B"] * n,
                "close": np.exp(np.cumsum(rng.normal(0.001, 0.02, 2 * n))),
            }
        )
        eq = _equity(n=n, seed=2).with_columns(event_time=pl.Series(days))
        out = T.build_tearsheet(eq, weights=weights, bars=bars, sleeve_map={"A": "s1", "B": "s2"})
        assert "attribution" in out
        assert out["attribution"]["research_only"] is True
        assert "by_sleeve" in out["attribution"]

    def test_safe_moment_degenerate(self) -> None:
        assert np.isnan(T._safe_moment(np.ones(10), 3))  # zero variance
        assert np.isnan(T._safe_moment(np.array([0.1, 0.2]), 3))  # too short


class TestMarkdown:
    def test_renders_all_blocks(self, tmp_path: Path) -> None:
        eq = _equity().with_columns(
            gross=pl.Series(np.full(60, 1.2)),
            net=pl.Series(np.full(60, 0.3)),
        )
        fills = pl.DataFrame(
            {"fill_time": [dt.datetime(2024, 1, 3)], "security_id": ["A"], "fee": [1.0]}
        )
        out = T.build_tearsheet(eq, fills=fills, synthetic=True)
        md = T.tearsheet_markdown(out)
        assert "SYNTHETIC" in md
        assert "## Summary" in md
        assert "## Drawdown" in md
        assert "## Risk" in md
        assert "## Execution" in md
        assert "## Exposure" in md
        assert "worst_episodes" in md
        assert "Period Returns" in md or "## Period" in md

    def test_attribution_markdown_block(self) -> None:
        # minimal fake attribution dict drives the markdown branch
        sheet = {
            "label": "X",
            "summary": {"sharpe": float("nan")},
            "attribution": {
                "total_gross_pnl": 1.0,
                "total_cost": 0.1,
                "total_net_pnl": 0.9,
                "top_contributors": [{"security_id": "A", "net_pnl": 1.0, "gross_pnl": 1.1}],
                "bottom_contributors": [{"security_id": "B", "net_pnl": -0.5, "gross_pnl": -0.4}],
                "by_sleeve": [{"sleeve": "s1", "net_pnl": 0.9}],
            },
        }
        md = T.tearsheet_markdown(sheet)
        assert "Attribution" in md
        assert "n/a" in md  # nan formatted
        assert "top contributors" in md and "sleeve s1" in md

    def test_write(self, tmp_path: Path) -> None:
        out = T.build_tearsheet(_equity())
        p = T.write_tearsheet_md(tmp_path / "sub" / "ts.md", out)
        assert p.exists() and "# Tearsheet" in p.read_text()
