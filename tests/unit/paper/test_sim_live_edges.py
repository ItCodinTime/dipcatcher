"""sim_live edge paths: equity-stat guards, the quantile-panel cache, and a
deterministic bench-only run covering composite specs and funding plumbing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.paper.quantile_signals import QuantilePolicy
from quant_fund.paper.sim_live import (
    StrategySlot,
    _bars_per_year,
    _equity_stats,
    _quantile_panel_cached,
    _trailing_window_sum,
    run_sim_live,
)

pytestmark = pytest.mark.synthetic


class TestEquityStatsGuards:
    def test_empty_and_missing_nav(self) -> None:
        assert _equity_stats(pl.DataFrame({"nav": []}), 365.25)["status"] == "no_equity"
        assert _equity_stats(pl.DataFrame({"x": [1.0]}), 365.25)["status"] == "no_equity"

    def test_degenerate_single_mark(self) -> None:
        out = _equity_stats(pl.DataFrame({"nav": [100.0]}), 365.25)
        assert out["status"] == "degenerate"

    def test_non_finite_marks_dropped(self) -> None:
        out = _equity_stats(pl.DataFrame({"nav": [100.0, float("nan"), 101.0]}), 365.25)
        assert out["status"] == "ok" and out["n_marks"] == 2


class TestWindowAndInterval:
    def test_trailing_sum_rejects_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _trailing_window_sum(np.ones(4), 0)

    def test_trailing_sum_is_causal(self) -> None:
        out = _trailing_window_sum(np.array([1.0, 2.0, 4.0, 8.0]), 2)
        assert np.allclose(out, [1.0, 3.0, 6.0, 12.0])

    def test_bars_per_year_table_and_default(self) -> None:
        assert _bars_per_year("1d") == 365.25
        assert _bars_per_year("4h") == 6.0 * 365.25
        assert _bars_per_year("1h") == 24.0 * 365.25
        assert _bars_per_year("15m") == 365.25


def test_quantile_panel_cache_roundtrip(tmp_path: Path) -> None:
    closes = 100.0 * np.cumprod(1.0 + np.random.default_rng(2).standard_normal(90) * 0.01)
    taus = np.asarray([0.1, 0.5, 0.9])
    panel1, stats1, digest1 = _quantile_panel_cached(
        cache_dir=tmp_path / "c",
        sid="AAA",
        spec="empirical",
        window=60,
        taus=taus,
        closes=closes,
        min_history=None,
    )
    assert list(tmp_path.glob("c/*.npz"))
    panel2, stats2, digest2 = _quantile_panel_cached(
        cache_dir=tmp_path / "c",
        sid="AAA",
        spec="empirical",
        window=60,
        taus=taus,
        closes=closes,
        min_history=None,
    )
    assert digest1 == digest2 and stats1 == stats2
    assert np.array_equal(panel1, panel2, equal_nan=True)


def _write_bars(root: Path, sym: str, n: int = 160, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.cumprod(1.0 + rng.standard_normal(n) * 0.01)
    times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in range(n)]
    pl.DataFrame(
        {
            "event_time": times,
            "open": closes * (1.0 - rng.standard_normal(n) * 0.001),
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1e6),
        }
    ).write_parquet(root / f"{sym.lower()}_1d.parquet")


def _cfg(root: Path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = root
    cfg.risk_gate.max_name = 0.9
    cfg.risk_gate.max_net = 0.9
    cfg.risk_gate.max_gross = 4.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.frictionless = True
    return cfg


def _slots(fund_cut: float | None = None) -> tuple[StrategySlot, list[StrategySlot]]:
    champion = StrategySlot("champ", "empirical", QuantilePolicy(fund_cut=fund_cut))
    challengers = [
        StrategySlot("ensemble", "vincent(empirical+ewma_emp)", QuantilePolicy(fund_cut=fund_cut)),
        StrategySlot("gated", "agree(empirical,ewma_emp)", QuantilePolicy(fund_cut=fund_cut)),
    ]
    return champion, challengers


def test_bench_only_receipt_with_composite_specs(tmp_path: Path) -> None:
    root = tmp_path / "bars"
    root.mkdir()
    _write_bars(root, "aaa", seed=1)
    _write_bars(root, "bbb", seed=2)
    champion, challengers = _slots()
    result = run_sim_live(
        bars_root=root,
        symbols=["AAA", "BBB"],
        interval="1d",
        config=_cfg(root),
        champion=champion,
        challengers=challengers,
        window=60,
        tail_bars=120,
        eval_tail_bars=80,
        out_dir=tmp_path / "out",
        run_id="edge",
        bench_only=True,
        n_jobs=1,
    )
    assert result.loop is None
    receipt = result.receipt
    assert receipt["kind"] == "sim_live_bench_receipt"
    assert receipt["live_pnl_claim"] is False
    assert receipt["loop_metrics"]["status"] == "bench_only_no_paper_loop"
    assert set(result.receipt["book_stats"]) == {"champ", "ensemble", "gated"}
    assert result.receipt_path.is_file()
    assert "AAA:empirical" in receipt["forecaster_stats"]


def test_funding_series_and_fund_cut(tmp_path: Path) -> None:
    root = tmp_path / "bars"
    root.mkdir()
    _write_bars(root, "aaa", seed=3)
    _write_bars(root, "bbb", seed=4)
    fund_times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d, hours=8) for d in range(160)]
    pl.DataFrame({"event_time": fund_times, "value": np.full(160, 0.0003)}).write_parquet(
        root / "btcusdt.funding.parquet"
    )
    champion, challengers = _slots(fund_cut=1.0)
    result = run_sim_live(
        bars_root=root,
        symbols=["AAA", "BBB"],
        interval="1d",
        config=_cfg(root),
        champion=champion,
        challengers=challengers,
        window=60,
        tail_bars=100,
        out_dir=tmp_path / "out",
        run_id="fund",
        bench_only=True,
        n_jobs=1,
    )
    assert result.receipt["kind"] == "sim_live_bench_receipt"


def test_fund_cut_without_funding_file(tmp_path: Path) -> None:
    root = tmp_path / "bars"
    root.mkdir()
    _write_bars(root, "aaa", seed=5)
    champion = StrategySlot("champ", "empirical", QuantilePolicy(fund_cut=0.5))
    result = run_sim_live(
        bars_root=root,
        symbols=["AAA"],
        interval="1d",
        config=_cfg(root),
        champion=champion,
        window=60,
        tail_bars=100,
        out_dir=tmp_path / "out",
        run_id="nofund",
        bench_only=True,
        n_jobs=1,
    )
    assert result.receipt["kind"] == "sim_live_bench_receipt"


def test_bench_failure_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from quant_fund.backtest import engine

    root = tmp_path / "bars"
    root.mkdir()
    _write_bars(root, "aaa", seed=6)
    monkeypatch.setattr(
        engine,
        "run_backtest",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    champion = StrategySlot("champ", "empirical", QuantilePolicy())
    result = run_sim_live(
        bars_root=root,
        symbols=["AAA"],
        interval="1d",
        config=_cfg(root),
        champion=champion,
        window=60,
        tail_bars=100,
        out_dir=tmp_path / "out",
        run_id="fail",
        bench_only=True,
        n_jobs=1,
    )
    stats = result.receipt["book_stats"]["champ"]
    assert stats["status"] == "bench_failed" and "boom" in stats["error"]


def test_joblib_parallel_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import joblib

    root = tmp_path / "bars"
    root.mkdir()
    _write_bars(root, "aaa", seed=7)
    monkeypatch.setattr(
        joblib,
        "Parallel",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no workers")),
    )
    champion = StrategySlot("champ", "empirical", QuantilePolicy())
    result = run_sim_live(
        bars_root=root,
        symbols=["AAA"],
        interval="1d",
        config=_cfg(root),
        champion=champion,
        window=60,
        tail_bars=100,
        out_dir=tmp_path / "out",
        run_id="serial",
        bench_only=True,
        n_jobs=1,
    )
    assert result.receipt["kind"] == "sim_live_bench_receipt"
    assert "AAA:empirical" in result.receipt["forecaster_stats"]
