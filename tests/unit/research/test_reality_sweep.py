"""Coverage: reality_sweep pure helpers (grid, locks, windows, stats, ledger).

The sweep driver itself needs network + backtest infra; this file covers the
deterministic machinery around it.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.research import reality_sweep as rs


def _spec() -> dict:
    cfg = AppConfig()
    return {
        "study_id": "test_study",
        "grid": {
            "sweep_reclaim": {
                "lookback": [2, 3],
                "hold_bars": [4, 5],
                "decay": [0.5, 0.9],
                "cluster_id": "dip",
                "family": "discovery",
                "gross_scale": 1.0,
                "max_name": 0.03,
            },
            "slow_trend": {
                "fast_bars": [5, 10],
                "slow_bars": [20, 40],
                "vol_window": [20],
                "cluster_id": "trend",
                "family": "bound",
                "gross_scale": 1.0,
                "max_name": 0.03,
            },
            "equal_weight_long": {
                "cluster_id": "base",
                "family": "calibration",
                "name_cap": 0.03,
                "net_cap": 0.9,
            },
        },
        "expected_trials": {
            "sweep_reclaim": 8,
            "slow_trend": 4,
            "equal_weight_long": 1,
            "total": 13,
        },
        "splits": {
            "train": ["2015-01-01", "2019-01-01"],
            "validation": ["2019-01-01", "2021-01-01"],
            "holdout": ["2021-01-01", "2023-01-01"],
        },
        "costs": {
            "commission_bps": cfg.costs.commission_bps,
            "half_spread_bps": cfg.costs.half_spread_bps,
            "impact_y": cfg.costs.impact_y,
            "bps_per_turnover": cfg.costs.bps_per_turnover,
            "borrow_bps_per_year": cfg.costs.borrow_bps_per_year,
            "financing_bps_per_year": cfg.costs.financing_bps_per_year,
            "participation_limit": cfg.costs.participation_limit,
            "fill": cfg.execution.fill.value,
            "initial_nav": 100_000.0,
        },
        "risk_gate": {
            "max_order_notional": cfg.risk_gate.max_order_notional,
            "max_gross": cfg.risk_gate.max_gross,
            "max_net": cfg.risk_gate.max_net,
            "max_name": cfg.risk_gate.max_name,
            "max_participation": cfg.risk_gate.max_participation,
            "max_predicted_vol": cfg.risk_gate.max_predicted_vol,
            "stale_price_bars": cfg.risk_gate.stale_price_bars,
        },
        "diagnostics": {
            "bootstrap_ci": {"periods": 252, "n_boot": 100, "seed": 3, "alpha": 0.05},
            "pbo": {"s_blocks": 8},
        },
    }


def _scored(
    strategy: str,
    validation: np.ndarray,
    *,
    periodic: float = 0.1,
    degenerate: bool = False,
    family: str = "discovery",
) -> rs.ScoredCell:
    cell = rs.Cell(strategy=strategy, cluster_id="c", family=family, params={})
    return rs.ScoredCell(
        cell=cell,
        trial_id=cell.trial_id("test_study"),
        by_window={
            "validation": {
                "periodic_ratio": periodic,
                "degenerate": degenerate,
                "n_obs": int(validation.size),
                "skew": 0.0,
                "kurtosis_raw": 3.0,
            }
        },
        validation_returns=validation,
        train_returns=validation.copy(),
        holdout_returns=validation.copy(),
        dates={},
        turnover={name: np.zeros(validation.size) for name in ("train", "validation", "holdout")},
        equity=pl.DataFrame(),
    )


def _bars(n_days: int = 60) -> pl.DataFrame:
    times = [
        datetime(2024, 1, 1, 16, 0, tzinfo=UTC) + __import__("datetime").timedelta(days=i)
        for i in range(n_days)
    ]
    rng = np.random.default_rng(2)
    rows = {
        "security_id": ["AAA"] * n_days + ["BBB"] * n_days,
        "event_time": sorted(times * 2),
        "open": np.tile(100 + rng.normal(0, 1, n_days).cumsum(), 2),
        "high": np.tile(101 + rng.normal(0, 1, n_days).cumsum(), 2),
        "low": np.tile(99 + rng.normal(0, 1, n_days).cumsum(), 2),
        "close": np.tile(100 + rng.normal(0, 1, n_days).cumsum(), 2),
        "volume": np.full(2 * n_days, 1_000_000.0),
        "source": ["synthetic"] * 2 * n_days,
    }
    return pl.DataFrame(rows)


class TestGrid:
    def test_grid_cells_counts_and_order(self) -> None:
        cells = rs.grid_cells(_spec())
        assert len(cells) == 13
        assert cells[0].strategy == "sweep_reclaim"
        assert cells[-1].strategy == "equal_weight_long"
        sr = [c for c in cells if c.strategy == "sweep_reclaim"]
        assert sr[0].params == {
            "decay": 0.5,
            "gross_scale": 1.0,
            "hold_bars": 4,
            "lookback": 2,
            "max_name": 0.03,
        }
        # fast >= slow pairs are skipped
        st = [c for c in cells if c.strategy == "slow_trend"]
        assert all(c.params["fast_bars"] < c.params["slow_bars"] for c in st)
        assert len(st) == 4

    def test_grid_cells_count_mismatch(self) -> None:
        spec = _spec()
        spec["expected_trials"]["sweep_reclaim"] = 9
        with pytest.raises(ValueError, match="cell count"):
            rs.grid_cells(spec)

    def test_grid_cells_total_mismatch(self) -> None:
        spec = _spec()
        spec["expected_trials"]["total"] = 99
        with pytest.raises(ValueError, match="grid has"):
            rs.grid_cells(spec)

    def test_trial_id_deterministic(self) -> None:
        a = rs.Cell("s", "c", "bound", {"x": 1})
        b = rs.Cell("s", "c", "bound", {"x": 1})
        assert a.trial_id("s1") == b.trial_id("s1")
        assert len(a.trial_id("s1")) == 64
        assert a.trial_id("s1") != a.trial_id("s2")


class TestCostLock:
    def test_lock_passes_on_defaults(self) -> None:
        rs.assert_cost_lock(AppConfig(), _spec())

    @pytest.mark.parametrize(
        "section,key",
        [
            ("costs", "commission_bps"),
            ("costs", "impact_y"),
            ("costs", "participation_limit"),
            ("risk_gate", "max_net"),
            ("risk_gate", "stale_price_bars"),
        ],
    )
    def test_lock_catches_drift(self, section: str, key: str) -> None:
        spec = _spec()
        spec[section][key] = float(spec[section][key]) + 1.0
        with pytest.raises(ValueError, match="cost lock"):
            rs.assert_cost_lock(AppConfig(), spec)

    def test_lock_catches_fill_drift(self) -> None:
        spec = _spec()
        spec["costs"]["fill"] = "vwap"
        with pytest.raises(ValueError, match="fill"):
            rs.assert_cost_lock(AppConfig(), spec)


class TestSelectWinner:
    def test_picks_highest_validation_ratio(self) -> None:
        rng = np.random.default_rng(0)
        rows = [
            _scored("a", rng.normal(0, 1, 50), periodic=0.1),
            _scored("b", rng.normal(0, 1, 50), periodic=0.5),
            _scored("c", rng.normal(0, 1, 50), periodic=9.0, degenerate=True),
        ]
        assert rs.select_winner(rows).cell.strategy == "b"

    def test_all_degenerate_raises(self) -> None:
        rows = [_scored("a", np.zeros(10), degenerate=True)]
        with pytest.raises(ValueError, match="degenerate"):
            rs.select_winner(rows)


class TestDatesAndSplits:
    def test_ny_date_requires_tz(self) -> None:
        aware = datetime(2024, 6, 3, 22, 0, tzinfo=UTC)
        assert rs.ny_date(aware) == date(2024, 6, 3)
        with pytest.raises(ValueError):
            rs.ny_date(datetime(2024, 6, 3, 22, 0))

    def test_as_date_variants(self) -> None:
        assert rs._as_date(datetime(2024, 1, 2, 3, 4)) == date(2024, 1, 2)
        assert rs._as_date(date(2024, 1, 2)) == date(2024, 1, 2)
        assert rs._as_date("2024-01-02") is None

    def test_window_bounds(self) -> None:
        bounds = rs.window_bounds(_spec())
        assert bounds["validation"] == (date(2019, 1, 1), date(2021, 1, 1))

    def test_split_arrays_assigns_sessions(self) -> None:
        # bars at 20:00 UTC on consecutive days -> same-date NY sessions
        times = [
            datetime(2018, 12, 31, 20, 0, tzinfo=UTC),
            datetime(2019, 6, 3, 20, 0, tzinfo=UTC),
            datetime(2022, 6, 3, 20, 0, tzinfo=UTC),
        ]
        equity = pl.DataFrame(
            {
                "event_time": times,
                "nav": [100.0, 110.0, 99.0],
                "turnover": [0.0, 0.5, 0.25],
            }
        )
        rets, turns, dates = rs._split_arrays(equity, rs.window_bounds(_spec()))
        assert rets["validation"].shape == (1,)
        assert rets["validation"][0] == pytest.approx(0.1)
        assert rets["holdout"][0] == pytest.approx(-0.1)
        assert turns["holdout"][0] == pytest.approx(0.25)
        # returns start at nav index 1, so the first bar never emits a date
        assert dates["validation"] == [date(2019, 6, 3)]

    def test_split_arrays_empty_frame(self) -> None:
        rets, turns, dates = rs._split_arrays(pl.DataFrame(), rs.window_bounds(_spec()))
        assert all(v.size == 0 for v in rets.values())
        assert dates["train"] == []

    def test_split_arrays_nan_on_zero_nav(self) -> None:
        equity = pl.DataFrame(
            {
                "event_time": [datetime(2019, 6, 3, 20, 0, tzinfo=UTC)] * 3,
                "nav": [0.0, 5.0, 6.0],
            }
        )
        rets, _, _ = rs._split_arrays(equity, {"validation": (date(2019, 1, 1), date(2021, 1, 1))})
        assert np.isnan(rets["validation"][0])


class TestWindowStats:
    def test_period_ratio_degenerate_cases(self) -> None:
        assert rs._period_ratio(np.array([0.01, 0.02]))[3]
        assert rs._period_ratio(np.ones(10))[3]  # sigma = 0 exactly
        assert rs._period_ratio(np.array([0.01, np.nan, 0.02, 0.0, 0.0]))[3]
        # near-constant floats leave a non-zero sigma residue — not degenerate
        assert not rs._period_ratio(np.full(10, 0.01))[3]

    def test_period_ratio_normal(self) -> None:
        rng = np.random.default_rng(1)
        ratio, skew, kurt, deg = rs._period_ratio(rng.normal(0.01, 0.02, 200))
        assert not deg
        assert math.isfinite(ratio) and math.isfinite(skew)
        assert 1.0 < kurt < 20.0

    def test_window_stats_no_bootstrap(self) -> None:
        rng = np.random.default_rng(3)
        stats = rs._window_stats(
            rng.normal(0.001, 0.01, 100),
            np.full(100, 0.2),
            periods=252,
            n_boot=50,
            seed=1,
            alpha=0.05,
            bootstrap=False,
        )
        assert stats["n_obs"] == 100
        assert math.isnan(stats["ratio_ci_low"])
        assert math.isfinite(stats["compounded_return"])
        assert stats["mean_turnover"] == pytest.approx(0.2)
        assert stats["annualized_ratio"] == pytest.approx(
            stats["periodic_ratio"] * math.sqrt(252.0)
        )

    def test_window_stats_with_bootstrap(self) -> None:
        rng = np.random.default_rng(5)
        stats = rs._window_stats(
            rng.normal(0.002, 0.01, 120),
            np.zeros(120),
            periods=252,
            n_boot=50,
            seed=2,
            alpha=0.10,
            bootstrap=True,
        )
        assert math.isfinite(stats["ratio_ci_low"])
        assert stats["ratio_ci_low"] <= stats["ratio_ci_point"] <= stats["ratio_ci_high"]

    def test_window_stats_degenerate(self) -> None:
        stats = rs._window_stats(
            np.array([0.0, 0.0]),
            np.array([]),
            periods=252,
            n_boot=10,
            seed=1,
            alpha=0.05,
            bootstrap=False,
        )
        assert stats["degenerate"]
        assert math.isnan(stats["annualized_ratio"])
        assert stats["compounded_return"] == 0.0  # prod(1+0)-1 over finite zeros
        assert stats["max_drawdown"] <= 0.0
        assert math.isnan(stats["mean_turnover"])


class TestWeightsAndBars:
    def test_equal_weight_long_caps(self) -> None:
        bars = _bars(30)
        out = rs.equal_weight_long(bars, name_cap=0.03, net_cap=1.0)
        # 2 names -> 1/2 each < cap(0.03)? No: 0.5 > 0.03 -> capped to 0.03 each
        w = out["target_weight"].to_numpy()
        assert np.all(w <= 0.03 + 1e-12)
        # gross 0.06 <= net_cap -> no rescale
        assert np.allclose(w, 0.03)
        # tight net cap forces per-day rescale to exactly net_cap
        out2 = rs.equal_weight_long(bars, name_cap=1.0, net_cap=0.5)
        per_day = out2.group_by("event_time").agg(pl.col("target_weight").sum())
        np.testing.assert_allclose(per_day["target_weight"].to_numpy(), 0.5)

    def test_equal_weight_long_invalid_caps(self) -> None:
        with pytest.raises(ValueError):
            rs.equal_weight_long(_bars(5), name_cap=0.0, net_cap=1.0)
        with pytest.raises(ValueError):
            rs.equal_weight_long(_bars(5), name_cap=1.0, net_cap=-1.0)

    def test_weights_for_dispatch(self) -> None:
        bars = _bars(80)
        cell = rs.Cell("equal_weight_long", "c", "bound", {})
        out = rs.weights_for(cell, bars, name_cap=0.03, net_cap=0.9)
        assert "target_weight" in out.columns
        # sweep_reclaim sleeve path
        cell2 = rs.Cell(
            "sweep_reclaim",
            "c",
            "bound",
            {"lookback": 5, "hold_bars": 3, "decay": 0.8, "max_name": 0.05, "gross_scale": 1.0},
        )
        out2 = rs.weights_for(cell2, bars, name_cap=0.03, net_cap=0.9)
        assert "target_weight" in out2.columns
        # slow_trend sleeve path
        cell3 = rs.Cell(
            "slow_trend",
            "c",
            "bound",
            {
                "fast_bars": 5,
                "slow_bars": 20,
                "max_name": 0.05,
                "gross_scale": 1.0,
                "vol_window": 10,
            },
        )
        out3 = rs.weights_for(cell3, bars, name_cap=0.03, net_cap=0.9)
        assert "target_weight" in out3.columns
        with pytest.raises(ValueError, match="unknown strategy"):
            rs.weights_for(rs.Cell("nope", "c", "bound", {}), bars, name_cap=1.0, net_cap=1.0)

    def test_prepare_bars_columns_and_causality(self) -> None:
        out = rs.prepare_bars(_bars(40))
        for col in ("adv", "vol_20", "close_total_return"):
            assert col in out.columns
        # adv is shifted: first bar uses first _dv fallback, no lookahead
        first = out.filter(pl.col("security_id") == "AAA").sort("event_time").head(1)
        assert float(first["adv"][0]) > 0
        assert float(first["vol_20"][0]) == pytest.approx(0.02)  # fallback
        assert out["close_total_return"].equals(out["close"])


class TestScoringHelpers:
    def test_returns_sha_deterministic(self) -> None:
        a = np.array([0.01, -0.02, 0.03])
        assert rs._returns_sha(a) == rs._returns_sha(a.copy())
        assert len(rs._returns_sha(a)) == 64
        assert rs._returns_sha(a) != rs._returns_sha(a[::-1])

    def test_pbo_on_pre_holdout(self) -> None:
        rng = np.random.default_rng(7)
        rows = [
            _scored("a", rng.normal(0, 1, 80)),
            _scored("b", rng.normal(0.1, 1, 80)),
            _scored("c", rng.normal(-0.1, 1, 80)),
        ]
        out = rs.pbo_on_pre_holdout(rows, _spec())
        assert out["n_periods"] == 160
        assert out["s_blocks"] == 8
        assert out["pbo"] is None or 0.0 <= out["pbo"] <= 1.0

    def test_pbo_length_mismatch(self) -> None:
        rows = [_scored("a", np.zeros(60)), _scored("b", np.zeros(70))]
        with pytest.raises(RuntimeError, match="lengths differ"):
            rs.pbo_on_pre_holdout(rows, _spec())

    def test_pbo_too_short(self) -> None:
        rows = [_scored("a", np.zeros(7)), _scored("b", np.zeros(7))]
        out = rs.pbo_on_pre_holdout(rows, _spec())
        assert out["pbo"] is None
        assert "shorter" in out["reason"]

    def test_raw_count_deflated(self) -> None:
        rng = np.random.default_rng(9)
        rows = [
            _scored("a", rng.normal(0, 1, 60), periodic=0.1),
            _scored("b", rng.normal(0, 1, 60), periodic=0.4),
            _scored("c", rng.normal(0, 1, 60), periodic=0.2),
        ]
        winner = rs.select_winner(rows)
        value = rs.raw_count_deflated(rows, winner)
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0

    def test_ledger_row_shape(self) -> None:
        row = _scored("sweep_reclaim", np.arange(30.0) * 0.001, periodic=0.3)
        ledger = rs._ledger_row(
            row, bundle_hash="ab" * 32, created_utc="2026-01-01T00:00:00Z", periods=252.0
        )
        assert ledger.family == "discovery"
        assert ledger.n_obs == 30
        assert ledger.sharpe_periodic == pytest.approx(0.3)


class TestSanitizeAndReport:
    def test_sanitize_recursion(self) -> None:
        raw = {
            "a": float("inf"),
            "b": [1.0, np.float64(-np.inf), {"c": np.int64(3)}],
            "d": "ok",
        }
        out = rs._sanitize(raw)
        assert out["a"] is None
        assert out["b"][1] is None
        assert out["b"][2]["c"] == 3
        assert out["d"] == "ok"

    def test_public_window_keys(self) -> None:
        stats = {
            k: 1.0
            for k in (
                "n_obs",
                "periodic_ratio",
                "annualized_ratio",
                "ratio_ci_low",
                "ratio_ci_high",
                "ratio_ci_point",
                "compounded_return",
                "max_drawdown",
                "mean_turnover",
                "degenerate",
                "extra",
            )
        }
        out = rs._public_window(stats)
        assert "extra" not in out
        assert set(out) == {
            "n_obs",
            "periodic_ratio",
            "annualized_ratio",
            "ratio_ci_low",
            "ratio_ci_high",
            "ratio_ci_point",
            "compounded_return",
            "max_drawdown",
            "mean_turnover",
            "degenerate",
        }

    def test_fmt_variants(self) -> None:
        assert rs._fmt(None) == "null"
        assert rs._fmt(float("nan")) == "null"
        assert rs._fmt(True) == "True"
        assert rs._fmt(0.123456789) == "0.123457"
        assert rs._fmt("x") == "x"

    def test_write_results_markdown(self, tmp_path: Path) -> None:
        win = {
            "n_obs": 10,
            "periodic_ratio": 0.1,
            "annualized_ratio": 1.5,
            "ratio_ci_low": 0.0,
            "ratio_ci_high": 0.2,
            "compounded_return": 0.05,
            "max_drawdown": -0.1,
            "mean_turnover": 0.3,
        }
        trial = {
            "trial_id": "t1",
            "strategy": "sweep_reclaim",
            "params": {"lookback": 2},
            "windows": {"train": win, "validation": win, "holdout": win},
        }
        receipt = {
            "reality_gate": {
                "verdict": "pass",
                "n_trials": 1,
                "best_trial_id": "t1",
                "n_effective_trials": 1,
                "dsr": 0.9,
                "psr": 0.8,
                "pbo": None,
            },
            "pbo": {"pbo": 0.1},
            "deflated_probability_raw_count": 0.85,
            "selected_trial_id": "t1",
            "provenance": {"dataset_sha256": "ab" * 8},
            "receipt_sha256": "cd" * 8,
            "trials": [trial],
            "symbols_included": ["AAA"],
            "symbols_excluded": [],
        }
        out = tmp_path / "sub" / "results.md"
        rs.write_results_markdown(out, receipt, "ef" * 8)
        text = out.read_text()
        assert "# Reality-filter trial" in text
        assert "`pass`" in text
        assert "| sweep_reclaim |" in text


class TestSpecLoading:
    def test_load_spec(self, tmp_path: Path) -> None:
        path = tmp_path / "spec.json"
        path.write_text('{"a": 1}')
        assert rs.load_spec(path) == {"a": 1}
        bad = tmp_path / "bad.json"
        bad.write_text("[1]")
        with pytest.raises(ValueError):
            rs.load_spec(bad)
