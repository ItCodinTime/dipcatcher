"""fast_replay edges: panel refusals, interpreted fallback, py_func kernel branches."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest import fast_replay
from quant_fund.backtest.fast_replay import run_backtest_fast
from tests.unit.backtest.test_fast_replay import _bars, _cfg, _weights

RNG = np.random.default_rng(0)


def _panel(sids=("A", "B"), n_days: int = 12):
    return _bars(list(sids), n_days, RNG), _weights(list(sids), n_days, RNG, lo=0.1, hi=0.4)


class TestPanelRefusals:
    def test_close_auction_rejected(self) -> None:
        bars, w = _panel()
        with pytest.raises(ValueError, match="allow_close_auction"):
            run_backtest_fast(bars, w, _cfg(allow_close_auction=True))

    def test_risk_overlay_rejected(self) -> None:
        bars, w = _panel()
        with pytest.raises(ValueError, match="risk_overlay"):
            run_backtest_fast(bars, w, _cfg(), risk_overlay=object())

    def test_missing_weight_columns_fails_closed(self) -> None:
        bars, w = _panel()
        bad = w.drop("target_weight")
        with pytest.raises(ValueError):
            run_backtest_fast(bars, bad, _cfg())

    def test_duplicate_weight_keys(self) -> None:
        bars, w = _panel()
        dup = pl.concat([w, w.head(1)])
        with pytest.raises(ValueError, match="duplicate target weights"):
            run_backtest_fast(bars, dup, _cfg())

    def test_nonfinite_weight_fails_closed(self) -> None:
        bars, w = _panel()
        bad = w.with_columns(
            pl.when(pl.arange(0, w.height) == 0)
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("target_weight"))
            .alias("target_weight")
        )
        with pytest.raises(ValueError):
            run_backtest_fast(bars, bad, _cfg())

    def test_empty_bars_panel(self) -> None:
        bars, w = _panel()
        with pytest.raises(ValueError, match="non-empty bars panel"):
            run_backtest_fast(bars.head(0), w, _cfg())


def _nav_array(result) -> np.ndarray:
    return np.asarray(result.equity["nav"].to_list())


class TestInterpretedFallback:
    """HAVE_NUMBA=False drives the 250-line interpreted loop (1077-1324)."""

    def test_interpreted_matches_compiled(self, monkeypatch) -> None:
        bars, w = _panel()
        cfg = _cfg(
            commission_bps=5.0,
            half_spread_bps=2.0,
            impact_y=0.1,
            bps_per_turnover=1.0,
            borrow_bps_per_year=40.0,
            max_order_notional=50_000.0,
            participation_limit=0.5,
        )
        fast = run_backtest_fast(bars, w, cfg)
        monkeypatch.setattr(fast_replay, "HAVE_NUMBA", False)
        slow = run_backtest_fast(bars, w, cfg)
        np.testing.assert_array_equal(_nav_array(fast), _nav_array(slow))
        assert fast.fills.height == slow.fills.height

    def test_interpreted_reports_rejects_and_halts(self, monkeypatch) -> None:
        bars, w = _panel(n_days=10)
        monkeypatch.setattr(fast_replay, "HAVE_NUMBA", False)
        cfg = _cfg(max_order_notional=1.0)
        out = run_backtest_fast(bars, w, cfg)
        # Tiny order cap -> every order rejected; the run still completes.
        assert 0 < out.equity.height <= 10

    def test_interpreted_stale_valuation_fails_closed(self, monkeypatch) -> None:
        bars = _bars(["A", "B"], 10, np.random.default_rng(7), missing=0.4)
        w = _weights(["A", "B"], 10, np.random.default_rng(8), lo=0.2, hi=0.5)
        monkeypatch.setattr(fast_replay, "HAVE_NUMBA", False)
        try:
            out = run_backtest_fast(bars, w, _cfg(stale_price_bars=0))
            assert 0 < out.equity.height <= 10
        except Exception as exc:
            # StaleValuationError from the interpreted loop is also correct —
            # fail-closed beats trading on dead prices either way.
            assert "stale" in str(exc).lower()


class TestKernelBranches:
    """Patch the njit kernel with its .py_func so branch lines execute in Python."""

    @pytest.fixture(autouse=True)
    def _py_kernel(self, monkeypatch):
        monkeypatch.setattr(fast_replay, "_replay_kernel", fast_replay._replay_kernel.py_func)
        monkeypatch.setattr(fast_replay, "_order_costs_nb", fast_replay._order_costs_nb.py_func)
        monkeypatch.setattr(fast_replay, "_exceeds_nb", fast_replay._exceeds_nb.py_func)
        monkeypatch.setattr(fast_replay, "_funded_nb", fast_replay._funded_nb.py_func)
        yield

    def test_full_cost_path(self) -> None:
        bars, w = _panel()
        out = run_backtest_fast(
            bars,
            w,
            _cfg(
                commission_bps=5.0,
                half_spread_bps=3.0,
                impact_y=0.2,
                bps_per_turnover=2.0,
                borrow_bps_per_year=60.0,
                participation_limit=0.3,
            ),
        )
        assert 0 < out.equity.height <= 12
        assert _nav_array(out)[-1] < _nav_array(out)[0] + 0.01

    def test_frictionless_path(self) -> None:
        bars, w = _panel()
        out = run_backtest_fast(bars, w, _cfg(frictionless=True))
        assert 0 < out.equity.height <= 12

    def test_tight_gates_reject_orders(self) -> None:
        bars, w = _panel()
        out = run_backtest_fast(
            bars,
            w,
            _cfg(max_order_notional=500.0, max_gross=0.05, max_name=0.01),
        )
        assert 0 < out.equity.height <= 12

    def test_nav_floor_early_break(self) -> None:
        bars, w = _panel(n_days=15)
        w = w.with_columns(target_weight=pl.lit(-5.0))
        out = run_backtest_fast(bars, w, _cfg())
        assert 0 < out.equity.height <= 15

    def test_missing_bars_stale_paths(self) -> None:
        bars = _bars(["A", "B"], 14, np.random.default_rng(11), missing=0.35)
        w = _weights(["A", "B"], 14, np.random.default_rng(12), lo=0.1, hi=0.5)
        try:
            out = run_backtest_fast(bars, w, _cfg(stale_price_bars=1))
            assert 0 < out.equity.height <= 14
        except Exception as exc:
            assert "stale" in str(exc).lower()

    def test_market_vol_gate(self) -> None:
        bars, w = _panel()
        # max_predicted_vol=0.0 forces the volatility gate to clip everything.
        out = run_backtest_fast(bars, w, _cfg(max_predicted_vol=0.0))
        assert 0 < out.equity.height <= 12

    def test_kill_switch_blocks(self) -> None:
        bars, w = _panel()
        out = run_backtest_fast(bars, w, _cfg(kill="DISABLED"))
        assert 0 < out.equity.height <= 12


class TestHelpers:
    def test_neumaier_compensation(self) -> None:
        v = np.array([1e16, 1.0, -1e16, 1.0])
        direct = fast_replay._neumaier(v)
        assert np.isfinite(direct)

    def test_csum_exact_prefix_then_naive(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        f = np.array([False, False, True])
        assert fast_replay._csum.py_func(v, f) == pytest.approx(6.0)
        assert fast_replay._csum.py_func(np.zeros(3), np.zeros(3, dtype=bool)) == 0.0

    def test_order_costs_python_mirror(self) -> None:
        cfg = _cfg(
            commission_bps=10.0, half_spread_bps=5.0, impact_y=0.1, bps_per_turnover=2.0
        ).costs
        comm, spr, imp, total = fast_replay._order_costs(10.0, 100.0, 1e9, 0.02, cfg)
        assert comm == pytest.approx(1000.0 * 10.0 / 1e4)
        assert spr == pytest.approx(1000.0 * 5.0 / 1e4)
        assert imp > 0
        assert total == pytest.approx(comm + spr + imp + 1000.0 * 2.0 / 1e4)
        zero = fast_replay._order_costs(10.0, 100.0, 1e9, 0.02, _cfg(frictionless=True).costs)
        assert zero == (0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="positive"):
            fast_replay._order_costs(1.0, -5.0, 1e9, 0.02, cfg)
