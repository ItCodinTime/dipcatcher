"""paper/quantile_signals edge paths: estimator history guards, QuantilePolicy
validation matrix, and load_deep_bars file/PIT/calendar branches."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.paper.quantile_signals import (
    QuantilePolicy,
    _ewma_next_sigma,
    _gpd_mom,
    _require_monotone,
    egarch_l_quantiles,
    empirical_quantiles,
    evt_quantiles,
    ewma_emp_quantiles,
    fhs_quantiles,
    garch_t_quantiles,
    load_deep_bars,
)

pytestmark = pytest.mark.synthetic

TAUS = np.asarray([0.05, 0.5, 0.95])


class TestEstimatorGuards:
    def test_ewma_next_sigma_degenerate(self) -> None:
        assert not np.isfinite(_ewma_next_sigma(np.array([1.0])))
        assert not np.isfinite(_ewma_next_sigma(np.zeros(10)))

    def test_require_monotone(self) -> None:
        with pytest.raises(ValueError, match="non-finite or mis-shaped"):
            _require_monotone(np.array([0.1, 0.2]), 3)
        with pytest.raises(ValueError, match="non-finite or mis-shaped"):
            _require_monotone(np.array([0.1, float("nan"), 0.3]), 3)
        with pytest.raises(ValueError, match="non-monotone"):
            _require_monotone(np.array([0.3, 0.2, 0.1]), 3)

    @pytest.mark.parametrize(
        ("fn", "need"),
        [
            (empirical_quantiles, 60),
            (ewma_emp_quantiles, 60),
            (garch_t_quantiles, 200),
            (fhs_quantiles, 200),
            (egarch_l_quantiles, 200),
        ],
    )
    def test_insufficient_history(self, fn, need) -> None:
        with pytest.raises(ValueError, match="insufficient history"):
            fn(np.zeros(need - 1), TAUS)

    def test_gpd_mom_guards(self) -> None:
        with pytest.raises(ValueError, match="insufficient exceedances"):
            _gpd_mom(np.array([0.01]))
        with pytest.raises(ValueError, match="unstable gpd moments"):
            _gpd_mom(np.ones(10))  # zero variance → v <= 0

    def test_evt_guards_and_fallback(self) -> None:
        with pytest.raises(ValueError, match="insufficient history"):
            evt_quantiles(np.zeros(99), TAUS)
        rng = np.random.default_rng(1)
        out = evt_quantiles(rng.normal(0, 0.01, size=300), np.asarray([0.05, 0.5, 0.95, 0.999]))
        assert np.all(np.diff(out) >= 0)  # monotone after accumulate


class TestPolicyValidation:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"mode": "x"}, "mode"),
            ({"gate_on": "x"}, "gate_on"),
            ({"sizing": "x"}, "sizing"),
            ({"book_vol_target": -1.0}, "book_vol_target"),
            ({"tail_gate": float("nan")}, "tail_gate"),
            ({"mkt_disp_cut": -0.5}, "mkt_disp_cut"),
            ({"fund_cut": -1.0}, "fund_cut"),
            ({"rvol_target": -0.1}, "rvol_target"),
            ({"persist_bars": 0}, "persist_bars"),
            ({"exit_persist": 0}, "exit_persist"),
            ({"top_k": 0}, "top_k"),
            ({"w_alpha": 0.0}, "w_alpha"),
            ({"w_alpha": 1.5}, "w_alpha"),
            ({"gate_out": -1.0}, "gate_out"),
            ({"meta_lookback": 1}, "meta_lookback"),
            ({"rebal_every": 0}, "rebal_every"),
            ({"edge_pow": 0.0}, "edge_pow"),
            ({"rvol_lookback": 1}, "rvol_lookback"),
            ({"kappa": float("nan")}, "kappa"),
            ({"gross_target": -0.5}, "gross_target"),
            ({"name_cap": -0.1}, "name_cap"),
            ({"cost_gate": -0.1}, "cost_gate"),
            ({"deadband": -0.1}, "deadband"),
            ({"band_lo": 0.0}, "band"),
            ({"band_hi": 1.0}, "band"),
            ({"band_lo": 0.9, "band_hi": 0.5}, "band"),
        ],
    )
    def test_post_init_rejects(self, overrides, match) -> None:
        with pytest.raises(ValueError, match=match):
            QuantilePolicy(**overrides)

    def test_defaults_ok(self) -> None:
        policy = QuantilePolicy()
        assert policy.mode == "long_flat"


class TestLoadDeepBars:
    def _bars(self, sym: str, n: int, start: datetime) -> pl.DataFrame:
        closes = 100 + np.arange(n, dtype=float)
        return pl.DataFrame(
            {
                "security_id": [sym] * n,
                "event_time": [start + timedelta(hours=i) for i in range(n)],
                "close": closes,
                "volume": [10.0] * n,
            }
        )

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no 1h bars"):
            load_deep_bars(tmp_path, ["AAA"], "1h")

    def test_shared_calendar_clips_ragged_edges(self, tmp_path: Path) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        a = self._bars("AAA", 30, t0)
        b = self._bars("BBB", 30, t0 + timedelta(hours=5))
        a.write_parquet(tmp_path / "aaa_1h_deep.parquet")
        b.write_parquet(tmp_path / "bbb_1h_deep.parquet")
        panel = load_deep_bars(tmp_path, ["AAA", "BBB"], "1h")
        assert panel.group_by("security_id").len()["len"].min() == 25
        assert "adv" in panel.columns
        assert "vol_20" in panel.columns

    def test_no_shared_calendar(self, tmp_path: Path) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        a = self._bars("AAA", 10, t0)
        b = self._bars("BBB", 10, t0 + timedelta(days=5))
        a.write_parquet(tmp_path / "aaa_1h_deep.parquet")
        b.write_parquet(tmp_path / "bbb_1h_deep.parquet")
        with pytest.raises(ValueError, match="shared calendar"):
            load_deep_bars(tmp_path, ["AAA", "BBB"], "1h")

    def test_pit_violation(self, tmp_path: Path) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        frame = self._bars("AAA", 10, t0).with_columns(
            (pl.col("event_time") - pl.duration(hours=2)).alias("available_time")
        )
        frame.write_parquet(tmp_path / "aaa_1h_deep.parquet")
        with pytest.raises(ValueError, match="PIT violation"):
            load_deep_bars(tmp_path, ["AAA"], "1h")

    def test_plain_suffix_and_existing_adv(self, tmp_path: Path) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        frame = self._bars("AAA", 10, t0).with_columns(
            pl.lit(1.0).alias("adv"), pl.lit(0.5).alias("vol_20")
        )
        frame.write_parquet(tmp_path / "aaa_1h.parquet")
        panel = load_deep_bars(tmp_path, ["AAA"], "1h")
        assert panel["adv"][0] == 1.0
        assert panel["vol_20"][0] == 0.5
