"""Edge paths for ``adaptive_mix``: validation matrices, softmax/anchor
combination, banded-target hysteresis, and the empty-rows early exit."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest import adaptive_mix

pytestmark = pytest.mark.synthetic


def _bars(n: int = 4) -> pl.DataFrame:
    dates = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in range(n)]
    return pl.DataFrame(
        {
            "event_time": [d for d in dates for _ in ("A", "B")],
            "security_id": ["A", "B"] * n,
        }
    ).sort(["event_time", "security_id"])


def _targets(weights: dict[str, float] | None = None) -> pl.DataFrame:
    rows = _bars(4)
    w = weights or {"A": 0.5, "B": 0.5}
    return rows.with_columns(
        pl.Series([w[sid] for sid in rows["security_id"]]).alias("target_weight")
    )


class TestDenseTargets:
    def test_missing_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="required columns"):
            adaptive_mix.dense_targets(pl.DataFrame({"x": [1]}), _targets())

    def test_duplicate_bars_raise(self) -> None:
        bars = pl.concat([_bars(1), _bars(1)])
        with pytest.raises(ValueError, match="duplicate bars"):
            adaptive_mix.dense_targets(bars, _targets())

    def test_duplicate_targets_raise(self) -> None:
        dup = pl.concat([_targets(), _targets()])
        with pytest.raises(ValueError, match="duplicate targets"):
            adaptive_mix.dense_targets(_bars(4), dup)

    def test_target_outside_panel_raises(self) -> None:
        stray = pl.DataFrame(
            {
                "event_time": [datetime(2025, 1, 1, tzinfo=UTC)],
                "security_id": ["A"],
                "target_weight": [0.5],
            }
        )
        with pytest.raises(ValueError, match="outside bar panel"):
            adaptive_mix.dense_targets(_bars(4), stray)

    def test_non_finite_weight_raises(self) -> None:
        t = _targets({"A": float("nan"), "B": 0.5})
        with pytest.raises(ValueError, match="non-finite"):
            adaptive_mix.dense_targets(_bars(4), t)

    def test_missing_proposal_fills_zero(self) -> None:
        t = _targets().filter(pl.col("security_id") == "A")
        dense = adaptive_mix.dense_targets(_bars(4), t)
        assert dense.height == 8
        assert dense.filter(pl.col("security_id") == "B")["target_weight"].to_list() == [0.0] * 4


class TestTrailingNavAllocations:
    def _equities(self, n: int = 8) -> dict[str, pl.DataFrame]:
        dates = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in range(n)]
        rng = np.random.default_rng(11)
        frames: dict[str, pl.DataFrame] = {}
        for name, drift in (("up", 1.02), ("down", 0.99)):
            nav = 100.0 * np.cumprod(drift + 0.001 * rng.standard_normal(n))
            frames[name] = pl.DataFrame({"event_time": dates, "nav": nav})
        return frames

    def test_empty_and_window_bounds(self) -> None:
        with pytest.raises(ValueError, match="min_obs <= window"):
            adaptive_mix.trailing_nav_allocations({}, window=5, min_obs=2)
        with pytest.raises(ValueError, match="min_obs <= window"):
            adaptive_mix.trailing_nav_allocations(self._equities(), window=3, min_obs=4)
        with pytest.raises(ValueError, match="min_obs <= window"):
            adaptive_mix.trailing_nav_allocations(self._equities(), window=1)

    def test_temperature_and_anchor_bounds(self) -> None:
        eq = self._equities()
        with pytest.raises(ValueError, match="temperature"):
            adaptive_mix.trailing_nav_allocations(eq, temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            adaptive_mix.trailing_nav_allocations(eq, temperature=float("nan"))
        with pytest.raises(ValueError, match="equal_anchor"):
            adaptive_mix.trailing_nav_allocations(eq, equal_anchor=1.5)
        with pytest.raises(ValueError, match="equal_anchor"):
            adaptive_mix.trailing_nav_allocations(eq, equal_anchor=float("nan"))

    def test_short_and_duplicate_calendar(self) -> None:
        dates = [datetime(2024, 1, 1, tzinfo=UTC)]
        one = pl.DataFrame({"event_time": dates, "nav": [100.0]})
        with pytest.raises(ValueError, match="two unique dates"):
            adaptive_mix.trailing_nav_allocations({"a": one})
        dup = pl.DataFrame(
            {
                "event_time": [datetime(2024, 1, 1, tzinfo=UTC)] * 2,
                "nav": [100.0, 100.0],
            }
        )
        with pytest.raises(ValueError, match="two unique dates"):
            adaptive_mix.trailing_nav_allocations({"a": dup})

    def test_calendar_mismatch_and_bad_nav(self) -> None:
        eq = self._equities()
        eq["down"] = eq["down"].with_columns(pl.col("event_time") + pl.duration(days=1))
        with pytest.raises(ValueError, match="calendar mismatch"):
            adaptive_mix.trailing_nav_allocations(eq)
        bad = self._equities()
        bad["down"] = bad["down"].with_columns(pl.col("nav") * -1.0)
        with pytest.raises(ValueError, match="invalid NAV"):
            adaptive_mix.trailing_nav_allocations(bad)

    def test_weights_sum_to_one_and_anchor_blends(self) -> None:
        alloc = adaptive_mix.trailing_nav_allocations(
            self._equities(), window=4, min_obs=2, equal_anchor=0.5
        )
        total = alloc.select("up", "down").sum_horizontal()
        assert np.allclose(total.to_numpy(), 1.0)
        # anchor=1.0 forces exact equal weights regardless of scores.
        flat = adaptive_mix.trailing_nav_allocations(
            self._equities(), window=4, min_obs=2, equal_anchor=1.0
        )
        assert np.allclose(flat["up"].to_numpy(), 0.5)
        # anchor=0 lets the winning sleeve dominate after min_obs.
        hot = adaptive_mix.trailing_nav_allocations(
            self._equities(), window=4, min_obs=2, equal_anchor=0.0
        )
        assert hot["up"][-1] > hot["down"][-1]


class TestMixTargets:
    def test_names_and_columns(self) -> None:
        with pytest.raises(ValueError, match="match target sleeves"):
            adaptive_mix.mix_targets({}, pl.DataFrame({"event_time": []}))
        alloc = pl.DataFrame({"event_time": [datetime(2024, 1, 1, tzinfo=UTC)], "a": [1.0]})
        with pytest.raises(ValueError, match="match target sleeves"):
            adaptive_mix.mix_targets({"b": _targets()}, alloc)

    def test_duplicate_allocation_dates(self) -> None:
        d = datetime(2024, 1, 1, tzinfo=UTC)
        alloc = pl.DataFrame({"event_time": [d, d], "a": [1.0, 1.0]})
        with pytest.raises(ValueError, match="duplicate allocation dates"):
            adaptive_mix.mix_targets({"a": _targets()}, alloc)

    def test_negative_and_unnormalized_weights(self) -> None:
        d = _bars(4)["event_time"].unique().sort()
        bad = pl.DataFrame({"event_time": d, "a": [-0.5] * d.len()})
        with pytest.raises(ValueError, match="invalid allocations"):
            adaptive_mix.mix_targets({"a": _targets()}, bad)
        unnorm = pl.DataFrame({"event_time": d, "a": [0.5] * d.len()})
        with pytest.raises(ValueError, match="sum to one"):
            adaptive_mix.mix_targets({"a": _targets()}, unnorm)

    def test_calendar_mismatch_and_missing_dates(self) -> None:
        t_a = _targets()
        t_b = _targets().with_columns(pl.col("event_time") + pl.duration(days=30))
        d = t_a["event_time"].unique().sort()
        alloc = pl.DataFrame({"event_time": d, "a": [0.5] * d.len(), "b": [0.5] * d.len()})
        with pytest.raises(ValueError, match="calendar mismatch"):
            adaptive_mix.mix_targets({"a": t_a, "b": t_b}, alloc)
        short = pl.DataFrame({"event_time": d.head(2), "a": [0.5] * 2, "b": [0.5] * 2})
        with pytest.raises(ValueError, match="missing target dates"):
            adaptive_mix.mix_targets({"a": t_a, "b": _targets()}, short)

    def test_weighted_sum(self) -> None:
        t_a = _targets({"A": 1.0, "B": 0.0})
        t_b = _targets({"A": 0.0, "B": 1.0})
        d = t_a["event_time"].unique().sort()
        alloc = pl.DataFrame({"event_time": d, "a": [0.25] * d.len(), "b": [0.75] * d.len()})
        mixed = adaptive_mix.mix_targets({"a": t_a, "b": t_b}, alloc)
        assert np.allclose(mixed["target_weight"].to_numpy(), [0.25, 0.75] * d.len())


class TestBandedTargets:
    def test_band_and_schema_guards(self) -> None:
        with pytest.raises(ValueError, match="nonnegative"):
            adaptive_mix.banded_targets(_targets(), band=-1.0)
        with pytest.raises(ValueError, match="required columns"):
            adaptive_mix.banded_targets(pl.DataFrame({"x": [1]}), band=0.1)
        dup = pl.concat([_targets(), _targets()])
        with pytest.raises(ValueError, match="duplicate targets"):
            adaptive_mix.banded_targets(dup, band=0.1)
        nan = _targets({"A": float("nan"), "B": 0.5})
        with pytest.raises(ValueError, match="non-finite"):
            adaptive_mix.banded_targets(nan, band=0.1)

    def test_band_zero_sorts_and_returns_all(self) -> None:
        out = adaptive_mix.banded_targets(_targets(), band=0.0)
        assert out.height == 8

    def test_suppression_and_exit(self) -> None:
        d = [datetime(2024, 1, d, tzinfo=UTC) for d in range(1, 5)]
        t = pl.DataFrame(
            {
                "event_time": d,
                "security_id": ["A"] * 4,
                "target_weight": [0.5, 0.52, 0.9, 0.0],
            }
        )
        out = adaptive_mix.banded_targets(t, band=0.1)
        assert out["target_weight"].to_list() == [0.5, 0.9, 0.0]

    def test_all_suppressed_returns_empty_frame(self) -> None:
        d = [datetime(2024, 1, d, tzinfo=UTC) for d in range(1, 4)]
        t = pl.DataFrame(
            {
                "event_time": d,
                "security_id": ["A"] * 3,
                "target_weight": [0.0, 0.0, 0.0],
            }
        )
        out = adaptive_mix.banded_targets(t, band=0.1)
        assert out.height == 0 and out.columns == t.columns
