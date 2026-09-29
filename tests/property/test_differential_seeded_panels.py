"""Seeded differential fuzzing: reference event loop vs vectorized fast replay.

Plain pytest + ``np.random.default_rng`` loops (50+ seeds). On supported
panels both engines must produce identical fills/positions/NAV — or raise the
same exception type. On unsupported panels the fast path must refuse with
``ValueError`` and the dispatcher must fall back without approximation.
Any one-sided failure or numeric divergence is a P4.2 conformance defect.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.engine import (
    _fast_replay_panel_supported,
    _run_backtest_event_loop,
    run_backtest,
)
from quant_fund.backtest.fast_replay import run_backtest_fast
from tests.property._differential import (
    T0,
    DiffWorkload,
    FaultFlags,
    _research_config,
    compare_results,
    run_pair,
)

FAULT_AXES = [
    "nan_marks",
    "bad_liquidity",
    "zero_vol",
    "flat",
    "extreme_px",
    "bar_gap",
    "sparse",
    "wild",
    "zero",
    "ghost",
    "nonbar",
    "shuffle",
]


def _bars(rng: np.random.Generator, n_days: int, n_assets: int, faults: set[str]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    lo, hi = (1e-3, 1e7) if "extreme_px" in faults else (10.0, 500.0)
    for k in range(n_assets):
        px = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        ctr = px
        for i in range(n_days):
            ret = 0.0 if "flat" in faults else float(rng.normal(0.0005, 0.02))
            open_px = px
            close = max(px * (1.0 + ret), 1e-9)
            ctr = ctr * (1.0 + ret)
            rows.append(
                {
                    "security_id": f"S{k}",
                    "event_time": T0 + timedelta(days=i),
                    "open": open_px,
                    "close": close,
                    "close_total_return": ctr,
                    "volume": 0.0 if "zero_vol" in faults else 1e6,
                    "adv": close * 1e6,
                    "vol_20": 0.02,
                    "source": "synthetic",
                }
            )
            px = close
    bars = pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    if "nan_marks" in faults:
        sid = f"S{int(rng.integers(0, n_assets))}"
        cut = int(rng.integers(1, n_days))
        col = str(rng.choice(["open", "close", "close_total_return"]))
        bars = bars.with_columns(
            pl.when(
                (pl.col("security_id") == sid) & (pl.col("event_time") >= T0 + timedelta(days=cut))
            )
            .then(float("nan"))
            .otherwise(pl.col(col))
            .alias(col)
        )
    if "bad_liquidity" in faults:
        for col in ("adv", "vol_20"):
            if rng.random() < 0.6:
                idx = int(rng.integers(1, bars.height))
                val = float(rng.choice([float("nan"), 0.0, -5.0]))
                bars = bars.with_columns(
                    pl.when(pl.int_range(bars.height) == idx)
                    .then(val)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
    if "bar_gap" in faults:
        # A security stops printing mid-run: marks freeze, fills must too.
        sid = f"S{int(rng.integers(0, n_assets))}"
        cut = int(rng.integers(1, n_days))
        bars = bars.filter(
            ~((pl.col("security_id") == sid) & (pl.col("event_time") >= T0 + timedelta(days=cut)))
        )
    if "shuffle" in faults:
        bars = bars.sample(fraction=1.0, seed=int(rng.integers(1e9)))
    return bars


def _weights(
    rng: np.random.Generator, n_days: int, n_assets: int, faults: set[str]
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(n_days):
        for k in range(n_assets):
            if "sparse" in faults and rng.random() < 0.7:
                continue
            tw = float(rng.uniform(-0.6, 0.6)) if "wild" in faults else 0.4 / n_assets
            rows.append(
                {
                    "event_time": T0 + timedelta(days=i),
                    "security_id": f"S{k}",
                    "target_weight": tw,
                }
            )
    if "ghost" in faults:
        rows.append(
            {"event_time": T0 + timedelta(days=1), "security_id": "GHOST", "target_weight": 0.7}
        )
    if "nonbar" in faults:
        rows.append(
            {"event_time": T0 + timedelta(hours=13), "security_id": "S0", "target_weight": 0.7}
        )
    if not rows:
        return pl.DataFrame(
            {"event_time": [], "security_id": [], "target_weight": []},
            schema={
                "event_time": pl.Datetime("us", "UTC"),
                "security_id": pl.String,
                "target_weight": pl.Float64,
            },
        )
    w = pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    if "zero" in faults:
        w = w.with_columns(pl.lit(0.0).alias("target_weight"))
    return w


def _workload(seed: int) -> DiffWorkload:
    rng = np.random.default_rng(seed)
    n_days = int(rng.integers(3, 21))
    n_assets = int(rng.integers(1, 6))
    faults = set(rng.choice(FAULT_AXES, size=int(rng.integers(1, 4)), replace=False).tolist())
    return DiffWorkload(
        bars=_bars(rng, n_days, n_assets, faults),
        weights=_weights(rng, n_days, n_assets, faults),
        config=_research_config(),
        faults=FaultFlags(),
        notes=sorted(faults),
    )


def _assert_pair(workload: DiffWorkload) -> None:
    ref, fast = run_pair(workload)
    if isinstance(ref, BaseException) and isinstance(fast, BaseException):
        assert type(ref) is type(fast), (
            f"{workload.notes}: ref raised {type(ref).__name__}({ref}) "
            f"but fast raised {type(fast).__name__}({fast})"
        )
        return
    if isinstance(ref, BaseException) or isinstance(fast, BaseException):
        pytest.fail(
            f"{workload.notes}: one engine refused — "
            f"ref={type(ref).__name__} fast={type(fast).__name__}"
        )
    report = compare_results(ref, fast)
    if not report.ok:
        pytest.fail(f"{workload.notes}: [{report.reason}] {report.detail}")


def test_seeded_adversarial_panels_pair_identical() -> None:
    """100+ seeded adversarial panels: identical outputs or identical refusal."""
    for seed in range(120):
        _assert_pair(_workload(seed))


def test_seeded_extreme_nav_and_books() -> None:
    """Initial NAV at the edges: zero, negative, NaN, inf, dust."""
    bars = _bars(np.random.default_rng(1), 5, 2, set())
    weights = _weights(np.random.default_rng(2), 5, 2, set())
    cfg = _research_config()
    for nav0 in (0.0, -100.0, float("nan"), float("inf"), 1e-9, 1e12):
        try:
            ref = _run_backtest_event_loop(bars, weights, cfg, initial_nav=nav0)
        except BaseException as exc:  # noqa: BLE001
            ref = exc
        try:
            fast = run_backtest_fast(bars, weights, cfg, initial_nav=nav0)
        except BaseException as exc:  # noqa: BLE001
            fast = exc
        if isinstance(ref, BaseException) or isinstance(fast, BaseException):
            assert type(ref) is type(fast), f"nav0={nav0}: {ref!r} vs {fast!r}"
        else:
            report = compare_results(ref, fast)
            assert report.ok, f"nav0={nav0}: [{report.reason}] {report.detail}"


def test_unsupported_panels_fast_refuses_never_approximates() -> None:
    """Empty/clock-mismatched panels: fast raises ValueError; dispatcher degrades."""
    rng = np.random.default_rng(3)
    bars = _bars(rng, 5, 2, set())
    weights = _weights(rng, 5, 2, set())
    cfg = _research_config()

    unsupported = {
        "empty_bars": (bars.clear(), weights),
        "weights_date_clock": (
            bars,
            weights.with_columns(pl.col("event_time").cast(pl.Date)),
        ),
        "weights_ms_clock": (
            bars,
            weights.with_columns(pl.col("event_time").cast(pl.Datetime("ms", "UTC"))),
        ),
        "duplicate_bar_keys": (pl.concat([bars, bars.head(1)]), weights),
    }
    for name, (b, w) in unsupported.items():
        assert not _fast_replay_panel_supported(b, w), name
        with pytest.raises(ValueError):
            run_backtest_fast(b, w, cfg)
        # Explicit fast=True is the same fail-closed contract.
        with pytest.raises(ValueError):
            run_backtest(b, w, cfg, fast=True)


def test_dispatcher_falls_back_without_approximation() -> None:
    """On unsupported panels run_backtest auto-routes to the event loop verbatim."""
    rng = np.random.default_rng(4)
    bars = _bars(rng, 6, 3, set())
    weights = _weights(rng, 6, 3, set())
    cfg = _research_config()
    for b, w in [
        (bars.clear(), weights),
        (bars, weights.with_columns(pl.col("event_time").cast(pl.Date))),
    ]:
        ref = _run_backtest_event_loop(b, w, cfg)
        auto = run_backtest(b, w, cfg)
        report = compare_results(ref, auto)
        assert report.ok, f"dispatcher diverged: [{report.reason}] {report.detail}"
