"""Old-versus-new equivalence for the data-ingest and feature hot paths.

SYNTHETIC series only. These checks are correctness bounds, not market evidence.
Float32 is intentionally not used on the public outputs: bar thresholds and
long EMA / Wilder recurrences leave the tolerance band.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quant_fund.config.models import UniverseConfig
from quant_fund.data.universe import (
    _vectorized_membership_panel,
    build_membership_panel,
    membership_asof,
)
from quant_fund.features.bars import (
    dollar_bars,
    dollar_imbalance_bars,
    tick_imbalance_bars,
    tick_run_bars,
    volume_bars,
)
from quant_fund.features.cross_sectional import apply_cross_sectional
from quant_fund.features.cycles import (
    cycle_periodogram,
    goertzel_power,
    hilbert_instantaneous_frequency,
    hilbert_transform_indicator,
)
from quant_fund.features.indicators import (
    _ema_causal,
    _wilder_smooth,
    aroon,
    bollinger,
    cci,
    cmo,
    ema,
    kama,
    macd,
    mfi,
    wma,
)
from quant_fund.microstructure.book_metrics import book_metrics_from_snapshot
from quant_fund.microstructure.synthetic_lob import (
    synthesize_l2_from_bars,
    synthesize_snapshots_from_bars,
)


def _prices(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, size=n)))


def _sizes(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 2.0, size=n)


def _reset_starts(weights: np.ndarray, threshold: float) -> np.ndarray:
    starts: list[int] = []
    acc = 0.0
    start = 0
    for t in range(weights.size):
        acc += float(weights[t])
        if acc >= threshold:
            starts.append(start)
            start = t + 1
            acc = 0.0
    return np.asarray([st for st in starts if st < weights.size - 1], dtype=float)


def _tick_signs_ref(prices: np.ndarray) -> np.ndarray:
    diff = np.diff(prices, prepend=prices[0])
    signs = np.sign(diff)
    for t in range(1, signs.size):
        if signs[t] == 0.0:
            signs[t] = signs[t - 1]
    signs[0] = 1.0 if signs[0] >= 0.0 else -1.0
    return signs


def _run_starts(prices: np.ndarray, expected_ticks: int, alpha: float) -> np.ndarray:
    signs = _tick_signs_ref(prices)
    starts: list[int] = []
    start = 0
    e_share = 0.5
    e_ticks = float(expected_ticks)
    for t in range(prices.size):
        is_buy = 1.0 if signs[t] > 0.0 else 0.0
        e_share = (1.0 - alpha) * e_share + alpha * is_buy
        run_buy = float(np.sum(signs[start : t + 1] > 0.0))
        run_sell = float(np.sum(signs[start : t + 1] < 0.0))
        run = max(run_buy, run_sell)
        thresh = e_ticks * max(e_share, 1.0 - e_share)
        if run >= max(thresh, 2.0):
            starts.append(start)
            start = t + 1
    return np.asarray([st for st in starts if st < prices.size - 1], dtype=float)


def _ema_ref(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.size, np.nan)
    alpha = 2.0 / (window + 1.0)
    out[window - 1] = values[:window].mean()
    for i in range(window, values.size):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _wilder_ref(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.size, np.nan)
    out[window - 1] = values[:window].mean()
    for i in range(window, values.size):
        out[i] = (out[i - 1] * (window - 1) + values[i]) / window
    return out


def _goertzel_ref(values: np.ndarray, period: int) -> float:
    demeaned = values - values.mean()
    coeff = 2.0 * np.cos(2.0 * np.pi / period)
    prev = 0.0
    prev2 = 0.0
    for sample in demeaned:
        state = sample + coeff * prev - prev2
        prev2 = prev
        prev = state
    power = prev2**2 + prev**2 - coeff * prev * prev2
    return float(power / (values.size * values.size))


def _quadrature_ref(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = values.size
    smooth = np.full(n, np.nan)
    for i in range(3, n):
        smooth[i] = (
            4.0 * values[i] + 3.0 * values[i - 1] + 2.0 * values[i - 2] + values[i - 3]
        ) / 10.0
    in_phase = np.full(n, np.nan)
    quad = np.full(n, np.nan)
    for i in range(6, n):
        window = smooth[i - 6 : i + 1]
        if not np.isfinite(window).all():
            continue
        quad[i] = (
            0.0962 * smooth[i]
            + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6]
        )
        in_phase[i] = smooth[i - 3]
    return in_phase, quad


def test_information_bar_bounds_match_reset_loops() -> None:
    prices = _prices(2500)
    sizes = _sizes(2500)
    volume = volume_bars(prices, sizes, 12.0)
    assert np.array_equal(volume["start"], _reset_starts(sizes, 12.0))
    dollar_threshold = float(np.mean(prices * sizes) * 25.0)
    dollar = dollar_bars(prices, sizes, dollar_threshold)
    assert np.array_equal(dollar["start"], _reset_starts(prices * sizes, dollar_threshold))
    # Open/high/low/close index the same ticks. Sums use reduceat (sequential)
    # versus ndarray.sum (pairwise); the gap stays under 1e-12 relative.
    starts = volume["start"].astype(int)
    ends = np.append(starts[1:], prices.size)
    manual_vol = np.array([sizes[a:b].sum() for a, b in zip(starts, ends, strict=True)])
    assert np.allclose(volume["volume"], manual_vol, rtol=1e-12, atol=1e-12)
    run = tick_run_bars(prices[:800], sizes[:800], expected_ticks=15, alpha_ewma=0.2)
    assert np.array_equal(run["start"], _run_starts(prices[:800], 15, 0.2))
    assert tick_imbalance_bars(prices, sizes, 30)["close"].size >= 1
    assert dollar_imbalance_bars(prices, sizes, 30)["close"].size >= 1


def test_ema_matches_scalar_loop_and_wilder_stays_inside_ulp_band() -> None:
    prices = _prices(1500, seed=4)
    assert np.nanmax(np.abs(ema(prices, 12) - _ema_ref(prices, 12))) == 0.0
    # lfilter evaluates v/n + prev*(n-1)/n. The scalar loop evaluates
    # (prev*(n-1) + v)/n. On this path the absolute gap stays ~1e-13.
    assert np.nanmax(np.abs(_wilder_smooth(prices, 14) - _wilder_ref(prices, 14))) < 1e-9
    gapped = prices.copy()
    gapped[:4] = np.nan
    causal = _ema_causal(gapped, 8)
    ref = _ema_ref(prices[4:], 8)
    assert np.nanmax(np.abs(causal[4:] - ref)) == 0.0
    macd_line = macd(prices)
    assert np.isfinite(macd_line["histogram"][-1])


def test_window_indicators_match_slice_reductions() -> None:
    prices = _prices(600, seed=5)
    high = prices * 1.002
    low = prices * 0.998
    volume = _sizes(prices.size, seed=6)
    bands = bollinger(prices, 20)
    manual_sd = np.array([prices[i - 19 : i + 1].std(ddof=0) for i in range(19, prices.size)])
    assert np.array_equal(bands["mid"][19:], _sma_ref(prices, 20)[19:])
    # Reconstructing the band width is within an ulp of the slice std.
    width = (bands["upper"][19:] - bands["mid"][19:]) / 2.0
    assert np.allclose(width, manual_sd, rtol=0.0, atol=1e-12)
    weights = np.arange(1, 11, dtype=float)
    manual_wma = np.array(
        [np.dot(prices[i - 9 : i + 1], weights) / weights.sum() for i in range(9, prices.size)]
    )
    # np.correlate and the slice dot product agree to about an ulp.
    assert np.allclose(wma(prices, 10)[9:], manual_wma, rtol=0.0, atol=1e-12)
    up = aroon(high, low, 25)["up"]
    manual_up = np.array(
        [100.0 * np.argmax(high[i - 24 : i + 1]) / 24.0 for i in range(24, high.size)]
    )
    assert np.array_equal(up[24:], manual_up)
    # KAMA efficiency ratio is a sliding sum of |diff|; the adaptive recursion is unchanged.
    assert np.isfinite(kama(prices, 10)[-1])
    assert np.isfinite(cci(high, low, prices, 20)[-1])
    assert np.isfinite(cmo(prices, 14)[-1])
    assert np.isfinite(mfi(high, low, prices, volume, 14)[-1])


def _sma_ref(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.size, np.nan)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    out[window - 1 :] = (cumulative[window:] - cumulative[:-window]) / window
    return out


def test_goertzel_and_hilbert_match_scalar_forms() -> None:
    prices = _prices(512, seed=8)
    for period in (8, 17, 40, 128):
        assert goertzel_power(prices, period) == pytest.approx(
            _goertzel_ref(prices, period), rel=0, abs=0
        )
    periods, power = cycle_periodogram(prices, np.array([8, 16, 32, 64]))
    manual = np.array([_goertzel_ref(prices, int(p)) for p in periods])
    assert np.max(np.abs(power - manual)) == 0.0
    in_phase, quad = _quadrature_ref(prices)
    transformed = hilbert_transform_indicator(prices)
    # np.convolve associates the 4-tap FIR differently from the scalar sum (~1 ulp).
    assert np.nanmax(np.abs(transformed["in_phase"] - in_phase)) < 1e-12
    assert np.nanmax(np.abs(transformed["quadrature"] - quad)) < 1e-12
    freq = hilbert_instantaneous_frequency(prices, smooth=1)
    assert np.isfinite(freq[20:]).any()


def test_float32_ema_and_bar_thresholds_leave_the_tolerance_band() -> None:
    """Public series stay float64.

    A float32 EMA on a few thousand bars drifts by more than 1e-7 relative to
    the exact float64 recurrence (measured about 3e-7). Rounding a weight of
    ``2**24 + 1`` to float32 drops the extra unit, so a threshold of that
    weight closes a bar one tick later. Both sit outside the equivalence
    band used above, so the hot paths do not downcast.
    """
    prices = _prices(4000, seed=9)
    narrow = prices.astype(np.float32)
    out = np.full(narrow.size, np.nan, dtype=np.float32)
    window = 20
    alpha = np.float32(2.0 / (window + 1.0))
    out[window - 1] = narrow[:window].mean()
    for i in range(window, narrow.size):
        out[i] = alpha * narrow[i] + (np.float32(1.0) - alpha) * out[i - 1]
    wide = ema(prices, window)
    rel = np.nanmax(np.abs(out.astype(np.float64) - wide) / np.maximum(np.abs(wide), 1e-12))
    # Float64 EMA matches the scalar loop exactly. Float32 is outside that band.
    assert rel > 1e-7
    assert ema(prices, window).dtype == np.float64
    # 2**24 + 1 is not a float32 value (ulp is 2). The float64 accumulator
    # closes on every tick; the rounded weights need two ticks to cross.
    coarse = np.array([2.0**24 + 1.0, 2.0**24 + 1.0, 2.0**24 + 1.0])
    rounded = coarse.astype(np.float32).astype(np.float64)
    threshold = float(2.0**24 + 1.0)
    assert not np.array_equal(_reset_starts(coarse, threshold), _reset_starts(rounded, threshold))


def _window_cross_section(
    frame: pl.DataFrame,
    columns: list[str],
    winsor_p: float,
    sector: str | None = None,
) -> pl.DataFrame:
    """Previous per-column ``over`` implementation, kept as the reference."""
    out = frame
    lo, hi = winsor_p, 1.0 - winsor_p
    eligible = pl.col("available_time") <= pl.col("event_time")
    for col in columns:
        source = pl.when(eligible).then(pl.col(col)).otherwise(None)
        q_lo = source.quantile(lo).over("event_time")
        q_hi = source.quantile(hi).over("event_time")
        clipped_base = source.clip(q_lo, q_hi)
        clipped = pl.when(eligible).then(clipped_base).otherwise(None)
        med = clipped_base.median().over("event_time")
        mad = (clipped_base - med).abs().median().over("event_time")
        sd = clipped_base.std().over("event_time")
        scale = pl.when(mad > 1e-12).then(1.4826 * mad).otherwise(sd)
        z = pl.when(scale > 1e-12).then((clipped_base - med) / scale).otherwise(0.0)
        robust_z = pl.when(eligible).then(z).otherwise(None)
        rank = source.rank("average").over("event_time")
        n = source.count().over("event_time")
        pct = pl.when(eligible).then((rank - 0.5) / n).otherwise(None)
        out = out.with_columns(
            clipped.alias(f"winsor_{col}"),
            robust_z.alias(f"cs_z_{col}"),
            pct.alias(f"cs_pct_{col}"),
        )
        if sector is not None and sector in out.columns:
            keys = ["event_time", sector]
            med_s = clipped_base.median().over(keys)
            mad_s = (clipped_base - med_s).abs().median().over(keys)
            sd_s = clipped_base.std().over(keys)
            scale_s = pl.when(mad_s > 1e-12).then(1.4826 * mad_s).otherwise(sd_s)
            z_s = pl.when(scale_s > 1e-12).then((clipped_base - med_s) / scale_s).otherwise(0.0)
            out = out.with_columns(
                pl.when(eligible).then(z_s).otherwise(None).alias(f"cs_z_sector_{col}")
            )
    return out


def test_cross_section_groupby_matches_window_form() -> None:
    rng = np.random.default_rng(3)
    n_time, n_name = 12, 25
    times = [datetime(2021, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n_time)]
    sectors = ["a", "b", None]
    frame = pl.DataFrame(
        {
            "event_time": [t for t in times for _ in range(n_name)],
            "available_time": [t for t in times for _ in range(n_name)],
            "security_id": [f"S{i:02d}" for _ in times for i in range(n_name)],
            "ret_1": rng.normal(size=n_time * n_name),
            "dv_z20": rng.normal(size=n_time * n_name),
            "sector": [sectors[i % 3] for _ in times for i in range(n_name)],
        }
    )
    # One late print must stay out of the cross-section in both forms.
    frame = frame.with_columns(
        pl.when(pl.col("security_id") == "S00")
        .then(pl.col("event_time") + timedelta(days=2))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    columns = ["ret_1", "dv_z20"]
    new = apply_cross_sectional(frame, columns, 0.01, sector="sector").sort(
        ["event_time", "security_id"]
    )
    old = _window_cross_section(frame, columns, 0.01, sector="sector").sort(
        ["event_time", "security_id"]
    )
    assert new.equals(old)


def _panel_bars(
    n_names: int = 8, n_days: int = 40
) -> tuple[pl.DataFrame, pl.DataFrame, list[datetime]]:
    rng = np.random.default_rng(11)
    times = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n_days)]
    ids = [f"S{i:02d}" for i in range(n_names)]
    security_id: list[str] = []
    event_time: list[datetime] = []
    close: list[float] = []
    volume: list[float] = []
    for i, sid in enumerate(ids):
        path = 20.0 + i + np.cumsum(rng.normal(0.0, 0.05, size=n_days))
        vol = rng.uniform(2e5, 8e5, size=n_days) * (1.0 + i)
        for t, px, sz in zip(times, path, vol, strict=True):
            security_id.append(sid)
            event_time.append(t)
            close.append(float(px))
            volume.append(float(sz))
    bars = pl.DataFrame(
        {
            "security_id": security_id,
            "event_time": event_time,
            "available_time": event_time,
            "open": close,
            "high": [c + 0.1 for c in close],
            "low": [c - 0.1 for c in close],
            "close": close,
            "volume": volume,
            "sector": ["tech"] * len(close),
        }
    )
    master = pl.DataFrame(
        {
            "security_id": ids,
            "exchange": ["XNYS"] * n_names,
            "sector": ["tech"] * n_names,
            "industry": ["soft"] * n_names,
            "security_type": ["common_stock"] * n_names,
            "ticker": [f"T{i}" for i in range(n_names)],
            "valid_from": [times[0]] * n_names,
            "valid_to": [None] * n_names,
            "available_time": [times[0]] * n_names,
        }
    )
    return bars, master, times


def _loop_panel(bars, master, timestamps, config, actions=None) -> pl.DataFrame:
    ordered = bars.sort(["security_id", "event_time"])
    frames = [
        membership_asof(ordered, master, t, config, actions=actions, presorted=True).with_columns(
            pl.lit(t).alias("asof")
        )
        for t in timestamps
    ]
    return pl.concat(frames, how="diagonal_relaxed")


def _assert_membership_close(fast: pl.DataFrame, ref: pl.DataFrame) -> None:
    assert set(fast.columns) == set(ref.columns)
    cols = list(ref.columns)
    got = fast.select(cols).sort(["asof", "security_id"])
    old = ref.select(cols).sort(["asof", "security_id"])
    assert got.height == old.height
    for col in cols:
        if col == "adv":
            # Full-series rolling_mean and tail(20).rolling_mean are the same
            # 20 bars. Polars reduces the longer series differently; the gap
            # is a few ulps (well under 1e-9 relative).
            assert np.allclose(
                got[col].to_numpy(),
                old[col].to_numpy(),
                rtol=1e-9,
                atol=1e-6,
                equal_nan=True,
            )
        else:
            assert got[col].equals(old[col]), col


def test_membership_panel_matches_per_timestamp_loop() -> None:
    bars, master, times = _panel_bars()
    config = UniverseConfig(
        min_price=1.0,
        min_adv=0.0,
        min_history_bars=5,
        top_n_adv=4,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )
    asofs = times[25::2]
    fast = _vectorized_membership_panel(bars, master, asofs, config)
    assert fast is not None
    _assert_membership_close(fast, _loop_panel(bars, master, asofs, config))
    built = build_membership_panel(bars, master, asofs, config)
    _assert_membership_close(built, fast)

    actions = pl.DataFrame(
        {
            "security_id": ["S00", "S01"],
            "event_time": [times[30], times[28]],
            "available_time": [times[30], times[28]],
            "action_type": ["delist", "ticker_change"],
            "new_ticker": [None, "NEWT"],
        }
    )
    open_universe = config.model_copy(update={"top_n_adv": None})
    fast_actions = _vectorized_membership_panel(bars, master, asofs, open_universe, actions=actions)
    assert fast_actions is not None
    _assert_membership_close(
        fast_actions, _loop_panel(bars, master, asofs, open_universe, actions=actions)
    )
    changed = fast_actions.filter(pl.col("security_id") == "S01")
    assert changed.filter(pl.col("asof") > times[28])["symbol"].unique().to_list() == ["NEWT"]
    assert changed.filter(pl.col("asof") < times[28])["symbol"].unique().to_list() == ["T1"]


def test_membership_panel_falls_back_when_availability_is_late() -> None:
    bars, master, times = _panel_bars(n_names=4, n_days=30)
    late = bars.with_columns(
        pl.when((pl.col("security_id") == "S00") & (pl.int_range(pl.len()) == 3))
        .then(pl.col("event_time") + timedelta(days=5))
        .otherwise(pl.col("available_time"))
        .alias("available_time")
    )
    config = UniverseConfig(
        min_price=1.0,
        min_adv=0.0,
        min_history_bars=1,
        top_n_adv=None,
        exchanges=["XNYS"],
        security_types=["common_stock"],
    )
    asofs = times[10:15]
    assert _vectorized_membership_panel(late, master, asofs, config) is None
    built = build_membership_panel(late, master, asofs, config)
    _assert_membership_close(built, _loop_panel(late, master, asofs, config))
    assert build_membership_panel(bars, master, [], config).is_empty()


def test_synthetic_l2_panel_matches_snapshot_metrics() -> None:
    rng = np.random.default_rng(12)
    times = [datetime(2024, 6, 3, tzinfo=UTC) + timedelta(days=i) for i in range(40)]
    rows: list[dict[str, object]] = []
    for sid in ("AAA", "BBB"):
        path = 40.0 + np.cumsum(rng.normal(0.0, 0.3, size=len(times)))
        for i, stamp in enumerate(times):
            close = float(abs(path[i]) + 8.0)
            rows.append(
                {
                    "security_id": sid,
                    "event_time": stamp,
                    "available_time": stamp,
                    "open": close * 0.995,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": float(500 + 10 * i),
                }
            )
    bars = pl.DataFrame(rows)
    panel = synthesize_l2_from_bars(bars, depth=5, seed=7, base_spread_bps=4.0)
    snaps = synthesize_snapshots_from_bars(bars, depth=5, seed=7, base_spread_bps=4.0)
    assert panel.height == len(snaps)
    for snap in snaps:
        metrics = book_metrics_from_snapshot(snap)
        row = panel.filter(
            (pl.col("security_id") == snap.security_id) & (pl.col("event_time") == snap.event_time)
        )
        assert row.height == 1
        for key, value in metrics.items():
            got = float(row[key][0])
            # Depth sums and log-slopes use a left fold / matmul. Snapshot
            # metrics use Python sum and np.dot. The gap is a fraction of an ulp.
            if np.isnan(value):
                assert np.isnan(got)
            else:
                assert got == pytest.approx(value, rel=1e-12, abs=1e-9), key
