"""Signal sleeves for the perp/spot books — each emits causal target weights.

Every function returns ``(event_time, security_id, target_weight)`` frames
consumable by ``run_perp_backtest`` / ``run_backtest``. All rolling statistics
are shift-guarded: a weight decided at bar t only sees data from bars strictly
before t's close, so fills at t+1 open are causally sound.

These produce *weight proposals* — leverage, margin and risk-gate enforcement
live in the engines, not here.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from quant_fund.northset.sweeps import liquidity_sweep_frame

_WCOLS = ("event_time", "security_id", "target_weight")


def _validate_bars(bars: pl.DataFrame) -> None:
    missing = {"event_time", "security_id", "close"} - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {sorted(missing)}")


def _cap_and_emit(
    frame: pl.DataFrame,
    raw_col: str,
    *,
    max_name: float,
    gross_scale: float,
) -> pl.DataFrame:
    """Clip names to ±max_name, scale the cross-section to gross_scale.

    ``gross_scale`` bounds sum(|w|) per timestamp — the engine's leverage cap
    still applies on top.
    """
    out = (
        frame.filter(pl.col(raw_col).is_not_null() & pl.col(raw_col).is_finite())
        .with_columns(
            (pl.col(raw_col).clip(-float(max_name), float(max_name))).alias("target_weight")
        )
        .with_columns(
            pl.when(pl.col("target_weight").abs().sum().over("event_time") > gross_scale)
            .then(
                pl.col("target_weight")
                * gross_scale
                / pl.col("target_weight").abs().sum().over("event_time")
            )
            .otherwise(pl.col("target_weight"))
            .alias("target_weight")
        )
        .filter(pl.col("target_weight").abs() > 1e-9)
        .select("event_time", "security_id", "target_weight")
    )
    return out.sort(["event_time", "security_id"])


def _per_symbol_vol(bars: pl.DataFrame, window: int) -> pl.DataFrame:
    """Causal per-bar volatility estimate: rolling std of log returns, shifted."""
    _validate_bars(bars)
    return (
        bars.sort(["security_id", "event_time"])
        .with_columns(
            pl.col("close").log().diff().alias("_r"),
        )
        .with_columns(
            pl.col("_r")
            .rolling_std(window, min_samples=max(2, window // 4))
            .shift(1)
            .over("security_id")
            .alias("_vol")
        )
        .select("security_id", "event_time", "_vol")
    )


def funding_carry_weights(
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    *,
    lookback_events: int = 3,
    max_name: float = 0.05,
    gross_scale: float = 2.0,
    vol_window: int = 48,
) -> pl.DataFrame:
    """Carry sleeve: long symbols whose funding is negative (shorts pay you),
    short symbols with positive funding — sized by rate z-score, inverse-vol.

    Uses only funding events strictly before the decision bar (``join_asof``
    backward on event_time < t). ``lookback_events`` averages the last N
    realized rates (~N×8h of history on Binance's standard grid).
    """
    _validate_bars(bars)
    if funding.height == 0:
        raise ValueError("funding frame must be non-empty")
    if "value" not in funding.columns:
        raise ValueError("funding frame needs a 'value' column (rate)")
    # Rolling mean of the last N realized rates per symbol, stamped at each
    # funding event; as-of join projects it forward to bar times.
    froll = (
        funding.sort(["security_id", "event_time"])
        .with_columns(
            pl.col("value")
            .rolling_mean(lookback_events, min_samples=1)
            .over("security_id")
            .alias("rate_ma")
        )
        .select("security_id", "event_time", "rate_ma")
        .sort(["security_id", "event_time"])
    )
    grid = bars.select("security_id", "event_time", "close").sort(["security_id", "event_time"])
    vols = _per_symbol_vol(bars, vol_window)
    joined = (
        grid.join_asof(
            froll,
            on="event_time",
            by="security_id",
            strategy="backward",
        )
        .join(vols, on=["security_id", "event_time"], how="left")
        .with_columns(
            pl.when(pl.col("_vol").is_not_null() & (pl.col("_vol") > 0))
            .then(-pl.col("rate_ma") / pl.col("_vol"))
            .otherwise(None)
            .alias("_raw")
        )
        .with_columns((pl.col("_raw") - pl.col("_raw").median().over("event_time")).alias("_raw"))
    )
    return _cap_and_emit(joined, "_raw", max_name=max_name, gross_scale=gross_scale)


def funding_spike_fade_weights(
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    *,
    lookback_events: int = 30,
    z_threshold: float = 2.0,
    max_name: float = 0.05,
    gross_scale: float = 1.0,
    vol_window: int = 48,
) -> pl.DataFrame:
    """Crowded-positioning fade: symbols whose funding rate is an extreme
    outlier vs their own history are over-levered on one side — fade them
    (short extreme-positive funding, long extreme-negative). Only acts beyond
    ``z_threshold``; between events the last extreme print's weight is carried
    flat by the backward as-of join until the next funding event re-evaluates."""
    _validate_bars(bars)
    if funding.height == 0:
        raise ValueError("funding frame must be non-empty")
    fz = (
        funding.sort(["security_id", "event_time"])
        .with_columns(
            (
                (
                    pl.col("value")
                    - pl.col("value")
                    .rolling_mean(lookback_events, min_samples=10)
                    .over("security_id")
                )
                / (
                    pl.col("value").rolling_std(lookback_events, min_samples=10).over("security_id")
                    + 1e-8
                )
            ).alias("_z")
        )
        .select("security_id", "event_time", "_z")
    )
    grid = bars.select("security_id", "event_time", "close").sort(["security_id", "event_time"])
    vols = _per_symbol_vol(bars, vol_window)
    joined = (
        grid.join_asof(fz, on="event_time", by="security_id", strategy="backward")
        .join(vols, on=["security_id", "event_time"], how="left")
        .with_columns(
            pl.when(pl.col("_z").abs() >= z_threshold)
            .then(
                -pl.col("_z").sign()
                / pl.when(pl.col("_vol") > 0).then(pl.col("_vol")).otherwise(None)
            )
            .otherwise(0.0)
            .alias("_raw")
        )
    )
    return _cap_and_emit(joined, "_raw", max_name=max_name, gross_scale=gross_scale)


def cross_sectional_momentum_weights(
    bars: pl.DataFrame,
    *,
    lookback_bars: int = 168,
    skip_bars: int = 4,
    max_name: float = 0.05,
    gross_scale: float = 2.0,
    vol_window: int = 48,
) -> pl.DataFrame:
    """Momentum sleeve: cross-sectional rank of return over
    ``[t-lookback, t-skip]`` (skip the freshest bars to blunt reversal),
    demeaned across the universe each bar, inverse-vol sized."""
    _validate_bars(bars)
    if lookback_bars <= skip_bars:
        raise ValueError("lookback_bars must exceed skip_bars")
    base = (
        bars.sort(["security_id", "event_time"])
        .with_columns(
            (pl.col("close").shift(skip_bars) / pl.col("close").shift(lookback_bars) - 1.0)
            .over("security_id")
            .alias("_mom")
        )
        .drop_nulls("_mom")
    )
    vols = _per_symbol_vol(bars, vol_window)
    frame = (
        base.join(vols, on=["security_id", "event_time"], how="left")
        .with_columns(
            (
                pl.col("_mom").rank().over("event_time") / pl.col("_mom").count().over("event_time")
                - 0.5
            ).alias("_rank")
        )
        .with_columns(
            pl.when(pl.col("_vol").is_not_null() & (pl.col("_vol") > 0))
            .then(pl.col("_rank") / pl.col("_vol"))
            .otherwise(None)
            .alias("_raw")
        )
        .with_columns(pl.col("_raw") - pl.col("_raw").median().over("event_time"))
    )
    return _cap_and_emit(frame, "_raw", max_name=max_name, gross_scale=gross_scale)


def sweep_reclaim_weights(
    bars: pl.DataFrame,
    *,
    lookback: int = 20,
    hold_bars: int = 8,
    decay: float = 0.75,
    max_name: float = 0.05,
    gross_scale: float = 1.0,
) -> pl.DataFrame:
    """Dipcatch sleeve: a swept-and-reclaimed prior extreme is a stop-run
    exhaustion — long after low-sweep reclaims, short after high-sweep
    reclaims. The pulse decays geometrically over ``hold_bars``."""
    _validate_bars(bars)
    if hold_bars < 1:
        raise ValueError("hold_bars must be >= 1")
    if not 0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    sweeps = liquidity_sweep_frame(bars, lookback=lookback).select(
        "security_id", "event_time", "sweep_reject_signed"
    )
    # Geometric hold: emit a decaying pulse for the next hold_bars bars.
    parts: list[pl.DataFrame] = []
    for h in range(hold_bars):
        parts.append(
            sweeps.select(
                "security_id",
                pl.col("event_time").alias("signal_time"),
                (pl.col("sweep_reject_signed") * (decay**h)).alias("_raw"),
            )
        )
        parts[-1] = parts[-1].with_columns(_h=pl.lit(h))
    pulses = pl.concat(parts).filter(pl.col("_raw") != 0.0)
    # Map each signal to bar index offsets per symbol.
    bar_idx = bars.sort(["security_id", "event_time"]).with_columns(
        pl.int_range(0, pl.len()).over("security_id").alias("_i")
    )
    sig = pulses.join(
        bar_idx.select("security_id", "event_time", "_i"),
        left_on=["security_id", "signal_time"],
        right_on=["security_id", "event_time"],
        how="inner",
    ).with_columns((pl.col("_i") + pl.col("_h")).alias("_ti"))
    placed = sig.join(
        bar_idx.select("security_id", pl.col("_i").alias("_ti"), "event_time"),
        on=["security_id", "_ti"],
        how="inner",
    )
    frame = (
        placed.group_by(["security_id", "event_time"])
        .agg(pl.col("_raw").sum())
        .sort(["security_id", "event_time"])
    )
    return _cap_and_emit(frame, "_raw", max_name=max_name, gross_scale=gross_scale)


def slow_trend_weights(
    bars: pl.DataFrame,
    *,
    fast_bars: int = 168,
    slow_bars: int = 720,
    max_name: float = 0.05,
    gross_scale: float = 1.0,
    vol_window: int = 48,
) -> pl.DataFrame:
    """Slow trend sleeve: sign(fast mean − slow mean) × inverse-vol,
    demeaned cross-sectionally so the book is roughly dollar-neutral."""
    _validate_bars(bars)
    if fast_bars >= slow_bars:
        raise ValueError("fast_bars must be < slow_bars")
    base = (
        bars.sort(["security_id", "event_time"])
        .with_columns(
            (
                pl.col("close")
                .rolling_mean(fast_bars, min_samples=fast_bars)
                .shift(1)
                .over("security_id")
                - pl.col("close")
                .rolling_mean(slow_bars, min_samples=slow_bars)
                .shift(1)
                .over("security_id")
            ).alias("_spread")
        )
        .drop_nulls("_spread")
    )
    vols = _per_symbol_vol(bars, vol_window)
    frame = (
        base.join(vols, on=["security_id", "event_time"], how="left")
        .with_columns(
            pl.when(pl.col("_vol").is_not_null() & (pl.col("_vol") > 0))
            .then(pl.col("_spread").sign() / pl.col("_vol"))
            .otherwise(None)
            .alias("_raw")
        )
        .with_columns(pl.col("_raw") - pl.col("_raw").median().over("event_time"))
    )
    return _cap_and_emit(frame, "_raw", max_name=max_name, gross_scale=gross_scale)


def basis_carry_weights(
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    *,
    lookback_events: int = 3,
    max_name: float = 0.15,
    gross_scale: float = 0.9,
    min_rate: float = 0.0,
) -> pl.DataFrame:
    """Delta-neutral carry weights: allocate the pair book across symbols
    whose trailing realized funding is positive, proportional to the rate.

    Weight ``w_i ∝ max(rate_ma_i − min_rate, 0)`` normalized to sum≈1 across
    qualifying names per timestamp, then per-name capped and gross-scaled.
    Only positive rates are eligible — negative-funding harvest needs spot
    borrow, which the carry book does not model.
    """
    _validate_bars(bars)
    if funding.height == 0:
        raise ValueError("funding frame must be non-empty")
    if "value" not in funding.columns:
        raise ValueError("funding frame needs a 'value' column (rate)")
    froll = (
        funding.sort(["security_id", "event_time"])
        .with_columns(
            pl.col("value")
            .rolling_mean(lookback_events, min_samples=1)
            .over("security_id")
            .alias("rate_ma")
        )
        .select("security_id", "event_time", "rate_ma")
        .sort(["security_id", "event_time"])
    )
    grid = bars.select("security_id", "event_time").sort(["security_id", "event_time"])
    joined = (
        grid.join_asof(froll, on="event_time", by="security_id", strategy="backward")
        .with_columns(
            pl.when(pl.col("rate_ma") > min_rate)
            .then(pl.col("rate_ma") - min_rate)
            .otherwise(0.0)
            .alias("_raw")
        )
        .with_columns(
            pl.when(pl.col("_raw").sum().over("event_time") > 0)
            .then(pl.col("_raw") / pl.col("_raw").sum().over("event_time"))
            .otherwise(0.0)
            .alias("_raw")
        )
    )
    return _cap_and_emit(joined, "_raw", max_name=max_name, gross_scale=gross_scale)


def basis_carry_hysteresis_weights(
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    *,
    enter_rate: float = 0.0003,
    exit_rate: float = 0.0,
    lookback_events: int = 3,
    name_weight: float = 0.08,
    max_names: int = 10,
    rebalance_band: float | None = None,
    rate_scale_ref: float | None = None,
    rate_scale_cap: float = 2.0,
    rate_scale_floor: float = 0.3,
    vol_lookback: int | None = None,
    vol_ref: float = 0.04,
    rate_exponent: float = 0.0,
) -> pl.DataFrame:
    """Event-driven carry membership book — the low-churn version.

    A symbol enters when its trailing mean realized funding ≥ ``enter_rate``
    and exits when it falls below ``exit_rate``; between membership changes no
    weight rows are emitted, so the engine holds the pair untouched (no daily
    rebalancing churn). Slots are capped at ``max_names``; contested slots go
    to the highest realized rates at each decision timestamp.

    ``rebalance_band`` bounds notional drift: a held pair's weight grows with
    price (fixed units), so when its mark drifts outside
    ``[name_weight / band, name_weight * band]`` a resize row re-emitting
    ``name_weight`` is emitted — sparse, only on breach. Without it, positions
    entered cheap can drift to many multiples of the intended weight and blow
    past the perp book's leverage/margin limits (the engine caps leverage only
    at order time). ``None`` keeps the original hold-untouched behaviour.

    ``rate_scale_ref`` enables regime-scaled sizing: when set, all weight rows
    emitted at a timestamp are multiplied by
    ``clip(book_rate / rate_scale_ref, rate_scale_floor, rate_scale_cap)``
    where ``book_rate`` is the mean trailing rate across the day's qualifying
    candidates (top ``max_names``) plus currently held names — the book
    self-sizes with funding dispersion (deploys more when yields are rich,
    thins when they compress) without adding churn between event days.

    ``vol_lookback`` enables per-name vol scaling: each emitted weight is also
    multiplied by ``min(1, vol_ref / vol_i)`` where ``vol_i`` is the symbol's
    daily close-close return std over the last ``vol_lookback`` bars. Wildest
    microcaps — the ones that drive basis-squeeze drawdowns — get diluted
    weight while calm names keep full size.

    ``rate_exponent`` tilts weights toward higher payers: each name's weight
    is multiplied by ``N * rate_i**rate_exponent / sum(rate_j**rate_exponent)``
    over the day's book, so the book's gross is unchanged but share follows
    rate^exponent. ``0`` = equal weight (default).
    """
    _validate_bars(bars)
    if funding.height == 0:
        raise ValueError("funding frame must be non-empty")
    if rebalance_band is not None and not (np.isfinite(rebalance_band) and rebalance_band > 1.0):
        raise ValueError("rebalance_band must be > 1 or None")
    froll = (
        funding.sort(["security_id", "event_time"])
        .with_columns(
            pl.col("value")
            .rolling_mean(lookback_events, min_samples=1)
            .over("security_id")
            .alias("rate_ma")
        )
        .select("security_id", "event_time", "rate_ma")
        .sort(["security_id", "event_time"])
    )
    grid = (
        bars.select("security_id", "event_time", "close")
        .sort(["security_id", "event_time"])
        .join_asof(froll, on="event_time", by="security_id", strategy="backward")
    )
    rates_by_time: dict[datetime, dict[str, float]] = {}
    px_by_time: dict[datetime, dict[str, float]] = {}
    for row in grid.iter_rows(named=True):
        rates_by_time.setdefault(row["event_time"], {})[str(row["security_id"])] = row["rate_ma"]
        px_by_time.setdefault(row["event_time"], {})[str(row["security_id"])] = row["close"]

    vols_by_time: dict[datetime, dict[str, float]] = {}
    if vol_lookback is not None:
        vroll = (
            bars.select("security_id", "event_time", "close")
            .sort(["security_id", "event_time"])
            .with_columns(
                pl.col("close")
                .pct_change()
                .rolling_std(vol_lookback, min_samples=5)
                .over("security_id")
                .alias("vol")
            )
            .select("security_id", "event_time", "vol")
        )
        for row in vroll.iter_rows(named=True):
            vols_by_time.setdefault(row["event_time"], {})[str(row["security_id"])] = row["vol"]

    active: dict[str, float] = {}
    entry_px: dict[str, float] = {}
    out_t: list[datetime] = []
    out_s: list[str] = []
    out_w: list[float] = []
    for t in sorted(rates_by_time):
        rates = rates_by_time[t]
        px_now = px_by_time.get(t, {})
        for sid in list(active):
            r = rates.get(sid)
            if r is None or r < exit_rate:
                out_t.append(t)
                out_s.append(sid)
                out_w.append(0.0)
                del active[sid]
                entry_px.pop(sid, None)
        cands = sorted(
            (
                (sid, r)
                for sid, r in rates.items()
                if r is not None and r >= enter_rate and sid not in active
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
        scale = 1.0
        if rate_scale_ref is not None and rate_scale_ref > 0:
            book_rates = [r for _, r in cands[:max_names]] + [
                rates[s] for s in active if rates.get(s) is not None
            ]
            if book_rates:
                scale = min(
                    rate_scale_cap,
                    max(rate_scale_floor, (sum(book_rates) / len(book_rates)) / rate_scale_ref),
                )
        w_now = name_weight * scale
        tilts: dict[str, float] = {}
        if rate_exponent > 0:
            book: dict[str, float | None] = {sid: r for sid, r in cands[:max_names]}
            for sid in active:
                book.setdefault(sid, rates.get(sid))
            rates_pos = {s: r for s, r in book.items() if isinstance(r, (int, float)) and r > 0}
            if rates_pos:
                n_book = max(len(book), 1)
                norm = n_book / sum(r**rate_exponent for r in rates_pos.values())
                tilts = {s: norm * r**rate_exponent for s, r in rates_pos.items()}
        vols_t = vols_by_time.get(t, {})

        def _vscale(sid: str, w: float, vols_t: dict[str, float] = vols_t) -> float:
            v = vols_t.get(sid)
            if v is None or v <= 0:
                return w
            return w * min(1.0, vol_ref / v)

        def _w(sid: str, w_now: float = w_now, tilts: dict[str, float] = tilts) -> float:
            return _vscale(sid, w_now * tilts.get(sid, 1.0))

        if rebalance_band is not None:
            for sid in list(active):
                p0 = entry_px.get(sid)
                p1 = px_now.get(sid)
                if p0 is None or p1 is None or not (p0 > 0 and p1 > 0):
                    continue
                drift = p1 / p0
                if drift >= rebalance_band or drift <= 1.0 / rebalance_band:
                    out_t.append(t)
                    out_s.append(sid)
                    out_w.append(_w(sid))
                    entry_px[sid] = p1
        for sid, _r in cands:
            if len(active) >= max_names:
                break
            w_sid = _w(sid)
            active[sid] = w_sid
            pe = px_now.get(sid)
            if pe is not None and pe > 0:
                entry_px[sid] = pe
            out_t.append(t)
            out_s.append(sid)
            out_w.append(w_sid)
    return pl.DataFrame({"event_time": out_t, "security_id": out_s, "target_weight": out_w}).sort(
        ["event_time", "security_id"]
    )


def blend_weights(
    sleeves: dict[str, pl.DataFrame],
    sleeve_weights: dict[str, float],
) -> pl.DataFrame:
    """Sum sleeve weight panels under named multipliers, then re-net
    duplicates (sleeve netting happens here — one weight per (t, sid))."""
    parts = []
    for name, frame in sleeves.items():
        mult = float(sleeve_weights.get(name, 0.0))
        if not np.isfinite(mult):
            raise ValueError(f"sleeve weight for {name} must be finite")
        if mult == 0.0:
            continue
        parts.append(frame.with_columns((pl.col("target_weight") * mult).alias("target_weight")))
    if not parts:
        return pl.DataFrame(
            schema={
                "event_time": pl.Datetime,
                "security_id": pl.String,
                "target_weight": pl.Float64,
            }
        )
    return (
        pl.concat(parts)
        .group_by(["event_time", "security_id"])
        .agg(pl.col("target_weight").sum())
        .sort(["event_time", "security_id"])
    )
