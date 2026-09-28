"""Point-in-time S&P 500 reality trial. Research diagnostic only.

The grid, costs, and membership rule are frozen in
``research/reality/survivorship/preregistration.json``. This module does not
edit reality-filter thresholds, round-1 ledger rows, or the round-1 receipt.
No live-trading claim.
"""

from __future__ import annotations

import csv
import io
import json
import math
import time
import urllib.error
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.backtest.engine import run_backtest
from quant_fund.backtest.sleeves import sweep_reclaim_weights
from quant_fund.config.loader import load_config
from quant_fund.config.models import AppConfig
from quant_fund.metrics.overfitting import deflated_sharpe
from quant_fund.research.reality_sweep import (
    Cell,
    ScoredCell,
    _attach_window_reports,
    _ledger_row,
    _public_window,
    _returns_sha,
    _sanitize,
    _split_arrays,
    assert_cost_lock,
    equal_weight_long,
    load_spec,
    pbo_on_pre_holdout,
    prepare_bars,
    select_winner,
    window_bounds,
)

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = _ROOT / "research" / "reality" / "survivorship" / "preregistration.json"
_MEMBERSHIP = _ROOT / "research" / "reality" / "survivorship" / "membership.json"
# Round-1 ledger was archived under studies/ after #201. The frozen hash still
# matches those bytes. Pending top-level trials.jsonl stays empty for the next
# undecided study.
_PRIOR_LEDGER = (
    _ROOT
    / "research"
    / "reality"
    / "studies"
    / "reality-us-liquid-daily-2026-09-27"
    / "trials.jsonl"
)
_PRIOR_CSV = (
    _ROOT / "research" / "reality" / "studies" / "reality-us-liquid-daily-2026-09-27" / "trials.csv"
)
_LEDGER = _PRIOR_LEDGER
_CSV = _PRIOR_CSV
_EXIT_SOURCE = "last_close_delisting_exit"
_MIN_LEG = 5
_RETURN_QUOTE = "quote_close"
_RETURN_TOTAL = "total_return"


def load_membership(path: Path | None = None) -> dict[str, Any]:
    """Load the vendored snapshot and refuse a file that is not newest-first."""
    payload = json.loads((path or _MEMBERSHIP).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("membership file must be a JSON object")
    changes = payload.get("changes")
    current = payload.get("current")
    if not isinstance(changes, list) or not isinstance(current, list):
        raise ValueError("membership file needs current and changes")
    dates = [str(event["date"]) for event in changes]
    if dates != sorted(dates, reverse=True):
        raise ValueError("membership changes must be newest-first")
    return payload


def members_asof(data: dict[str, Any], day: str) -> set[str]:
    """Index members at the close of ``day``. Changes dated ``day`` are in effect.

    Walks the change log newest-first and stops at the first event on or
    before ``day``. Newer events are undone. This is the pinned dataset's
    rule; it does not look at prices.
    """
    date.fromisoformat(day)
    members = {str(symbol) for symbol in data["current"]}
    for event in data["changes"]:
        if str(event["date"]) <= day:
            break
        added = str(event.get("added") or "")
        removed = str(event.get("removed") or "")
        if added:
            members.discard(added)
        if removed:
            members.add(removed)
    return members


def yahoo_symbol(ticker: str) -> str:
    """Yahoo share-class form. The membership ticker is otherwise unchanged."""
    return ticker.replace(".", "-")


def fetch_universe(data: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Tickers to request. SPY is appended by the caller, not here."""
    data_spec = spec["data"]
    days = {str(data_spec["fetch_start_utc"])[:10], "2024-12-31"}
    start = str(data_spec["fetch_start_utc"])[:10]
    end = str(data_spec["fetch_end_utc"])[:10]
    for event in data["changes"]:
        stamp = str(event["date"])
        if start <= stamp <= end:
            days.add(stamp)
    names: set[str] = set()
    for day in days:
        names |= members_asof(data, day)
    return sorted(names)


def grid_cells(spec: dict[str, Any]) -> list[Cell]:
    """Expand the frozen survivorship grid. Order is the pre-registration's."""
    grid = spec["grid"]
    cells: list[Cell] = []
    momentum = grid["xs_momentum"]
    for lookback in momentum["lookback"]:
        for skip in momentum["skip"]:
            for quantile in momentum["quantile"]:
                cells.append(
                    Cell(
                        strategy="xs_momentum",
                        cluster_id=str(momentum["cluster_id"]),
                        family=str(momentum["family"]),
                        params={
                            "gross_scale": momentum["gross_scale"],
                            "lookback": lookback,
                            "max_name": momentum["max_name"],
                            "quantile": quantile,
                            "rebalance_bars": momentum["rebalance_bars"],
                            "skip": skip,
                        },
                    )
                )
    reversal = grid["st_reversal"]
    for lookback in reversal["lookback"]:
        for skip in reversal["skip"]:
            for quantile in reversal["quantile"]:
                cells.append(
                    Cell(
                        strategy="st_reversal",
                        cluster_id=str(reversal["cluster_id"]),
                        family=str(reversal["family"]),
                        params={
                            "gross_scale": reversal["gross_scale"],
                            "lookback": lookback,
                            "max_name": reversal["max_name"],
                            "quantile": quantile,
                            "rebalance_bars": lookback,
                            "skip": skip,
                        },
                    )
                )
    low_vol = grid["low_vol"]
    for window in low_vol["vol_window"]:
        for quantile in low_vol["quantile"]:
            cells.append(
                Cell(
                    strategy="low_vol",
                    cluster_id=str(low_vol["cluster_id"]),
                    family=str(low_vol["family"]),
                    params={
                        "quantile": quantile,
                        "rebalance_bars": low_vol["rebalance_bars"],
                        "vol_window": window,
                    },
                )
            )
    dip = grid["dip_regime"]
    for lookback in dip["lookback"]:
        for hold_bars in dip["hold_bars"]:
            for decay in dip["decay"]:
                for regime_lookback in dip["regime_lookback"]:
                    cells.append(
                        Cell(
                            strategy="dip_regime",
                            cluster_id=str(dip["cluster_id"]),
                            family=str(dip["family"]),
                            params={
                                "decay": decay,
                                "gross_scale": dip["gross_scale"],
                                "hold_bars": hold_bars,
                                "lookback": lookback,
                                "max_name": dip["max_name"],
                                "regime_lookback": regime_lookback,
                            },
                        )
                    )
    base = grid["equal_weight_pit"]
    cells.append(
        Cell(
            strategy="equal_weight_pit",
            cluster_id=str(base["cluster_id"]),
            family=str(base["family"]),
            params={"name_cap": base["name_cap"], "net_cap": base["net_cap"]},
        )
    )
    expected = spec["expected_trials"]
    for name in (
        "xs_momentum",
        "st_reversal",
        "low_vol",
        "dip_regime",
        "equal_weight_pit",
    ):
        count = sum(1 for cell in cells if cell.strategy == name)
        if count != int(expected[name]):
            raise ValueError(f"{name} cell count {count} != pre-registered {expected[name]}")
    if len(cells) != int(expected["new_total"]):
        raise ValueError(
            f"grid has {len(cells)} cells, pre-registration says {expected['new_total']}"
        )
    return cells


def formation_return(bars: pl.DataFrame, *, lookback: int, skip: int) -> pl.DataFrame:
    """Trailing simple return that does not use the decision row's close."""
    if lookback < 1 or skip < 0:
        raise ValueError("lookback must be >= 1 and skip >= 0")
    return bars.sort(["security_id", "event_time"]).with_columns(
        (
            pl.col("close").shift(1 + skip).over("security_id")
            / pl.col("close").shift(1 + skip + lookback).over("security_id")
            - 1.0
        ).alias("formation")
    )


def trailing_volatility(bars: pl.DataFrame, *, window: int) -> pl.DataFrame:
    """Causal close-to-close volatility. The decision row's close is excluded."""
    if window < 2:
        raise ValueError("vol window must be >= 2")
    minimum = max(10, window // 2)
    return (
        bars.sort(["security_id", "event_time"])
        .with_columns(pl.col("close").pct_change().over("security_id").alias("_ret"))
        .with_columns(
            pl.col("_ret")
            .rolling_std(window, min_samples=minimum)
            .shift(1)
            .over("security_id")
            .alias("volatility")
        )
    )


def _clip_gross(frame: pl.DataFrame, *, max_name: float, gross_scale: float) -> pl.DataFrame:
    clipped = frame.with_columns(
        pl.col("target_weight").clip(-float(max_name), float(max_name)).alias("target_weight")
    )
    scaled = clipped.with_columns(
        pl.col("target_weight").abs().sum().over("event_time").alias("_gross")
    ).with_columns(
        pl.when(pl.col("_gross") > gross_scale)
        .then(pl.col("target_weight") * gross_scale / pl.col("_gross"))
        .otherwise(pl.col("target_weight"))
        .alias("target_weight")
    )
    return scaled.filter(pl.col("target_weight").abs() > 1e-12).select(
        "event_time", "security_id", "target_weight"
    )


def long_short_weights(
    signaled: pl.DataFrame,
    *,
    signal: str,
    quantile: float,
    higher_is_long: bool,
    max_name: float,
    gross_scale: float,
    min_leg: int = _MIN_LEG,
) -> pl.DataFrame:
    """Quintile long/short among rows that already carry a finite signal.

    ``signaled`` must contain only names eligible on that session. Percentile
    is average-rank over the count of finite signals that session.
    """
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must be in (0, 0.5)")
    finite = signaled.filter(pl.col(signal).is_not_null() & pl.col(signal).is_finite())
    if finite.is_empty():
        return _empty_weights()
    ranked = finite.with_columns(
        (
            pl.col(signal).rank(method="average").over("event_time")
            / pl.col(signal).count().over("event_time")
        ).alias("_pct")
    )
    long_cut = 1.0 - quantile
    if higher_is_long:
        long_flag = pl.col("_pct") > long_cut
        short_flag = pl.col("_pct") <= quantile
    else:
        long_flag = pl.col("_pct") <= quantile
        short_flag = pl.col("_pct") > long_cut
    sided = ranked.with_columns(
        pl.when(long_flag).then(1.0).when(short_flag).then(-1.0).otherwise(0.0).alias("_side")
    ).with_columns(
        pl.when(pl.col("_side") > 0).then(1).otherwise(0).sum().over("event_time").alias("_n_long"),
        pl.when(pl.col("_side") < 0)
        .then(1)
        .otherwise(0)
        .sum()
        .over("event_time")
        .alias("_n_short"),
    )
    weighted = sided.with_columns(
        pl.when(
            (pl.col("_n_long") < min_leg) | (pl.col("_n_short") < min_leg) | (pl.col("_side") == 0)
        )
        .then(0.0)
        .when(pl.col("_side") > 0)
        .then(0.5 / pl.col("_n_long"))
        .otherwise(-0.5 / pl.col("_n_short"))
        .alias("target_weight")
    )
    return _clip_gross(
        weighted.filter(pl.col("target_weight") != 0.0),
        max_name=max_name,
        gross_scale=gross_scale,
    )


def low_vol_weights(
    signaled: pl.DataFrame,
    *,
    quantile: float,
    name_cap: float,
    net_cap: float,
    min_leg: int = _MIN_LEG,
) -> pl.DataFrame:
    """Long-only calm quintile, then the baseline name/net caps."""
    finite = signaled.filter(pl.col("volatility").is_not_null() & (pl.col("volatility") > 0))
    if finite.is_empty():
        return _empty_weights()
    ranked = finite.with_columns(
        (
            pl.col("volatility").rank(method="average").over("event_time")
            / pl.col("volatility").count().over("event_time")
        ).alias("_pct")
    )
    chosen = ranked.filter(pl.col("_pct") <= quantile)
    counts = chosen.group_by("event_time").agg(pl.len().alias("_n"))
    chosen = chosen.join(counts, on="event_time").filter(pl.col("_n") >= min_leg)
    if chosen.is_empty():
        return _empty_weights()
    return equal_weight_long(chosen, name_cap=name_cap, net_cap=net_cap)


def spy_regime(spy: pl.DataFrame, *, lookback: int) -> pl.DataFrame:
    """True when the shifted SPY trailing return is strictly negative."""
    formed = formation_return(spy, lookback=lookback, skip=0)
    return formed.select(
        "event_time",
        (
            pl.col("formation").is_not_null()
            & pl.col("formation").is_finite()
            & (pl.col("formation") < 0.0)
        ).alias("regime_on"),
    )


def gate_dip_weights(weights: pl.DataFrame, regime: pl.DataFrame) -> pl.DataFrame:
    """Keep positive dip weights only on sessions where the regime is on."""
    if weights.is_empty():
        return _empty_weights()
    joined = weights.join(regime, on="event_time", how="left")
    gated = joined.with_columns(
        pl.when(pl.col("regime_on").fill_null(False) & (pl.col("target_weight") > 0.0))
        .then(pl.col("target_weight"))
        .otherwise(0.0)
        .alias("target_weight")
    )
    return gated.filter(pl.col("target_weight") != 0.0).select(
        "event_time", "security_id", "target_weight"
    )


def _empty_weights() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "event_time": pl.Datetime(time_unit="us", time_zone="UTC"),
            "security_id": pl.String,
            "target_weight": pl.Float64,
        }
    )


def emit_weight_rows(
    raw_by_day: dict[datetime, dict[str, float]],
    calendar: list[datetime],
    every: int,
    eligible: dict[datetime, set[str]],
) -> pl.DataFrame:
    """Sample targets on the rebalance grid and flatten names that drop out.

    A session that emits any row replaces the engine's carried map. An empty
    book after a non-empty one emits explicit zeros so the engine does not
    keep the previous names.
    """
    if every < 1:
        raise ValueError("rebalance_bars must be >= 1")
    snapshot: dict[str, float] = {}
    previous: set[str] = set()
    times: list[datetime] = []
    names: list[str] = []
    values: list[float] = []
    for index, stamp in enumerate(calendar):
        allowed = eligible.get(stamp, set())
        if index % every == 0:
            fresh = raw_by_day.get(stamp, {})
            snapshot = {
                sid: weight
                for sid, weight in fresh.items()
                if sid in allowed and weight != 0.0 and math.isfinite(weight)
            }
        else:
            snapshot = {sid: weight for sid, weight in snapshot.items() if sid in allowed}
        if snapshot:
            for sid, weight in sorted(snapshot.items()):
                times.append(stamp)
                names.append(sid)
                values.append(weight)
            previous = set(snapshot)
        elif previous:
            for sid in sorted(previous):
                times.append(stamp)
                names.append(sid)
                values.append(0.0)
            previous = set()
    if not times:
        return _empty_weights()
    return pl.DataFrame({"event_time": times, "security_id": names, "target_weight": values})


def frame_to_day_map(frame: pl.DataFrame) -> dict[datetime, dict[str, float]]:
    grouped: dict[datetime, dict[str, float]] = {}
    if frame.is_empty():
        return grouped
    for stamp, sid, weight in frame.select(
        "event_time", "security_id", "target_weight"
    ).iter_rows():
        if not isinstance(stamp, datetime):
            raise ValueError("event_time must be a datetime")
        grouped.setdefault(stamp, {})[str(sid)] = float(weight)
    return grouped


def with_eligibility(real: pl.DataFrame, membership: dict[str, Any]) -> pl.DataFrame:
    """Mark bars that may receive a non-zero target.

    Membership is the close-of-session set. The name's last real print is
    not eligible: the target there is zero so a flatten can fill on the
    following exit bar. Pre-membership bars stay in the frame for formation
    math and are simply not eligible.
    """
    stamped = real.with_columns(
        pl.col("event_time").dt.convert_time_zone("America/New_York").dt.date().alias("ny_date"),
        (pl.col("event_time") == pl.col("event_time").max().over("security_id")).alias(
            "is_last_real"
        ),
    )
    sessions = [day for day in stamped["ny_date"].unique().to_list() if isinstance(day, date)]
    ids: list[str] = []
    days: list[date] = []
    for session in sessions:
        for symbol in members_asof(membership, session.isoformat()):
            ids.append(symbol)
            days.append(session)
    member_frame = pl.DataFrame({"security_id": ids, "ny_date": days}).with_columns(
        pl.lit(True).alias("in_index")
    )
    joined = stamped.join(member_frame, on=["security_id", "ny_date"], how="left")
    return joined.with_columns(
        (
            pl.col("in_index").fill_null(False)
            & ~pl.col("is_last_real")
            & (pl.col("source") == "yahoo")
        ).alias("eligible")
    )


def eligible_map(frame: pl.DataFrame) -> dict[datetime, set[str]]:
    grouped: dict[datetime, set[str]] = {}
    rows = frame.filter(pl.col("eligible")).select("event_time", "security_id")
    for stamp, sid in rows.iter_rows():
        if isinstance(stamp, datetime):
            grouped.setdefault(stamp, set()).add(str(sid))
    return grouped


def append_delisting_exits(real: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Copy the last close onto the next panel session when the tape ends early.

    The copied bar is not a market print. Signals must not see it. It exists
    so the engine can flatten a name whose Yahoo series ended.
    """
    if real.is_empty():
        return real, 0
    times = sorted(
        stamp for stamp in real["event_time"].unique().to_list() if isinstance(stamp, datetime)
    )
    position = {stamp: index for index, stamp in enumerate(times)}
    ranked = real.with_columns(
        pl.col("event_time")
        .rank(method="ordinal", descending=True)
        .over("security_id")
        .alias("_ord")
    )
    last = ranked.filter(pl.col("_ord") == 1).drop("_ord")
    present = set(real.select("security_id", "event_time").iter_rows())
    extras: list[dict[str, Any]] = []
    for row in last.iter_rows(named=True):
        stamp = row["event_time"]
        if not isinstance(stamp, datetime):
            continue
        index = position.get(stamp)
        if index is None or index + 1 >= len(times):
            continue
        nxt = times[index + 1]
        key = (row["security_id"], nxt)
        if key in present:
            continue
        copied = dict(row)
        copied["event_time"] = nxt
        copied["source"] = _EXIT_SOURCE
        copied["close_total_return"] = copied["close"]
        extras.append(copied)
    if not extras:
        return real, 0
    extra = pl.DataFrame(extras).select(real.columns)
    combined = pl.concat([real, extra], how="vertical").sort(["security_id", "event_time"])
    duplicate = combined.select(
        pl.struct(["security_id", "event_time"]).is_duplicated().any()
    ).item()
    if duplicate:
        raise RuntimeError("delisting exit created a duplicate bar")
    return combined, len(extras)


def anchor_coverage(
    real: pl.DataFrame, membership: dict[str, Any], anchors: list[str]
) -> list[dict[str, Any]]:
    """Share of index members with a real bar on the first session at/after each anchor."""
    stamped = real.with_columns(
        pl.col("event_time").dt.convert_time_zone("America/New_York").dt.date().alias("ny_date")
    )
    sessions = sorted(day for day in stamped["ny_date"].unique().to_list() if isinstance(day, date))
    reports: list[dict[str, Any]] = []
    for anchor in anchors:
        start = date.fromisoformat(anchor)
        session = next((day for day in sessions if day >= start), None)
        if session is None:
            raise RuntimeError(f"no panel session on or after {anchor}")
        members = members_asof(membership, session.isoformat())
        have = set(
            stamped.filter(pl.col("ny_date") == session)["security_id"].cast(pl.String).to_list()
        )
        covered = members & have
        reports.append(
            {
                "anchor": anchor,
                "session": session.isoformat(),
                "members": len(members),
                "with_bar": len(covered),
                "coverage": (len(covered) / len(members)) if members else 0.0,
                "missing": sorted(members - have),
            }
        )
    return reports


def _align_clock(weights: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    if weights.is_empty():
        return _empty_weights().with_columns(pl.col("event_time").cast(bars.schema["event_time"]))
    return weights.with_columns(pl.col("event_time").cast(bars.schema["event_time"])).sort(
        ["event_time", "security_id"]
    )


def _assert_on_calendar(weights: pl.DataFrame, calendar: list[datetime], strategy: str) -> None:
    if weights.is_empty():
        return
    known = set(calendar)
    foreign = [stamp for stamp in weights["event_time"].unique().to_list() if stamp not in known]
    if foreign:
        raise RuntimeError(f"{strategy} produced {len(foreign)} timestamps off the bar calendar")


def weights_for_cell(
    cell: Cell,
    real: pl.DataFrame,
    spy: pl.DataFrame,
    *,
    name_cap: float,
    net_cap: float,
) -> pl.DataFrame:
    """Daily or rebalanced targets on the real-bar panel. Exit bars are absent."""
    calendar = sorted(
        stamp for stamp in real["event_time"].unique().to_list() if isinstance(stamp, datetime)
    )
    eligible = eligible_map(real)
    params = cell.params
    if cell.strategy in {"xs_momentum", "st_reversal"}:
        formed = formation_return(real, lookback=int(params["lookback"]), skip=int(params["skip"]))
        signaled = formed.filter(pl.col("eligible"))
        raw = long_short_weights(
            signaled,
            signal="formation",
            quantile=float(params["quantile"]),
            higher_is_long=cell.strategy == "xs_momentum",
            max_name=float(params["max_name"]),
            gross_scale=float(params["gross_scale"]),
        )
        _assert_on_calendar(raw, calendar, cell.strategy)
        return emit_weight_rows(
            frame_to_day_map(raw),
            calendar,
            int(params["rebalance_bars"]),
            eligible,
        )
    if cell.strategy == "low_vol":
        formed = trailing_volatility(real, window=int(params["vol_window"]))
        signaled = formed.filter(pl.col("eligible"))
        raw = low_vol_weights(
            signaled,
            quantile=float(params["quantile"]),
            name_cap=name_cap,
            net_cap=net_cap,
        )
        _assert_on_calendar(raw, calendar, cell.strategy)
        return emit_weight_rows(
            frame_to_day_map(raw),
            calendar,
            int(params["rebalance_bars"]),
            eligible,
        )
    if cell.strategy == "dip_regime":
        sleeve = sweep_reclaim_weights(
            real.drop(
                [
                    column
                    for column in ("ny_date", "is_last_real", "in_index", "eligible")
                    if column in real.columns
                ]
            ),
            lookback=int(params["lookback"]),
            hold_bars=int(params["hold_bars"]),
            decay=float(params["decay"]),
            max_name=float(params["max_name"]),
            gross_scale=float(params["gross_scale"]),
        )
        regime = spy_regime(spy, lookback=int(params["regime_lookback"]))
        gated = gate_dip_weights(sleeve.filter(pl.col("target_weight") > 0.0), regime)
        _assert_on_calendar(gated, calendar, cell.strategy)
        # Drop names that are not eligible that session.
        if not gated.is_empty() and "eligible" in real.columns:
            allowed = real.filter(pl.col("eligible")).select("event_time", "security_id")
            gated = gated.join(allowed, on=["event_time", "security_id"], how="inner")
        return emit_weight_rows(frame_to_day_map(gated), calendar, 1, eligible)
    if cell.strategy == "equal_weight_pit":
        tradable = real.filter(pl.col("eligible"))
        if tradable.is_empty():
            return _empty_weights()
        return equal_weight_long(tradable, name_cap=name_cap, net_cap=net_cap)
    raise ValueError(f"unknown strategy {cell.strategy}")


def _score_weights(
    cell: Cell,
    bars: pl.DataFrame,
    weights: pl.DataFrame,
    config: AppConfig,
    spec: dict[str, Any],
) -> tuple[ScoredCell, dict[str, int]]:
    aligned = _align_clock(weights, bars)
    result = run_backtest(
        bars,
        aligned,
        config,
        initial_nav=float(spec["costs"]["initial_nav"]),
    )
    returns, turns, dates = _split_arrays(result.equity, window_bounds(spec))
    validation = {
        "periodic_ratio": 0.0,
        "skew": 0.0,
        "kurtosis_raw": 3.0,
        "degenerate": True,
        "n_obs": int(returns["validation"].size),
    }
    from quant_fund.research.reality_sweep import _period_ratio

    ratio, skew, kurt, degenerate = _period_ratio(returns["validation"])
    validation = {
        "periodic_ratio": ratio,
        "skew": skew,
        "kurtosis_raw": kurt,
        "degenerate": degenerate,
        "n_obs": int(returns["validation"].size),
    }
    metrics = result.metrics
    rejects = {
        "risk_gate_rejects": _metric_int(metrics, "risk_gate_rejects"),
        "cash_rejects": _metric_int(metrics, "cash_rejects"),
    }
    scored = ScoredCell(
        cell=cell,
        trial_id=cell.trial_id(str(spec["study_id"])),
        by_window={"validation": validation},
        validation_returns=returns["validation"],
        train_returns=returns["train"],
        holdout_returns=returns["holdout"],
        dates={name: [day.isoformat() for day in dates[name]] for name in dates},
        turnover=turns,
        equity=result.equity,
    )
    return scored, rejects


def assert_prior_ledger(path: Path, spec: dict[str, Any]) -> bytes:
    """Refuse to append unless the round-1 file is still the frozen bytes."""
    from quant_fund.proofcore.contracts import sha256_hex_bytes

    raw = path.read_bytes()
    digest = sha256_hex_bytes(raw)
    expected = str(spec["prior_study"]["ledger_sha256"])
    if digest != expected:
        raise RuntimeError(
            f"trial ledger sha256 is {digest}, pre-registration froze {expected}. "
            "Refusing to rewrite or double-append."
        )
    lines = raw.splitlines()
    if len(lines) != int(spec["prior_study"]["n_trials"]):
        raise RuntimeError(f"trial ledger has {len(lines)} lines, expected the original 29")
    if not raw.endswith(b"\n"):
        raise RuntimeError("trial ledger must end in a newline")
    return raw


def append_ledger_lines(path: Path, prefix: bytes, lines: list[str]) -> None:
    """Append JSON lines. The prior bytes stay the prefix."""
    payload = "".join(line if line.endswith("\n") else line + "\n" for line in lines).encode()
    with path.open("ab") as handle:
        handle.write(payload)
    updated = path.read_bytes()
    if not updated.startswith(prefix):
        raise RuntimeError("append changed the existing ledger prefix")
    if updated[len(prefix) :] != payload:
        raise RuntimeError("appended ledger bytes do not match the new rows")


def append_csv_rows(path: Path, rows: list[Any]) -> None:
    """Append CSV rows. Existing lines, including the header, stay put."""
    prefix = path.read_bytes()
    if not prefix.endswith(b"\n"):
        raise RuntimeError("trial CSV must end in a newline")
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "trial_id",
            "bundle_hash",
            "family",
            "strategy",
            "cluster_id",
            "n_obs",
            "periods_per_year",
            "sharpe_periodic",
            "skew",
            "kurtosis_raw",
            "returns_sha256",
            "created_utc",
        ),
        lineterminator="\n",
    )
    for row in rows:
        payload = row.model_dump(mode="json")
        writer.writerow({name: payload[name] for name in writer.fieldnames})
    extra = buffer.getvalue().encode()
    with path.open("ab") as handle:
        handle.write(extra)
    updated = path.read_bytes()
    if not updated.startswith(prefix) or updated[len(prefix) :] != extra:
        raise RuntimeError("CSV append changed existing rows")


def _fetch_one(
    ticker: str,
    *,
    start: datetime,
    end: datetime,
    cache: Path,
    events: str | None = None,
) -> tuple[pl.DataFrame | None, pl.DataFrame]:
    from quant_fund.data.adapters.yahoo_eod import fetch_yahoo_chart, parse_yahoo_chart
    from quant_fund.research.total_return import parse_yahoo_corporate_actions

    symbol = yahoo_symbol(ticker)
    cached = cache / "bars" / f"{ticker}.parquet"
    actions_cached = cache / "actions" / f"{ticker}.parquet"
    missing = cache / "missing" / f"{ticker}.json"
    empty_actions = pl.DataFrame(
        schema={
            "security_id": pl.String(),
            "event_time": pl.Datetime(time_zone="UTC"),
            "available_time": pl.Datetime(time_zone="UTC"),
            "action_type": pl.String(),
            "amount": pl.Float64(),
            "factor": pl.Float64(),
            "source": pl.String(),
            "revision_id": pl.String(),
        }
    )
    if cached.exists() and (events is None or actions_cached.exists()):
        frame = pl.read_parquet(cached)
        actions = pl.read_parquet(actions_cached) if actions_cached.exists() else empty_actions
        if frame.height == 0:
            return None, empty_actions
        return frame, actions
    if missing.exists():
        return None, empty_actions
    try:
        payload = fetch_yahoo_chart(
            symbol, start=start, end=end, retries=3, timeout=30.0, events=events
        )
        frame = parse_yahoo_chart(payload, security_id=ticker, yahoo_symbol=symbol)
        actions = (
            parse_yahoo_corporate_actions(payload, security_id=ticker, yahoo_symbol=symbol)
            if events is not None
            else empty_actions
        )
    except urllib.error.HTTPError as exc:
        if exc.code in {404, 400}:
            missing.parent.mkdir(parents=True, exist_ok=True)
            missing.write_text(
                json.dumps({"ticker": ticker, "yahoo_symbol": symbol, "status": exc.code}),
                encoding="utf-8",
            )
            return None, empty_actions
        raise
    if frame.is_empty():
        missing.parent.mkdir(parents=True, exist_ok=True)
        missing.write_text(
            json.dumps({"ticker": ticker, "yahoo_symbol": symbol, "status": "empty"}),
            encoding="utf-8",
        )
        return None, empty_actions
    cached.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(cached)
    if events is not None:
        actions_cached.parent.mkdir(parents=True, exist_ok=True)
        actions.write_parquet(actions_cached)
    return frame, actions


def fetch_yahoo_members(
    tickers: list[str],
    spec: dict[str, Any],
    cache: Path,
    *,
    events: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list[dict[str, str]]]:
    """Download real Yahoo bars. Failures are recorded. Prices are not invented.

    Pass ``events`` (for example ``div,split``) to also cache corporate actions.
    """
    data = spec["data"]
    start = datetime.fromisoformat(str(data["fetch_start_utc"]))
    end = datetime.fromisoformat(str(data["fetch_end_utc"]))
    frames: list[pl.DataFrame] = []
    action_frames: list[pl.DataFrame] = []
    failures: list[dict[str, str]] = []
    symbols = list(tickers)
    regime = str(data["regime_symbol"])
    if regime not in symbols:
        symbols.append(regime)
    for index, ticker in enumerate(symbols):
        if index:
            time.sleep(0.15)
        if index % 25 == 0:
            print(f"fetch {index}/{len(symbols)} {ticker}", flush=True)
        try:
            frame, actions = _fetch_one(ticker, start=start, end=end, cache=cache, events=events)
        except Exception as exc:
            failures.append({"ticker": ticker, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if frame is None:
            failures.append({"ticker": ticker, "error": "no bars"})
            continue
        frames.append(frame)
        if actions.height:
            action_frames.append(actions)
    if not frames:
        raise RuntimeError("Yahoo returned no bars; stopping without a synthetic substitute")
    panel = pl.concat(frames, how="diagonal_relaxed")
    panel = panel.unique(subset=["security_id", "event_time"], keep="last").sort(
        ["security_id", "event_time"]
    )
    if action_frames:
        actions_panel = pl.concat(action_frames, how="diagonal_relaxed")
        actions_panel = actions_panel.unique(
            subset=["security_id", "event_time", "action_type"], keep="last"
        ).sort(["security_id", "event_time", "action_type"])
    else:
        actions_panel = pl.DataFrame(
            schema={
                "security_id": pl.String(),
                "event_time": pl.Datetime(time_zone="UTC"),
                "available_time": pl.Datetime(time_zone="UTC"),
                "action_type": pl.String(),
                "amount": pl.Float64(),
                "factor": pl.Float64(),
                "source": pl.String(),
                "revision_id": pl.String(),
            }
        )
    spy = panel.filter(pl.col("security_id") == regime)
    if spy.is_empty():
        raise RuntimeError("SPY regime series is missing; stopping")
    stocks = panel.filter(pl.col("security_id") != regime)
    if stocks.is_empty():
        raise RuntimeError("no constituent bars; stopping")
    return stocks, spy, actions_panel, failures


def _ledger_best_raw_dsr(rows: list[Any]) -> float:
    ratios = np.asarray([float(row.sharpe_periodic) for row in rows], dtype=float)
    variance = float(np.var(ratios, ddof=1)) if ratios.size > 1 else 0.0
    best = max(rows, key=lambda row: (float(row.sharpe_periodic), row.trial_id))
    return deflated_sharpe(
        float(best.sharpe_periodic),
        int(best.n_obs),
        float(best.skew),
        float(best.kurtosis_raw),
        len(rows),
        variance,
    )


def _prior_baseline(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    row = next(item for item in receipt["trials"] if item["strategy"] == "equal_weight_long")
    return {
        "trial_id": row["trial_id"],
        "validation_compounded": row["windows"]["validation"]["compounded_return"],
        "holdout_compounded": row["windows"]["holdout"]["compounded_return"],
        "validation_annualized_ratio": row["windows"]["validation"]["annualized_ratio"],
        "holdout_annualized_ratio": row["windows"]["holdout"]["annualized_ratio"],
        "note": "Copied from the round-1 receipt. Not recomputed.",
    }


def write_results_markdown(path: Path, receipt: dict[str, Any]) -> None:
    """Results note. Figures are copied from the receipt."""
    gate = receipt["reality_gate"]
    prior = receipt["prior_equal_weight_long"]
    pit = next(row for row in receipt["trials"] if row["strategy"] == "equal_weight_pit")
    lines = [
        "# Survivorship-aware reality trial",
        "",
        "Research diagnostic. Annualized ratio means a Sharpe ratio:",
        "per-period mean over sample standard deviation, times the square",
        "root of the periods-per-year count. It is not a promotion and not",
        "a live-trading claim. Reality-filter thresholds were not edited.",
        "Round-1 ledger rows were not rewritten.",
        "",
        f"Verdict: `{gate['verdict']}`.",
        f"Trials recorded: {gate['n_trials']}.",
        f"Effective trials (cluster count): {gate['n_effective_trials']}.",
        f"Gate best ledger row: `{gate['best_trial_id']}`.",
        f"Gate deflated probability: {gate['dsr']}.",
        f"Gate PSR of the best ledger row: {gate['psr']}.",
        f"Gate PBO: {gate['pbo']} (unset; the JSONL ledger has no return series).",
        f"CSCV PBO on the 13 new train-then-validation paths: {receipt['pbo']['pbo']}.",
        f"Deflated probability using the raw trial count: {receipt['deflated_probability_raw_count']}.",
        f"New-study winner: `{receipt['selected_trial_id']}` ({receipt['selected_strategy']}).",
        f"Dataset sha256: `{receipt['provenance']['dataset_sha256']}`.",
        f"Receipt sha256: `{receipt['receipt_sha256']}`.",
        "",
        "## Survivorship and the equal-weight baseline",
        "",
        "Both books use the frozen net cap of 0.2 and the same cost model.",
        "Round-1 figures are copied from that receipt. The point-in-time",
        "figures are this study's equal-weight cell.",
        "",
        "| book | universe | validation compounded | holdout compounded | validation annualized ratio | holdout annualized ratio |",
        "|---|---|---:|---:|---:|---:|",
        "| equal_weight_long | present-day US_LIQUID | {v} | {h} | {va} | {ha} |".format(
            v=_fmt(prior["validation_compounded"]),
            h=_fmt(prior["holdout_compounded"]),
            va=_fmt(prior["validation_annualized_ratio"]),
            ha=_fmt(prior["holdout_annualized_ratio"]),
        ),
        "| equal_weight_pit | point-in-time S&P 500 members | {v} | {h} | {va} | {ha} |".format(
            v=_fmt(pit["windows"]["validation"]["compounded_return"]),
            h=_fmt(pit["windows"]["holdout"]["compounded_return"]),
            va=_fmt(pit["windows"]["validation"]["annualized_ratio"]),
            ha=_fmt(pit["windows"]["holdout"]["annualized_ratio"]),
        ),
        "",
        "## Every new trial",
        "",
        "| strategy | params | validation annualized | holdout annualized | validation compounded | holdout compounded |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in receipt["trials"]:
        lines.append(
            "| {strategy} | `{params}` | {val} | {hold} | {val_c} | {hold_c} |".format(
                strategy=row["strategy"],
                params=json.dumps(row["params"], sort_keys=True),
                val=_fmt(row["windows"]["validation"]["annualized_ratio"]),
                hold=_fmt(row["windows"]["holdout"]["annualized_ratio"]),
                val_c=_fmt(row["windows"]["validation"]["compounded_return"]),
                hold_c=_fmt(row["windows"]["holdout"]["compounded_return"]),
            )
        )
    coverage = receipt["coverage"]
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| anchor | session | members | with a bar | coverage |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in coverage:
        lines.append(
            f"| {row['anchor']} | {row['session']} | {row['members']} | {row['with_bar']} | {_fmt(row['coverage'])} |"
        )
    lines.extend(
        [
            "",
            f"Fetch failures: {receipt['n_fetch_failures']}.",
            f"Delisting-exit bars (copied last close, not a print): {receipt['n_delisting_exit_bars']}.",
            f"Names with at least one real bar: {receipt['n_names_with_bars']}.",
            "",
            "## Caveats",
            "",
        ]
    )
    for item in receipt["remaining_bias"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "Holdout was summarized after the new trials were appended.",
            "Selection used the validation per-period ratio only.",
            "The CSCV figure covers the 13 new paths. It does not include",
            "the original 29, whose return series were not stored.",
            "The gate deflates the best row in the full ledger.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _metric_int(metrics: object, key: str) -> int:
    if not isinstance(metrics, dict):
        return 0
    value = metrics.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _fmt(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "null" if value is None else str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "null"
    return f"{float(value):.6g}"


def run_study(
    *,
    spec_path: Path | None = None,
    membership_path: Path | None = None,
    ledger_path: Path | None = None,
    csv_path: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Fetch, score every new cell, append every cell, write the receipt."""
    spec_file = spec_path or _SPEC
    spec = load_spec(spec_file)
    from quant_fund.proofcore.contracts import sha256_hex_bytes

    membership_file = membership_path or _MEMBERSHIP
    member_hash = sha256_hex_bytes(membership_file.read_bytes())
    if member_hash != str(spec["data"]["membership_sha256"]):
        raise RuntimeError("membership file hash does not match the pre-registration")
    membership = load_membership(membership_file)
    cells = grid_cells(spec)
    config = load_config(_ROOT / "configs" / "backtest.yaml")
    assert_cost_lock(config, spec)
    ledger = ledger_path or _LEDGER
    prefix = assert_prior_ledger(ledger, spec)
    cache = cache_dir or (_ROOT / "data" / "reality_sweep_survivorship")
    tickers = fetch_universe(membership, spec)
    stocks, spy, _actions, failures = fetch_yahoo_members(tickers, spec, cache)
    prepared = prepare_bars(stocks)
    prepared = with_eligibility(prepared, membership)
    coverage = anchor_coverage(prepared, membership, list(spec["data"]["coverage_anchors"]))
    floor = float(spec["data"]["coverage_floor"])
    for report in coverage:
        print(
            f"coverage {report['anchor']} session={report['session']} "
            f"{report['with_bar']}/{report['members']}={report['coverage']:.4f}",
            flush=True,
        )
        if float(report["coverage"]) < floor:
            raise RuntimeError(
                f"coverage {report['coverage']:.4f} on {report['session']} is below {floor}; "
                "not scoring and not appending"
            )
    real = prepared.filter(pl.col("source") == "yahoo")
    panel, n_exit = append_delisting_exits(
        real.drop(
            [
                column
                for column in ("ny_date", "is_last_real", "in_index", "eligible")
                if column in real.columns
            ]
        )
    )
    name_cap = float(spec["risk_gate"]["max_name"])
    net_cap = float(spec["risk_gate"]["max_net"])
    # Signals see real bars only, including the eligibility columns.
    scored: list[ScoredCell] = []
    rejects: dict[str, dict[str, int]] = {}
    for cell in cells:
        print(f"score {cell.strategy} {json.dumps(cell.params, sort_keys=True)}", flush=True)
        weights = weights_for_cell(cell, real, spy, name_cap=name_cap, net_cap=net_cap)
        scored_row, row_rejects = _score_weights(cell, panel, weights, config, spec)
        scored.append(scored_row)
        rejects[scored_row.trial_id] = row_rejects
    if len(scored) != len(cells):
        raise RuntimeError("a cell was dropped before recording")
    winner = select_winner(scored)
    lengths = {int(row.validation_returns.size) for row in scored}
    if len(lengths) != 1:
        raise RuntimeError(f"validation lengths differ: {sorted(lengths)}")

    from quant_fund.proof.bundle import build_bundle
    from quant_fund.proofcore.contracts import (
        DataAccessRecord,
        DataManifestSummary,
        merkle_root_hex,
    )

    signal = pl.DataFrame(
        {
            "trial_id": [row.trial_id for row in scored],
            "strategy": [row.cell.strategy for row in scored],
            "cluster_id": [row.cell.cluster_id for row in scored],
            "params_json": [json.dumps(row.cell.params, sort_keys=True) for row in scored],
        }
    )
    trade = winner.equity.select(pl.col("event_time").alias("fill_time"), pl.col("nav"))
    parquet_path = cache / "bars.parquet"
    cache.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(parquet_path)
    content_hash = sha256_hex_bytes(parquet_path.read_bytes())
    last_stamp = panel["event_time"].max()
    if not isinstance(last_stamp, datetime):
        raise RuntimeError("panel event_time max is not a datetime")
    manifest = DataManifestSummary(
        reads=(
            DataAccessRecord(
                dataset="reality/sp500_pit_yahoo_daily",
                asof_utc=last_stamp.astimezone(UTC).isoformat(),
                params={
                    "membership_sha256": member_hash,
                    "revision_id": "YAHOO_VENDOR_ADJ",
                    "source": "yahoo",
                    "study_id": str(spec["study_id"]),
                },
                rows=int(panel.height),
                content_sha256=content_hash,
            ),
        ),
        merkle_root=merkle_root_hex([content_hash]),
        n_reads=1,
    )
    bundle = build_bundle(
        run_kind="research",
        data_manifest=manifest,
        config_dump=spec,
        seed=int(spec["diagnostics"]["bootstrap_ci"]["seed"]),
        signal_log=signal,
        trade_log=trade,
        engine_metrics={
            "n_names": float(real["security_id"].n_unique()),
            "n_trials": float(len(scored)),
        },
        bundle_dir=cache / "bundle",
    )
    periods = float(spec["ledger"]["periods_per_year"])
    ledger_rows = [
        _ledger_row(
            row, bundle_hash=bundle.bundle_id, created_utc=bundle.created_utc, periods=periods
        )
        for row in scored
    ]
    new_ids = {row.trial_id for row in ledger_rows}
    old_ids = {json.loads(line)["trial_id"] for line in prefix.splitlines()}
    overlap = new_ids & old_ids
    if overlap:
        raise RuntimeError(f"new trial ids collide with the existing ledger: {sorted(overlap)[:3]}")
    append_ledger_lines(
        ledger,
        prefix,
        [json.dumps(row.model_dump(mode="json"), sort_keys=True) for row in ledger_rows],
    )
    append_csv_rows(csv_path or _CSV, ledger_rows)
    # Holdout summaries are attached only after the append. Selection already
    # used the validation ratio, and this helper refuses if that ratio moves.
    _attach_window_reports(scored, spec)

    pbo = pbo_on_pre_holdout(scored, spec)
    from quant_fund.proofcore.contracts import TrialLedgerRow
    from quant_fund.reality.report import build_reality_report

    all_rows = [
        TrialLedgerRow.model_validate(json.loads(line))
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(all_rows) != int(spec["expected_trials"]["ledger_total"]):
        raise RuntimeError(f"ledger has {len(all_rows)} rows after append")
    gate = build_reality_report(all_rows, q=float(spec["reality_filter"]["fdr_q"]))
    raw_dsr = _ledger_best_raw_dsr(all_rows)

    return_rows: list[dict[str, Any]] = []
    for scored_row in scored:
        series_by_window = {
            "train": scored_row.train_returns,
            "validation": scored_row.validation_returns,
            "holdout": scored_row.holdout_returns,
        }
        for window, series in series_by_window.items():
            session_dates = scored_row.by_window[window]["dates"]
            for day, value in zip(session_dates, series, strict=True):
                return_rows.append(
                    {
                        "trial_id": scored_row.trial_id,
                        "strategy": scored_row.cell.strategy,
                        "window": window,
                        "session_date": day,
                        "simple_return": float(value),
                    }
                )
    returns_frame = pl.DataFrame(return_rows)
    returns_file = cache / "returns.parquet"
    returns_frame.write_parquet(returns_file)
    returns_sha = sha256_hex_bytes(returns_file.read_bytes())
    from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256

    prior = _prior_baseline(_prior_receipt_path(spec))
    trials_public = [
        {
            "trial_id": row.trial_id,
            "strategy": row.cell.strategy,
            "cluster_id": row.cell.cluster_id,
            "params": row.cell.params,
            "returns_sha256": _returns_sha(row.validation_returns),
            "risk_gate_rejects": rejects[row.trial_id]["risk_gate_rejects"],
            "cash_rejects": rejects[row.trial_id]["cash_rejects"],
            "windows": {name: _public_window(row.by_window[name]) for name in row.by_window},
        }
        for row in sorted(scored, key=lambda item: item.trial_id)
    ]
    coverage_public = [
        {key: value for key, value in row.items() if key != "missing"}
        | {"n_missing": len(row["missing"])}
        for row in coverage
    ]
    missing_union = sorted({symbol for row in coverage for symbol in row["missing"]})
    receipt_body: dict[str, Any] = {
        "schema": "dipcatcher.reality_survivorship_receipt.v1",
        "data_source": "yahoo",
        "live_pnl_claim": False,
        "study_id": spec["study_id"],
        "preregistration_sha256": sha256_hex_bytes(spec_file.read_bytes()),
        "membership_sha256": member_hash,
        "n_names_with_bars": int(real["security_id"].n_unique()),
        "n_fetch_failures": len(failures),
        "fetch_failures": failures,
        "coverage": coverage_public,
        "missing_members_on_anchors": missing_union,
        "n_delisting_exit_bars": n_exit,
        "n_trials_new": len(scored),
        "n_trials_ledger": len(all_rows),
        "selected_trial_id": winner.trial_id,
        "selected_strategy": winner.cell.strategy,
        "bundle_id": bundle.bundle_id,
        "pbo": pbo,
        "deflated_probability_raw_count": raw_dsr,
        "reality_gate": gate.model_dump(mode="json"),
        "prior_equal_weight_long": prior,
        "returns_parquet_sha256": returns_sha,
        "remaining_bias": list(spec["data"]["remaining_bias"]),
        "leakage_checks": {
            "formation_uses_shift": True,
            "decision_close_excluded": True,
            "exit_bars_excluded_from_signals": True,
            "holdout_used_for_selection": False,
            "membership_undoes_only_later_events": True,
            "thresholds_unchanged": True,
        },
        "trials": trials_public,
        "provenance": {
            "run_id": spec["study_id"],
            "git_revision": git_revision(),
            "git_worktree_sha256": git_worktree_sha256(),
            "config_sha256": sha256_hex_bytes(spec_file.read_bytes()),
            "dataset_sha256": content_hash,
            "dataset_content_sha256": content_hash,
            "execution_claim": "historical_backtest_simulation",
            "point_in_time_membership": True,
            "point_in_time_prices": False,
        },
    }
    safe = _sanitize(receipt_body)
    if not isinstance(safe, dict):
        raise RuntimeError("receipt sanitizer dropped the object")
    encoded = json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    safe["receipt_sha256"] = sha256_hex_bytes(encoded)
    receipt_file = _ROOT / str(spec["outputs"]["receipt"])
    receipt_file.parent.mkdir(parents=True, exist_ok=True)
    receipt_file.write_text(
        json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    from quant_fund.audit.ledger import AuditLedger
    from quant_fund.audit.record import record_research_receipt
    from quant_fund.audit.signing import Ed25519Signer

    audit_root = _ROOT / str(spec["outputs"]["audit_dir"])
    if (audit_root / "entries.jsonl").exists():
        raise RuntimeError("survivorship audit ledger already exists; refusing to re-sign")
    signer = Ed25519Signer.generate()
    audit = AuditLedger(audit_root, signer=signer, sign_every=1)
    record_research_receipt(audit, receipt_file)
    (audit_root / "trust_pub.hex").write_text(signer.public_key_hex + "\n", encoding="utf-8")
    write_results_markdown(_ROOT / str(spec["outputs"]["results"]), safe)
    print(
        f"verdict={gate.verdict} n_trials={gate.n_trials} dsr={gate.dsr} "
        f"pbo={pbo.get('pbo')} bundle={bundle.bundle_id}",
        flush=True,
    )
    return safe


def _prior_receipt_path(spec: dict[str, Any]) -> Path:
    """Round-1 receipt moved under studies/ after archival; fall back to the frozen path."""
    frozen = _ROOT / str(spec["prior_study"]["receipt"])
    if frozen.exists():
        return frozen
    archived = (
        _ROOT
        / "research"
        / "reality"
        / "studies"
        / str(spec["prior_study"]["study_id"])
        / "receipt.json"
    )
    if archived.exists():
        return archived
    raise FileNotFoundError(f"prior receipt missing at {frozen} and {archived}")


def _trial_public_row(row: ScoredCell, rejects: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {
        "trial_id": row.trial_id,
        "strategy": row.cell.strategy,
        "cluster_id": row.cell.cluster_id,
        "params": row.cell.params,
        "returns_sha256": _returns_sha(row.validation_returns),
        "risk_gate_rejects": rejects[row.trial_id]["risk_gate_rejects"],
        "cash_rejects": rejects[row.trial_id]["cash_rejects"],
        "windows": {name: _public_window(row.by_window[name]) for name in row.by_window},
    }


def _score_panel(
    *,
    cells: list[Cell],
    real: pl.DataFrame,
    spy: pl.DataFrame,
    panel: pl.DataFrame,
    config: AppConfig,
    spec: dict[str, Any],
    name_cap: float,
    net_cap: float,
    label: str,
) -> tuple[list[ScoredCell], dict[str, dict[str, int]]]:
    scored: list[ScoredCell] = []
    rejects: dict[str, dict[str, int]] = {}
    for cell in cells:
        print(
            f"score[{label}] {cell.strategy} {json.dumps(cell.params, sort_keys=True)}",
            flush=True,
        )
        weights = weights_for_cell(cell, real, spy, name_cap=name_cap, net_cap=net_cap)
        scored_row, row_rejects = _score_weights(cell, panel, weights, config, spec)
        scored.append(scored_row)
        rejects[scored_row.trial_id] = row_rejects
    if len(scored) != len(cells):
        raise RuntimeError(f"{label}: a cell was dropped before recording")
    return scored, rejects


def _prepare_eligible_panel(
    stocks: pl.DataFrame,
    membership: dict[str, Any],
    spec: dict[str, Any],
    *,
    actions: pl.DataFrame | None,
) -> tuple[pl.DataFrame, pl.DataFrame, list[dict[str, Any]], int]:
    prepared = prepare_bars(stocks, actions)
    prepared = with_eligibility(prepared, membership)
    coverage = anchor_coverage(prepared, membership, list(spec["data"]["coverage_anchors"]))
    floor = float(spec["data"]["coverage_floor"])
    for report in coverage:
        print(
            f"coverage {report['anchor']} session={report['session']} "
            f"{report['with_bar']}/{report['members']}={report['coverage']:.4f}",
            flush=True,
        )
        if float(report["coverage"]) < floor:
            raise RuntimeError(
                f"coverage {report['coverage']:.4f} on {report['session']} is below {floor}; "
                "not scoring"
            )
    real = prepared.filter(pl.col("source") == "yahoo")
    panel, n_exit = append_delisting_exits(
        real.drop(
            [
                column
                for column in ("ny_date", "is_last_real", "in_index", "eligible")
                if column in real.columns
            ]
        )
    )
    return real, panel, coverage, n_exit


def compare_return_bases(
    *,
    spec_path: Path | None = None,
    membership_path: Path | None = None,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Score the frozen 13-cell grid on quote closes and on total-return marks.

    One Yahoo fetch with ``events=div,split`` feeds both panels. Does not
    append to the trial ledger, does not re-sign the sealed audit chain, and
    does not edit reality-filter thresholds. Research diagnostic only.
    """
    from quant_fund.proofcore.contracts import sha256_hex_bytes
    from quant_fund.research.total_return import YAHOO_CHART_EVENTS

    spec_file = spec_path or _SPEC
    spec = load_spec(spec_file)
    membership_file = membership_path or _MEMBERSHIP
    member_hash = sha256_hex_bytes(membership_file.read_bytes())
    if member_hash != str(spec["data"]["membership_sha256"]):
        raise RuntimeError("membership file hash does not match the pre-registration")
    membership = load_membership(membership_file)
    cells = grid_cells(spec)
    config = load_config(_ROOT / "configs" / "backtest.yaml")
    assert_cost_lock(config, spec)
    cache = cache_dir or (_ROOT / "data" / "reality_sweep_survivorship_tr")
    out_dir = output_dir or (_ROOT / "research" / "reality" / "survivorship" / "total_return")
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = fetch_universe(membership, spec)
    stocks, spy, actions, failures = fetch_yahoo_members(
        tickers, spec, cache, events=YAHOO_CHART_EVENTS
    )
    # Regime series stays on quote closes for the causal SPY filter in both
    # modes so the on/off mask is identical; only constituent marks change.
    name_cap = float(spec["risk_gate"]["max_name"])
    net_cap = float(spec["risk_gate"]["max_net"])
    modes: dict[str, dict[str, Any]] = {}
    for mode, mode_actions in ((_RETURN_QUOTE, None), (_RETURN_TOTAL, actions)):
        print(f"prepare mode={mode}", flush=True)
        real, panel, coverage, n_exit = _prepare_eligible_panel(
            stocks, membership, spec, actions=mode_actions
        )
        scored, rejects = _score_panel(
            cells=cells,
            real=real,
            spy=spy,
            panel=panel,
            config=config,
            spec=spec,
            name_cap=name_cap,
            net_cap=net_cap,
            label=mode,
        )
        winner = select_winner(scored)
        pbo = pbo_on_pre_holdout(scored, n_splits=int(spec["pbo"]["n_splits"]))
        modes[mode] = {
            "return_basis": mode,
            "n_dividend_rows": int(
                actions.filter(
                    pl.col("action_type").is_in(["cash_dividend", "special_dividend"])
                ).height
            )
            if mode == _RETURN_TOTAL
            else 0,
            "n_names_with_bars": int(real["security_id"].n_unique()),
            "n_delisting_exit_bars": n_exit,
            "selected_trial_id": winner.trial_id,
            "selected_strategy": winner.cell.strategy,
            "pbo": pbo,
            "coverage": [
                {key: value for key, value in row.items() if key != "missing"}
                | {"n_missing": len(row["missing"])}
                for row in coverage
            ],
            "trials": [
                _trial_public_row(row, rejects)
                for row in sorted(scored, key=lambda item: item.trial_id)
            ],
        }

    sealed = json.loads(
        (_ROOT / "research" / "reality" / "survivorship" / "receipt.json").read_text(
            encoding="utf-8"
        )
    )
    comparison_rows = []
    quote_by_id = {row["trial_id"]: row for row in modes[_RETURN_QUOTE]["trials"]}
    total_by_id = {row["trial_id"]: row for row in modes[_RETURN_TOTAL]["trials"]}
    sealed_by_id = {row["trial_id"]: row for row in sealed["trials"]}
    for trial_id in sorted(quote_by_id):
        quote = quote_by_id[trial_id]
        total = total_by_id[trial_id]
        published = sealed_by_id.get(trial_id)
        comparison_rows.append(
            {
                "trial_id": trial_id,
                "strategy": quote["strategy"],
                "params": quote["params"],
                "published_quote_holdout_compounded": (
                    None
                    if published is None
                    else published["windows"]["holdout"]["compounded_return"]
                ),
                "recomputed_quote_holdout_compounded": quote["windows"]["holdout"][
                    "compounded_return"
                ],
                "total_return_holdout_compounded": total["windows"]["holdout"]["compounded_return"],
                "published_quote_validation_annualized": (
                    None
                    if published is None
                    else published["windows"]["validation"]["annualized_ratio"]
                ),
                "recomputed_quote_validation_annualized": quote["windows"]["validation"][
                    "annualized_ratio"
                ],
                "total_return_validation_annualized": total["windows"]["validation"][
                    "annualized_ratio"
                ],
                "published_quote_holdout_annualized": (
                    None
                    if published is None
                    else published["windows"]["holdout"]["annualized_ratio"]
                ),
                "recomputed_quote_holdout_annualized": quote["windows"]["holdout"][
                    "annualized_ratio"
                ],
                "total_return_holdout_annualized": total["windows"]["holdout"]["annualized_ratio"],
            }
        )

    from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256

    body: dict[str, Any] = {
        "schema": "dipcatcher.reality_survivorship_total_return_comparison.v1",
        "live_pnl_claim": False,
        "study_id": spec["study_id"],
        "note": (
            "Paired quote-close vs cash-dividend total-return marks on one Yahoo "
            "fetch. Does not append to the trial ledger. Not a live-trading claim."
        ),
        "preregistration_sha256": sha256_hex_bytes(spec_file.read_bytes()),
        "membership_sha256": member_hash,
        "n_fetch_failures": len(failures),
        "fetch_failures": failures,
        "n_corporate_action_rows": int(actions.height),
        "n_cash_dividend_rows": int(
            actions.filter(
                pl.col("action_type").is_in(["cash_dividend", "special_dividend"])
            ).height
        ),
        "modes": modes,
        "comparison": comparison_rows,
        "sealed_quote_receipt_sha256": sealed.get("receipt_sha256"),
        "provenance": {
            "git_revision": git_revision(),
            "git_worktree_sha256": git_worktree_sha256(),
            "execution_claim": "historical_backtest_simulation",
            "point_in_time_membership": True,
            "point_in_time_prices": False,
            "return_bases": [_RETURN_QUOTE, _RETURN_TOTAL],
        },
    }
    safe = _sanitize(body)
    if not isinstance(safe, dict):
        raise RuntimeError("comparison sanitizer dropped the object")
    encoded = json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    safe["receipt_sha256"] = sha256_hex_bytes(encoded)
    receipt_path = out_dir / "comparison.json"
    receipt_path.write_text(
        json.dumps(safe, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_comparison_markdown(
        _ROOT / "docs" / "REALITY_TRIAL_SURVIVORSHIP_TOTAL_RETURN_2026.md", safe
    )
    print(
        f"comparison written n_trials={len(comparison_rows)} "
        f"dividends={safe['n_cash_dividend_rows']} receipt={safe['receipt_sha256']}",
        flush=True,
    )
    return safe


def write_comparison_markdown(path: Path, receipt: dict[str, Any]) -> None:
    """Before/after note for quote closes vs dividend total-return marks."""
    lines = [
        "# Survivorship trial: quote close vs total return",
        "",
        "Research diagnostic only. Same point-in-time S&P membership grid as",
        "`docs/REALITY_TRIAL_SURVIVORSHIP_2026.md`, scored twice on one Yahoo",
        "fetch: once on quote closes (default), once after cash-dividend",
        "reinvestment through `prepare_bars(..., actions)`. Reality-filter",
        "thresholds and the sealed ledger were not edited. Not a live-trading",
        "claim.",
        "",
        f"Comparison receipt sha256: `{receipt['receipt_sha256']}`.",
        f"Cash-dividend rows on the fetch: {receipt['n_cash_dividend_rows']}.",
        f"Fetch failures: {receipt['n_fetch_failures']}.",
        "",
        "## Equal-weight pit baseline",
        "",
        "| return basis | validation compounded | holdout compounded | validation annualized | holdout annualized |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in (_RETURN_QUOTE, _RETURN_TOTAL):
        pit = next(
            row for row in receipt["modes"][mode]["trials"] if row["strategy"] == "equal_weight_pit"
        )
        lines.append(
            "| {mode} | {v} | {h} | {va} | {ha} |".format(
                mode=mode,
                v=_fmt(pit["windows"]["validation"]["compounded_return"]),
                h=_fmt(pit["windows"]["holdout"]["compounded_return"]),
                va=_fmt(pit["windows"]["validation"]["annualized_ratio"]),
                ha=_fmt(pit["windows"]["holdout"]["annualized_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Every cell: holdout compounded return",
            "",
            "| strategy | params | published quote | recomputed quote | total return | delta (TR − quote) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in receipt["comparison"]:
        quote = float(row["recomputed_quote_holdout_compounded"])
        total = float(row["total_return_holdout_compounded"])
        published = row["published_quote_holdout_compounded"]
        lines.append(
            "| {strategy} | `{params}` | {published} | {quote} | {total} | {delta} |".format(
                strategy=row["strategy"],
                params=json.dumps(row["params"], sort_keys=True),
                published=_fmt(published),
                quote=_fmt(quote),
                total=_fmt(total),
                delta=_fmt(total - quote),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Published quote figures are the sealed survivorship receipt",
            "  (quote closes, dividends omitted).",
            "- Recomputed quote and total-return figures share one Yahoo download",
            "  with `events=div,split`, so the delta isolates dividend reinvestment.",
            "- Fills stay at the open on the same price basis as the mark.",
            "- Annualized ratio is mean/std × √252 on net simple returns; it is an",
            "  overfitting diagnostic here, not a promotion.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare-return-bases":
        compare_return_bases()
        return
    run_study()


if __name__ == "__main__":
    main()
