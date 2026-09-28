"""Helpers for the event-loop ↔ fast_replay differential fuzzer.

Stated tolerances are the contract the fuzzer enforces. On the matched-class
panel the engines are expected to land far inside these bands (often
bit-identical); the numbers below are the public acceptance floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.backtest.engine import BacktestResult, _run_backtest_event_loop
from quant_fund.backtest.fast_replay import run_backtest_fast
from quant_fund.config.models import (
    AppConfig,
    CostConfig,
    DataConfig,
    ExecutionConfig,
    FillConvention,
    KillSwitchConfig,
    RiskGateConfig,
)

# --- stated tolerances -------------------------------------------------------
NAV_ATOL = 1e-9
NAV_RTOL = 0.0
FILL_QTY_ATOL = 1e-12
FILL_PX_ATOL = 1e-12
POSITION_ATOL = 1e-12

T0 = datetime(2024, 1, 2, tzinfo=UTC)

# Isolated data root so GARCH overlay artifacts never trip the fast-path
# completeness refusal during property generation.
_DIFF_DATA_ROOT = Path("/tmp/dipcatcher-differential-fuzzer")


@dataclass(frozen=True)
class FaultFlags:
    """Which adversarial injections are present on a generated workload."""

    missing_prints: bool = False
    trading_halt: bool = False
    bad_prices: bool = False  # zero / negative / NaN
    duplicate_bars: bool = False
    splits: bool = False
    dividends: bool = False


@dataclass
class DiffWorkload:
    bars: pl.DataFrame
    weights: pl.DataFrame
    config: AppConfig
    faults: FaultFlags
    notes: list[str] = field(default_factory=list)


def _research_config(
    *,
    kill: str = "ENABLED",
    commission_bps: float = 5.0,
    half_spread_bps: float = 1.0,
    impact_y: float = 0.0,
    frictionless: bool = False,
    participation_limit: float = 1.0,
    stale_price_bars: int = 5,
    max_gross: float = 2.0,
    max_net: float = 1.0,
    max_name: float = 1.0,
) -> AppConfig:
    return AppConfig(
        data=DataConfig(root=_DIFF_DATA_ROOT),
        costs=CostConfig(
            commission_bps=commission_bps,
            half_spread_bps=half_spread_bps,
            impact_y=impact_y,
            bps_per_turnover=0.0,
            borrow_bps_per_year=0.0,
            frictionless=frictionless,
            participation_limit=participation_limit,
        ),
        risk_gate=RiskGateConfig(
            max_order_notional=1e15,
            max_gross=max_gross,
            max_net=max_net,
            max_name=max_name,
            max_participation=1.0,
            max_predicted_vol=100.0,
            stale_price_bars=stale_price_bars,
            stale_model_hours=1e9,
        ),
        execution=ExecutionConfig(
            fill=FillConvention.NEXT_OPEN,
            allow_close_auction=False,
        ),
        kill_switch=KillSwitchConfig(state=kill),
    )


def build_clean_panel(
    *,
    n_assets: int,
    n_days: int,
    seed: int,
    weight: float = 0.4,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministic clean multi-name daily panel (no faults)."""
    rng = np.random.default_rng(seed)
    sids = [f"S{k}" for k in range(n_assets)]
    bar_rows: list[dict[str, Any]] = []
    for k, sid in enumerate(sids):
        px = 50.0 + 25.0 * k
        ctr = px
        for i in range(n_days):
            ret = float(rng.normal(0.0005, 0.01))
            open_px = px
            close = px * (1.0 + ret)
            ctr = ctr * (1.0 + ret)
            bar_rows.append(
                {
                    "security_id": sid,
                    "event_time": T0 + timedelta(days=i),
                    "open": open_px,
                    "close": close,
                    "close_total_return": ctr,
                    "volume": 1_000_000.0,
                    "adv": close * 1_000_000.0,
                    "vol_20": 0.02,
                    "source": "synthetic",
                }
            )
            px = close
    bars = pl.DataFrame(bar_rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    w_rows = [
        {
            "event_time": T0 + timedelta(days=i),
            "security_id": sid,
            "target_weight": weight / n_assets,
        }
        for i in range(n_days)
        for sid in sids
    ]
    weights = pl.DataFrame(w_rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))
    return bars, weights


def inject_missing_prints(
    bars: pl.DataFrame,
    *,
    seed: int,
    rate: float,
) -> pl.DataFrame:
    """Drop a fraction of mid-panel bar rows (missing exchange prints)."""
    if bars.height == 0 or rate <= 0.0:
        return bars
    rng = np.random.default_rng(seed)
    keep: list[bool] = []
    times = bars["event_time"].to_list()
    t0, t1 = min(times), max(times)
    for row in bars.iter_rows(named=True):
        et = row["event_time"]
        # Keep first/last day for every name so the panel stays non-empty
        # and staleness has somewhere to start/stop.
        if et in (t0, t1):
            keep.append(True)
        else:
            keep.append(rng.random() >= rate)
    if not any(keep):
        return bars
    return bars.filter(pl.Series(keep))


def inject_trading_halt_days(
    bars: pl.DataFrame,
    *,
    seed: int,
    n_halts: int,
) -> pl.DataFrame:
    """Null the open on selected days — no executable print (halted session)."""
    if n_halts <= 0 or bars.height == 0:
        return bars
    rng = np.random.default_rng(seed)
    dates = sorted(set(bars["event_time"].to_list()))
    # Never halt the first day: decisions need a prior close.
    candidates = dates[1:] if len(dates) > 1 else dates
    if not candidates:
        return bars
    pick = set(rng.choice(candidates, size=min(n_halts, len(candidates)), replace=False).tolist())
    return bars.with_columns(
        pl.when(pl.col("event_time").is_in(list(pick)))
        .then(None)
        .otherwise(pl.col("open"))
        .alias("open")
    )


def inject_bad_prices(
    bars: pl.DataFrame,
    *,
    seed: int,
    n_bad: int,
    kinds: tuple[str, ...] = ("zero", "negative", "nan"),
) -> pl.DataFrame:
    """Overwrite selected OHLCV cells with zero / negative / NaN prices."""
    if n_bad <= 0 or bars.height == 0:
        return bars
    rng = np.random.default_rng(seed)
    cols = ["open", "close", "close_total_return"]
    rows = bars.to_dicts()
    # Avoid first row so we can still enter a position before poison.
    idxs = list(range(1, len(rows))) if len(rows) > 1 else list(range(len(rows)))
    if not idxs:
        return bars
    chosen = rng.choice(idxs, size=min(n_bad, len(idxs)), replace=False)
    for i in np.atleast_1d(chosen):
        col = str(rng.choice(cols))
        kind = str(rng.choice(kinds))
        if kind == "zero":
            rows[int(i)][col] = 0.0
        elif kind == "negative":
            rows[int(i)][col] = -abs(float(rows[int(i)].get(col) or 1.0))
        else:
            rows[int(i)][col] = float("nan")
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us", "UTC")))


def inject_duplicate_bars(
    bars: pl.DataFrame,
    *,
    seed: int,
    conflict: bool,
) -> pl.DataFrame:
    """Append a second row for an existing (event_time, security_id) key.

    When ``conflict`` is True the duplicate carries a different open so a
    last-write-wins engine would diverge from first-write-wins.
    """
    if bars.height == 0:
        return bars
    rng = np.random.default_rng(seed)
    i = int(rng.integers(0, bars.height))
    row = bars.row(i, named=True)
    dup = dict(row)
    if conflict:
        base = (
            float(dup["open"])
            if dup["open"] is not None and math.isfinite(float(dup["open"]))
            else 100.0
        )
        dup["open"] = base * 1.1
    return pl.concat([bars, pl.DataFrame([dup]).cast(bars.schema)], how="vertical_relaxed")


def inject_split(
    bars: pl.DataFrame,
    *,
    sid: str,
    day_index: int,
    factor: float = 2.0,
) -> pl.DataFrame:
    """Apply a ``factor``-for-1 split on ``sid`` from ``day_index`` onward.

    Raw open/close are divided by ``factor``; ``close_total_return`` is left
    continuous on the pre-split basis (the engines mark via CTR).
    """
    dates = sorted(set(bars["event_time"].to_list()))
    if day_index < 0 or day_index >= len(dates):
        return bars
    ex = dates[day_index]
    return bars.with_columns(
        [
            pl.when((pl.col("security_id") == sid) & (pl.col("event_time") >= ex))
            .then(pl.col("open") / factor)
            .otherwise(pl.col("open"))
            .alias("open"),
            pl.when((pl.col("security_id") == sid) & (pl.col("event_time") >= ex))
            .then(pl.col("close") / factor)
            .otherwise(pl.col("close"))
            .alias("close"),
            # CTR stays continuous — already built on the adjusted path in
            # build_clean_panel; re-scale post-ex CTR so the jump from the
            # unadjusted close does not appear in the mark series.
            pl.when((pl.col("security_id") == sid) & (pl.col("event_time") >= ex))
            .then(pl.col("close_total_return"))  # already continuous
            .otherwise(pl.col("close_total_return"))
            .alias("close_total_return"),
        ]
    )


def inject_dividend(
    bars: pl.DataFrame,
    *,
    sid: str,
    day_index: int,
    amount: float,
) -> pl.DataFrame:
    """Cash dividend on ``sid`` at ``day_index``.

    Raw close drops by ``amount``; ``close_total_return`` is held flat across
    the ex-date so reinvested wealth is conserved in the mark.
    """
    dates = sorted(set(bars["event_time"].to_list()))
    if day_index <= 0 or day_index >= len(dates) or amount <= 0:
        return bars
    ex = dates[day_index]
    # Pull the pre-ex CTR so we can pin the ex-date mark.
    pre = bars.filter((pl.col("security_id") == sid) & (pl.col("event_time") < ex)).sort(
        "event_time"
    )
    if pre.height == 0:
        return bars
    pre_ctr = float(pre["close_total_return"][-1])
    return bars.with_columns(
        [
            pl.when((pl.col("security_id") == sid) & (pl.col("event_time") == ex))
            .then(pl.col("close") - amount)
            .otherwise(pl.col("close"))
            .alias("close"),
            pl.when((pl.col("security_id") == sid) & (pl.col("event_time") == ex))
            .then(pl.lit(pre_ctr))
            .otherwise(pl.col("close_total_return"))
            .alias("close_total_return"),
        ]
    )


def assemble_workload(
    *,
    seed: int,
    n_assets: int,
    n_days: int,
    missing_rate: float,
    n_halt_days: int,
    n_bad_prices: int,
    duplicate: bool,
    duplicate_conflict: bool,
    split: bool,
    dividend: bool,
    kill_halt: bool,
) -> DiffWorkload:
    """Build one adversarial workload from independent fault knobs."""
    bars, weights = build_clean_panel(n_assets=n_assets, n_days=n_days, seed=seed)
    notes: list[str] = []
    flags = FaultFlags(
        missing_prints=missing_rate > 0,
        trading_halt=n_halt_days > 0 or kill_halt,
        bad_prices=n_bad_prices > 0,
        duplicate_bars=duplicate,
        splits=split,
        dividends=dividend,
    )

    if missing_rate > 0:
        bars = inject_missing_prints(bars, seed=seed + 11, rate=missing_rate)
        notes.append(f"missing_prints rate={missing_rate:.3f}")

    if n_halt_days > 0:
        bars = inject_trading_halt_days(bars, seed=seed + 22, n_halts=n_halt_days)
        notes.append(f"halt_days={n_halt_days}")

    if n_bad_prices > 0:
        bars = inject_bad_prices(bars, seed=seed + 33, n_bad=n_bad_prices)
        notes.append(f"bad_prices={n_bad_prices}")

    sids = sorted(set(bars["security_id"].to_list()))
    if split and sids:
        bars = inject_split(bars, sid=sids[0], day_index=max(1, n_days // 3), factor=2.0)
        notes.append(f"split 2:1 on {sids[0]}")

    if dividend and sids:
        bars = inject_dividend(bars, sid=sids[0], day_index=max(2, n_days // 2), amount=1.0)
        notes.append(f"dividend $1 on {sids[0]}")

    if duplicate:
        bars = inject_duplicate_bars(bars, seed=seed + 44, conflict=duplicate_conflict)
        notes.append(f"duplicate_bars conflict={duplicate_conflict}")

    cfg = _research_config(kill="HALT_NEW_ORDERS" if kill_halt else "ENABLED")
    if kill_halt:
        notes.append("kill_switch=HALT_NEW_ORDERS")

    return DiffWorkload(bars=bars, weights=weights, config=cfg, faults=flags, notes=notes)


def positions_from_fills(fills: pl.DataFrame) -> dict[str, float]:
    """Terminal share counts reconstructed from signed fill quantities."""
    if fills.height == 0:
        return {}
    out: dict[str, float] = {}
    for row in fills.iter_rows(named=True):
        sid = str(row["security_id"])
        out[sid] = out.get(sid, 0.0) + float(row["quantity"])
    return out


def _finite_close(a: float, b: float, *, atol: float, rtol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


@dataclass
class DiffReport:
    ok: bool
    reason: str = ""
    detail: str = ""


def compare_results(
    ref: BacktestResult,
    fast: BacktestResult,
    *,
    nav_atol: float = NAV_ATOL,
    nav_rtol: float = NAV_RTOL,
    fill_qty_atol: float = FILL_QTY_ATOL,
    fill_px_atol: float = FILL_PX_ATOL,
    position_atol: float = POSITION_ATOL,
) -> DiffReport:
    """Assert fills, reconstructed positions, and NAV agree within tolerance."""
    if ref.equity.height != fast.equity.height:
        return DiffReport(
            False,
            "equity_height",
            f"ref={ref.equity.height} fast={fast.equity.height}",
        )
    if ref.equity.height:
        for col in ("nav", "gross", "net", "turnover"):
            r = np.asarray(ref.equity[col].to_list(), dtype=float)
            f = np.asarray(fast.equity[col].to_list(), dtype=float)
            for i, (a, b) in enumerate(zip(r, f, strict=True)):
                if not _finite_close(float(a), float(b), atol=nav_atol, rtol=nav_rtol):
                    return DiffReport(
                        False,
                        f"equity.{col}",
                        f"i={i} ref={a!r} fast={b!r}",
                    )
        if ref.equity["event_time"].to_list() != fast.equity["event_time"].to_list():
            return DiffReport(False, "equity.event_time", "timestamp mismatch")

    if ref.fills.height != fast.fills.height:
        return DiffReport(
            False,
            "fills_height",
            f"ref={ref.fills.height} fast={fast.fills.height}",
        )
    if ref.fills.height:
        if ref.fills["security_id"].to_list() != fast.fills["security_id"].to_list():
            return DiffReport(False, "fills.security_id", "order/id mismatch")
        for col, atol in (
            ("quantity", fill_qty_atol),
            ("price", fill_px_atol),
            ("fee", fill_px_atol),
            ("spread_cost", fill_px_atol),
            ("impact_cost", fill_px_atol),
            ("turnover_cost", fill_px_atol),
        ):
            r = np.asarray(ref.fills[col].to_list(), dtype=float)
            f = np.asarray(fast.fills[col].to_list(), dtype=float)
            for i, (a, b) in enumerate(zip(r, f, strict=True)):
                if not _finite_close(float(a), float(b), atol=atol, rtol=0.0):
                    return DiffReport(
                        False,
                        f"fills.{col}",
                        f"i={i} ref={a!r} fast={b!r}",
                    )

    pos_r = positions_from_fills(ref.fills)
    pos_f = positions_from_fills(fast.fills)
    keys = sorted(set(pos_r) | set(pos_f))
    for sid in keys:
        a, b = pos_r.get(sid, 0.0), pos_f.get(sid, 0.0)
        if not _finite_close(a, b, atol=position_atol, rtol=0.0):
            return DiffReport(False, "positions", f"{sid}: ref={a!r} fast={b!r}")

    return DiffReport(True)


def run_pair(
    workload: DiffWorkload,
) -> tuple[BacktestResult | BaseException, BacktestResult | BaseException]:
    """Run both engines; capture exceptions as values for differential checks."""
    try:
        ref: BacktestResult | BaseException = _run_backtest_event_loop(
            workload.bars, workload.weights, workload.config
        )
    except BaseException as exc:  # noqa: BLE001 — differential must see any raise
        ref = exc
    try:
        fast: BacktestResult | BaseException = run_backtest_fast(
            workload.bars, workload.weights, workload.config
        )
    except BaseException as exc:  # noqa: BLE001
        fast = exc
    return ref, fast
