"""Rate-tiered delta-neutral carry book — adaptive size per funding level.

Same hysteresis membership as ``basis_carry_hysteresis_weights`` but position
size scales with the trailing realized funding rate: bigger when funding is
rich, smaller when thin. Size changes are emitted only on TIER transitions
(quantized levels), so churn stays low like the flat-weight book.

Tier ladder: w = clip(floor(rate_ma / tier_bp) * tier_w, 0, w_cap) where the
emitted weight is the LADDER STEP below the raw score — a resize row fires
only when the raw score crosses a step boundary. A held name also exits on
rate_ma < exit_rate and resizes on price-drift breach of rebalance_band.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl


def tiered_carry_weights(
    bars: pl.DataFrame,
    funding: pl.DataFrame,
    *,
    enter_rate: float = 0.0003,
    exit_rate: float = 0.0,
    lookback_events: int = 9,
    step_rate: float = 0.0003,  # funding increment per size step
    step_weight: float = 0.04,  # weight per step
    w_cap: float = 0.20,
    max_names: int = 15,
    rebalance_band: float | None = 1.3,
) -> pl.DataFrame:
    if funding.height == 0:
        raise ValueError("funding frame must be non-empty")
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

    def _steps(rate: float) -> int:
        if rate is None or rate < enter_rate:
            return 0
        return min(int(rate / step_rate), int(w_cap / step_weight))

    active: dict[str, int] = {}  # sid -> current step count
    entry_px: dict[str, float] = {}
    out_t: list[datetime] = []
    out_s: list[str] = []
    out_w: list[float] = []
    for t in sorted(rates_by_time):
        rates = rates_by_time[t]
        px_now = px_by_time.get(t, {})
        # exits first
        for sid in list(active):
            r = rates.get(sid)
            if r is None or r < exit_rate:
                out_t.append(t)
                out_s.append(sid)
                out_w.append(0.0)
                del active[sid]
                entry_px.pop(sid, None)
        # tier resize on held names (only on step change or drift breach)
        for sid in list(active):
            r = rates.get(sid)
            if r is None:
                continue
            want = _steps(r)
            if want <= 0:
                continue
            p0 = entry_px.get(sid)
            p1 = px_now.get(sid)
            drift = (p1 / p0) if (p0 and p1 and p0 > 0 and p1 > 0) else 1.0
            band_breach = rebalance_band is not None and (
                drift >= rebalance_band or drift <= 1.0 / rebalance_band
            )
            if want != active[sid] or band_breach:
                out_t.append(t)
                out_s.append(sid)
                out_w.append(min(w_cap, want * step_weight))
                active[sid] = want
                if band_breach and p1:
                    entry_px[sid] = p1
        # entries: highest-rate candidates for open slots
        slots = max_names - len(active)
        if slots > 0:
            cands = sorted(
                (
                    (sid, r)
                    for sid, r in rates.items()
                    if r is not None and r >= enter_rate and sid not in active
                ),
                key=lambda kv: kv[1],
                reverse=True,
            )[:slots]
            for sid, r in cands:
                steps = _steps(r)
                if steps <= 0:
                    continue
                active[sid] = steps
                pe = px_now.get(sid)
                if pe is not None and pe > 0:
                    entry_px[sid] = pe
                out_t.append(t)
                out_s.append(sid)
                out_w.append(min(w_cap, steps * step_weight))
    return pl.DataFrame({"event_time": out_t, "security_id": out_s, "target_weight": out_w}).sort(
        ["event_time", "security_id"]
    )


def main() -> int:
    import json
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from carry_research import SPLIT, load_carry, make_cfg

    from quant_fund.backtest.carry_engine import run_carry_backtest

    perp, spot, fund = load_carry()
    dev_p = perp.filter(pl.col("event_time") < SPLIT)
    dev_s = spot.filter(pl.col("event_time") < SPLIT)
    dev_f = fund.filter(pl.col("event_time") < SPLIT)

    grid = [
        dict(enter_rate=e, lookback_events=lb, step_rate=sr, step_weight=sw, w_cap=wc, max_names=mx)
        for e in (0.0002, 0.0003)
        for lb in (3, 9)
        for sr in (0.0003, 0.0005)
        for sw in (0.04, 0.06)
        for wc in (0.15, 0.20)
        for mx in (10, 15)
    ]
    out = {}
    for g in grid:
        key = "_".join(f"{k}{v}" for k, v in g.items())
        try:
            w = tiered_carry_weights(
                dev_p,
                dev_f,
                exit_rate=0.0,
                rebalance_band=1.3,
                **g,
            )
            res = run_carry_backtest(dev_p, dev_s, dev_f, w, make_cfg(), initial_nav=1e6)
            m = res.metrics
            out[key] = {
                k: m.get(k)
                for k in (
                    "total_return",
                    "cagr",
                    "sharpe",
                    "max_drawdown",
                    "mean_turnover",
                    "funding_net",
                    "liquidation_count",
                    "margin_rejects",
                )
            }
            print(key, json.dumps(out[key], default=str), flush=True)
        except Exception as e:  # noqa: BLE001
            print(key, "ERR", type(e).__name__, e, flush=True)
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/carry_tiered_dev.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print("wrote artifacts/carry_tiered_dev.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
