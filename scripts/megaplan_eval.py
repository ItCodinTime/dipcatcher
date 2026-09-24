"""Megaplan Phase E harness on the rebuilt daily Binance book.

Phase C sleeves + Phase D overlay scalers, frozen-config protocol:
dev-halves grid selection -> freeze -> ONE locked holdout eval at SPLIT.

Stages (all scored on dev halves only, score = min(sharpe_h1, sharpe_h2)):
  A. per-sleeve param grid -> best variant per sleeve
  B. blend-weight grid over the surviving sleeves (ex-ante combos)
  C. overlay target-vol grid (DD governor fixed at megaplan spec 3%/5%)

Writes a JSON receipt with per-segment metrics + frozen config + input hashes.
Failure is a result: if holdout does not clear Sharpe>5 / MDD<5%, the receipt
says so. No holdout retuning.

Usage: uv run python scripts/megaplan_eval.py [--data-dir data/binance_carry]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import polars as pl  # noqa: E402

from quant_fund.backtest.overlay import (  # noqa: E402
    CompositeScaler,
    DrawdownGovernor,
    VolTargetScaler,
)
from quant_fund.backtest.perp_engine import run_perp_backtest  # noqa: E402
from quant_fund.backtest.sleeves import (  # noqa: E402
    blend_weights,
    cross_sectional_momentum_weights,
    funding_carry_weights,
    funding_spike_fade_weights,
    slow_trend_weights,
    sweep_reclaim_weights,
)
from quant_fund.config.loader import load_config  # noqa: E402

SPLIT = datetime(2025, 1, 1, tzinfo=UTC)
PPY = 365.25
METRIC_KEYS = (
    "total_return",
    "cagr",
    "sharpe",
    "max_drawdown",
    "n",
    "periods_per_year",
    "mean_turnover",
    "funding_paid_total",
    "funding_received_total",
    "liquidation_count",
    "margin_rejects",
    "ruined",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slice(df: pl.DataFrame, lo: datetime, hi: datetime) -> pl.DataFrame:
    return df.filter((pl.col("event_time") >= lo) & (pl.col("event_time") < hi))


def _metrics(m: dict) -> dict:
    return {k: m.get(k) for k in METRIC_KEYS if k in m}


def _sharpe(m: dict) -> float:
    s = m.get("sharpe")
    return float(s) if s is not None and s == s else float("-inf")


def _score(h1: dict, h2: dict, *, mdd_gate: bool) -> float:
    """min-half Sharpe; hard-ruin always disqualifies; MDD>5.1% disqualifies
    only when mdd_gate is set — i.e. once the Phase-D overlay (the designed
    MDD control) is applied. Raw sleeves are scored on predictive power alone."""
    s = min(_sharpe(h1), _sharpe(h2))
    if h1.get("ruined") or h2.get("ruined"):
        return float("-inf")
    if mdd_gate:
        for h in (h1, h2):
            mdd = h.get("max_drawdown")
            if mdd is not None and abs(float(mdd)) > 0.051:
                return float("-inf")
    return s


class Evaluator:
    def __init__(
        self,
        bars: pl.DataFrame,
        funding: pl.DataFrame,
        cfg,
        nav: float,
        vol_window: int,
        ppy: float,
    ) -> None:
        self.bars, self.funding, self.cfg, self.nav = bars, funding, cfg, nav
        self.vol_window, self.ppy = vol_window, ppy
        self._wcache: dict[str, pl.DataFrame] = {}
        self._elig: dict[tuple, set[str]] = {}
        self.exclusions: dict[tuple, dict[str, str]] = {}
        self.n_evals = 0
        self.times_by_sid = {
            (sid[0] if isinstance(sid, tuple) else sid): df["event_time"].sort().to_list()
            for sid, df in bars.group_by("security_id")
        }
        self.bar_times = sorted(bars["event_time"].unique().to_list())
        self.step_s = (
            (self.bar_times[1] - self.bar_times[0]).total_seconds()
            if len(self.bar_times) > 1
            else 3600.0
        )
        self.stale_bound = int(cfg.risk_gate.stale_price_bars) + 1

    def _funding_on_bars(self, f: pl.DataFrame) -> pl.DataFrame:
        """Assign each funding event to the latest bar at-or-before it.

        The perp engine applies funding only on exact bar-timestamp matches
        (`fund_map.get(dt)`); on a coarser grain (e.g. 8h events on daily
        bars) the intraday events would be dropped, losing 2/3 of carry
        income. Backward-asof lands every event on its containing bar."""
        if f.height == 0:
            return f
        grid = pl.DataFrame({"bar_time": self.bar_times})
        return (
            f.sort("event_time")
            .join_asof(grid, left_on="event_time", right_on="bar_time", strategy="backward")
            .drop_nulls("bar_time")
            .drop("event_time")
            .rename({"bar_time": "event_time"})
        )

    def eligible(self, lo: datetime, hi: datetime) -> set[str]:
        """Sids markable through the whole segment: prints in-segment, no
        internal spacing > stale_bound bars, no dead tail reaching `hi`."""
        key = (lo, hi)
        if key in self._elig:
            return self._elig[key]
        ok, reasons = set(), {}
        for sid, ts in self.times_by_sid.items():
            seg = [t for t in ts if lo <= t < hi]
            reason = None
            if not seg:
                reason = "no prints in segment"
            else:
                for a, b in zip(seg, seg[1:], strict=False):
                    gap = (b - a).total_seconds() / self.step_s
                    if gap > self.stale_bound:
                        reason = f"print gap {gap:.0f} bars exceeds stale bound"
                        break
                if (
                    reason is None
                    and (hi - seg[-1]).total_seconds() / self.step_s > self.stale_bound
                ):
                    reason = "prints die before segment end"
            if reason is None:
                ok.add(sid)
            else:
                reasons[sid] = reason
        self._elig[key] = ok
        self.exclusions[key] = reasons
        return ok

    def weights(self, key: str, builder) -> pl.DataFrame:
        if key not in self._wcache:
            self._wcache[key] = builder()
        return self._wcache[key]

    def run(
        self,
        w: pl.DataFrame,
        lo: datetime,
        hi: datetime,
        target_vol: float | None,
    ) -> dict:
        scaler = None
        if target_vol is not None:
            scaler = CompositeScaler(
                [
                    VolTargetScaler(
                        target_ann_vol=target_vol,
                        window=self.vol_window,
                        periods_per_year=self.ppy,
                        max_scale=3.0,
                    ),
                    DrawdownGovernor(dd_soft=0.03, dd_hard=0.05, floor=0.25),
                ]
            )
        elig = sorted(self.eligible(lo, hi))
        b = _slice(self.bars, lo, hi).filter(pl.col("security_id").is_in(elig))
        f = self._funding_on_bars(_slice(self.funding, lo, hi))
        wseg = w.filter(
            (pl.col("event_time") >= lo)
            & (pl.col("event_time") < hi)
            & pl.col("security_id").is_in(elig)
        )
        if b.height == 0 or wseg.height == 0:
            return {"skipped": True, "n": 0, "sharpe": None}
        self.n_evals += 1
        res = run_perp_backtest(b, f, wseg, self.cfg, initial_nav=self.nav, scaler=scaler)
        m = _metrics(res.metrics)
        m["funding_events_dropped"] = res.metrics.get("funding_events_dropped")
        return m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/binance_carry"))
    p.add_argument("--initial-nav", type=float, default=1_000_000.0)
    p.add_argument("--config", default="configs/research.yaml")
    p.add_argument("--target-vols", type=float, nargs="*", default=None)
    p.add_argument("--freq", choices=["1d", "1h"], default="1d")
    p.add_argument("--out", type=Path, default=Path(".dsh-24x7/evidence-megaplan.json"))
    p.add_argument("--limit-symbols", type=int, default=None)
    args = p.parse_args()
    if args.target_vols is None:
        args.target_vols = [0.15, 0.25, 0.40] if args.freq == "1d" else [0.10, 0.20, 0.30]

    bars = pl.read_parquet(args.data_dir / "perp_bars.parquet")
    funding = pl.read_parquet(args.data_dir / "funding.parquet")
    if args.limit_symbols:
        keep = sorted(bars["security_id"].unique().to_list())[: args.limit_symbols]
        bars = bars.filter(pl.col("security_id").is_in(keep))
        funding = funding.filter(pl.col("security_id").is_in(keep))

    t0, t1 = bars["event_time"].min(), bars["event_time"].max()
    dev_times = sorted(bars.filter(pl.col("event_time") < SPLIT)["event_time"].unique().to_list())
    mid = dev_times[len(dev_times) // 2]

    cfg = load_config(args.config)
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_gross = 6.0
    cfg.risk_gate.max_net = 6.0
    cfg.risk_gate.max_order_notional = 1e12

    is_1h = args.freq == "1h"
    ppy = 8766.0 if is_1h else PPY
    vol_w = 168 if is_1h else 60
    ev = Evaluator(bars, funding, cfg, args.initial_nav, vol_w, ppy)

    def score_weights(wkey: str, w: pl.DataFrame, *, mdd_gate=False) -> tuple[float, dict]:
        best = (float("-inf"), {})
        h1 = ev.run(w, t0, mid, None)
        h2 = ev.run(w, mid, SPLIT, None)
        s = _score(h1, h2, mdd_gate=mdd_gate)
        if s > best[0]:
            best = (s, {"tv": None, "h1": h1, "h2": h2})
        return best

    receipt: dict = {
        "schema": "megaplan_eval.v1",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "data_dir": str(args.data_dir),
        "split": str(SPLIT),
        "dev_mid": str(mid),
        "bar_span": [str(t0), str(t1)],
        "n_symbols": int(bars["security_id"].n_unique()),
        "n_bars": bars.height,
        "n_funding": funding.height,
        "live_pnl_claim": False,
        "freq": args.freq,
        "protocol": "dev-half grid -> freeze -> one locked holdout eval; failure is a result",
        "stageA": {},
        "stageB": {},
        "log": [],
    }

    t_start = time.time()

    # ---- Stage A: per-sleeve param grid --------------------------------
    candidates: dict[str, tuple[str, object]] = {}
    if is_1h:
        carry_lb, carry_vw = [3, 9], [48, 168]
        fade_lb, fade_z = [30], [2.0, 3.0]
        mom_lb, mom_sk = [72, 168, 336], [4, 24]
        sweep_lb, sweep_h = [20, 48], [8, 24]
        trend_fa, trend_sl = [168, 336], [720, 1440]
        svw = 48
    else:
        carry_lb, carry_vw = [3, 9], [20, 60]
        fade_lb, fade_z = [10, 30], [2.0, 3.0]
        mom_lb, mom_sk = [20, 60, 120], [0, 5]
        sweep_lb, sweep_h = [10, 20], [3, 8]
        trend_fa, trend_sl = [20, 60], [120, 240]
        svw = 20

    sleeve_grids: dict[str, list[tuple[str, object]]] = {
        "carry": [
            (
                f"carry_lb{lb}_vw{vw}",
                (
                    lambda lb=lb, vw=vw: funding_carry_weights(
                        bars, funding, lookback_events=lb, vol_window=vw, gross_scale=2.0
                    )
                ),
            )
            for lb, vw in itertools.product(carry_lb, carry_vw)
        ],
        "spike_fade": [
            (
                f"fade_l{lb}_z{z}",
                (
                    lambda lb=lb, z=z: funding_spike_fade_weights(
                        bars,
                        funding,
                        lookback_events=lb,
                        z_threshold=z,
                        vol_window=svw,
                        gross_scale=1.0,
                    )
                ),
            )
            for lb, z in itertools.product(fade_lb, fade_z)
        ],
        "momentum": [
            (
                f"mom_l{lb}_s{sk}",
                (
                    lambda lb=lb, sk=sk: cross_sectional_momentum_weights(
                        bars, lookback_bars=lb, skip_bars=sk, vol_window=svw, gross_scale=1.5
                    )
                ),
            )
            for lb, sk in itertools.product(mom_lb, mom_sk)
        ],
        "sweep": [
            (
                f"sweep_l{lb}_h{hb}",
                (
                    lambda lb=lb, hb=hb: sweep_reclaim_weights(
                        bars, lookback=lb, hold_bars=hb, gross_scale=0.5
                    )
                ),
            )
            for lb, hb in itertools.product(sweep_lb, sweep_h)
        ],
        "trend": [
            (
                f"trend_{fa}_{sl}",
                (
                    lambda fa=fa, sl=sl: slow_trend_weights(
                        bars, fast_bars=fa, slow_bars=sl, vol_window=svw, gross_scale=1.0
                    )
                ),
            )
            for fa, sl in itertools.product(trend_fa, trend_sl)
        ],
    }

    for fam, grid in sleeve_grids.items():
        best = (float("-inf"), None, None)
        for key, builder in grid:
            w = ev.weights(key, builder)
            s, info = score_weights(key, w)
            receipt["stageA"][key] = {"score": s, "h1": info.get("h1"), "h2": info.get("h2")}
            if s > best[0]:
                best = (s, key, w)
        if best[1] is not None:
            candidates[fam] = (best[1], best[2])
        receipt["log"].append(f"stageA {fam}: best={best[1]} score={best[0]:.3f}")
        print(f"stageA {fam}: best={best[1]} score={best[0]:.3f} ({ev.n_evals} evals)", flush=True)

    # ---- Stage B: blend grid over survivors -----------------------------
    fams = list(candidates)
    blend_cfgs: list[tuple[str, dict[str, float]]] = [(f"only_{f}", {f: 1.0}) for f in fams]
    if len(fams) >= 2:
        blend_cfgs.append(("equal", {f: 1.0 / len(fams) for f in fams}))
        for f in fams:
            rest = 0.5 / (len(fams) - 1)
            blend_cfgs.append((f"heavy_{f}", {f: 0.5, **{g: rest for g in fams if g != f}}))
    for bname, mix in blend_cfgs:
        w = blend_weights({f: candidates[f][1] for f in mix}, mix)
        s, info = score_weights(bname, w)
        receipt["stageB"][bname] = {
            "score": s,
            "mix": mix,
            "h1": info.get("h1"),
            "h2": info.get("h2"),
        }
    bestB = max(receipt["stageB"].items(), key=lambda kv: kv[1]["score"])
    receipt["log"].append(f"stageB best={bestB[0]} score={bestB[1]['score']:.3f}")
    print(f"stageB best={bestB[0]} score={bestB[1]['score']:.3f} ({ev.n_evals} evals)", flush=True)

    # ---- Stage C: overlay target-vol on the stage-B winner -------------
    champ_w = blend_weights({f: candidates[f][1] for f in bestB[1]["mix"]}, bestB[1]["mix"])
    stageC = {}
    for tv in args.target_vols:
        h1 = ev.run(champ_w, t0, mid, tv)
        h2 = ev.run(champ_w, mid, SPLIT, tv)
        stageC[tv] = {"score": _score(h1, h2, mdd_gate=True), "h1": h1, "h2": h2}
        print(
            f"  tv={tv}: score={stageC[tv]['score']:.3f} h1={h1.get('sharpe')} h2={h2.get('sharpe')}",
            flush=True,
        )
    # include no-overlay as a candidate
    h1 = ev.run(champ_w, t0, mid, None)
    h2 = ev.run(champ_w, mid, SPLIT, None)
    stageC["none"] = {"score": _score(h1, h2, mdd_gate=True), "h1": h1, "h2": h2}
    receipt["stageC"] = {str(k): v for k, v in stageC.items()}
    bestC = max(stageC.items(), key=lambda kv: kv[1]["score"])
    qualified = bestC[1]["score"] > float("-inf")
    champ_tv = None if bestC[0] == "none" else float(bestC[0])
    receipt["frozen"] = {
        "blend": bestB[0],
        "mix": bestB[1]["mix"],
        "target_vol": champ_tv,
        "qualified_on_dev": qualified,
    }
    print(f"frozen: {receipt['frozen']}", flush=True)

    # ---- Locked eval: one shot on dev + holdout -------------------------
    # holdout runs through t1 + one bar step so the dataset's last bar is
    # inside the (lo, hi) window rather than truncated off.
    final = {}
    hold_end = t1 + timedelta(seconds=ev.step_s)
    for label, lo, hi in (("dev", t0, SPLIT), ("holdout", SPLIT, hold_end)):
        final[label] = ev.run(champ_w, lo, hi, champ_tv)
        print(f"{label}: {final[label]}", flush=True)
    receipt["final"] = final

    gate = (
        _sharpe(final["holdout"]) > 5.0
        and abs(float(final["holdout"].get("max_drawdown") or 1.0)) < 0.05
    )
    h_mdd = final["holdout"].get("max_drawdown")
    mdd_ok = h_mdd is not None and abs(float(h_mdd)) < 0.05
    gate = _sharpe(final["holdout"]) > 5.0 and mdd_ok
    receipt["gate"] = {
        "sharpe_gt_5": _sharpe(final["holdout"]) > 5.0,
        "mdd_lt_5pct": mdd_ok,
        "qualified_on_dev": qualified,
        "PROVEN": gate and qualified,
    }
    receipt["input_hashes"] = {f.name: _sha256(f) for f in sorted(args.data_dir.glob("*.parquet"))}
    receipt["segment_exclusions"] = {
        f"{lo}->{hi}": {"n_excluded": len(v), "n_eligible": len(ev._elig[k])}
        for k, v in ev.exclusions.items()
        for lo, hi in [k]
    }
    receipt["n_evals"] = ev.n_evals
    receipt["wall_seconds"] = round(time.time() - t_start, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"receipt -> {args.out} | evals={ev.n_evals} | gate={'PROVEN' if gate else 'NOT PROVEN'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
