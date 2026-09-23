"""Evaluate the delta-neutral carry book (short perp + long spot) on real data.

Pairs ``{sym}_{interval}.perp.parquet`` with ``{sym}_{interval}.spot.parquet``
plus funding, builds proportional-to-rate carry weights, runs
``run_carry_backtest`` on dev + locked holdout, writes a JSON receipt.

Symbols without a spot file cannot form pairs — they are excluded by
construction (honest: no cross-venue stale hedges, no perp-only directional
substitute).

Research diagnostic only — ``live_pnl_claim`` is always false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from typing import cast  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from quant_fund.backtest.carry_engine import run_carry_backtest  # noqa: E402
from quant_fund.backtest.overlay import (  # noqa: E402
    CompositeScaler,
    DrawdownGovernor,
    VolTargetScaler,
)
from quant_fund.backtest.sleeves import (  # noqa: E402
    basis_carry_hysteresis_weights,
    basis_carry_weights,
)
from quant_fund.config.loader import load_config  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load(sources: Path, pattern: str) -> dict[str, pl.DataFrame]:
    return {p.name.split("_")[0].upper(): pl.read_parquet(p) for p in sorted(sources.glob(pattern))}


def _split(bars: pl.DataFrame, holdout_frac: float) -> tuple[pl.DataFrame, pl.DataFrame, object]:
    times = sorted(bars["event_time"].unique().to_list())
    cut = times[int(len(times) * (1.0 - holdout_frac))]
    return (
        bars.filter(pl.col("event_time") < cut),
        bars.filter(pl.col("event_time") >= cut),
        cut,
    )


def _segment_eligible(
    joint_times: list[datetime],
    seg0: datetime,
    seg1: datetime,
    step_s: float,
    bound_bars: int,
) -> str | None:
    """Return an exclusion reason, or None if the pair is markable through the
    whole segment: no internal joint-print spacing > bound_bars+1 (the engine
    raises StaleValuationError once a held pair's marks are stale for
    bound_bars+1 consecutive bars) and no dead tail reaching segment end
    (perp delisted/halted mid-hold). A pair that starts printing late is fine —
    it simply cannot be held before its first mark."""
    seg = [t for t in joint_times if seg0 <= t < seg1]
    if not seg:
        return "no joint prints in segment"
    max_spacing = bound_bars + 1
    for i in range(len(seg) - 1):
        gap = (seg[i + 1] - seg[i]).total_seconds() / step_s
        if gap > max_spacing:
            return f"joint-print gap {gap:.0f} bars exceeds stale bound"
    tail = (seg1 - seg[-1]).total_seconds() / step_s
    if tail > max_spacing:
        return f"joint prints die {tail:.0f} bars before segment end"
    return None


def _metrics(m: dict) -> dict:
    keys = (
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "n",
        "periods_per_year",
        "mean_turnover",
        "funding_received_total",
        "funding_paid_total",
        "funding_net",
        "liquidation_count",
        "margin_rejects",
        "ruined",
        "flag_high_sharpe",
        "risk_gate_rejects",
    )
    out = {k: m.get(k) for k in keys if k in m}
    pa = m.get("pnl_attribution")
    if pa:
        out["pnl_attribution_totals"] = pa["totals"]
        out["pnl_attribution_conservation_error"] = pa["conservation_error"]
        out["pnl_attribution_by_symbol"] = pa["by_symbol"]
        out["liquidation_events"] = pa["liquidation_events"]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--interval", default="1d")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--initial-nav", type=float, default=1_000_000.0)
    p.add_argument("--config", default="configs/research.yaml")
    p.add_argument("--lookback-events", type=int, default=3)
    p.add_argument("--max-name", type=float, default=0.15)
    p.add_argument("--gross-scale", type=float, default=0.9)
    p.add_argument("--min-rate", type=float, default=0.0)
    p.add_argument(
        "--mode",
        choices=["proportional", "hysteresis"],
        default="hysteresis",
        help="proportional = daily rebalanced rate-proportional book; "
        "hysteresis = event-driven membership book (low churn)",
    )
    p.add_argument("--enter-rate", type=float, default=0.0003)
    p.add_argument("--exit-rate", type=float, default=0.0)
    p.add_argument("--name-weight", type=float, default=0.08)
    p.add_argument("--max-names", type=int, default=10)
    p.add_argument(
        "--rebalance-band",
        type=float,
        default=1.5,
        help="hysteresis mode: re-emit name_weight when a held pair's drifted "
        "weight leaves [name_weight/band, name_weight*band]; bounds notional "
        "drift so fixed units cannot grow past the margin cliff (0=disable)",
    )
    p.add_argument(
        "--stale-bars",
        type=int,
        default=8,
        help="risk_gate.stale_price_bars override: held pairs carry last marks "
        "through bounded feed gaps (hold-and-report); beyond it the engine "
        "still raises StaleValuationError",
    )
    p.add_argument("--target-vol", type=float, default=0.20)
    p.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    sources = args.data_root / "raw" / "sources"
    perp = _load(sources, f"*_{args.interval}.perp.parquet")
    spot = _load(sources, f"*_{args.interval}.spot.parquet")
    candidates = sorted(set(perp) & set(spot))
    # Coverage gate: a pair is only tradeable while BOTH venues print bars.
    # Spot-delisted symbols (e.g. XMR, delisted Feb-2024) end mid-history —
    # holding them past spot end would make the hedge untradeable, so they
    # are excluded entirely and reported, not silently carried on stale marks.
    paired, coverage_excluded = [], []
    for s in candidates:
        smin = cast(datetime, spot[s]["event_time"].min())
        smax = cast(datetime, spot[s]["event_time"].max())
        pmax = cast(datetime, perp[s]["event_time"].max())
        p_in_span = perp[s].filter(pl.col("event_time") >= smin).height
        n_joint = (
            spot[s]
            .select("event_time")
            .join(
                perp[s].filter(pl.col("event_time") >= smin).select("event_time"),
                on="event_time",
                how="inner",
            )
            .height
        )
        cov = n_joint / max(p_in_span, 1)
        if smax >= pmax and cov >= 0.98:
            paired.append(s)
        else:
            coverage_excluded.append(
                {
                    "symbol": s,
                    "spot_max": str(smax),
                    "perp_max": str(pmax),
                    "coverage": round(cov, 4),
                }
            )
    perp_only = sorted(set(perp) - set(spot))
    if not paired:
        raise SystemExit(f"no symbols with both perp+spot {args.interval} files")
    funding = pl.concat(
        [pl.read_parquet(p) for p in sorted(sources.glob("*.funding.parquet"))]
    ).filter(pl.col("security_id").is_in(paired))

    perp_bars = pl.concat([perp[s] for s in paired])
    spot_bars = pl.concat([spot[s] for s in paired])

    cfg = load_config(args.config)
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_gross = 4.0
    cfg.risk_gate.max_net = 4.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.stale_price_bars = args.stale_bars

    dev_p, hold_p, cut = _split(perp_bars, args.holdout_frac)
    dev_s, hold_s, _ = _split(spot_bars, args.holdout_frac)
    dev_f = funding.filter(pl.col("event_time") < pl.lit(cut))
    hold_f = funding.filter(pl.col("event_time") >= pl.lit(cut))

    # Per-segment eligibility: a pair must be markable for the WHOLE segment —
    # no internal joint-print gap beyond the stale bound and no dead tail
    # (e.g. MARSCOIN perp delisted mid-holdout). Held through a larger gap the
    # engine would fail closed, so ineligible names are excluded per segment
    # and reported, never silently carried on stale marks.
    all_times = sorted(perp_bars["event_time"].unique().to_list())
    step_s = float(
        np.median(
            np.diff(np.array([t.timestamp() for t in all_times], dtype=float))
        )
    )
    seg_bounds = {
        "dev": (all_times[0], cast(datetime, cut)),
        "holdout": (cast(datetime, cut), all_times[-1]),
    }
    seg_excluded: dict[str, list[dict]] = {"dev": [], "holdout": []}
    seg_eligible: dict[str, set[str]] = {"dev": set(), "holdout": set()}
    for s in paired:
        joint = sorted(
            spot[s]
            .select("event_time")
            .join(perp[s].select("event_time"), on="event_time", how="inner")["event_time"]
            .to_list()
        )
        for label, (t0, t1) in seg_bounds.items():
            reason = _segment_eligible(joint, t0, t1, step_s, args.stale_bars)
            if reason is None:
                seg_eligible[label].add(s)
            else:
                seg_excluded[label].append({"symbol": s, "reason": reason})

    def fresh_scaler() -> CompositeScaler | None:
        if not args.overlay:
            return None
        return CompositeScaler(
            [
                VolTargetScaler(target_ann_vol=args.target_vol, window=168),
                DrawdownGovernor(dd_soft=0.05, dd_hard=0.10, floor=0.25),
            ]
        )

    receipt: dict = {
        "schema": "carry_eval.v1",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "interval": args.interval,
        "book": "delta_neutral_carry (short perp + long spot, funding harvest)",
        "holdout_cut": str(cut),
        "n_paired_symbols": len(paired),
        "n_perp_only_excluded": len(perp_only),
        "n_coverage_excluded": len(coverage_excluded),
        "coverage_excluded": coverage_excluded,
        "perp_only_symbols": perp_only,
        "params": {
            "mode": args.mode,
            "lookback_events": args.lookback_events,
            "max_name": args.max_name,
            "gross_scale": args.gross_scale,
            "min_rate": args.min_rate,
            "enter_rate": args.enter_rate,
            "exit_rate": args.exit_rate,
            "name_weight": args.name_weight,
            "max_names": args.max_names,
            "rebalance_band": args.rebalance_band if args.mode == "hysteresis" else None,
            "stale_price_bars": args.stale_bars,
            "overlay": args.overlay,
            "target_vol": args.target_vol if args.overlay else None,
        },
        "segment_excluded": seg_excluded,
        "bar_span": [str(perp_bars["event_time"].min()), str(perp_bars["event_time"].max())],
        "live_pnl_claim": False,
        "survivorship_note": (
            "universe = currently-listed perps with a spot pair; delisted "
            "symbols absent — returns are upward-biased vs a PIT universe"
        ),
        "segments": {},
    }

    for label, pb, sb, f in (
        ("dev", dev_p, dev_s, dev_f),
        ("holdout", hold_p, hold_s, hold_f),
    ):
        elig = seg_eligible[label]
        pb = pb.filter(pl.col("security_id").is_in(elig))
        sb = sb.filter(pl.col("security_id").is_in(elig))
        f = f.filter(pl.col("security_id").is_in(elig))
        fctx = funding.filter(pl.col("security_id").is_in(elig))
        # Weights are built per segment over the segment's bar grid but with
        # the FULL funding history — rate_ma context is proper, and
        # membership books re-enter qualifying names at the segment start.
        if args.mode == "hysteresis":
            wseg = basis_carry_hysteresis_weights(
                pb,
                fctx,
                enter_rate=args.enter_rate,
                exit_rate=args.exit_rate,
                lookback_events=args.lookback_events,
                name_weight=args.name_weight,
                max_names=args.max_names,
                rebalance_band=args.rebalance_band or None,
            )
        else:
            wseg = basis_carry_weights(
                pb,
                fctx,
                lookback_events=args.lookback_events,
                max_name=args.max_name,
                gross_scale=args.gross_scale,
                min_rate=args.min_rate,
            )
        res = run_carry_backtest(
            pb, sb, f, wseg, cfg, initial_nav=args.initial_nav, scaler=fresh_scaler()
        )
        receipt["segments"][label] = _metrics(res.metrics)
        receipt["segments"][label]["n_eligible_symbols"] = len(elig)
        print(f"[{label}] done")

    receipt["input_hashes"] = {p.name: _sha256(p) for p in sorted(sources.glob("*.parquet"))[:400]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, default=str))
    print(f"receipt -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
