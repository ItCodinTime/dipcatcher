"""Multi-asset daily book on Yahoo bars — honest Sharpe-maximization engine.

Sleeves (all causal; decision at close t -> fill at next bar open):
  tsmom : per-asset time-series momentum, mean sign over {21,63,126,252}d,
          inverse-vol sized.
  rev   : short-term reversal on equity classes; z = -r5/(vol60*sqrt5),
          gated |z|>gate, inverse-vol sized.
  volrp : short-vol premium via SVXY — long only when VIX < gate AND
          SVXY > its 126d trend; small inverse-vol weight (tail risk).

Book: summed sleeves -> gross cap -> in-engine causal CompositeScaler
(VolTargetScaler on realized NAV + DrawdownGovernor). Shorts allowed on
liquid ETFs; borrow cost via config costs.borrow_bps_per_year.

Honesty contract: dev/holdout split by union timeline; tune on dev only;
frozen config -> single holdout run. All metrics incl. flag_high_sharpe.
Research-only; live_pnl_claim=false always.
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

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from quant_fund.backtest.overlay import (  # noqa: E402
    CompositeScaler,
    DrawdownGovernor,
    VolTargetScaler,
)
from quant_fund.backtest.perp_engine import run_perp_backtest  # noqa: E402
from quant_fund.config.loader import load_config  # noqa: E402

CLASSES: dict[str, str] = {
    "SPY": "eq", "QQQ": "eq", "DIA": "eq", "IWM": "eq", "EFA": "eq", "EEM": "eq",
    "VEA": "eq", "MTUM": "eq", "USMV": "eq", "QUAL": "eq", "VLUE": "eq",
    "XLF": "eqsec", "XLE": "eqsec", "XLK": "eqsec", "XLV": "eqsec",
    "XLP": "eqsec", "XLU": "eqsec", "XLI": "eqsec", "XLB": "eqsec",
    "XLY": "eqsec", "XLC": "eqsec", "XLRE": "eqsec",
    "VNQ": "reit", "IYR": "reit",
    "TLT": "bond", "IEF": "bond", "IEI": "bond", "SHY": "bond", "BIL": "bond",
    "LQD": "bond", "HYG": "bond", "TIP": "bond", "EMB": "bond", "MUB": "bond",
    "AGG": "bond", "VCIT": "bond", "BND": "bond", "TLH": "bond",
    "GLD": "cmd", "SLV": "cmd", "USO": "cmd", "UNG": "cmd", "DBC": "cmd",
    "PPLT": "cmd", "CPER": "cmd", "DBB": "cmd",
    "UUP": "fx", "FXE": "fx", "FXY": "fx", "FXB": "fx", "FXF": "fx", "FXA": "fx",
    "SVXY": "vol", "VXX": "vol",
    "IAU": "cmd", "IVV": "eq", "VOO": "eq", "SCHF": "eq",
    "BIV": "bond", "BSV": "bond", "SGOV": "cash", "SHV": "cash",
}
FEATURE_SYMS = {"VIX": "vix", "TNX": "tnx", "IRX": "irx", "FVX": "fvx",
                "GSPC": "gspc", "IXIC": "ixic", "DJI": "dji", "RUT": "rut"}
REV_CLASSES = {"eq", "eqsec", "reit"}
TSMOM_LOOKBACKS = (21, 63, 126, 252)
# Near-duplicate ETF pairs — tight cointegration, mean-reversion legs.
PAIRS = [("TLT", "TLH"), ("AGG", "BND"), ("EFA", "VEA"), ("GLD", "IAU"),
         ("SPY", "IVV"), ("SPY", "VOO")]
XSEC_CLASSES = {"eqsec": 3, "bond": 3, "cmd": 2, "fx": 2}
CASH_LEGS = ("SGOV", "BIL", "SHV")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load_universe(src: Path, stocks_src: Path | None = None) -> tuple[pl.DataFrame, dict[str, pl.Series]]:
    """Load traded symbols -> normalized-daily bars; feature series -> date->value.

    Files under ``stocks_src`` get class "stk" (single-name equity L/S pool).
    """
    bars_parts: list[pl.DataFrame] = []
    features: dict[str, pl.Series] = {}
    files = sorted(src.glob("*_1d.parquet"))
    if stocks_src is not None and stocks_src.exists():
        files += sorted(stocks_src.glob("*_1d.parquet"))
    for p in files:
        sym = p.stem.replace("_1d", "").upper()
        if stocks_src is not None and p.parent == stocks_src and sym not in CLASSES:
            CLASSES[sym] = "stk"
        df = pl.read_parquet(p).with_columns(
            pl.col("event_time").dt.truncate("1d").alias("event_time")
        )
        if sym in FEATURE_SYMS:
            features[FEATURE_SYMS[sym]] = df.select("event_time", "close")
            continue
        if sym not in CLASSES:
            continue
        if df.height < 260:
            continue  # insufficient history for any signal — exclude cleanly
        df = df.with_columns(
            pl.col("close").alias("close_total_return"),
            pl.lit("yahoo").alias("source"),
        )
        bars_parts.append(df.select(
            "security_id", "event_time", "open", "high", "low", "close",
            "close_total_return", "volume", "source",
        ))
    if not bars_parts:
        raise SystemExit(f"no traded symbol bars under {src}")
    return pl.concat(bars_parts), features


def _close_matrix(bars: pl.DataFrame) -> tuple[list[datetime], list[str], np.ndarray]:
    piv = bars.pivot(on="security_id", index="event_time", values="close",
                     aggregate_function="last").sort("event_time")
    times = piv["event_time"].to_list()
    sids = [c for c in piv.columns if c != "event_time"]
    mat = piv.select(sids).to_numpy()
    return times, sids, mat


def build_weights(
    times: list[datetime],
    sids: list[str],
    close: np.ndarray,
    feats_by_date: dict[str, dict[datetime, float]],
    args: argparse.Namespace,
) -> dict[datetime, dict[str, float]]:
    """Per-date pre-governor target weights (fraction of NAV), dense panel."""
    n, m = close.shape
    logc = np.where(close > 0, np.log(np.where(close > 0, close, 1.0)), np.nan)
    ret1 = np.full_like(logc, np.nan)
    ret1[1:] = logc[1:] - logc[:-1]
    win = np.lib.stride_tricks.sliding_window_view(ret1, 60, axis=0)  # (n-59, m, 60)
    with np.errstate(invalid="ignore"):
        vol60 = np.full_like(close, np.nan)
        vol60[60:] = np.nanstd(win[:-1], axis=2, ddof=1)  # row i <- ret1[i-60:i]
        vol60_d = vol60 * np.sqrt(252.0)

    sleeves = set(args.sleeves.split(","))
    out: dict[datetime, dict[str, float]] = {}
    svxy_j = sids.index("SVXY") if "SVXY" in sids else -1
    vix_map = feats_by_date.get("vix", {})
    pair_idx = [
        (sids.index(a), sids.index(b), a, b)
        for a, b in PAIRS
        if a in sids and b in sids
    ]
    xsec_groups: dict[str, list[int]] = {}
    for j, sid in enumerate(sids):
        cls = CLASSES.get(sid)
        if cls in XSEC_CLASSES:
            xsec_groups.setdefault(cls, []).append(j)
    for i in range(1, n):
        dt = times[i]
        w: dict[str, float] = {}
        for j, sid in enumerate(sids):
            if CLASSES.get(sid) == "cash":
                continue
            v = vol60_d[i, j]
            if not np.isfinite(v) or v <= 5e-3:
                continue
            acc = 0.0
            if "tsmom" in sleeves:
                signs = []
                for L in TSMOM_LOOKBACKS:
                    if i - L >= 0 and np.isfinite(logc[i, j]) and np.isfinite(logc[i - L, j]):
                        signs.append(np.sign(logc[i, j] - logc[i - L, j]))
                if signs:
                    acc += float(np.mean(signs)) * args.tsmom_risk / v
            if "rev" in sleeves and CLASSES.get(sid) in REV_CLASSES and i >= 5:
                r5 = logc[i, j] - logc[i - 5, j] if np.isfinite(logc[i, j]) and np.isfinite(
                    logc[i - 5, j]) else np.nan
                if np.isfinite(r5) and v > 0:
                    z = float(np.clip(-r5 / (v / np.sqrt(252.0) * np.sqrt(5.0)), -1.5, 1.5))
                    if abs(z) > args.rev_gate:
                        acc += z * args.rev_risk / v
            if acc != 0.0:
                w[sid] = acc
        if "pairs" in sleeves and i >= 63:
            for ja, jb, a, b in pair_idx:
                spread = logc[:, ja] - logc[:, jb]
                win = spread[i - 60 : i]
                if not np.isfinite(win).all():
                    continue
                mu, sd = float(np.mean(win)), float(np.std(win, ddof=1))
                if sd <= 1e-6 or not np.isfinite(spread[i]):
                    continue
                z = float(np.clip((spread[i] - mu) / sd, -3.0, 3.0))
                if abs(z) >= args.pair_gate:
                    leg = z * args.pair_risk / (sd * np.sqrt(252.0))
                    w[a] = w.get(a, 0.0) - leg
                    w[b] = w.get(b, 0.0) + leg
        if "xsec" in sleeves and i >= 63:
            for cls, k in XSEC_CLASSES.items():
                idxs = xsec_groups.get(cls, [])
                mom = np.array(
                    [logc[i, j] - logc[i - 63, j]
                     if np.isfinite(logc[i, j]) and np.isfinite(logc[i - 63, j]) else np.nan
                     for j in idxs]
                )
                finite = np.isfinite(mom)
                if finite.sum() < k + 1:
                    continue
                order = np.argsort(np.where(finite, mom, -np.inf))
                for j_idx in order[-k:]:
                    j = idxs[j_idx]
                    v = vol60_d[i, j]
                    if np.isfinite(v) and v > 5e-3:
                        w[sids[j]] = w.get(sids[j], 0.0) + args.xsec_risk / v
                for j_idx in order[:k]:
                    j = idxs[j_idx]
                    v = vol60_d[i, j]
                    if np.isfinite(v) and v > 5e-3:
                        w[sids[j]] = w.get(sids[j], 0.0) - args.xsec_risk * 0.5 / v
        if "stkrev" in sleeves and i >= 66:
            idxs = [j for j, s in enumerate(sids) if CLASSES.get(s) == "stk"]
            r5 = np.array(
                [
                    logc[i, j] - logc[i - 5, j]
                    if np.isfinite(logc[i, j]) and np.isfinite(logc[i - 5, j])
                    else np.nan
                    for j in idxs
                ]
            )
            fin = np.isfinite(r5)
            if fin.sum() >= 20:
                med = float(np.nanmedian(r5))
                mad = float(np.nanmedian(np.abs(r5 - med))) * 1.4826
                if mad > 1e-6:
                    z = np.clip((r5 - med) / mad, -3.0, 3.0)
                    for k_j, j in enumerate(idxs):
                        if not fin[k_j] or abs(z[k_j]) < args.stk_gate:
                            continue
                        v = vol60_d[i, j]
                        if not np.isfinite(v) or v <= 5e-3:
                            continue
                        w[sids[j]] = w.get(sids[j], 0.0) - z[k_j] * args.stk_risk / v
        if "stkmom" in sleeves and i >= 252:
            idxs = [j for j, s in enumerate(sids) if CLASSES.get(s) == "stk"]
            mom = np.array(
                [
                    logc[i - 21, j] - logc[i - 252, j]
                    if np.isfinite(logc[i - 21, j]) and np.isfinite(logc[i - 252, j])
                    else np.nan
                    for j in idxs
                ]
            )
            fin = np.isfinite(mom)
            if fin.sum() >= 20:
                med = float(np.nanmedian(mom))
                mad = float(np.nanmedian(np.abs(mom - med))) * 1.4826
                if mad > 1e-6:
                    z = np.clip((mom - med) / mad, -3.0, 3.0)
                    for k_j, j in enumerate(idxs):
                        if not fin[k_j] or abs(z[k_j]) < 0.5:
                            continue
                        v = vol60_d[i, j]
                        if not np.isfinite(v) or v <= 5e-3:
                            continue
                        w[sids[j]] = w.get(sids[j], 0.0) + z[k_j] * args.stkmom_risk / v
        if "volrp" in sleeves and svxy_j >= 0 and i >= 126:
            vix = vix_map.get(dt)
            sv = close[:, svxy_j]
            sma = np.nanmean(sv[i - 126 : i]) if np.isfinite(sv[i - 126 : i]).any() else np.nan
            v = vol60_d[i, svxy_j]
            mode = args.vrp_mode
            spike_ok = False
            if mode in ("spike", "both") and vix is not None:
                recent = [
                    vix_map.get(times[k]) for k in range(max(0, i - 63), i)
                ]
                spike_ok = (
                    any(x is not None and x > args.vix_spike for x in recent)
                    and vix < args.vix_gate
                )
            trend_ok = (
                mode in ("trend", "both")
                and vix is not None
                and vix < args.vix_gate
                and np.isfinite(sma)
                and close[i, svxy_j] > sma
            )
            if (trend_ok or spike_ok) and np.isfinite(v) and v > 1e-4:
                w["SVXY"] = w.get("SVXY", 0.0) + args.vrp_risk / v
        gross = sum(abs(x) for x in w.values())
        if gross > args.gross_cap and gross > 0:
            f = args.gross_cap / gross
            w = {k: x * f for k, x in w.items()}
        if "cash" in sleeves:
            leftover = max(0.0, 1.0 - sum(abs(x) for x in w.values()))
            if leftover > 1e-6:
                for leg in CASH_LEGS:
                    if leg in sids:
                        w[leg] = w.get(leg, 0.0) + leftover
                        break
        out[dt] = w
    return out


def _weights_frame(
    wmap: dict[datetime, dict[str, float]], times: list[datetime], sids: list[str]
) -> pl.DataFrame:
    """Dense panel: explicit 0.0 for every traded sid on every segment date.

    run_perp_backtest carries absent (dt, sid) targets forward — zeros are
    the only way to flatten a stale position or let the governor de-risk on
    a no-signal day.
    """
    rows = []
    for dt in times:
        per = wmap.get(dt, {})
        for sid in sids:
            rows.append((dt, sid, per.get(sid, 0.0)))
    return pl.DataFrame(
        rows, schema=["event_time", "security_id", "target_weight"], orient="row"
    )


def _metrics(result_metrics: dict) -> dict:
    keys = (
        "total_return", "cagr", "sharpe", "max_drawdown", "n",
        "periods_per_year", "mean_turnover", "liquidation_count",
        "margin_rejects", "ruined", "flag_high_sharpe", "risk_gate_rejects",
    )
    return {k: result_metrics.get(k) for k in keys if k in result_metrics}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--config", default="configs/research.yaml")
    ap.add_argument("--holdout-frac", type=float, default=0.22)
    ap.add_argument("--holdout", action="store_true", help="evaluate locked tail")
    ap.add_argument("--sleeves", default="tsmom,rev,volrp")
    ap.add_argument("--tsmom-risk", type=float, default=0.10)
    ap.add_argument("--rev-risk", type=float, default=0.04)
    ap.add_argument("--rev-gate", type=float, default=0.5)
    ap.add_argument("--vrp-risk", type=float, default=0.10)
    ap.add_argument("--vix-gate", type=float, default=22.0)
    ap.add_argument("--vix-spike", type=float, default=32.0)
    ap.add_argument("--vrp-mode", choices=["trend", "spike", "both"], default="both")
    ap.add_argument("--pair-risk", type=float, default=0.02)
    ap.add_argument("--pair-gate", type=float, default=2.0)
    ap.add_argument("--xsec-risk", type=float, default=0.05)
    ap.add_argument("--stk-risk", type=float, default=0.015)
    ap.add_argument("--stk-gate", type=float, default=1.0)
    ap.add_argument("--stkmom-risk", type=float, default=0.02)
    ap.add_argument("--stocks-dir", type=str, default="sources_stocks")
    ap.add_argument("--gross-cap", type=float, default=2.0)
    ap.add_argument("--name-cap", type=float, default=0.35)
    ap.add_argument("--target-vol", type=float, default=0.12)
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--max-scale", type=float, default=3.0)
    ap.add_argument("--dd-soft", type=float, default=0.025)
    ap.add_argument("--dd-hard", type=float, default=0.045)
    ap.add_argument("--dd-floor", type=float, default=0.15)
    ap.add_argument("--initial-nav", type=float, default=1_000_000.0)
    ap.add_argument("--commission-bps", type=float, default=1.0)
    ap.add_argument("--half-spread-bps", type=float, default=2.0)
    ap.add_argument("--borrow-bps", type=float, default=50.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bars, feats = _load_universe(
        args.data_root / "raw" / "sources_yahoo",
        args.data_root / "raw" / args.stocks_dir if args.stocks_dir else None,
    )
    # feature series -> {date: value} maps
    feats_by_date: dict[str, dict[datetime, float]] = {}
    for name, fr in feats.items():
        feats_by_date[name] = {
            t: float(v)
            for t, v in zip(fr["event_time"].to_list(), fr["close"].to_list(), strict=True)
        }

    all_times = sorted(set(bars["event_time"].to_list()))
    cut = all_times[int(len(all_times) * (1.0 - args.holdout_frac))]
    seg0, seg1 = (all_times[0], cut) if not args.holdout else (cut, all_times[-1])
    bars_seg = bars.filter(
        (pl.col("event_time") >= seg0) & (pl.col("event_time") <= seg1)
        if args.holdout
        else pl.col("event_time") < seg1
    )
    seg_times = sorted(set(bars_seg["event_time"].to_list()))
    sids = sorted(set(bars_seg["security_id"].to_list()))
    # Signals computed on the FULL panel so holdout-start warmups see history
    # (PIT-legal: history before the segment boundary is past data).
    times_full, sids_full, close_full = _close_matrix(bars)
    wmap_full = build_weights(times_full, sids_full, close_full, feats_by_date, args)
    wmap = {dt: wmap_full.get(dt, {}) for dt in seg_times}
    weights = _weights_frame(wmap, seg_times, sids)

    cfg = load_config(args.config)
    cfg.perp.funding_enabled = False
    cfg.perp.periods_per_year_override = 252
    cfg.perp.bar_seconds_hint = 86400.0
    cfg.costs.commission_bps = args.commission_bps
    cfg.costs.half_spread_bps = args.half_spread_bps
    cfg.costs.borrow_bps_per_year = args.borrow_bps
    cfg.costs.frictionless = False
    cfg.constraints.gross_leverage = max(args.gross_cap * args.max_scale, 1.0)
    cfg.constraints.name_max = args.name_cap * args.max_scale
    cfg.constraints.net_exposure = 0.95
    cfg.risk_gate.max_gross = max(args.gross_cap * args.max_scale, 1.0)
    cfg.risk_gate.max_net = 2.0 * args.max_scale
    cfg.risk_gate.max_name = args.name_cap * args.max_scale
    cfg.risk_gate.max_predicted_vol = 10.0
    cfg.risk_gate.max_participation = 0.10

    scaler = CompositeScaler(
        [
            VolTargetScaler(
                target_ann_vol=args.target_vol,
                window=args.vol_window,
                periods_per_year=252.0,
                min_scale=0.0,
                max_scale=args.max_scale,
            ),
            DrawdownGovernor(
                dd_soft=args.dd_soft, dd_hard=args.dd_hard, floor=args.dd_floor
            ),
        ]
    )

    t0 = datetime.now(UTC)
    res = run_perp_backtest(
        bars_seg, None, weights, cfg, initial_nav=args.initial_nav, scaler=scaler
    )
    elapsed = (datetime.now(UTC) - t0).total_seconds()
    m = res.metrics
    receipt = {
        "schema": "multi_asset_book.v1",
        "live_pnl_claim": False,
        "segment": "holdout" if args.holdout else "dev",
        "seg0": str(seg0),
        "seg1": str(seg1),
        "sleeves": args.sleeves,
        "params": {
            "tsmom_risk": args.tsmom_risk, "rev_risk": args.rev_risk,
            "rev_gate": args.rev_gate, "vrp_risk": args.vrp_risk,
            "vix_gate": args.vix_gate, "gross_cap": args.gross_cap,
            "pair_risk": args.pair_risk, "pair_gate": args.pair_gate,
            "xsec_risk": args.xsec_risk,
            "name_cap": args.name_cap, "target_vol": args.target_vol,
            "vol_window": args.vol_window, "max_scale": args.max_scale,
            "dd_soft": args.dd_soft, "dd_hard": args.dd_hard,
            "dd_floor": args.dd_floor,
            "commission_bps": args.commission_bps,
            "half_spread_bps": args.half_spread_bps,
            "borrow_bps_per_year": args.borrow_bps,
        },
        "metrics": _metrics(m),
        "n_weight_rows": weights.height,
        "n_bars": bars_seg.height,
        "elapsed_s": round(elapsed, 2),
        "data_sha256": {
            p.name: _sha256(p)
            for d in (
                args.data_root / "raw" / "sources_yahoo",
                args.data_root / "raw" / args.stocks_dir if args.stocks_dir else None,
            )
            if d is not None and d.exists()
            for p in sorted(d.glob("*_1d.parquet"))
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps(receipt["metrics"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
