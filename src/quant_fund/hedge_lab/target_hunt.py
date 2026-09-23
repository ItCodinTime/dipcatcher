"""Causal hunt for high-Sharpe / low-DD streams on the file tape.

Ports the OU/Kalman pairs idea from NDAR123909/ou-statarb and
mo-479/statarb-pairs-trading-2 (walk-forward, delay 1, costs), plus
overnight / TSMOM / vol-target overlays. Every path is delay-shifted.
Not a live claim. Numbers are whatever the tape gives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.hedge_lab.directional import (
    ETF_BASKET,
    antonacci_returns,
    close_matrix,
    risk_parity_blend,
    topk_long_returns,
    tsmom_book_returns,
)
from quant_fund.hedge_lab.lightspeed_book import reconstruct_3x, risk_book_returns
from quant_fund.hedge_lab.resources import cap_blas_threads
from quant_fund.hedge_lab.scoreboard import book_economic_scoreboard
from quant_fund.lightspeed.momentum import momentum_target_weights
from quant_fund.lightspeed.rotation import tqqq_target_weights
from quant_fund.lightspeed.specs import (
    HOLDOUT_START,
    MomentumSpec,
    MomentumUniverse,
    stock_momentum_v1,
)
from quant_fund.risk.gates import GateSpec, apply_gate_stack, dd_halt, vol_target

Array = NDArray[np.float64]
_EPS = 1e-12

# Re-export for tests that import hunt helpers.
__all__ = [
    "ECONOMIC_PAIRS",
    "dd_halt",
    "kalman_beta",
    "pair_spread_returns",
    "run_target_hunt",
    "vol_target",
]

# Declared economic pairs (not a 55-choose-2 scan). Same-sector / ETF hedges.
ECONOMIC_PAIRS: tuple[tuple[str, str], ...] = (
    ("JPM", "GS"),
    ("JPM", "BAC"),
    ("XOM", "CVX"),
    ("MSFT", "AAPL"),
    ("GOOGL", "META"),
    ("NVDA", "AMD"),
    ("QQQ", "XLK"),
    ("GLD", "SLV"),
    ("TLT", "IEF"),
    ("XLF", "JPM"),
    ("XLE", "XOM"),
    ("SPY", "QQQ"),
    ("IWM", "SPY"),
)


def kalman_beta(y: Array, x: Array, *, delta: float = 1e-5, ve: float = 1e-3) -> Array:
    """Causal dynamic-regression beta. Row t uses observations through t."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    beta = np.full(n, np.nan)
    q = delta / max(1.0 - delta, _EPS)
    r_state = np.eye(2) * q
    p = np.eye(2)
    theta = np.zeros(2)
    for i in range(n):
        if not (np.isfinite(y[i]) and np.isfinite(x[i])):
            beta[i] = theta[1]
            continue
        f = np.array([1.0, x[i]])
        p = p + r_state
        pred = float(f @ theta)
        sys = float(f @ p @ f + ve)
        k = (p @ f) / max(sys, _EPS)
        theta = theta + k * (y[i] - pred)
        p = p - np.outer(k, f) @ p
        beta[i] = theta[1]
    return beta


def pair_spread_returns(
    y: Array,
    x: Array,
    *,
    z_win: int = 60,
    entry: float = 2.0,
    exit_z: float = 0.5,
    delay: int = 1,
    one_way_cost: float = 0.001,
) -> Array:
    """Kalman spread z-score. Position at t uses data through t-delay."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    b = kalman_beta(y, x)
    spread = y - b * x
    n = len(y)
    pos = np.zeros(n)
    cur = 0.0
    for i in range(z_win, n):
        w = spread[i - z_win + 1 : i + 1]
        mu = float(np.mean(w))
        sd = float(np.std(w, ddof=1))
        if sd < _EPS:
            pos[i] = cur
            continue
        z = (spread[i] - mu) / sd
        if cur == 0.0 and abs(z) >= entry:
            cur = -1.0 if z > 0 else 1.0
        elif cur != 0.0 and abs(z) <= exit_z:
            cur = 0.0
        pos[i] = cur
    if delay > 0:
        shifted = np.zeros(n)
        shifted[delay:] = pos[:-delay]
        pos = shifted
    ry = np.zeros(n)
    rx = np.zeros(n)
    ry[1:] = y[1:] / np.maximum(y[:-1], _EPS) - 1.0
    rx[1:] = x[1:] / np.maximum(x[:-1], _EPS) - 1.0
    # Dollar-neutral: long/short 0.5 each side when in a trade.
    pnl = 0.5 * pos * (ry - rx)
    turn = np.zeros(n)
    turn[1:] = np.abs(pos[1:] - pos[:-1])
    pnl = pnl - 2.0 * float(one_way_cost) * turn
    return pnl


def _pivot(gold: pl.DataFrame, col: str) -> tuple[list[object], dict[str, Array]]:
    sub = gold.select(["event_time", "security_id", col]).drop_nulls()
    piv = sub.pivot(index="event_time", on="security_id", values=col).sort("event_time")
    dates = piv["event_time"].to_list()
    closes = {c: piv[c].to_numpy().astype(float) for c in piv.columns if c != "event_time"}
    return dates, closes


def _card(name: str, returns: Array, dates: list[object] | None = None) -> dict[str, Any]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    econ = book_economic_scoreboard(r, data_source="file")
    econ["name"] = name
    if dates is not None and len(returns) == len(dates) and len(dates) >= 3:
        from quant_fund.hedge_lab.lightspeed_book import split_by_holdout

        parts = split_by_holdout(dates, np.asarray(returns[1:], dtype=float))
        ho = book_economic_scoreboard(parts["holdout"], data_source="file")
        sel = book_economic_scoreboard(parts["is"], data_source="file")
        econ["holdout_sharpe"] = ho.get("sharpe")
        econ["holdout_max_drawdown"] = ho.get("max_drawdown")
        econ["holdout_cagr"] = ho.get("cagr")
        econ["holdout_n"] = ho.get("n_returns")
        econ["selection_sharpe"] = sel.get("sharpe")
        econ["selection_max_drawdown"] = sel.get("max_drawdown")
    return econ


def run_target_hunt(
    labels_path: str = "data/file_us/gold/labels.parquet",
    *,
    one_way_cost: float = 0.001,
    dd_limit: float = 0.05,
    vol_tgt: float = 0.025,
    artifact_name: str = "target_hunt.json",
) -> dict[str, Any]:
    cap_blas_threads(0.6)
    gold = pl.read_parquet(labels_path)
    price = "close_total_return" if "close_total_return" in gold.columns else "close"
    dates, closes = _pivot(gold, price)
    names = sorted(closes)
    # Align to common calendar (inner).
    n = min(len(v) for v in closes.values())
    for k in list(closes):
        closes[k] = np.asarray(closes[k][:n], dtype=float)
    dates = dates[:n]

    def card(name: str, returns: Array) -> dict[str, Any]:
        return _card(name, returns, dates)

    paths: dict[str, Array] = {}
    cards: list[dict[str, Any]] = []

    # Equal-weight close-to-close (delay 1 from membership: always in).
    ew_px = np.column_stack([closes[s] for s in names])
    rets = np.zeros_like(ew_px)
    rets[1:] = ew_px[1:] / np.maximum(ew_px[:-1], _EPS) - 1.0
    ew = np.nanmean(rets, axis=1)
    ew[0] = 0.0
    paths["ew_close"] = ew
    cards.append(card("ew_close", ew))

    if {"open", "close"} <= set(gold.columns):
        _, opens = _pivot(gold, "open")
        _, raw_c = _pivot(gold, "close")
        common = [s for s in names if s in opens and s in raw_c]
        on_rows = []
        intra_rows = []
        for s in common:
            o = np.asarray(opens[s][:n], dtype=float)
            c = np.asarray(raw_c[s][:n], dtype=float)
            on = np.zeros(n)
            intra = np.zeros(n)
            on[1:] = o[1:] / np.maximum(c[:-1], _EPS) - 1.0
            intra[:] = np.where(o > _EPS, c / o - 1.0, 0.0)
            on_rows.append(on)
            intra_rows.append(intra)
        on_ew = np.nanmean(np.column_stack(on_rows), axis=1)
        intra_ew = np.nanmean(np.column_stack(intra_rows), axis=1)
        # Overnight-only book: enter prior close, exit open. Two sides / day.
        on_costed = on_ew - 2.0 * float(one_way_cost)
        paths["ew_overnight_frictionless"] = on_ew
        paths["ew_overnight_20bp"] = on_costed
        paths["ew_intraday_frictionless"] = intra_ew
        cards.append(card("ew_overnight_frictionless", on_ew))
        cards.append(card("ew_overnight_20bp", on_costed))
        cards.append(card("ew_intraday_frictionless", intra_ew))

    # Economic Kalman pairs, equal risk across live pairs.
    pair_pnl = []
    used_pairs = []
    for a, b in ECONOMIC_PAIRS:
        if a not in closes or b not in closes:
            continue
        pnl = pair_spread_returns(
            closes[a], closes[b], one_way_cost=float(one_way_cost)
        )
        pair_pnl.append(pnl)
        used_pairs.append(f"{a}/{b}")
        paths[f"pair_{a}_{b}"] = pnl
        cards.append(card(f"pair_{a}_{b}", pnl))
    if pair_pnl:
        blend = np.nanmean(np.column_stack(pair_pnl), axis=1)
        vt = vol_target(blend, target=vol_tgt)
        paths["pairs_equal"] = blend
        paths["pairs_voltgt_2p5"] = vt
        paths["pairs_voltgt_ddhalt"] = dd_halt(vt, limit=dd_limit)
        cards.append(card("pairs_equal", blend))
        cards.append(card("pairs_voltgt_2p5", vt))
        cards.append(card("pairs_voltgt_ddhalt", paths["pairs_voltgt_ddhalt"]))

    # Lightspeed momentum, then vol-target to the 5% rail.
    risk = tuple(s for s in stock_momentum_v1().universe.risk if s in closes)
    if len(risk) >= 2:
        spec = MomentumSpec(
            family="stock-momentum-v1",
            name="stock-momentum-v1",
            status="frozen",
            universe=MomentumUniverse(risk=risk, defensive="SGOV"),
            params=stock_momentum_v1().params,
            live_disabled=True,
        )
        c_m = {s: closes[s] for s in risk}
        c_m["SGOV"] = np.full(n, 100.0)
        w = momentum_target_weights(c_m, spec)
        mom = risk_book_returns(w, c_m, one_way_cost=float(one_way_cost))
        # Align: risk_book_returns starts at bar 1.
        mom_full = np.zeros(n)
        mom_full[1 : 1 + len(mom)] = mom[: n - 1]
        vt_m = vol_target(mom_full, target=vol_tgt)
        paths["lightspeed_mom"] = mom_full
        paths["lightspeed_mom_voltgt_2p5"] = vt_m
        paths["lightspeed_mom_voltgt_ddhalt"] = dd_halt(vt_m, limit=dd_limit)
        cards.append(card("lightspeed_mom", mom_full))
        cards.append(card("lightspeed_mom_voltgt_2p5", vt_m))
        cards.append(card("lightspeed_mom_voltgt_ddhalt", paths["lightspeed_mom_voltgt_ddhalt"]))

    if "QQQ" in closes:
        tqqq = reconstruct_3x(closes["QQQ"])
        rot = tqqq_target_weights(closes["QQQ"], tqqq)
        rot_px = {"TQQQ": tqqq, "SGOV": np.full(n, 100.0)}
        tr = risk_book_returns(rot, rot_px, one_way_cost=float(one_way_cost))
        t_full = np.zeros(n)
        t_full[1 : 1 + len(tr)] = tr[: n - 1]
        vt_t = vol_target(t_full, target=vol_tgt)
        paths["tqqq_3x"] = t_full
        paths["tqqq_3x_voltgt_2p5"] = vt_t
        paths["tqqq_3x_voltgt_ddhalt"] = dd_halt(vt_t, limit=dd_limit)
        cards.append(card("tqqq_3x", t_full))
        cards.append(card("tqqq_3x_voltgt_2p5", vt_t))
        cards.append(card("tqqq_3x_voltgt_ddhalt", paths["tqqq_3x_voltgt_ddhalt"]))

    combo_keys = [k for k in ("pairs_voltgt_2p5", "lightspeed_mom_voltgt_2p5", "tqqq_3x_voltgt_2p5") if k in paths]
    if combo_keys:
        combo = np.nanmean(np.column_stack([paths[k] for k in combo_keys]), axis=1)
        paths["combo_voltgt"] = combo
        paths["combo_voltgt_ddhalt"] = dd_halt(combo, limit=dd_limit)
        cards.append(card("combo_voltgt", combo))
        cards.append(card("combo_voltgt_ddhalt", paths["combo_voltgt_ddhalt"]))

    # Directional total-return books (not CS-idio LS). Delay 1, monthly rebalance.
    name_list = list(names)
    _, px_all = close_matrix(closes, name_list)
    tsmom_lo = tsmom_book_returns(px_all, long_only=True, one_way_cost=float(one_way_cost))
    tsmom_ls = tsmom_book_returns(px_all, long_only=False, max_gross=1.0, one_way_cost=float(one_way_cost))
    topk = topk_long_returns(px_all, top_k=5, one_way_cost=float(one_way_cost))
    topk_trend = topk_long_returns(
        px_all,
        top_k=5,
        one_way_cost=float(one_way_cost),
        sma=200,
        crash_lookback=10,
        crash_return=-0.2,
    )
    paths["tsmom_lo"] = tsmom_lo
    paths["tsmom_ls"] = tsmom_ls
    paths["topk5_12_1"] = topk
    paths["topk5_trend_crash"] = topk_trend
    cards.append(card("tsmom_lo", tsmom_lo))
    cards.append(card("tsmom_ls", tsmom_ls))
    cards.append(card("topk5_12_1", topk))
    cards.append(card("topk5_trend_crash", topk_trend))
    for tgt in (0.10, 0.15, 0.20):
        tag = str(int(round(tgt * 100)))
        for key, series in (("topk5_12_1", topk), ("topk5_trend_crash", topk_trend)):
            scaled = vol_target(series, target=tgt)
            name = f"{key}_voltgt_{tag}"
            paths[name] = scaled
            cards.append(card(name, scaled))
    etf_names = [s for s in ETF_BASKET if s in closes]
    if len(etf_names) >= 4:
        _, px_etf = close_matrix(closes, etf_names)
        etf_lo = tsmom_book_returns(px_etf, long_only=True, one_way_cost=float(one_way_cost))
        paths["tsmom_etf_lo"] = etf_lo
        cards.append(card("tsmom_etf_lo", etf_lo))
    if "SPY" in closes and "TLT" in closes:
        dual = antonacci_returns(closes["SPY"], closes["TLT"], one_way_cost=float(one_way_cost))
        paths["antonacci_spy_tlt"] = dual
        cards.append(card("antonacci_spy_tlt", dual))
        paths["spy_buy_hold"] = np.zeros(n)
        spy_r = np.zeros(n)
        spy_r[1:] = closes["SPY"][1:] / np.maximum(closes["SPY"][:-1], _EPS) - 1.0
        paths["spy_buy_hold"] = spy_r
        cards.append(card("spy_buy_hold", spy_r))

    dir_keys = [k for k in ("tsmom_lo", "tsmom_etf_lo", "antonacci_spy_tlt", "topk5_12_1", "tqqq_3x") if k in paths]
    if len(dir_keys) >= 2:
        blend = risk_parity_blend({k: paths[k] for k in dir_keys}, delay=1, one_way_cost=0.0)
        paths["dir_riskparity"] = blend
        cards.append(card("dir_riskparity", blend))
        vt_d = vol_target(blend, target=vol_tgt)
        paths["dir_riskparity_voltgt"] = vt_d
        cards.append(card("dir_riskparity_voltgt", vt_d))
        cards.append(card("dir_riskparity_voltgt_ddhalt", dd_halt(vt_d, limit=dd_limit)))

    stack = GateSpec(vol_target=vol_tgt, dd_limit=dd_limit)
    stack_stepm = GateSpec(vol_target=vol_tgt, dd_limit=dd_limit, stepm_enable=True)
    for raw_key in ("ew_close", "pairs_equal", "lightspeed_mom", "tqqq_3x", "tsmom_lo", "topk5_12_1", "antonacci_spy_tlt", "dir_riskparity"):
        if raw_key not in paths:
            continue
        gated = apply_gate_stack(paths[raw_key], stack)
        paths[f"{raw_key}_riskstack"] = gated.returns
        gated_card = card(f"{raw_key}_riskstack", gated.returns)
        gated_card["gate"] = gated.snapshot()
        cards.append(gated_card)
        stepm = apply_gate_stack(paths[raw_key], stack_stepm, stepm_excess=paths[raw_key])
        paths[f"{raw_key}_stepm"] = stepm.returns
        stepm_card = card(f"{raw_key}_stepm", stepm.returns)
        stepm_card["gate"] = stepm.snapshot()
        cards.append(stepm_card)

    cards.sort(key=lambda c: float(c.get("sharpe") or float("-inf")), reverse=True)
    best = cards[0] if cards else {}
    receipt = {
        "catalog": "hedge_lab_analytics",
        "research_only": True,
        "live_pnl_claim": False,
        "one_way_cost": float(one_way_cost),
        "dd_limit": float(dd_limit),
        "vol_target": float(vol_tgt),
        "pairs": used_pairs,
        "n_dates": n,
        "n_names": len(names),
        "cards": cards,
        "best_abs_sharpe": best.get("name"),
        "target": {"sharpe": 5.0, "max_dd": -0.05, "note": "joint target; not forced"},
        "hit_sharpe5": bool(any((c.get("sharpe") or 0) > 5.0 for c in cards)),
        "hit_dd5": bool(any((c.get("max_drawdown") or 0) > -0.05 and (c.get("sharpe") or 0) > 1.0 for c in cards)),
        "note": (
            "Kalman/OU pairs after NDAR123909/ou-statarb + delay-1 costs. "
            "Directional TSMOM is Moskowitz–Ooi–Pedersen 12–1 on total-return "
            "closes (long-only / LS / ETF basket / Antonacci GEM), not the "
            "CS-idio ranker. topk5_trend_crash adds a 200-day SMA and a "
            "−20%/10d crash flatten, checked daily, delay 1. Vol targets "
            "10/15/20% are a Pareto sweep, not a promotion. Wide tape is "
            "survivorship-biased. Vol-target is causal. DD halt flattens after -5% "
            "and stays cash. Overnight 20bp assumes a close-to-open round trip "
            "every session. Sharpe is invariant to constant leverage. "
            "Champion stays public ridge. blend_weight stays 0."
        ),
        "holdout_start": HOLDOUT_START,
    }
    out = Path("artifacts") / "hedge_lab" / artifact_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    receipt["artifact_path"] = str(out)
    return receipt
