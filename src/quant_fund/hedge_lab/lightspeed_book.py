"""Run frozen Lightspeed books on the file tape. Research only.

Two honest cards, neither auto-promoted:

1. CS challenger ``nautica`` (a priori ``cs_z_mom_60``) vs public ridge
   on the 55-name panel. Promotion still needs pairwise DM of −IC vs
   ridge plus White RC / SPA / StepM.
2. The actual concentrated momentum engine (frozen 63d / 200 SMA /
   crash) on names that exist on the tape, plus a reconstructed 3× QQQ
   TQQQ rotation. Yahoo expected-metrics from the private repo are not
   copied as Dipcatcher P&L.

``blend_weight`` stays 0. Champion stays public ridge. No Alpaca.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.hedge_lab.mirror import long_short_path
from quant_fund.hedge_lab.promotion import ALPHA, clears_cs_promotion
from quant_fund.hedge_lab.scoreboard import book_economic_scoreboard
from quant_fund.lightspeed.metalabel import meta_label_gate
from quant_fund.lightspeed.momentum import momentum_target_weights
from quant_fund.lightspeed.rotation import tqqq_target_weights
from quant_fund.lightspeed.specs import (
    HOLDOUT_START,
    SELECTION_END,
    MomentumSpec,
    MomentumUniverse,
    nautica_momentum_v1,
    stock_momentum_v1,
    tqqq_long_full_v1,
)
from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.metrics.inference import diebold_mariano, overlap_aware_hac_lags
from quant_fund.metrics.snooping import reality_check, spa_test, stepm

Array = NDArray[np.float64]
_EPS = 1e-12


def reconstruct_3x(signal: Array) -> Array:
    """Daily-reset 3× of a signal close. LETF path, not 3× of long-horizon return."""
    px = np.asarray(signal, dtype=float)
    out = np.full(len(px), np.nan, dtype=float)
    if len(px) == 0 or not np.isfinite(px[0]) or px[0] <= 0:
        return out
    out[0] = 100.0
    for i in range(1, len(px)):
        if not (np.isfinite(px[i]) and np.isfinite(px[i - 1]) and px[i - 1] > 0):
            out[i] = out[i - 1]
            continue
        out[i] = out[i - 1] * (1.0 + 3.0 * (px[i] / px[i - 1] - 1.0))
        if not np.isfinite(out[i]) or out[i] <= 0:
            out[i] = out[i - 1]
    return out


def close_panel(
    gold: pl.DataFrame,
    symbols: tuple[str, ...],
    *,
    price_col: str | None = None,
) -> tuple[list[object], dict[str, Array]]:
    """Inner-join close paths for ``symbols``. Prefers total-return close."""
    col = price_col
    if col is None:
        col = "close_total_return" if "close_total_return" in gold.columns else "close"
    if col not in gold.columns:
        raise ValueError(f"gold panel missing {col}")
    present = [s for s in symbols if s in set(gold["security_id"].unique().to_list())]
    if not present:
        raise ValueError("none of the requested symbols are on the tape")
    sub = gold.filter(pl.col("security_id").is_in(present)).select(
        ["event_time", "security_id", col]
    )
    piv = sub.pivot(index="event_time", on="security_id", values=col).sort("event_time")
    keep = [s for s in present if s in piv.columns]
    piv = piv.drop_nulls(subset=keep)
    dates = piv["event_time"].to_list()
    closes = {s: piv[s].to_numpy().astype(float) for s in keep}
    return dates, closes


def risk_book_returns(
    weights: dict[str, Array],
    closes: dict[str, Array],
    *,
    defensive: str = "SGOV",
    one_way_cost: float = 0.001,
) -> Array:
    """Close-to-close book after delay-shifted weights. Residual sleeve is cash."""
    if not np.isfinite(one_way_cost) or one_way_cost < 0.0:
        raise ValueError("one_way_cost must be finite and non-negative")
    risk = [s for s in weights if s != defensive and s in closes]
    if not risk:
        return np.asarray([], dtype=float)
    n = len(weights[risk[0]])
    rets: list[float] = []
    prev = {s: 0.0 for s in risk}
    for t in range(1, n):
        pnl = 0.0
        turn = 0.0
        for s in risk:
            px = closes[s]
            w = float(weights[s][t])
            if np.isfinite(px[t]) and np.isfinite(px[t - 1]) and px[t - 1] > _EPS:
                pnl += w * (px[t] / px[t - 1] - 1.0)
            turn += abs(w - prev[s])
            prev[s] = w
        rets.append(pnl - float(one_way_cost) * turn)
    return np.asarray(rets, dtype=float)


def _ic_card(name: str, scores: Array, y: Array, dates: Array, hac: int | None) -> dict[str, Any]:
    mask = np.isfinite(scores) & np.isfinite(y)
    ic = date_ic_series(scores[mask], y[mask], dates[mask], min_names=5, hac_lags=hac)
    return {
        "name": name,
        "mean_ic": ic.mean_pearson,
        "mean_rank_ic": ic.mean_spearman,
        "t_ic": ic.t_pearson,
        "p_ic": ic.p_pearson,
        "n_dates": ic.n_dates,
        "ic_series": [float(v) for v in ic.pearson.tolist()],
        "ic_dates": [str(d) for d in ic.dates],
    }


def _align_ic(cards: list[dict[str, Any]]) -> dict[str, Array]:
    common: set[str] | None = None
    by_date: dict[str, dict[str, float]] = {}
    for card in cards:
        dates = [str(d) for d in card["ic_dates"]]
        common = set(dates) if common is None else common & set(dates)
        by_date[card["name"]] = dict(zip(dates, card["ic_series"], strict=True))
    if not common:
        return {card["name"]: np.asarray([], dtype=float) for card in cards}
    order = sorted(common)
    return {
        card["name"]: np.asarray([by_date[card["name"]][d] for d in order], dtype=float)
        for card in cards
    }


def _gates(
    aligned: dict[str, Array],
    *,
    benchmark: str,
    n_boot: int,
    lags: int | None = None,
) -> dict[str, Any]:
    if benchmark not in aligned:
        raise ValueError("benchmark IC series missing")
    bench = aligned[benchmark]
    names = [n for n in aligned if n != benchmark]
    if not names or bench.size < 10:
        return {
            "dm": {},
            "reality_check_p": float("nan"),
            "spa_p_consistent": float("nan"),
            "stepm_rejected": [],
            "promote": False,
            "cleared": [],
        }
    dm: dict[str, Any] = {}
    cols = []
    means: dict[str, float] = {}
    kept: list[str] = []
    for name in names:
        series = aligned[name]
        n = min(len(series), len(bench))
        if n < 10:
            continue
        # Negative IC as DM loss: higher IC ⇒ lower loss.
        result = diebold_mariano(-series[:n], -bench[:n], lags=lags, name_a=name, name_b=benchmark)
        chunk = series[:n]
        finite = chunk[np.isfinite(chunk)]
        means[name] = float(np.mean(finite)) if finite.size else float("nan")
        dm[name] = {
            "statistic": result.statistic,
            "p_value": result.p_value,
            "preferred": result.preferred,
            "mean_loss_diff": result.mean_loss_diff,
            "mean_ic": means[name],
            "lags": int(result.lags),
        }
        cols.append(chunk - bench[:n])
        kept.append(name)
    if not cols:
        return {
            "dm": dm,
            "reality_check_p": float("nan"),
            "spa_p_consistent": float("nan"),
            "stepm_rejected": [],
            "promote": False,
            "cleared": [],
        }
    f = np.column_stack(cols)
    rc = reality_check(f, n_boot=n_boot, seed=7)
    spa = spa_test(f, n_boot=n_boot, seed=7)
    step = stepm(f, n_boot=n_boot, seed=7)
    rejected = [kept[i] for i, flag in enumerate(step.rejected) if flag]
    cleared = [
        name
        for name, row in dm.items()
        if clears_cs_promotion(
            name=name,
            mean_ic=means[name],
            dm_preferred=str(row["preferred"]),
            dm_p=float(row["p_value"]),
            stepm_rejected=rejected,
            rc_p=float(rc.p_value),
            spa_p=float(spa.p_consistent),
            alpha=ALPHA,
        )
    ]
    return {
        "dm": dm,
        "reality_check_p": rc.p_value,
        "spa_p_consistent": spa.p_consistent,
        "stepm_rejected": rejected,
        "n_boot": int(n_boot),
        "hac_lags": lags,
        "promote": bool(cleared),
        "cleared": cleared,
        "note": (
            "promote is True only if the same challenger has mean IC > 0, "
            "wins pairwise DM of −IC vs ridge at 5%, and is inside the StepM "
            "rejection set, while White RC and SPA also reject no-skill. "
            "A less-negative IC does not clear. This runner never writes a "
            "champion alias and does not move blend_weight."
        ),
    }


def _date_key(value: object) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())[:10]
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def split_by_holdout(
    dates: list[object],
    returns: Array,
    *,
    holdout_start: str = HOLDOUT_START,
) -> dict[str, Array]:
    """Align close-to-close returns (starts at date 1) to the frozen holdout cut."""
    if len(returns) != max(len(dates) - 1, 0):
        raise ValueError("returns must align with dates[1:]")
    keys = [_date_key(d) for d in dates[1:]]
    r = np.asarray(returns, dtype=float)
    is_mask = np.array([k < holdout_start for k in keys])
    oos_mask = np.array([k >= holdout_start for k in keys])
    return {"is": r[is_mask], "holdout": r[oos_mask], "full": r}


def _window_cards(dates: list[object], returns: Array) -> dict[str, Any]:
    parts = split_by_holdout(dates, returns)
    return {
        "selection_end": SELECTION_END,
        "holdout_start": HOLDOUT_START,
        "full": book_economic_scoreboard(parts["full"], data_source="file"),
        "is": book_economic_scoreboard(parts["is"], data_source="file"),
        "holdout": book_economic_scoreboard(parts["holdout"], data_source="file"),
    }


def _momentum_spec_on_tape(names: tuple[str, ...], family: str) -> MomentumSpec:
    parent = nautica_momentum_v1() if family.startswith("nautica") else stock_momentum_v1()
    return MomentumSpec(
        family=parent.family,
        name=parent.name,
        status=parent.status,
        universe=MomentumUniverse(risk=names, defensive="SGOV"),
        params=parent.params,
        live_disabled=True,
    )


def run_lightspeed_file_book(
    config_path: str = "configs/hedge_lab.yaml",
    label: str = "future_idio_return_1",
    *,
    one_way_cost: float = 0.001,
    n_boot: int = 1000,
    k_frac: float = 0.2,
) -> dict[str, Any]:
    """CS nautica vs ridge plus frozen Lightspeed books on the file tape."""
    from quant_fund.config import load_config
    from quant_fund.models.ranking import PUBLIC_FEATURES, available_features, drop_oracle_columns
    from quant_fund.pipeline.dataset import design_matrix, panel
    from quant_fund.pipeline.train import _label_horizon
    from quant_fund.research.benches import oos_rank_scores

    cfg = load_config(config_path)
    raw_gold = drop_oracle_columns(panel(cfg))
    gold = raw_gold.filter(raw_gold["security_id"] != cfg.data.benchmark_id)
    feats = available_features(gold.columns, PUBLIC_FEATURES)
    x, y, dates, used, ids = design_matrix(gold, label, feats)
    horizon = _label_horizon(label)
    n_dates = len({str(d) for d in dates.tolist()})
    hac = overlap_aware_hac_lags(int(n_dates), int(horizon)) if n_dates else None

    engines = {
        "ridge": oos_rank_scores(
            "ridge", cfg, x, y, dates, ids, horizon_bars=horizon, feature_names=used
        ),
        "nautica": oos_rank_scores(
            "nautica", cfg, x, y, dates, ids, horizon_bars=horizon, feature_names=used
        ),
        "classic": oos_rank_scores(
            "classic", cfg, x, y, dates, ids, horizon_bars=horizon, feature_names=used
        ),
    }
    ic_cards = [_ic_card(name, scores, y, dates, hac) for name, scores in engines.items()]
    aligned = _align_ic(ic_cards)
    gates = _gates(aligned, benchmark="ridge", n_boot=int(n_boot), lags=hac)

    ls_cards = {}
    for name, scores in engines.items():
        path = long_short_path(
            scores, y, dates, ids, k_frac=k_frac, one_way_cost=float(one_way_cost)
        )
        ls_cards[name] = path["economic"]

    tape_ids = set(raw_gold["security_id"].unique().to_list())
    stock_names = tuple(s for s in stock_momentum_v1().universe.risk if s in tape_ids)
    nautica_names = tuple(s for s in nautica_momentum_v1().universe.risk if s in tape_ids)
    books: dict[str, Any] = {}
    for family, names in (
        ("stock-momentum-v1", stock_names),
        ("nautica-momentum-v1", nautica_names),
    ):
        if len(names) < 2:
            continue
        dates_c, closes = close_panel(raw_gold, names)
        closes["SGOV"] = np.full(len(dates_c), 100.0, dtype=float)
        spec = _momentum_spec_on_tape(names, family)
        weights = momentum_target_weights(closes, spec)
        r = risk_book_returns(weights, closes, one_way_cost=float(one_way_cost))
        last = {k: float(v[-1]) for k, v in weights.items()}
        spy_windows = None
        if "SPY" in tape_ids:
            spy_dates, spy_closes = close_panel(raw_gold, ("SPY",))
            spy_map = {_date_key(d): i for i, d in enumerate(spy_dates)}
            spy_r = []
            for d0, d1 in zip(dates_c[:-1], dates_c[1:], strict=True):
                i0 = spy_map.get(_date_key(d0))
                i1 = spy_map.get(_date_key(d1))
                px = spy_closes["SPY"]
                if i0 is None or i1 is None or px[i0] <= _EPS:
                    spy_r.append(0.0)
                else:
                    spy_r.append(float(px[i1] / px[i0] - 1.0))
            spy_windows = _window_cards(dates_c, np.asarray(spy_r, dtype=float))
        books[family] = {
            "n_risk_names": len(names),
            "risk_names": list(names),
            "n_dates": len(dates_c),
            "last_weights": last,
            "windows": _window_cards(dates_c, r),
            "spy_buyhold_windows": spy_windows,
            "one_way_cost": float(one_way_cost),
            "defensive": "cash_residual (SGOV not on tape)",
        }

    rotation = None
    if "QQQ" in tape_ids:
        q_dates, q_closes = close_panel(raw_gold, ("QQQ",))
        qqq = q_closes["QQQ"]
        tqqq = reconstruct_3x(qqq)
        rot = tqqq_target_weights(qqq, tqqq)
        rot_closes = {"TQQQ": tqqq, "SGOV": np.full(len(qqq), 100.0)}
        r_rot = risk_book_returns(rot, rot_closes, one_way_cost=float(one_way_cost))
        q_ret = np.zeros(len(qqq))
        q_ret[1:] = qqq[1:] / np.maximum(qqq[:-1], _EPS) - 1.0
        gap = np.zeros(len(qqq))
        gap[1:] = (qqq[1:] - qqq[:-1]) / np.maximum(qqq[:-1], _EPS)
        gate = meta_label_gate(gap, q_ret)
        gated = tqqq_target_weights(qqq, tqqq, risk_multiplier=gate.multiplier)
        r_gated = risk_book_returns(gated, rot_closes, one_way_cost=float(one_way_cost))
        spy_bh = None
        if "SPY" in tape_ids:
            _s_dates, spy_closes = close_panel(raw_gold, ("SPY",))
            if len(spy_closes["SPY"]) == len(qqq):
                spy = spy_closes["SPY"]
                spy_r_arr = np.asarray(spy[1:] / np.maximum(spy[:-1], _EPS) - 1.0, dtype=float)
                spy_bh = _window_cards(q_dates, spy_r_arr)
        rotation = {
            "family": tqqq_long_full_v1().family,
            "tqqq": "reconstructed_3x_qqq_daily",
            "defensive": "cash_residual (SGOV not on tape)",
            "last_weights": {k: float(v[-1]) for k, v in rot.items()},
            "windows": _window_cards(q_dates, r_rot),
            "metalabel_windows": _window_cards(q_dates, r_gated),
            "spy_buyhold_windows": spy_bh,
            "metalabel_last_multiplier": float(gate.multiplier[-1]),
            "one_way_cost": float(one_way_cost),
        }

    receipt: dict[str, Any] = {
        "catalog": "hedge_lab_analytics",
        "source": "https://github.com/cosmic-hydra/lightspeed",
        "config": config_path,
        "label": label,
        "n_names": int(gold["security_id"].n_unique()),
        "n_rows": int(x.shape[0]),
        "features": used,
        "scheme": cfg.validation.scheme,
        "train_bars": int(cfg.validation.train_bars),
        "one_way_cost": float(one_way_cost),
        "cs_ic": [
            {k: v for k, v in c.items() if k not in {"ic_series", "ic_dates"}} for c in ic_cards
        ],
        "cs_ls": ls_cards,
        "gates": gates,
        "momentum_books": books,
        "tqqq_rotation": rotation,
        "execution_claim": "paper_backtest",
        "live_pnl_claim": False,
        "blend_weight": 0.0,
        "champion": "ridge",
        "champion_alias": False,
        "research_only": True,
        "broker": None,
        "note": (
            "Lightspeed formulas on the Yahoo session-close tape. "
            "TQQQ is a daily-reset 3× reconstruction of QQQ, not the ETF. "
            "SGOV is a zero-yield residual. Momentum/TQQQ params were frozen "
            f"on IS through {SELECTION_END}; holdout starts {HOLDOUT_START} "
            "and is not used to retune. CS nautica vs ridge is walk-forward OOS. "
            "Not a live P&L claim."
        ),
    }
    root = Path(cfg.data.root)
    out = root / "metadata" / "lightspeed_book.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    public = Path("artifacts") / "hedge_lab" / "lightspeed_book.json"
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    receipt["receipt_path"] = str(out)
    receipt["artifact_path"] = str(public)
    return receipt
