"""Sign-flip paper books. Costs do not flip. Not a live P&L claim.

If a long-short score has frictionless return series ``r``, the mirror is
``-r``. Sharpe is odd under that map (and leverage-invariant when rf=0).
Spread, commission, and impact are even: both books pay them. Searching for
the worst in-sample strategy and flipping it is the same search as picking
the best in-sample strategy. ``blend_weight`` stays 0. A 5% drawdown halt
cannot coexist with −150% total return on the same path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.hedge_lab.scoreboard import book_economic_scoreboard
from quant_fund.models.asset_pricing import date_groups
from quant_fund.models.cs_papers import _mean_date_ic
from quant_fund.validation.walk_forward import timestamp_ns, walk_forward

Array = NDArray[np.float64]


def negate_target_weights(weights: pl.DataFrame) -> pl.DataFrame:
    """Flip every ``target_weight``. Gross is unchanged; net changes sign."""
    if "target_weight" not in weights.columns:
        raise ValueError("weight panel requires target_weight")
    return weights.with_columns((-pl.col("target_weight")).alias("target_weight"))


def dollar_neutral_weights(scores: Array, k_frac: float = 0.2) -> Array:
    """Equal-weight long top ``k_frac``, short bottom ``k_frac``. Gross = 1."""
    if not np.isfinite(k_frac) or not 0.0 < k_frac <= 0.5:
        raise ValueError("k_frac must be in (0, 0.5]")
    s = np.asarray(scores, dtype=float)
    n = int(s.size)
    w = np.zeros(n, dtype=float)
    finite = np.isfinite(s)
    if int(finite.sum()) < 4:
        return w
    idx = np.where(finite)[0]
    k = max(1, int(np.floor(idx.size * float(k_frac))))
    if 2 * k > idx.size:
        k = max(1, idx.size // 4)
    order = idx[np.argsort(s[idx], kind="mergesort")]
    short = order[:k]
    long = order[-k:]
    w[long] = 0.5 / float(long.size)
    w[short] = -0.5 / float(short.size)
    return w


def long_short_path(
    scores: Array,
    y: Array,
    dates: NDArray[Any],
    ids: NDArray[Any] | None = None,
    *,
    k_frac: float = 0.2,
    one_way_cost: float = 0.0,
) -> dict[str, Any]:
    """Date-level dollar-neutral long-short of ``scores`` vs forward ``y``.

    ``one_way_cost`` is a fraction of NAV charged on one-way turnover
    ``sum |Δw|``. First date turns over from cash. Not a live P&L claim.
    """
    if not np.isfinite(one_way_cost) or one_way_cost < 0.0:
        raise ValueError("one_way_cost must be finite and non-negative")
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    if scores.shape[0] != y.shape[0] or len(dates) != scores.shape[0]:
        raise ValueError("scores, y, dates must align")
    if ids is not None and len(ids) != scores.shape[0]:
        raise ValueError("ids must align with scores")
    groups = date_groups(np.asarray(dates))
    id_arr = None if ids is None else np.asarray(ids)
    rets: list[float] = []
    turns: list[float] = []
    kept_dates: list[str] = []
    prev_map: dict[str, float] | None = None
    for idx in groups:
        w = dollar_neutral_weights(scores[idx], k_frac=k_frac)
        if float(np.sum(np.abs(w))) < 1e-12:
            continue
        yi = np.where(np.isfinite(y[idx]), y[idx], 0.0)
        gross_r = float(np.dot(w, yi))
        if prev_map is None or id_arr is None:
            turn = float(np.sum(np.abs(w)))
        else:
            cur_ids = [str(v) for v in id_arr[idx].tolist()]
            aligned = np.array([prev_map.get(sid, 0.0) for sid in cur_ids], dtype=float)
            current = {sid: float(ww) for sid, ww in zip(cur_ids, w, strict=True)}
            dropped = float(sum(abs(val) for key, val in prev_map.items() if key not in current))
            turn = float(np.sum(np.abs(w - aligned)) + dropped)
        rets.append(gross_r - float(one_way_cost) * turn)
        turns.append(turn)
        stamp = dates[int(idx[0])]
        if hasattr(stamp, "isoformat"):
            kept_dates.append(str(stamp.isoformat())[:10])
        else:
            text = str(stamp)
            kept_dates.append(text[:10] if len(text) >= 10 else text)
        if id_arr is not None:
            prev_map = {
                str(sid): float(ww) for sid, ww in zip(id_arr[idx].tolist(), w, strict=True)
            }
        else:
            prev_map = {str(i): float(ww) for i, ww in enumerate(w)}
    r = np.asarray(rets, dtype=float)
    economic = book_economic_scoreboard(r, data_source="paper_ls")
    economic["mean_turnover"] = float(np.mean(turns)) if turns else float("nan")
    economic["one_way_cost"] = float(one_way_cost)
    economic["k_frac"] = float(k_frac)
    economic["n_dates"] = int(r.size)
    return {
        "returns": r,
        "turnover": np.asarray(turns, dtype=float),
        "dates": kept_dates,
        "economic": economic,
    }


def pair_book_and_mirror(
    scores: Array,
    y: Array,
    dates: NDArray[Any],
    ids: NDArray[Any] | None = None,
    *,
    k_frac: float = 0.2,
    one_way_cost: float = 0.0,
) -> dict[str, Any]:
    """Frictionless-odd / cost-even identity for one score card."""
    book = long_short_path(scores, y, dates, ids, k_frac=k_frac, one_way_cost=one_way_cost)
    mirror = long_short_path(
        -np.asarray(scores, dtype=float),
        y,
        dates,
        ids,
        k_frac=k_frac,
        one_way_cost=one_way_cost,
    )
    s_book = float(book["economic"]["sharpe"])
    s_mirror = float(mirror["economic"]["sharpe"])
    return {
        "book": book,
        "mirror": mirror,
        "sharpe_sum": s_book + s_mirror,
        "costs_even_under_sign_flip": True,
        "one_way_cost": float(one_way_cost),
        "execution_claim": "paper_backtest",
        "live_pnl_claim": False,
        "blend_weight": 0.0,
    }


def nested_worst_univariate_scores(
    x: Array,
    y: Array,
    dates: NDArray[Any],
    config: Any,
    *,
    horizon_bars: int = 1,
) -> Array:
    """Walk-forward: trade the train-fold column with the *lowest* date IC.

    Mirror of this score is the best-train-IC univariate. Train-only selection.
    Not OOS-tuned, not a live claim.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    times = sorted(set(np.asarray(dates).tolist()))
    folds = walk_forward(
        times,
        config.validation,
        horizon_bars=int(horizon_bars),
        embargo_bars=config.embargo_bars(),
    )
    pred = np.full(len(y), np.nan, dtype=float)
    date_ns = timestamp_ns(dates)
    if not folds:
        fallback_ics = [_mean_date_ic(x[:, j], y, dates) for j in range(x.shape[1])]
        j_star = int(np.nanargmin(np.asarray(fallback_ics, dtype=float)))
        pred[:] = x[:, j_star]
        return pred
    for fold in folds:
        tr = np.isin(date_ns, timestamp_ns(fold.train_times))
        te = np.isin(date_ns, timestamp_ns(fold.test_times))
        if not tr.any() or not te.any():
            continue
        ics = np.array(
            [_mean_date_ic(x[tr, j], y[tr], np.asarray(dates)[tr]) for j in range(x.shape[1])],
            dtype=float,
        )
        if not np.isfinite(ics).any():
            continue
        j_star = int(np.nanargmin(ics))
        pred[te] = x[te, j_star]
    return pred


def random_date_scores(dates: NDArray[Any], seed: int = 7) -> Array:
    """Per-date Gaussian scores. Both signs lose to costs; no edge."""
    rng = np.random.default_rng(int(seed))
    out = np.zeros(len(dates), dtype=float)
    for idx in date_groups(np.asarray(dates)):
        out[idx] = rng.normal(size=idx.size)
    return out


def summarize_pair(name: str, pair: dict[str, Any]) -> dict[str, Any]:
    book = pair["book"]["economic"]
    mirror = pair["mirror"]["economic"]
    return {
        "name": name,
        "book_sharpe": book["sharpe"],
        "mirror_sharpe": mirror["sharpe"],
        "sharpe_sum": pair["sharpe_sum"],
        "book_total_return": book.get("total_return"),
        "mirror_total_return": mirror.get("total_return"),
        "book_max_drawdown": book["max_drawdown"],
        "mirror_max_drawdown": mirror["max_drawdown"],
        "book_cagr": book["cagr"],
        "mirror_cagr": mirror["cagr"],
        "mean_turnover": pair["book"]["economic"].get("mean_turnover"),
        "one_way_cost": pair["one_way_cost"],
        "flag_high_sharpe_book": book.get("flag_high_sharpe"),
        "flag_high_sharpe_mirror": mirror.get("flag_high_sharpe"),
        "live_pnl_claim": False,
        "blend_weight": 0.0,
    }


def run_file_tape_mirror(
    config_path: str = "configs/hedge_lab.yaml",
    label: str = "future_idio_return_1",
    *,
    one_way_cost: float = 0.001,
    k_frac: float = 0.2,
) -> dict[str, Any]:
    """Walk-forward public scores, then book vs sign-flip LS with a cost haircut.

    Not a live P&L claim. Champion stays public ridge. ``blend_weight`` 0.
    """
    from quant_fund.config import load_config
    from quant_fund.models.ranking import PUBLIC_FEATURES, available_features, drop_oracle_columns
    from quant_fund.pipeline.dataset import design_matrix, panel
    from quant_fund.pipeline.train import _label_horizon
    from quant_fund.research.benches import oos_rank_scores

    cfg = load_config(config_path)
    gold = drop_oracle_columns(panel(cfg))
    gold = gold.filter(gold["security_id"] != cfg.data.benchmark_id)
    feats = available_features(gold.columns, PUBLIC_FEATURES)
    x, y, dates, used, ids = design_matrix(gold, label, feats)
    horizon = _label_horizon(label)
    engines: dict[str, Array] = {
        "reversal": oos_rank_scores(
            "reversal", cfg, x, y, dates, ids, horizon_bars=horizon, feature_names=used
        ),
        "classic_st": oos_rank_scores(
            "classic_st", cfg, x, y, dates, ids, horizon_bars=horizon, feature_names=used
        ),
        "ridge": oos_rank_scores(
            "ridge", cfg, x, y, dates, ids, horizon_bars=horizon, feature_names=used
        ),
        "combo_ic": oos_rank_scores(
            "combo_ic", cfg, x, y, dates, ids, horizon_bars=horizon, feature_names=used
        ),
        "anti_univ": nested_worst_univariate_scores(x, y, dates, cfg, horizon_bars=horizon),
        "random": random_date_scores(dates, seed=7),
    }
    rows = []
    cards = {}
    for name, scores in engines.items():
        for cost in (0.0, float(one_way_cost)):
            pair = pair_book_and_mirror(scores, y, dates, ids, k_frac=k_frac, one_way_cost=cost)
            key = f"{name}_c{cost:.4f}"
            cards[key] = pair
            row = summarize_pair(key, pair)
            rows.append(row)
            print(
                f"{key:24s}  book S={row['book_sharpe']:.3f}  "
                f"mirror S={row['mirror_sharpe']:.3f}  sum={row['sharpe_sum']:.3f}  "
                f"book tot={row['book_total_return']:.3f}  "
                f"mirror tot={row['mirror_total_return']:.3f}  "
                f"book DD={row['book_max_drawdown']:.3f}",
                flush=True,
            )
    receipt = {
        "catalog": "hedge_lab_analytics",
        "config": config_path,
        "label": label,
        "n_names": int(gold["security_id"].n_unique()),
        "n_rows": int(x.shape[0]),
        "features": used,
        "one_way_cost": float(one_way_cost),
        "k_frac": float(k_frac),
        "scheme": cfg.validation.scheme,
        "train_bars": int(cfg.validation.train_bars),
        "rows": rows,
        "execution_claim": "paper_backtest",
        "live_pnl_claim": False,
        "blend_weight": 0.0,
        "champion_alias": False,
        "note": (
            "Sign-flip identity on walk-forward public CS scores. "
            "Frictionless Sharpe is odd; costs are even. Searching for the worst "
            "train univariate and flipping it is the best-train univariate. "
            "A 5% DD halt cannot coexist with -150% total return. "
            "Not a live P&L claim."
        ),
    }
    root = cfg.data.root
    out = Path(root) / "metadata" / f"mirror_anti_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    public = Path("artifacts") / "hedge_lab" / "mirror_latest.json"
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    receipt["receipt_path"] = str(out)
    receipt["artifact_path"] = str(public)
    return receipt
