"""Ranking benches or ranking and calibration training.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.config.models import AppConfig
from quant_fund.metrics.cross_section import _date_keys, date_ic_series, decile_portfolios
from quant_fund.metrics.inference import overlap_aware_hac_lags, pairwise_diebold_mariano
from quant_fund.metrics.scoring import pearson_ic
from quant_fund.models.alpha import HistoricalMeanAlpha, RidgeAlpha
from quant_fund.models.ranking import (
    ORACLE_FEATURES,
    PUBLIC_FEATURES,
    RidgeRanker,
    available_features,
)
from quant_fund.pipeline.dataset import design_matrix
from quant_fund.pipeline.train import _fit_ranker, _label_horizon, _make_ranker, _predict_ranker
from quant_fund.validation.walk_forward import timestamp_ns, walk_forward

from .common import (
    _BANDIT_MAX_DATES,
    _aligned_col,
    _holdout,
    _public_features_in,
    _tail_date_mask,
)


def _policy_planted_oracle(
    frame: pl.DataFrame, dates: NDArray[Any], ids: NDArray[Any]
) -> tuple[NDArray[np.float64] | None, str | None]:
    for name in ("cs_z_planted_signal", "planted_signal"):
        col = _aligned_col(frame, dates, ids, name)
        if col is not None and np.isfinite(col).any():
            return col, name
    return None, None


def _public_leak_diagnostic(
    frame: pl.DataFrame,
    dates: NDArray[Any],
    ids: NDArray[Any],
    y: NDArray[np.float64],
) -> dict[str, Any]:
    """Date-level IC of public mom vs planted / next-day residual. SYNTHETIC."""
    mom_name = next(
        (
            c
            for c in ("cs_z_mom_20", "cs_pct_mom_20", "mom_20", "sector_relative_mom_20")
            if c in frame.columns
        ),
        None,
    )
    out: dict[str, Any] = {
        "leak_label": "SYNTHETIC",
        "leak_mom_feature": mom_name,
        "leak_ic_mom_vs_planted": float("nan"),
        "leak_ic_mom_vs_residual": float("nan"),
    }
    if mom_name is None:
        return out
    mom = _aligned_col(frame, dates, ids, mom_name)
    if mom is None:
        return out
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(mom) & np.isfinite(y)
    if int(mask.sum()) >= 10:
        ic_y = date_ic_series(mom[mask], y[mask], dates[mask], min_names=5)
        out["leak_ic_mom_vs_residual"] = float(ic_y.mean_pearson)
    planted, _ = _policy_planted_oracle(frame, dates, ids)
    if planted is not None:
        m2 = np.isfinite(mom) & np.isfinite(planted)
        if int(m2.sum()) >= 10:
            ic_p = date_ic_series(mom[m2], planted[m2], dates[m2], min_names=5)
            out["leak_ic_mom_vs_planted"] = float(ic_p.mean_pearson)
    return out


def _policy_topk_mean(y: NDArray[np.float64], scores: NDArray[np.float64], k: int) -> float:
    k = max(1, min(int(k), int(y.size)))
    idx = np.argsort(np.asarray(scores, dtype=float))[-k:]
    return float(np.mean(y[idx]))


def _policy_ridge_topk_rewards(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dates: NDArray[Any],
    *,
    top_k: int = 3,
    ridge_alpha: float = 1.0,
) -> tuple[NDArray[np.float64], list[str]]:
    """Causal expanding RidgeRanker top-k mean y on public features. Deterministic."""
    keys = _date_keys(dates)
    order = sorted(set(keys))
    min_n = max(int(top_k) * 2, 4)
    min_fit = max(int(x.shape[1]) + 2, min_n)
    rewards: list[float] = []
    kept: list[str] = []
    prefix_x: list[NDArray[np.float64]] = []
    prefix_y: list[NDArray[np.float64]] = []
    n_prefix = 0
    for key in order:
        row_mask = np.array([item == key for item in keys], dtype=bool)
        if int(row_mask.sum()) < min_n:
            continue
        xx, yy = x[row_mask], y[row_mask]
        finite = np.isfinite(yy) & np.isfinite(xx).all(axis=1)
        if int(finite.sum()) < min_n:
            continue
        xx, yy = xx[finite], yy[finite]
        if n_prefix >= min_fit:
            try:
                ranker = RidgeRanker(alpha=ridge_alpha)
                ranker.fit(np.concatenate(prefix_x, axis=0), np.concatenate(prefix_y, axis=0))
                scores = np.asarray(ranker.predict(xx), dtype=float)
            except (ValueError, np.linalg.LinAlgError):
                scores = (
                    np.where(np.isfinite(xx[:, 0]), xx[:, 0], 0.0)
                    if xx.shape[1]
                    else np.zeros(yy.size, dtype=float)
                )
        elif xx.shape[1] > 0:
            scores = np.where(np.isfinite(xx[:, 0]), xx[:, 0], 0.0)
        else:
            scores = np.zeros(yy.size, dtype=float)
        rewards.append(_policy_topk_mean(yy, scores, top_k))
        kept.append(key)
        prefix_x.append(xx)
        prefix_y.append(yy)
        n_prefix += int(yy.size)
    return np.asarray(rewards, dtype=float), kept


def _policy_paired_rewards(
    policy_dates: list[str],
    policy: NDArray[np.float64],
    ridge_dates: list[str],
    ridge: NDArray[np.float64],
    uniform: NDArray[np.float64],
    oracle_r: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    lookup = {d: i for i, d in enumerate(ridge_dates)}
    p_idx: list[int] = []
    r_idx: list[int] = []
    for i, d in enumerate(policy_dates):
        j = lookup.get(d)
        if j is not None:
            p_idx.append(i)
            r_idx.append(j)
    if p_idx:
        pi = np.asarray(p_idx)
        ri = np.asarray(r_idx)
        return policy[pi], ridge[ri], uniform[pi], oracle_r[pi]
    n = min(int(policy.size), int(ridge.size))
    return policy[:n], ridge[:n], uniform[:n], oracle_r[:n]


def _policy_bandit_metrics(
    *,
    policy: NDArray[np.float64],
    oracle_r: NDArray[np.float64],
    uniform: NDArray[np.float64],
    policy_dates: list[str],
    ridge: NDArray[np.float64],
    ridge_dates: list[str],
    used: list[str],
    oracle_definition: str,
    oracle_column: str | None,
    leak: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pol, rid, uni, ora = _policy_paired_rewards(
        policy_dates, policy, ridge_dates, ridge, uniform, oracle_r
    )
    gap = (pol - rid) if pol.size and rid.size else np.asarray([], dtype=float)
    out: dict[str, Any] = {
        "n_dates": int(policy.size),
        "mean_policy_reward": float(np.mean(policy)),
        "mean_oracle_reward": float(np.mean(oracle_r)),
        "mean_uniform_reward": float(np.mean(uniform)),
        "mean_ridge_reward": float(np.mean(rid)) if rid.size else float("nan"),
        "final_regret_vs_oracle": (
            float(np.cumsum(oracle_r - policy)[-1]) if policy.size else float("nan")
        ),
        "mean_regret_vs_oracle": float(np.mean(oracle_r - policy)),
        "mean_advantage_vs_uniform": float(np.mean(policy - uniform)),
        "mean_advantage_vs_ridge": (
            float(np.mean(pol - rid)) if pol.size and rid.size else float("nan")
        ),
        "policy_reward": pol.astype(float).tolist(),
        "ridge_reward": rid.astype(float).tolist(),
        "uniform_reward": uni.astype(float).tolist(),
        "oracle_reward_by_date": ora.astype(float).tolist(),
        "reward_gap_series": gap.astype(float).tolist(),
        "features": used,
        "oracle_definition": oracle_definition,
        "oracle_column": oracle_column,
        **leak,
    }
    if extra:
        out.update(extra)
    return out


def _policy_public_design(frame: pl.DataFrame, label: str) -> dict[str, Any]:
    """PUBLIC_FEATURES panel, shared date tail, planted-or-y-greedy oracle."""
    feats = _public_features_in(frame)
    if not feats or label not in frame.columns:
        return {}
    x, y, dates, used, ids = design_matrix(frame, label, feats)
    if x.size == 0:
        return {}
    mask = _tail_date_mask(dates, _BANDIT_MAX_DATES)
    x, y, dates, ids = x[mask], y[mask], dates[mask], ids[mask]
    oracle, oracle_col = _policy_planted_oracle(frame, dates, ids)
    return {
        "x": x,
        "y": y,
        "dates": dates,
        "ids": ids,
        "features": used,
        "oracle": oracle,
        "oracle_column": oracle_col,
        "oracle_definition": "planted" if oracle is not None else "y_greedy",
        "leak": _public_leak_diagnostic(frame, dates, ids, y),
    }


def oos_rank_scores(
    model_name: str,
    config: AppConfig,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dates: NDArray[Any],
    ids: NDArray[Any] | None = None,
    *,
    horizon_bars: int = 1,
    feature_names: list[str] | None = None,
) -> NDArray[np.float64]:
    """Purged walk-forward OOS scores. ``horizon_bars`` is the label horizon.

    Expanding vs rolling follows ``config.validation.scheme``. A 5- or 20-bar
    forward label reaches ``horizon_bars`` sessions past its decision date, so
    purging must use that horizon, not 1.
    """
    times = sorted(set(dates.tolist()))
    folds = walk_forward(
        times,
        config.validation,
        horizon_bars=int(horizon_bars),
        embargo_bars=config.embargo_bars(),
    )
    pred = np.full(len(y), np.nan, dtype=float)
    date_ns = timestamp_ns(dates)
    if not folds:
        cut = max(len(times) - 40, len(times) // 2)
        tr = np.isin(date_ns, timestamp_ns(times[:cut]))
        te = np.isin(date_ns, timestamp_ns(times[cut:]))
        model = _make_ranker(model_name, config)
        _fit_ranker(
            model,
            model_name,
            x[tr],
            y[tr],
            dates[tr],
            None if ids is None else ids[tr],
            features=feature_names,
        )
        pred[te] = _predict_ranker(
            model, model_name, x[te], dates[te], None if ids is None else ids[te]
        )
        return pred
    for fold in folds:
        tr = np.isin(date_ns, timestamp_ns(fold.train_times))
        te = np.isin(date_ns, timestamp_ns(fold.test_times))
        if not tr.any() or not te.any():
            continue
        model = _make_ranker(model_name, config)
        _fit_ranker(
            model,
            model_name,
            x[tr],
            y[tr],
            dates[tr],
            None if ids is None else ids[tr],
            features=feature_names,
        )
        pred[te] = _predict_ranker(
            model, model_name, x[te], dates[te], None if ids is None else ids[te]
        )
    return pred


def _paper_ranker_public_row(item: tuple[Any, ...], _seed: int) -> dict[str, Any] | None:
    """One paper ranker's public-feature OOS row. ``_seed`` is not consumed.

    Rankers draw from ``config.train.random_seed`` inside the model. Passing
    an index-dependent seed into the fit would change published scores.
    """
    (
        model_name,
        config,
        x_pub,
        y_pub,
        dates_pub,
        ids_pub,
        horizon,
        public_feats,
        n_buckets,
        hac,
    ) = item
    row_name = f"{model_name}_public"
    try:
        scores = oos_rank_scores(
            model_name,
            config,
            x_pub,
            y_pub,
            dates_pub,
            ids_pub,
            horizon_bars=horizon,
            feature_names=public_feats,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    mask = np.isfinite(scores) & np.isfinite(y_pub)
    if int(mask.sum()) < 5:
        return None
    ic = date_ic_series(scores[mask], y_pub[mask], dates_pub[mask], min_names=5, hac_lags=hac)
    dec = decile_portfolios(
        scores[mask],
        y_pub[mask],
        dates_pub[mask],
        n_buckets=n_buckets,
        min_names=5,
        hac_lags=hac,
    )
    return {
        "name": row_name,
        "feature_set": "public",
        "engine": model_name,
        "mean_ic": ic.mean_pearson,
        "mean_rank_ic": ic.mean_spearman,
        "t_ic": ic.t_pearson,
        "p_ic": ic.p_pearson,
        "icir": ic.icir_pearson,
        "n_dates": ic.n_dates,
        "n_folds": ic.n_dates,
        "decile_monotonicity": dec.monotonicity,
        "ls_mean": dec.mean_ls,
        "ls_t": dec.t_ls,
        "ls_p": dec.p_ls,
        "decile_means": dec.mean_returns,
        "ic_series": [float(v) for v in ic.pearson.tolist()],
        "ic_dates": [str(date) for date in ic.dates],
    }


def bench_ranking(frame: pl.DataFrame, config: AppConfig, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, str, list[str] | None, str | None]] = [
        ("oracle_raw", "oracle", None, "cs_z_planted_signal"),
        ("ridge_oracle", "oracle", ORACLE_FEATURES, None),
        ("ridge_public", "public", PUBLIC_FEATURES, None),
        ("ridge_combined", "combined", None, None),
    ]
    n_names = frame["security_id"].n_unique() if "security_id" in frame.columns else 8
    n_buckets = 5 if n_names < 20 else 10
    horizon = _label_horizon(label, default=1)
    n_dates_all = frame["event_time"].n_unique() if "event_time" in frame.columns else 0
    # Overlapping h-bar labels make the date IC series serially dependent.
    hac = overlap_aware_hac_lags(int(n_dates_all), int(horizon)) if n_dates_all else None
    for name, fset, wanted, raw_col in specs:
        if raw_col is not None:
            col = raw_col if raw_col in frame.columns else "planted_signal"
            if col not in frame.columns:
                continue
            sub = frame.select(["event_time", label, col]).drop_nulls()
            scores = sub[col].to_numpy().astype(float)
            y = sub[label].to_numpy().astype(float)
            dates = sub["event_time"].to_numpy()
        else:
            feats = available_features(frame.columns, wanted)
            if not feats:
                continue
            x, y, dates, used, _ = design_matrix(frame, label, feats)
            scores = oos_rank_scores(
                "ridge", config, x, y, dates, horizon_bars=horizon, feature_names=used
            )
        mask = np.isfinite(scores) & np.isfinite(y)
        ic = date_ic_series(scores[mask], y[mask], dates[mask], min_names=5, hac_lags=hac)
        dec = decile_portfolios(
            scores[mask], y[mask], dates[mask], n_buckets=n_buckets, min_names=5, hac_lags=hac
        )
        rows.append(
            {
                "name": name,
                "feature_set": fset,
                "mean_ic": ic.mean_pearson,
                "mean_rank_ic": ic.mean_spearman,
                "t_ic": ic.t_pearson,
                "p_ic": ic.p_pearson,
                "icir": ic.icir_pearson,
                "n_dates": ic.n_dates,
                "n_folds": ic.n_dates,  # date-level OOS folds for promotion stability
                "decile_monotonicity": dec.monotonicity,
                "ls_mean": dec.mean_ls,
                "ls_t": dec.t_ls,
                "ls_p": dec.p_ls,
                "decile_means": dec.mean_returns,
                # Negative IC as DM loss (higher IC ⇒ lower loss). Keep the
                # date keys so pairwise contrasts use only truly common dates.
                "ic_series": [float(x) for x in ic.pearson.tolist()],
                "ic_dates": [str(date) for date in ic.dates],
            }
        )
    public_feats = available_features(frame.columns, PUBLIC_FEATURES)
    if public_feats:
        x_pub, y_pub, dates_pub, _, ids_pub = design_matrix(frame, label, public_feats)
        n_dates_pub = len({str(d) for d in dates_pub.tolist()})
        # Tiny CI panels skip the paper universe. A serious panel (enough
        # dates and names for managed-portfolio estimators) always runs it.
        run_paper = n_dates_pub >= 80 and n_names >= 12
        if run_paper:
            # Paper rankers are independent, but on this machine a process
            # pool oversubscribed BLAS and made the sweep slower. They stay
            # in-process. ``_seed`` is ignored; models use ``random_seed``.
            for model_name in config.train.paper_rankers:
                row = _paper_ranker_public_row(
                    (
                        model_name,
                        config,
                        x_pub,
                        y_pub,
                        dates_pub,
                        ids_pub,
                        horizon,
                        public_feats,
                        n_buckets,
                        hac,
                    ),
                    0,
                )
                if row is not None:
                    rows.append(row)
    # Pairwise Diebold–Mariano on -IC series across rankers. Align by the
    # intersection of date keys, never by positional truncation: different
    # feature sets can have different missing-date patterns.
    ic_payloads = {
        str(r["name"]): dict(zip(r["ic_dates"], r["ic_series"], strict=True))
        for r in rows
        if r.get("ic_series") and r.get("ic_dates")
    }
    common_dates = (
        set.intersection(*(set(payload) for payload in ic_payloads.values()))
        if ic_payloads
        else set()
    )
    loss_map = {
        name: -np.asarray([payload[date] for date in sorted(common_dates)], dtype=float)
        for name, payload in ic_payloads.items()
    }
    if len(loss_map) >= 2 and common_dates:
        dm_rows = pairwise_diebold_mariano(loss_map)
        for r in rows:
            r["pairwise_dm"] = [d for d in dm_rows if d["a"] == r["name"] or d["b"] == r["name"]]
        rows.append(
            {
                "name": "_pairwise_dm_summary",
                "feature_set": "contrast",
                "pairwise_dm_all": dm_rows,
                "mean_ic": float("nan"),
                "n_dates": 0,
            }
        )
    return rows


def bench_alpha(frame: pl.DataFrame, config: AppConfig, label: str) -> dict[str, Any]:
    feats = _public_features_in(frame)
    if not feats:
        return {}
    x, y, dates, used, ids = design_matrix(frame, label, feats)
    if x.size == 0:
        return {}
    tr, te = _holdout(dates, horizon=_label_horizon(label))
    if not tr.any() or not te.any():
        return {}
    mean_m = HistoricalMeanAlpha().fit(x[tr], y[tr])
    ridge = RidgeAlpha(config.train.ridge_alpha).fit(x[tr], y[tr])
    p_mean = mean_m.predict(x[te])
    p_ridge = ridge.predict(x[te])
    yy = y[te]
    mse_mean = float(np.mean((p_mean - yy) ** 2))
    mse_ridge = float(np.mean((p_ridge - yy) ** 2))
    out: dict[str, Any] = {
        "mse_historical_mean": mse_mean,
        "mse_ridge": mse_ridge,
        "ic_ridge": pearson_ic(p_ridge, yy),
        "ic_mean": pearson_ic(p_mean, yy),
        "features": used,
        **_public_leak_diagnostic(frame, dates, ids, y),
    }
    planted, pcol = _policy_planted_oracle(frame, dates, ids)
    if planted is not None and pcol is not None:
        ic_p = date_ic_series(planted, y, dates, min_names=5)
        out["ic_oracle_planted"] = float(ic_p.mean_pearson)
        out["oracle_column_diagnostic"] = pcol
    return out


__all__ = [
    "bench_alpha",
    "bench_ranking",
    "oos_rank_scores",
]
