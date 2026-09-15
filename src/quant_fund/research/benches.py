"""Scientific family benches. Scores are proper rules, not Sharpe."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.config.models import AppConfig
from quant_fund.execution.almgren_chriss import (
    almgren_chriss_trajectory,
    expected_shortfall_ac,
    slice_trades,
    twap_trajectory,
)
from quant_fund.metrics.conformal import (
    assign_terciles,
    conditional_coverage,
    covered,
    set_metrics,
    worst_slice_coverage,
)
from quant_fund.metrics.cross_section import _date_keys, date_ic_series, decile_portfolios
from quant_fund.metrics.evalues import bench_e_coverage, e_process
from quant_fund.metrics.inference import diebold_mariano
from quant_fund.metrics.probability import (
    brier_score,
    expected_calibration_error,
    kupiec_pof,
    log_loss,
    pit_ks,
)
from quant_fund.metrics.scoring import (
    coverage,
    crps_from_quantiles,
    mean_pinball,
    pearson_ic,
    pit_values,
    qlike,
    quantile_crossing_rate,
)
from quant_fund.models.alpha import HistoricalMeanAlpha, RidgeAlpha
from quant_fund.models.conformal import (
    AdaptiveConformal,
    MondrianACI,
    MondrianCQR,
    SplitCQR,
    SplitOneSided,
)
from quant_fund.models.crc import ConformalRiskControl, bench_crc_var, loss_hit
from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
    LinearQuantileDistribution,
    ScaledGaussianDistribution,
    ScaledStudentTDistribution,
    fit_operational_wrappee,
    select_scaled_wrappee,
)
from quant_fund.models.jackknife_plus import JackknifePlus, jackknife_plus_coverage_level
from quant_fund.models.quantile_bandit import QuantileThompson
from quant_fund.models.ranking import (
    ORACLE_FEATURES,
    PUBLIC_FEATURES,
    RidgeRanker,
    available_features,
)
from quant_fund.models.regime import GaussianHMMRegime, SingleStateRegime, VolThresholdRegime
from quant_fund.models.rl import run_linucb_panel
from quant_fund.models.tail import DrawdownClassifier, HistoricalTail, ScaledHistoricalTail
from quant_fund.models.weighted_conformal import WeightedSplitCQR
from quant_fund.pipeline.dataset import design_matrix
from quant_fund.pipeline.train import _fit_ranker, _make_ranker
from quant_fund.portfolio.interval_risk import (
    bench_interval_caps,
    cap_from_interval,
    equal_weight_per_date,
    interval_refs,
)
from quant_fund.validation.walk_forward import walk_forward


def _holdout(n: int, frac: float = 0.3) -> tuple[slice, slice]:
    cut = max(int(n * (1.0 - frac)), n // 2)
    return slice(0, cut), slice(cut, n)


def _triple_split(n: int) -> tuple[slice, slice, slice]:
    """Chronological train / calibration / test. Calibration is never test."""
    if n < 30:
        a, b = max(n // 3, 1), max(2 * n // 3, 2)
        return slice(0, a), slice(a, b), slice(b, n)
    a = max(int(0.5 * n), 8)
    b = max(int(0.7 * n), a + 5)
    return slice(0, a), slice(a, b), slice(b, n)


def _aligned_col(
    frame: pl.DataFrame, dates: NDArray[Any], ids: NDArray[Any], name: str
) -> NDArray[np.float64] | None:
    if name not in frame.columns:
        return None
    sub = frame.select(["event_time", "security_id", name]).drop_nulls()
    lookup = {
        (d, i): float(v)
        for d, i, v in zip(
            _date_keys(sub["event_time"].to_numpy()),
            np.asarray(sub["security_id"].to_numpy()).astype(str),
            sub[name].to_numpy().astype(float),
            strict=False,
        )
    }
    out = np.array(
        [
            lookup.get((d, i), np.nan)
            for d, i in zip(_date_keys(dates), np.asarray(ids).astype(str), strict=True)
        ],
        dtype=float,
    )
    return out


def _scaled_fill(vol: NDArray[np.float64]) -> NDArray[np.float64]:
    finite = vol[np.isfinite(vol)]
    med = float(np.median(finite)) if finite.size else 1e-8
    if not np.isfinite(med) or med <= 0.0:
        med = 1e-8
    return np.where(np.isfinite(vol), vol, med)


def _abs_y_labels(y: NDArray[np.float64]) -> NDArray[Any]:
    terc = np.full(y.size, "mid", dtype=object)
    mag = np.abs(y)
    if mag.size >= 15:
        q1, q2 = np.nanquantile(mag, [1.0 / 3.0, 2.0 / 3.0])
        terc[mag <= q1] = "low_|y|"
        terc[mag > q2] = "high_|y|"
    return terc


def _public_features_in(frame: pl.DataFrame) -> list[str]:
    return available_features(list(frame.columns), PUBLIC_FEATURES)


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
) -> NDArray[np.float64]:
    times = sorted(set(dates.tolist()))
    folds = walk_forward(
        times, config.validation, horizon_bars=1, embargo_bars=config.embargo_bars()
    )
    pred = np.full(len(y), np.nan, dtype=float)
    if not folds:
        cut = max(len(times) - 40, len(times) // 2)
        tr = np.isin(dates, times[:cut])
        te = np.isin(dates, times[cut:])
        model = _make_ranker(model_name, config)
        _fit_ranker(model, model_name, x[tr], y[tr], dates[tr])
        pred[te] = model.predict(x[te])
        return pred
    for fold in folds:
        tr = np.isin(dates, np.array(fold.train_times, dtype=object))
        te = np.isin(dates, np.array(fold.test_times, dtype=object))
        if not tr.any() or not te.any():
            continue
        model = _make_ranker(model_name, config)
        _fit_ranker(model, model_name, x[tr], y[tr], dates[tr])
        pred[te] = model.predict(x[te])
    return pred


def bench_ranking(
    frame: pl.DataFrame, config: AppConfig, label: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, str, list[str] | None, str | None]] = [
        ("oracle_raw", "oracle", None, "cs_z_planted_signal"),
        ("ridge_oracle", "oracle", ORACLE_FEATURES, None),
        ("ridge_public", "public", PUBLIC_FEATURES, None),
        ("ridge_combined", "combined", None, None),
    ]
    n_names = frame["security_id"].n_unique() if "security_id" in frame.columns else 8
    n_buckets = 5 if n_names < 20 else 10
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
            x, y, dates, _, _ = design_matrix(frame, label, feats)
            scores = oos_rank_scores("ridge", config, x, y, dates)
        mask = np.isfinite(scores) & np.isfinite(y)
        ic = date_ic_series(scores[mask], y[mask], dates[mask], min_names=5)
        dec = decile_portfolios(
            scores[mask], y[mask], dates[mask], n_buckets=n_buckets, min_names=5
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
                "decile_monotonicity": dec.monotonicity,
                "ls_mean": dec.mean_ls,
                "ls_t": dec.t_ls,
                "ls_p": dec.p_ls,
                "decile_means": dec.mean_returns,
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
    tr, te = _holdout(x.shape[0])
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


def bench_volatility(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    label = config.train.volatility_target
    if label not in frame.columns:
        cands = [c for c in frame.columns if c.startswith("future_realized_var")]
        if not cands:
            return {}
        label = cands[0]
    need = [c for c in [label, "vol_20", "vol_ewma"] if c in frame.columns]
    if label not in need or "vol_20" not in need:
        return {}
    sub = frame.select(["event_time", *need]).drop_nulls().sort("event_time")
    y = sub[label].to_numpy().astype(float)
    roll = np.clip(sub["vol_20"].to_numpy().astype(float) ** 2, config.train.qlike_floor, None)
    if "vol_ewma" in sub.columns:
        ewma = np.clip(sub["vol_ewma"].to_numpy().astype(float) ** 2, config.train.qlike_floor, None)
    else:
        ewma = roll
    _, te = _holdout(y.size)
    yy = np.clip(y[te], config.train.qlike_floor, None)
    q_roll = qlike(yy, roll[te], config.train.qlike_floor)
    q_ewma = qlike(yy, ewma[te], config.train.qlike_floor)
    loss_e = (np.sqrt(yy) - np.sqrt(ewma[te])) ** 2
    loss_r = (np.sqrt(yy) - np.sqrt(roll[te])) ** 2
    dm = diebold_mariano(loss_e, loss_r, name_a="ewma", name_b="rolling")
    return {
        "qlike_ewma": q_ewma,
        "qlike_rolling": q_roll,
        "dm_preferred": dm.preferred,
        "dm_p": dm.p_value,
        "dm_stat": dm.statistic,
    }


def _distribution_horizon_scores(
    frame: pl.DataFrame, label: str, taus: list[float]
) -> dict[str, Any] | None:
    if label not in frame.columns:
        return None
    x, y, dates, _feats, ids = design_matrix(frame, label)
    if x.size == 0:
        return None
    tr, te = _holdout(x.shape[0])
    g = GaussianDistribution(taus).fit(x[tr], y[tr])
    e = EmpiricalDistribution(taus).fit(x[tr], y[tr])
    qg, qe = g.predict(x[te]), e.predict(x[te])
    yy = y[te]
    mid = taus.index(min(taus, key=lambda t: abs(t - 0.5)))
    lo_i = 0
    hi_i = len(taus) - 1
    pits = pit_values(yy, qg, np.array(taus))
    ks, ks_p = pit_ks(pits)
    out: dict[str, Any] = {
        "target": label,
        "pinball_gaussian": mean_pinball(yy, qg[:, mid], taus[mid]),
        "pinball_empirical": mean_pinball(yy, qe[:, mid], taus[mid]),
        "crps_gaussian": crps_from_quantiles(yy, qg, np.array(taus)),
        "crps_empirical": crps_from_quantiles(yy, qe, np.array(taus)),
        "coverage_gaussian": coverage(yy, qg[:, lo_i], qg[:, hi_i]),
        "nominal_coverage": taus[hi_i] - taus[lo_i],
        "crossing_gaussian": quantile_crossing_rate(qg, np.array(taus)),
        "pit_ks": ks,
        "pit_ks_p": ks_p,
    }
    vol = _aligned_col(frame, dates, ids, "vol_20")
    if vol is not None and np.isfinite(vol[tr]).any() and np.isfinite(vol[te]).any():
        sc = _scaled_fill(vol)
        sg = ScaledGaussianDistribution(taus).fit(y[tr], sc[tr])
        qs = sg.predict(sc[te])
        pits_s = pit_values(yy, qs, np.array(taus))
        ks_s, ks_sp = pit_ks(pits_s)
        out["coverage_scaled_gaussian"] = coverage(yy, qs[:, lo_i], qs[:, hi_i])
        out["pinball_scaled_gaussian"] = mean_pinball(yy, qs[:, mid], taus[mid])
        out["crps_scaled_gaussian"] = crps_from_quantiles(yy, qs, np.array(taus))
        out["pit_ks_scaled"] = ks_s
        out["pit_ks_p_scaled"] = ks_sp
        st = ScaledStudentTDistribution(taus).fit(y[tr], sc[tr])
        qt = st.predict(sc[te])
        pits_t = pit_values(yy, qt, np.array(taus))
        ks_t, ks_tp = pit_ks(pits_t)
        out["coverage_scaled_student_t"] = coverage(yy, qt[:, lo_i], qt[:, hi_i])
        out["pinball_scaled_student_t"] = mean_pinball(yy, qt[:, mid], taus[mid])
        out["crps_scaled_student_t"] = crps_from_quantiles(yy, qt, np.array(taus))
        out["pit_ks_scaled_t"] = ks_t
        out["pit_ks_p_scaled_t"] = ks_tp
        out["nu_scaled_student_t"] = float(st.nu)
        wname, _ = fit_operational_wrappee(
            taus, y[tr], sc[tr], nominal_coverage=float(out["nominal_coverage"])
        )
        out["wrappee"] = wname
    return out


def bench_distribution(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    label = config.train.distribution_target
    if label not in frame.columns:
        cands = [c for c in frame.columns if c.startswith("future_log_return")]
        if not cands:
            return {}
        label = cands[0]
    taus = config.quantiles.levels
    primary = _distribution_horizon_scores(frame, label, taus)
    if not primary:
        return {}
    out = dict(primary)
    by_horizon: dict[str, Any] = {label: dict(primary)}
    for lab in ("future_log_return_1", "future_log_return_5"):
        if lab in frame.columns and lab not in by_horizon:
            scores = _distribution_horizon_scores(frame, lab, taus)
            if scores:
                by_horizon[lab] = scores
    out["by_horizon"] = by_horizon
    one = by_horizon.get("future_log_return_1")
    if isinstance(one, dict):
        out["wrappee_1d"] = one.get("wrappee")
        out["pit_ks_p_scaled_1d"] = one.get("pit_ks_p_scaled")
        out["pit_ks_p_scaled_t_1d"] = one.get("pit_ks_p_scaled_t")
        out["coverage_scaled_gaussian_1d"] = one.get("coverage_scaled_gaussian")
        out["coverage_scaled_student_t_1d"] = one.get("coverage_scaled_student_t")
    five = by_horizon.get("future_log_return_5")
    if isinstance(five, dict):
        out["wrappee_5d"] = five.get("wrappee")
    return out


def bench_regime(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    cols = [c for c in ["mkt_ret_1", "mkt_vol_20", "cs_dispersion", "breadth"] if c in frame.columns]
    if len(cols) < 2:
        return {}
    sub = frame.select(["event_time", *cols]).unique("event_time").drop_nulls().sort("event_time")
    x = sub.select(cols).to_numpy().astype(float)
    if x.shape[0] < 40:
        return {}
    tr, te = _holdout(x.shape[0], 0.25)
    hmm = GaussianHMMRegime(config.train.n_hmm_states, config.train.random_seed).fit(x[tr])
    thr = VolThresholdRegime().fit(x[tr])
    one = SingleStateRegime().fit(x[tr])
    extra = hmm.aic_bic(x[tr])
    xx_te = hmm.scaler.transform(np.where(np.isfinite(x[te]), x[te], 0.0))
    holdout_ll = float(hmm.model.score(xx_te) / max(xx_te.shape[0], 1))
    _ = (thr, one)
    return {
        **extra,
        "holdout_avg_ll": holdout_ll,
        "labels": {str(k): v for k, v in hmm.labels.items()},
    }


def bench_tail(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    label = next((c for c in frame.columns if c.startswith("future_return_1")), None)
    if label is None:
        return {}
    x, y, dates, _feats, ids = design_matrix(frame, label)
    if x.size == 0:
        return {}
    tr, te = _holdout(x.shape[0])
    hist = HistoricalTail(0.95).fit(x[tr], y[tr])
    var, es = hist.predict_var_es()
    losses = -y[te]
    hits = (losses >= var).astype(float)
    rate, lr, p = kupiec_pof(hits, 0.05)
    tail = losses[losses >= var] if np.isfinite(var) else np.array([])
    realized_es = float(np.mean(tail)) if tail.size else float("nan")
    out: dict[str, Any] = {
        "var_95": var,
        "es_95": es,
        "hit_rate_unscaled": rate,
        "hit_rate": rate,
        "nominal_hit_rate": 0.05,
        "kupiec_lr_unscaled": lr,
        "kupiec_p_unscaled": p,
        "kupiec_lr": lr,
        "kupiec_p": p,
        "realized_es": realized_es,
    }
    vol = _aligned_col(frame, dates, ids, "vol_20")
    if vol is not None and np.isfinite(vol[tr]).any() and np.isfinite(vol[te]).any():
        med = float(np.nanmedian(vol[np.isfinite(vol)]))
        sc = np.where(np.isfinite(vol), vol, med)
        scaled = ScaledHistoricalTail(0.95).fit(y[tr], sc[tr])
        var_s, es_s = scaled.predict_var_es(sc[te])
        hits_s = (losses >= var_s).astype(float)
        rate_s, lr_s, p_s = kupiec_pof(hits_s, 0.05)
        out["var_95_scaled"] = float(np.mean(var_s))
        out["es_95_scaled"] = float(np.mean(es_s))
        out["hit_rate"] = rate_s
        out["kupiec_lr"] = lr_s
        out["kupiec_p"] = p_s
        out["hit_rate_scaled"] = rate_s
        out["kupiec_p_scaled"] = p_s
        out["kupiec_lr_scaled"] = lr_s
    return out


def bench_drawdown(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    ev = next((c for c in frame.columns if c.startswith("future_tail_event")), None)
    dd = next((c for c in frame.columns if c.startswith("future_max_drawdown")), None)
    if ev is None or dd is None:
        return {}
    feats = _public_features_in(frame)
    if not feats:
        return {}
    x, y_ev, dates, used, ids = design_matrix(frame, ev, feats)
    if x.size == 0:
        return {}
    y_dd = _aligned_col(frame, dates, ids, dd)
    tr, te = _holdout(x.shape[0])
    clf = DrawdownClassifier().fit(x[tr], y_ev[tr])
    p = clf.predict_proba(x[te])
    yy = (y_ev[te] > 0.5).astype(float)
    baseline = np.full_like(p, float(np.mean(y_ev[tr] > 0.5)))
    if y_dd is not None:
        mae = float(np.mean(np.abs(y_dd[te] - np.nanmean(y_dd[tr]))))
    else:
        mae = float("nan")
    te_keys = _date_keys(dates[te])
    brier_clf_d: list[float] = []
    brier_base_d: list[float] = []
    for key in sorted(set(te_keys)):
        m = np.array([d == key for d in te_keys], dtype=bool)
        if int(m.sum()) < 2:
            continue
        brier_clf_d.append(brier_score(p[m], yy[m]))
        brier_base_d.append(brier_score(baseline[m], yy[m]))
    return {
        "brier": brier_score(p, yy),
        "brier_base_rate": brier_score(baseline, yy),
        "log_loss": log_loss(p, yy),
        "ece": expected_calibration_error(p, yy),
        "mae_drawdown_mean": mae,
        "event_rate": float(np.mean(yy)),
        "features": used,
        "n_dates": len(brier_clf_d),
        "brier_by_date": brier_clf_d,
        "brier_base_by_date": brier_base_d,
    }


def bench_liquidity(frame: pl.DataFrame) -> dict[str, Any]:
    cols = [c for c in ["amihud", "ret_1", "adv", "volume"] if c in frame.columns]
    out: dict[str, Any] = {}
    if "amihud" in cols and "ret_1" in cols:
        sub = frame.select(["amihud", "ret_1"]).drop_nulls()
        a = sub["amihud"].to_numpy().astype(float)
        r = np.abs(sub["ret_1"].to_numpy().astype(float))
        out["corr_amihud_abs_ret"] = pearson_ic(a, r)
    ac = almgren_chriss_trajectory(
        1.0, 5, sigma=0.02, eta=1e-4, gamma=1e-5, risk_aversion=1e-3
    )
    tw = twap_trajectory(1.0, 5)
    ac_c = expected_shortfall_ac(
        ac, slice_trades(ac), arrival=100.0, eta=1e-4, gamma=1e-5, sigma=0.02
    )
    tw_c = expected_shortfall_ac(
        tw, slice_trades(tw), arrival=100.0, eta=1e-4, gamma=1e-5, sigma=0.02
    )
    out["ac_expected_is"] = ac_c["expected_is"]
    out["twap_expected_is"] = tw_c["expected_is"]
    out["ac_vs_twap"] = ac_c["expected_is"] - tw_c["expected_is"]
    return out


def bench_rl(frame: pl.DataFrame, label: str) -> dict[str, Any]:
    design = _policy_public_design(frame, label)
    if not design:
        return {}
    x, y, dates = design["x"], design["y"], design["dates"]
    trace = run_linucb_panel(x, y, dates, oracle=design["oracle"], top_k=3, alpha=1.0, seed=7)
    if trace.policy_reward.size < 5:
        return {}
    ridge, ridge_dates = _policy_ridge_topk_rewards(x, y, dates, top_k=3)
    return _policy_bandit_metrics(
        policy=trace.policy_reward,
        oracle_r=trace.oracle_reward,
        uniform=trace.uniform_reward,
        policy_dates=trace.dates,
        ridge=ridge,
        ridge_dates=ridge_dates,
        used=design["features"],
        oracle_definition=design["oracle_definition"],
        oracle_column=design["oracle_column"],
        leak=design["leak"],
    )


def bench_conformal(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    """CQR vs raw Gaussian vs ACI. Coverage and width only."""
    split = _gaussian_interval_split(frame, config)
    if split is None:
        return {}
    label = str(split["label"])
    alpha = float(split["alpha"])
    x = split["x"]
    y = split["y"]
    dates = split["dates"]
    tr = split["tr"]
    cal = split["cal"]
    te = split["te"]
    q_cal = split["q_cal"]
    q_te = split["q_te"]
    q_raw_cal = split["q_raw_cal"]
    q_raw_te = split["q_raw_te"]
    q_sg_te = split["q_sg_te"]
    q_st_te = split["q_st_te"]
    scale = np.asarray(split["covariate"], dtype=float)
    covariate = scale
    cov_name = str(split.get("cov_name", "vol_20"))
    wrappee_name = str(split.get("wrappee", "scaled_gaussian"))
    taus = [alpha / 2.0, 1.0 - alpha / 2.0]
    raw = set_metrics(y[te], q_raw_te[:, 0], q_raw_te[:, 1])
    sg_m = set_metrics(y[te], q_sg_te[:, 0], q_sg_te[:, 1])
    st_m = set_metrics(y[te], q_st_te[:, 0], q_st_te[:, 1])
    scaled_raw = set_metrics(y[te], q_te[:, 0], q_te[:, 1])
    cqr_raw_m = SplitCQR(alpha).calibrate(y[cal], q_raw_cal[:, 0], q_raw_cal[:, 1])
    lo_rr, hi_rr = cqr_raw_m.predict_sets(q_raw_te[:, 0], q_raw_te[:, 1])
    cqr_raw_metrics = set_metrics(y[te], lo_rr, hi_rr)
    aci_raw = AdaptiveConformal(alpha=alpha, gamma=0.05, score_window=400)
    aci_raw.initialize(y[cal], q_raw_cal[:, 0], q_raw_cal[:, 1])
    path_raw = aci_raw.run(y[te], q_raw_te[:, 0], q_raw_te[:, 1], dates[te])
    aci_raw_metrics = set_metrics(y[te], path_raw.lower, path_raw.upper)
    cqr = SplitCQR(alpha).calibrate(y[cal], q_cal[:, 0], q_cal[:, 1])
    lo_c, hi_c = cqr.predict_sets(q_te[:, 0], q_te[:, 1])
    cqr_m = set_metrics(y[te], lo_c, hi_c)
    qr_m: dict[str, float] | None = None
    if 80 <= int(tr.stop - tr.start) <= 4000:
        try:
            qr = LinearQuantileDistribution(taus).fit(x[tr], y[tr])
            qqc = qr.predict(x[cal])
            qqt = qr.predict(x[te])
            cqr_qr = SplitCQR(alpha).calibrate(y[cal], qqc[:, 0], qqc[:, 1])
            lo_q, hi_q = cqr_qr.predict_sets(qqt[:, 0], qqt[:, 1])
            qm = set_metrics(y[te], lo_q, hi_q)
            qr_m = {"coverage": qm.coverage, "mean_width": qm.mean_width, "qhat": cqr_qr.qhat}
        except Exception:
            qr_m = None
    aci = AdaptiveConformal(alpha=alpha, gamma=0.05, score_window=400)
    aci.initialize(y[cal], q_cal[:, 0], q_cal[:, 1])
    path = aci.run(y[te], q_te[:, 0], q_te[:, 1], dates[te])
    aci_m = set_metrics(y[te], path.lower, path.upper)
    misses = 1.0 - path.covered
    misses = misses[np.isfinite(misses)]
    miss_rate, lr, kp = kupiec_pof(misses, alpha)
    terc = _abs_y_labels(y[te])
    cond = conditional_coverage(y[te], path.lower, path.upper, terc)
    naci = AdaptiveConformal(alpha=alpha, gamma=0.05, score_window=400)
    naci.initialize(y[cal], q_cal[:, 0], q_cal[:, 1], scale[cal])
    npath = naci.run(y[te], q_te[:, 0], q_te[:, 1], dates[te], scale[te])
    naci_m = set_metrics(y[te], npath.lower, npath.upper)
    ncond = conditional_coverage(y[te], npath.lower, npath.upper, terc)
    cuts_src = covariate[cal]
    finite_cuts = cuts_src[np.isfinite(cuts_src)]
    if finite_cuts.size >= 6:
        frozen = np.nanquantile(finite_cuts, [1.0 / 3.0, 2.0 / 3.0])
    else:
        frozen = None
    lab_all, _cuts = assign_terciles(covariate, frozen, prefix=cov_name)
    mondrian = MondrianCQR(alpha).calibrate(
        y[cal], q_cal[:, 0], q_cal[:, 1], lab_all[cal], scale[cal]
    )
    mlo, mhi = mondrian.predict_sets(q_te[:, 0], q_te[:, 1], lab_all[te], scale[te])
    mcqr_m = set_metrics(y[te], mlo, mhi)
    maci = MondrianACI(alpha=alpha, gamma=0.05, score_window=400)
    maci.initialize(y[cal], q_cal[:, 0], q_cal[:, 1], lab_all[cal], scale[cal])
    mpath = maci.run(y[te], q_te[:, 0], q_te[:, 1], dates[te], lab_all[te], scale[te])
    maci_m = set_metrics(y[te], mpath.lower, mpath.upper)
    m_x = conditional_coverage(y[te], mpath.lower, mpath.upper, lab_all[te])
    m_y = conditional_coverage(y[te], mpath.lower, mpath.upper, terc)
    m_miss = 1.0 - mpath.covered
    m_miss = m_miss[np.isfinite(m_miss)]
    m_rate, m_lr, m_kp = kupiec_pof(m_miss, alpha)
    high_key = next((k for k in m_x if k.startswith("high_")), None)
    high_hits = None
    high_kupiec: tuple[float, float, float] = (float("nan"), float("nan"), float("nan"))
    if high_key is not None:
        sel_h = lab_all[te] == high_key
        if int(sel_h.sum()) >= 10:
            hh = (1.0 - mpath.covered[sel_h]).astype(float)
            high_kupiec = kupiec_pof(hh, alpha)
            high_hits = float(m_x.get(high_key, float("nan")))
    hmm_cond: dict[str, float] = {}
    rcols = [c for c in ["mkt_ret_1", "mkt_vol_20", "cs_dispersion", "breadth"] if c in frame.columns]
    if len(rcols) >= 2:
        try:
            rsub = (
                frame.select(["event_time", *rcols]).unique("event_time").drop_nulls().sort("event_time")
            )
            rx = rsub.select(rcols).to_numpy().astype(float)
            rdates = rsub["event_time"].to_numpy()
            if rx.shape[0] >= 40:
                rtr, _ = _holdout(rx.shape[0], 0.3)
                hmm = GaussianHMMRegime(
                    min(int(config.train.n_hmm_states), 3), config.train.random_seed
                ).fit(rx[rtr])
                xx_all = hmm.scaler.transform(np.where(np.isfinite(rx), rx, 0.0))
                states = np.asarray(hmm.model.predict(xx_all))
                date_state = {
                    _date_keys(np.array([d]))[0]: int(s)
                    for d, s in zip(rdates, states, strict=False)
                }
                te_keys = _date_keys(dates[te])
                labs = np.array(
                    [
                        hmm.labels.get(date_state[k], str(date_state[k]))
                        if k in date_state
                        else "unknown"
                        for k in te_keys
                    ],
                    dtype=object,
                )
                hmm_cond = conditional_coverage(y[te], path.lower, path.upper, labs)
        except Exception:
            hmm_cond = {}
    # one-sided conformal tail: same scaled wrappee as CRC (H11 is CRC)
    tail_out: dict[str, Any] = {}
    try:
        losses_cal = -y[cal]
        losses_te = -y[te]
        b_cal = -q_cal[:, 0]
        b_te = -q_te[:, 0]
        one = SplitOneSided(0.05).calibrate(losses_cal, b_cal)
        bound = one.predict_bound(b_te)
        hit = (losses_te > bound).astype(float)
        rate, tlr, tp = kupiec_pof(hit, 0.05)
        tail_out = {
            "var_hit_rate": rate,
            "nominal": 0.05,
            "kupiec_p": tp,
            "kupiec_lr": tlr,
            "qhat": one.qhat,
            "wrappee": wrappee_name,
            "note": "same scaled wrappee as CRC; H11 is CRC",
        }
    except Exception:
        tail_out = {}
    dd_out: dict[str, Any] = {}
    dd_lab = next((c for c in frame.columns if c.startswith("future_max_drawdown")), None)
    if dd_lab is not None:
        xd, yd, d_dates, _, d_ids = design_matrix(frame, dd_lab)
        if xd.shape[0] >= 40:
            dtr, dcal, dte = _triple_split(xd.shape[0])
            vol_dd = _aligned_col(frame, d_dates, d_ids, "vol_20")
            if vol_dd is not None and np.isfinite(vol_dd).any():
                sc_dd = _scaled_fill(vol_dd)
                dd_name, dd_model = select_scaled_wrappee(
                    taus,
                    yd[dtr],
                    sc_dd[dtr],
                    yd[dcal],
                    sc_dd[dcal],
                    nominal_coverage=1.0 - alpha,
                )
                qdc = dd_model.predict(sc_dd[dcal])
                qdt = dd_model.predict(sc_dd[dte])
                cqr_dd = SplitCQR(alpha).calibrate(yd[dcal], qdc[:, 0], qdc[:, 1])
                lo_d, hi_d = cqr_dd.predict_sets(qdt[:, 0], qdt[:, 1])
                dm = set_metrics(yd[dte], lo_d, hi_d)
                dd_out = {
                    "kind": "scaled_interval",
                    "wrappee": dd_name,
                    "coverage": dm.coverage,
                    "mean_width": dm.mean_width,
                    "qhat": cqr_dd.qhat,
                }
            else:
                mu = float(np.nanmean(yd[dtr]))
                lo_cal = np.full(yd[dcal].shape, mu)
                hi_cal = np.full(yd[dcal].shape, mu)
                cqr_dd = SplitCQR(alpha).calibrate(yd[dcal], lo_cal, hi_cal)
                lo_d, hi_d = cqr_dd.predict_sets(
                    np.full(yd[dte].shape, mu), np.full(yd[dte].shape, mu)
                )
                dm = set_metrics(yd[dte], lo_d, hi_d)
                dd_out = {
                    "kind": "cqr_around_mean",
                    "note": "cqr_around_mean diagnostic, not a drawdown model",
                    "coverage": dm.coverage,
                    "mean_width": dm.mean_width,
                    "qhat": cqr_dd.qhat,
                }
    abs_y_lift = (
        float(m_y["high_|y|"] - cond["high_|y|"])
        if "high_|y|" in m_y and "high_|y|" in cond
        else None
    )
    out: dict[str, Any] = {
        "target": label,
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "wrappee": wrappee_name,
        "nu": split.get("nu"),
        "gaussian_raw": {
            "coverage": raw.coverage,
            "mean_width": raw.mean_width,
            "n": raw.n,
        },
        "scaled_gaussian": {
            "coverage": sg_m.coverage,
            "mean_width": sg_m.mean_width,
            "n": sg_m.n,
        },
        "scaled_student_t": {
            "coverage": st_m.coverage,
            "mean_width": st_m.mean_width,
            "n": st_m.n,
            "nu": split.get("nu_student"),
        },
        "scaled": {
            "family": wrappee_name,
            "coverage": scaled_raw.coverage,
            "mean_width": scaled_raw.mean_width,
            "n": scaled_raw.n,
            "nu": split.get("nu"),
        },
        "cqr_raw": {
            "coverage": cqr_raw_metrics.coverage,
            "mean_width": cqr_raw_metrics.mean_width,
            "median_width": cqr_raw_metrics.median_width,
            "qhat": cqr_raw_m.qhat,
            "n": cqr_raw_metrics.n,
        },
        "aci_raw": {
            "coverage": aci_raw_metrics.coverage,
            "mean_width": aci_raw_metrics.mean_width,
            "median_width": aci_raw_metrics.median_width,
            "n": aci_raw_metrics.n,
        },
        "cqr": {
            "coverage": cqr_m.coverage,
            "mean_width": cqr_m.mean_width,
            "median_width": cqr_m.median_width,
            "qhat": cqr.qhat,
            "n": cqr_m.n,
        },
        "aci": {
            "coverage": aci_m.coverage,
            "mean_width": aci_m.mean_width,
            "median_width": aci_m.median_width,
            "mean_alpha_t": float(np.mean(path.alpha_t)) if path.alpha_t.size else float("nan"),
            "final_alpha_t": float(path.alpha_t[-1]) if path.alpha_t.size else float("nan"),
            "miss_rate": miss_rate,
            "kupiec_lr": lr,
            "kupiec_p": kp,
            "n": aci_m.n,
            "n_dates": int(path.alpha_t.size),
        },
        "aci_conditional_coverage": cond,
        "aci_hmm_conditional_coverage": hmm_cond,
        "normalized_aci": {
            "coverage": naci_m.coverage,
            "mean_width": naci_m.mean_width,
            "median_width": naci_m.median_width,
            "scale": cov_name,
            "n": naci_m.n,
        },
        "mondrian_cqr": {
            "coverage": mcqr_m.coverage,
            "mean_width": mcqr_m.mean_width,
            "qhat": mondrian.qhat,
            "n": mcqr_m.n,
        },
        "mondrian_aci": {
            "coverage": maci_m.coverage,
            "mean_width": maci_m.mean_width,
            "median_width": maci_m.median_width,
            "miss_rate": m_rate,
            "kupiec_lr": m_lr,
            "kupiec_p": m_kp,
            "covariate": cov_name,
            "x_coverage": m_x,
            "worst_x_coverage": worst_slice_coverage(m_x),
            "high_x_coverage": high_hits,
            "high_x_kupiec_p": high_kupiec[2],
            "high_x_kupiec_lr": high_kupiec[1],
            "n": maci_m.n,
        },
        "selected_slice_abs_y": {
            "note": "selected |Y| slice; not a conformal X-validity guarantee",
            "aci": cond,
            "mondrian": m_y,
            "normalized_aci_high_|y|": ncond.get("high_|y|"),
            "high_|y|_coverage": m_y.get("high_|y|"),
            "high_|y|_lift_vs_aci": abs_y_lift,
        },
        "tail_conformal": tail_out,
        "drawdown_conformal": dd_out,
    }
    if qr_m is not None:
        out["cqr_linear_qr"] = qr_m
    return out


_JP_MAX_CAL = 400
_JP_MAX_TEST = 400
_EV_MAX_DATES = 120
_BANDIT_MAX_DATES = 80


def _interval_label(frame: pl.DataFrame) -> str | None:
    if "future_log_return_1" in frame.columns:
        return "future_log_return_1"
    cands = [c for c in frame.columns if c.startswith("future_log_return")]
    if not cands:
        cands = [c for c in frame.columns if c.startswith("future_idio_return")]
    return cands[0] if cands else None


def _tail_date_mask(dates: NDArray[Any], max_dates: int) -> NDArray[np.bool_]:
    keys = _date_keys(dates)
    uniq = sorted(set(keys))
    if len(uniq) <= max_dates:
        return np.ones(len(keys), dtype=bool)
    keep = set(uniq[-max_dates:])
    return np.asarray([k in keep for k in keys], dtype=bool)


def _even_take(*arrays: NDArray[Any], n: int) -> tuple[NDArray[Any], ...]:
    """Evenly spaced rows across a window. Does not keep only the high-vol tail."""
    m = int(arrays[0].shape[0])
    if m <= n:
        return arrays
    idx = np.linspace(0, m - 1, n, dtype=int)
    return tuple(np.asarray(a)[idx] for a in arrays)


def _vol_or_width(
    frame: pl.DataFrame,
    dates: NDArray[Any],
    ids: NDArray[Any],
    q_cal: NDArray[np.float64],
    q_te: NDArray[np.float64],
    cal: slice,
    te: slice,
    n: int,
) -> tuple[NDArray[np.float64], str]:
    vol = _aligned_col(frame, dates, ids, "vol_20")
    if vol is not None and np.isfinite(vol).any():
        return _scaled_fill(vol), "vol_20"
    scale = np.full(n, 1e-8)
    scale[cal] = np.maximum(q_cal[:, 1] - q_cal[:, 0], 1e-8)
    scale[te] = np.maximum(q_te[:, 1] - q_te[:, 0], 1e-8)
    return scale, "pred_width"


def _gaussian_interval_split(
    frame: pl.DataFrame, config: AppConfig, *, alpha: float = 0.10
) -> dict[str, Any] | None:
    """Train / cal / test split with operational scaled wrappee plus raw Gaussian bands."""
    _ = config
    label = _interval_label(frame)
    if label is None:
        return None
    x, y, dates, _feats, ids = design_matrix(frame, label)
    if x.shape[0] < 40:
        return None
    taus = [alpha / 2.0, 1.0 - alpha / 2.0]
    tr, cal, te = _triple_split(x.shape[0])
    gauss = GaussianDistribution(taus).fit(x[tr], y[tr])
    q_raw_cal = gauss.predict(x[cal])
    q_raw_te = gauss.predict(x[te])
    covariate, cov_name = _vol_or_width(frame, dates, ids, q_raw_cal, q_raw_te, cal, te, y.size)
    wrappee_name, wrappee = select_scaled_wrappee(
        taus,
        y[tr],
        covariate[tr],
        y[cal],
        covariate[cal],
        nominal_coverage=1.0 - alpha,
    )
    q_cal = wrappee.predict(covariate[cal])
    q_te = wrappee.predict(covariate[te])
    sg = ScaledGaussianDistribution(taus).fit(y[tr], covariate[tr])
    st = ScaledStudentTDistribution(taus).fit(y[tr], covariate[tr])
    nu = float(getattr(wrappee, "nu", float("nan"))) if wrappee_name == "scaled_student_t" else None
    return {
        "label": label,
        "alpha": alpha,
        "y": y,
        "x": x,
        "dates": dates,
        "ids": ids,
        "tr": tr,
        "cal": cal,
        "te": te,
        "q_cal": q_cal,
        "q_te": q_te,
        "q_raw_cal": q_raw_cal,
        "q_raw_te": q_raw_te,
        "q_sg_te": sg.predict(covariate[te]),
        "q_st_te": st.predict(covariate[te]),
        "covariate": covariate,
        "cov_name": cov_name,
        "wrappee": wrappee_name,
        "nu": nu,
        "nu_student": float(st.nu),
    }


def bench_evalues(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Anytime-valid miss e-process on ACI sets from the research panel."""
    split = _gaussian_interval_split(frame, config)
    if split is None:
        return {}
    alpha = float(split["alpha"])
    y = split["y"]
    dates = split["dates"]
    cal = split["cal"]
    te = split["te"]
    q_cal = split["q_cal"]
    q_te = split["q_te"]
    mask = _tail_date_mask(dates[te], _EV_MAX_DATES)
    y_te = y[te][mask]
    q_te_m = q_te[mask]
    d_te = dates[te][mask]
    if y_te.size < 10:
        return {}
    aci = AdaptiveConformal(alpha=alpha, gamma=0.05, score_window=400)
    aci.initialize(y[cal], q_cal[:, 0], q_cal[:, 1])
    path = aci.run(y_te, q_te_m[:, 0], q_te_m[:, 1], d_te)
    flags = path.covered[np.isfinite(path.covered)]
    ev = bench_e_coverage(flags, alpha)
    misses = 1.0 - flags
    e_path = e_process(misses, alpha)
    e_sup = float(np.max(e_path)) if e_path.size else 1.0
    return {
        "target": split["label"],
        "alpha": alpha,
        "coverage": ev["coverage"],
        "e_final": ev["e_final"],
        "e_sup": e_sup,
        "ever_cross": ev["ever_cross"],
        "n": ev["n"],
        "threshold": 20.0,
        "wrappee": split.get("wrappee"),
    }


def bench_jackknife_plus(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Jackknife+ on vol-scaled residuals. Coverage floor is 1-2α, not 1-α."""
    split = _gaussian_interval_split(frame, config)
    if split is None:
        return {}
    alpha = float(split["alpha"])
    y = split["y"]
    cal = split["cal"]
    te = split["te"]
    covariate = np.maximum(np.asarray(split["covariate"], dtype=float), 1e-8)
    y_cal, sc_cal = _even_take(y[cal], covariate[cal], n=_JP_MAX_CAL)
    y_te, sc_te = _even_take(y[te], covariate[te], n=_JP_MAX_TEST)
    if y_cal.size < 2 or y_te.size < 8:
        return {}
    jp = JackknifePlus(alpha).fit(y_cal / sc_cal)
    lo_z, hi_z = jp.predict_interval(np.zeros_like(y_te), np.ones_like(y_te))
    lo, hi = lo_z * sc_te, hi_z * sc_te
    metrics = set_metrics(y_te, lo, hi)
    hits = 1.0 - covered(y_te, lo, hi)
    hits = hits[np.isfinite(hits)]
    if hits.size >= 10:
        rate, lr, kp = kupiec_pof(hits, 2.0 * alpha)
    else:
        rate, lr, kp = float("nan"), float("nan"), float("nan")
    floor = jackknife_plus_coverage_level(alpha)
    return {
        "target": split["label"],
        "alpha": alpha,
        "coverage": metrics.coverage,
        "mean_width": metrics.mean_width,
        "median_width": metrics.median_width,
        "coverage_floor": floor,
        "n": metrics.n,
        "miss_rate": rate,
        "kupiec_lr": lr,
        "kupiec_p": kp,
        "wrappee": split.get("wrappee"),
    }


def bench_crc(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    """CRC expected hit risk on scaled wrappee VaR bounds. Homoskedastic Gaussian is diagnostic."""
    split = _gaussian_interval_split(frame, config)
    if split is None:
        return {}
    y = split["y"]
    x = split["x"]
    tr = split["tr"]
    cal = split["cal"]
    te = split["te"]
    q_cal = split["q_cal"]
    q_te = split["q_te"]
    covariate = np.asarray(split["covariate"], dtype=float)
    wrappee_name = str(split.get("wrappee", "scaled_gaussian"))
    alpha = 0.05
    losses_cal = -y[cal]
    losses_te = -y[te]
    b_cal = -q_cal[:, 0]
    b_te = -q_te[:, 0]
    cal_bench = bench_crc_var(losses_cal, b_cal, alpha=alpha)
    crc = ConformalRiskControl(alpha).calibrate(losses_cal, b_cal)
    pred = crc.predict_bound(b_te)
    hits = loss_hit(losses_te, pred)
    finite = np.isfinite(hits)
    hits = hits[finite]
    if hits.size == 0:
        return {}
    risk = float(np.mean(hits))
    n = int(hits.size)
    _rate, lr, kp = kupiec_pof(hits, alpha)
    te_scale = covariate[te]
    finite_sc = te_scale[np.isfinite(te_scale)]
    med = float(np.median(finite_sc)) if finite_sc.size else 0.0
    high = np.isfinite(te_scale) & (te_scale >= med)
    low = np.isfinite(te_scale) & (te_scale < med)
    high_b = float(np.mean(pred[high])) if int(high.sum()) else float("nan")
    low_b = float(np.mean(pred[low])) if int(low.sum()) else float("nan")
    gauss = GaussianDistribution([0.95]).fit(x[tr], -y[tr])
    b_raw_cal = gauss.predict(x[cal])[:, 0]
    b_raw_te = gauss.predict(x[te])[:, 0]
    raw_crc = ConformalRiskControl(alpha).calibrate(losses_cal, b_raw_cal)
    raw_pred = raw_crc.predict_bound(b_raw_te)
    raw_hits = loss_hit(losses_te, raw_pred)
    raw_finite = raw_hits[np.isfinite(raw_hits)]
    raw_risk = float(np.mean(raw_finite)) if raw_finite.size else float("nan")
    return {
        "target": split["label"],
        "wrappee": wrappee_name,
        "risk": risk,
        "nominal": alpha,
        "n": float(n),
        "lambda_hat": float(crc.lambda_hat),
        "crc_stat": float((n * risk + 1.0) / (n + 1)),
        "kupiec_lr": float(lr),
        "kupiec_p": float(kp),
        "cal_risk": cal_bench.get("risk"),
        "cal_lambda_hat": cal_bench.get("lambda_hat"),
        "cal_crc_stat": cal_bench.get("crc_stat"),
        "high_vol_mean_bound": high_b,
        "low_vol_mean_bound": low_b,
        "gaussian_raw": {
            "risk": raw_risk,
            "lambda_hat": float(raw_crc.lambda_hat),
            "n": float(raw_finite.size),
        },
    }


def bench_weighted_conformal(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Weighted split CQR vs exchangeable CQR on the panel vol covariate."""
    split = _gaussian_interval_split(frame, config)
    if split is None:
        return {}
    alpha = float(split["alpha"])
    y = split["y"]
    cal = split["cal"]
    te = split["te"]
    q_cal = split["q_cal"]
    q_te = split["q_te"]
    x_all = split["covariate"]
    x_cal = x_all[cal]
    x_te = x_all[te]
    model = WeightedSplitCQR(alpha).calibrate(y[cal], q_cal[:, 0], q_cal[:, 1], x_cal)
    wlo, whi = model.predict_sets(q_te[:, 0], q_te[:, 1], x_te)
    weighted = set_metrics(y[te], wlo, whi)
    plain = SplitCQR(alpha).calibrate(y[cal], q_cal[:, 0], q_cal[:, 1])
    ulo, uhi = plain.predict_sets(q_te[:, 0], q_te[:, 1])
    unweighted = set_metrics(y[te], ulo, uhi)
    misses = 1.0 - covered(y[te], wlo, whi)
    misses = misses[np.isfinite(misses)]
    if misses.size >= 10:
        _rate, lr, kp = kupiec_pof(misses, alpha)
    else:
        _rate, lr, kp = float("nan"), float("nan"), float("nan")
    qhat = np.asarray(model.qhat, dtype=float)
    return {
        "target": split["label"],
        "alpha": alpha,
        "coverage": weighted.coverage,
        "mean_width": weighted.mean_width,
        "median_width": weighted.median_width,
        "unweighted_coverage": unweighted.coverage,
        "unweighted_mean_width": unweighted.mean_width,
        "qhat_mean": float(np.mean(qhat)) if qhat.size else float("nan"),
        "n": weighted.n,
        "kupiec_lr": lr,
        "kupiec_p": kp,
        "covariate": "vol_20",
    }


def bench_interval_risk(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Name-cap binding from CQR intervals. Equal-weight per date, then caps. No P&L."""
    split = _gaussian_interval_split(frame, config)
    if split is None:
        return {}
    alpha = float(split["alpha"])
    y = split["y"]
    cal = split["cal"]
    te = split["te"]
    q_cal = split["q_cal"]
    q_te = split["q_te"]
    cqr = SplitCQR(alpha).calibrate(y[cal], q_cal[:, 0], q_cal[:, 1])
    lo_cal, hi_cal = cqr.predict_sets(q_cal[:, 0], q_cal[:, 1])
    lo, hi = cqr.predict_sets(q_te[:, 0], q_te[:, 1])
    wr, dr = interval_refs(lo_cal, hi_cal, multiple=4.0)
    max_w = float(config.constraints.name_max)
    dates_te = np.asarray(split["dates"])[te]
    w_eq = equal_weight_per_date(dates_te)
    # Same 1/n on the date, inside the name box; interval caps clip further.
    weights = np.minimum(w_eq, max_w)
    caps = cap_from_interval(lo, hi, max_weight=max_w, width_ref=wr, downside_ref=dr)
    out = bench_interval_caps(
        lo, hi, weights, max_weight=max_w, width_ref=wr, downside_ref=dr
    )
    width = hi - lo
    finite_w = np.isfinite(width)
    med_w = float(np.nanmedian(width[finite_w])) if int(finite_w.sum()) else 0.0
    wide = finite_w & (width >= med_w)
    tight = finite_w & (width < med_w)
    bind = np.abs(weights) > caps
    n_wide = int(wide.sum())
    n_tight = int(tight.sum())
    n_bind_wide = int(np.sum(bind[wide])) if n_wide else 0
    n_bind_tight = int(np.sum(bind[tight])) if n_tight else 0
    bind_wide = float(n_bind_wide / n_wide) if n_wide else float("nan")
    bind_tight = float(n_bind_tight / n_tight) if n_tight else float("nan")
    date_keys = np.asarray(_date_keys(dates_te))
    gaps: list[float] = []
    for d in np.unique(date_keys):
        m = date_keys == d
        w_m = wide & m
        t_m = tight & m
        if int(w_m.sum()) == 0 or int(t_m.sum()) == 0:
            continue
        gaps.append(float(np.mean(bind[w_m])) - float(np.mean(bind[t_m])))
    return {
        "target": split["label"],
        "alpha": alpha,
        "max_weight": max_w,
        "width_ref": wr,
        "downside_ref": dr,
        "weight_rule": "equal_weight_per_date",
        "bind_wide": bind_wide,
        "bind_tight": bind_tight,
        "n_wide": n_wide,
        "n_tight": n_tight,
        "n_bind_wide": n_bind_wide,
        "n_bind_tight": n_bind_tight,
        "bind_gap_by_date": gaps,
        "n_dates": int(np.unique(date_keys).size),
        **out,
        "frac_binding": float(np.mean(bind)) if bind.size else float("nan"),
    }


def bench_quantile_bandit(frame: pl.DataFrame, label: str) -> dict[str, Any]:
    """Quantile Thompson vs static public ridge, oracle, and uniform on a shared date tail."""
    design = _policy_public_design(frame, label)
    if not design:
        return {}
    x, y, dates = design["x"], design["y"], design["dates"]
    if x.shape[0] < 20:
        return {}
    trace = QuantileThompson(n_quantiles=5, ridge=1.0, seed=7).run_panel(
        x, y, dates, k=3, oracle=design["oracle"]
    )
    if trace.policy_reward.size < 5:
        return {}
    ridge, ridge_dates = _policy_ridge_topk_rewards(x, y, dates, top_k=3)
    return _policy_bandit_metrics(
        policy=trace.policy_reward,
        oracle_r=trace.oracle_reward,
        uniform=trace.uniform_reward,
        policy_dates=trace.dates,
        ridge=ridge,
        ridge_dates=ridge_dates,
        used=design["features"],
        oracle_definition=design["oracle_definition"],
        oracle_column=design["oracle_column"],
        leak=design["leak"],
        extra={"n_quantiles": 5},
    )
