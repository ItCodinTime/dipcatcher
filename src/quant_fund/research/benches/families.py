"""Remaining forecast-family trainers or scientific benches.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.execution.almgren_chriss import (
    almgren_chriss_trajectory,
    expected_shortfall_ac,
    slice_trades,
    twap_trajectory,
)
from quant_fund.metrics.cross_section import _date_keys
from quant_fund.metrics.evalues import e_process_dm
from quant_fund.metrics.inference import diebold_mariano, overlap_aware_hac_lags
from quant_fund.metrics.probability import (
    acerbi_szekely_z1,
    acerbi_szekely_z2,
    brier_score,
    christoffersen_cc,
    christoffersen_independence,
    expected_calibration_error,
    kupiec_pof,
    log_loss,
    pit_ks,
)
from quant_fund.metrics.scoring import (
    coverage,
    crps_from_quantiles,
    date_level_equal_weight,
    mean_crps_gaussian,
    mean_crps_student_t,
    mean_fissler_ziegel,
    mean_pinball,
    nonoverlapping_origin_mask,
    pearson_ic,
    pit_values,
    qlike,
    quantile_crossing_rate,
)
from quant_fund.models.distribution import (
    EmpiricalDistribution,
    GaussianDistribution,
    ScaledEmpiricalDistribution,
    ScaledGaussianDistribution,
    ScaledStudentTDistribution,
    fit_operational_wrappee,
)
from quant_fund.models.regime import GaussianHMMRegime, SingleStateRegime, VolThresholdRegime
from quant_fund.models.rl import run_linucb_panel
from quant_fund.models.tail import DrawdownClassifier, HistoricalTail, ScaledHistoricalTail
from quant_fund.pipeline.dataset import design_matrix
from quant_fund.pipeline.train import _label_horizon

from .common import (
    _aligned_col,
    _crps_from_quantiles_obs,
    _holdout,
    _public_features_in,
    _scaled_fill,
)
from .ranking import _policy_bandit_metrics, _policy_public_design, _policy_ridge_topk_rewards


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
    dates_d, y_d = date_level_equal_weight(
        sub["event_time"].to_numpy(), sub[label].to_numpy().astype(float)
    )
    _, roll_d = date_level_equal_weight(
        sub["event_time"].to_numpy(), sub["vol_20"].to_numpy().astype(float) ** 2
    )
    if "vol_ewma" in sub.columns:
        _, ewma_d = date_level_equal_weight(
            sub["event_time"].to_numpy(), sub["vol_ewma"].to_numpy().astype(float) ** 2
        )
    else:
        ewma_d = roll_d
    if dates_d.size == 0 or dates_d.size != roll_d.size or dates_d.size != ewma_d.size:
        return {}
    horizon = _label_horizon(label)
    _, te = _holdout(int(dates_d.size))
    yy = np.clip(y_d[te], config.train.qlike_floor, None)
    roll_te = np.clip(roll_d[te], config.train.qlike_floor, None)
    ewma_te = np.clip(ewma_d[te], config.train.qlike_floor, None)
    q_roll = qlike(yy, roll_te, config.train.qlike_floor)
    q_ewma = qlike(yy, ewma_te, config.train.qlike_floor)
    loss_e = (np.sqrt(yy) - np.sqrt(ewma_te)) ** 2
    loss_r = (np.sqrt(yy) - np.sqrt(roll_te)) ** 2
    dm_lags = overlap_aware_hac_lags(int(yy.size), horizon)
    dm = diebold_mariano(loss_e, loss_r, lags=dm_lags, name_a="ewma", name_b="rolling")
    # Research-only Choe–Ramdas e-process on the same loss differential (Wave4 helper).
    # Never a live capital / promotion claim — diagnostic keys only.
    ep = e_process_dm(loss_e, loss_r)
    keep = nonoverlapping_origin_mask(np.arange(int(yy.size), dtype=int), horizon)
    if int(keep.sum()) >= 1:
        q_roll_non = qlike(yy[keep], roll_te[keep], config.train.qlike_floor)
        q_ewma_non = qlike(yy[keep], ewma_te[keep], config.train.qlike_floor)
        dm_non = diebold_mariano(
            loss_e[keep],
            loss_r[keep],
            lags=overlap_aware_hac_lags(int(keep.sum()), 1),
            name_a="ewma",
            name_b="rolling",
        )
        dm_non_preferred = dm_non.preferred
        dm_non_p = dm_non.p_value
        dm_non_stat = dm_non.statistic
    else:
        q_roll_non = float("nan")
        q_ewma_non = float("nan")
        dm_non_preferred = "inconclusive"
        dm_non_p = float("nan")
        dm_non_stat = float("nan")
    return {
        "qlike_ewma": q_ewma,
        "qlike_rolling": q_roll,
        "dm_preferred": dm.preferred,
        "dm_p": dm.p_value,
        "dm_stat": dm.statistic,
        "e_dm_final": float(ep["e_final"]),  # type: ignore[arg-type]
        "e_dm_reject": bool(ep["reject"]),
        "e_dm_n": int(cast(Any, ep["n"])),
        "scoring_scope": "date_level_equal_weight",
        "horizon_bars": int(horizon),
        "dm_lags": int(dm.lags),
        "n_dates": int(yy.size),
        "n_origins_nonoverlapping": int(keep.sum()),
        "qlike_ewma_nonoverlapping": q_ewma_non,
        "qlike_rolling_nonoverlapping": q_roll_non,
        "dm_preferred_nonoverlapping": dm_non_preferred,
        "dm_p_nonoverlapping": dm_non_p,
        "dm_stat_nonoverlapping": dm_non_stat,
        "research_only": True,  # no live_pnl_claim key (pnl token forbidden)
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
    n_te = int(yy.shape[0])
    mu_g = np.full(n_te, float(g.mu), dtype=float)
    sig_g = np.full(n_te, float(g.sig), dtype=float)
    # Quantile-approx keys kept; closed-form reported beside them (honest dual view).
    out: dict[str, Any] = {
        "target": label,
        "pinball_gaussian": mean_pinball(yy, qg[:, mid], taus[mid]),
        "pinball_empirical": mean_pinball(yy, qe[:, mid], taus[mid]),
        "crps_gaussian": crps_from_quantiles(yy, qg, np.array(taus)),
        "crps_empirical": crps_from_quantiles(yy, qe, np.array(taus)),
        "crps_gaussian_closed": mean_crps_gaussian(yy, mu_g, sig_g),
        "coverage_gaussian": coverage(yy, qg[:, lo_i], qg[:, hi_i]),
        "nominal_coverage": taus[hi_i] - taus[lo_i],
        "crossing_gaussian": quantile_crossing_rate(qg, np.array(taus)),
        "pit_ks": ks,
        "pit_ks_p": ks_p,
    }
    # DM on per-obs quantile-CRPS losses: gaussian vs empirical (research-only).
    if n_te >= 3:
        loss_g = _crps_from_quantiles_obs(yy, qg, taus)
        loss_e = _crps_from_quantiles_obs(yy, qe, taus)
        dm = diebold_mariano(loss_g, loss_e, name_a="gaussian", name_b="empirical")
        out["dm_crps_preferred"] = dm.preferred
        out["dm_crps_p"] = dm.p_value
        out["dm_crps_stat"] = dm.statistic
        # DayWave18: Choe–Ramdas e-process on the same CRPS loss differential.
        # Prefixed e_dm_crps_* to avoid clashing with vol-bench e_dm_* if blobs merge.
        # Prefer omit keys on failure (fail-closed); never mint live_pnl_claim.
        try:
            ep = e_process_dm(loss_g, loss_e)
            out["e_dm_crps_final"] = float(ep["e_final"])  # type: ignore[arg-type]
            out["e_dm_crps_reject"] = bool(ep["reject"])
            out["e_dm_crps_n"] = int(cast(Any, ep["n"]))
        except (TypeError, ValueError, FloatingPointError, KeyError):
            # Day Wave 20: emit presence sentinels (NaN/False/0) so soft verify
            # can require key presence beside dm_crps_*; never mint live_pnl_claim.
            out["e_dm_crps_final"] = float("nan")
            out["e_dm_crps_reject"] = False
            out["e_dm_crps_n"] = 0
        out["research_only"] = True  # no live_pnl_claim key (pnl token forbidden)
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
        # Closed-form: μ + z_sig * scale[te] as σ_t (homoskedastic z-scale).
        sig_t = float(sg.z_sig) * np.maximum(np.asarray(sc[te], dtype=float), 1e-8)
        mu_s = np.full(n_te, float(sg.mu), dtype=float)
        out["crps_scaled_gaussian_closed"] = mean_crps_gaussian(yy, mu_s, sig_t)
        out["pit_ks_scaled"] = ks_s
        out["pit_ks_p_scaled"] = ks_sp
        st = ScaledStudentTDistribution(taus).fit(y[tr], sc[tr])
        qt = st.predict(sc[te])
        pits_t = pit_values(yy, qt, np.array(taus))
        ks_t, ks_tp = pit_ks(pits_t)
        out["coverage_scaled_student_t"] = coverage(yy, qt[:, lo_i], qt[:, hi_i])
        out["pinball_scaled_student_t"] = mean_pinball(yy, qt[:, mid], taus[mid])
        out["crps_scaled_student_t"] = crps_from_quantiles(yy, qt, np.array(taus))
        # Closed-form Student-t CRPS beside quantile Riemann (Day Wave 43).
        sig_st = float(st.z_sig) * np.maximum(np.asarray(sc[te], dtype=float), 1e-8)
        mu_st = np.full(n_te, float(st.mu), dtype=float)
        out["crps_scaled_student_t_closed"] = mean_crps_student_t(yy, mu_st, sig_st, float(st.nu))
        out["pit_ks_scaled_t"] = ks_t
        out["pit_ks_p_scaled_t"] = ks_tp
        out["nu_scaled_student_t"] = float(st.nu)
        se = ScaledEmpiricalDistribution(taus).fit(y[tr], sc[tr])
        qe = se.predict(sc[te])
        pits_e = pit_values(yy, qe, np.array(taus))
        ks_e, ks_ep = pit_ks(pits_e)
        out["coverage_scaled_empirical"] = coverage(yy, qe[:, lo_i], qe[:, hi_i])
        out["pinball_scaled_empirical"] = mean_pinball(yy, qe[:, mid], taus[mid])
        out["crps_scaled_empirical"] = crps_from_quantiles(yy, qe, np.array(taus))
        out["pit_ks_scaled_empirical"] = ks_e
        out["pit_ks_p_scaled_empirical"] = ks_ep
        # DayWave19: multi-model DM + e-process on scaled gauss vs scaled t CRPS
        # (diagnostics only; do not change wrappee selection below).
        if n_te >= 3:
            loss_sg = _crps_from_quantiles_obs(yy, qs, taus)
            loss_st = _crps_from_quantiles_obs(yy, qt, taus)
            loss_se = _crps_from_quantiles_obs(yy, qe, taus)
            dm_s = diebold_mariano(
                loss_sg, loss_st, name_a="scaled_gaussian", name_b="scaled_student_t"
            )
            out["dm_crps_scaled_preferred"] = dm_s.preferred
            out["dm_crps_scaled_p"] = dm_s.p_value
            out["dm_crps_scaled_stat"] = dm_s.statistic
            dm_se = diebold_mariano(
                loss_se, loss_st, name_a="scaled_empirical", name_b="scaled_student_t"
            )
            out["dm_crps_empirical_vs_student_preferred"] = dm_se.preferred
            out["dm_crps_empirical_vs_student_p"] = dm_se.p_value
            out["dm_crps_empirical_vs_student_stat"] = dm_se.statistic
            try:
                ep_s = e_process_dm(loss_sg, loss_st)
                out["e_dm_crps_scaled_final"] = float(ep_s["e_final"])  # type: ignore[arg-type]
                out["e_dm_crps_scaled_reject"] = bool(ep_s["reject"])
                out["e_dm_crps_scaled_n"] = int(cast(Any, ep_s["n"]))
            except (TypeError, ValueError, FloatingPointError, KeyError):
                # Day Wave 20: presence sentinels beside dm_crps_scaled_* (soft verify).
                out["e_dm_crps_scaled_final"] = float("nan")
                out["e_dm_crps_scaled_reject"] = False
                out["e_dm_crps_scaled_n"] = 0
            out["research_only"] = True  # no live_pnl_claim key (pnl token forbidden)
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
    cols = [
        c for c in ["mkt_ret_1", "mkt_vol_20", "cs_dispersion", "breadth"] if c in frame.columns
    ]
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
    """Historical / scaled-historical VaR–ES holdout diagnostics (research-only).

    Kupiec POF + Christoffersen independence / conditional coverage on VaR hits,
    plus Acerbi–Székely Z1/Z2 and mean Fissler–Ziegel FZ0 on the same holdout
    losses. Alpha convention matches ``var_backtest_hooks``: coverage
    ``alpha_cov=0.95`` (VaR level); miss level ``p_miss=1-alpha_cov`` for Kupiec /
    Christoffersen / Acerbi Z1; FZ uses coverage ``alpha_cov``. Primary keys prefer
    the scaled path when ``vol_20`` is available (same pattern as Kupiec), with
    ``*_unscaled`` / ``*_scaled`` mirrors. Empty / short / no hits / non-positive ES
    → honest NaN from helpers. Never a live capital / promotion claim.
    """
    label = next((c for c in frame.columns if c.startswith("future_return_1")), None)
    if label is None:
        return {}
    x, y, dates, _feats, ids = design_matrix(frame, label)
    if x.size == 0:
        return {}
    tr, te = _holdout(x.shape[0])
    alpha_cov = 0.95
    p_miss = 1.0 - alpha_cov
    hist = HistoricalTail(alpha_cov).fit(x[tr], y[tr])
    var, es = hist.predict_var_es()
    losses = -y[te]
    n_te = int(losses.size)
    hits = (losses >= var).astype(float)
    rate, lr, p = kupiec_pof(hits, p_miss)
    ind_lr_u, ind_p_u, _ = christoffersen_independence(hits)
    cc_lr_u, cc_p_u, _ = christoffersen_cc(hits, p_miss)
    tail = losses[losses >= var] if np.isfinite(var) else np.array([])
    realized_es = float(np.mean(tail)) if tail.size else float("nan")
    # Scalar VaR/ES broadcast to holdout length for ES diagnostics.
    va_u = np.full(n_te, float(var), dtype=float)
    es_u = np.full(n_te, float(es), dtype=float)
    z1_u, n_hits_u = acerbi_szekely_z1(losses, va_u, es_u, p_miss)
    z2_u, _ = acerbi_szekely_z2(losses, va_u, es_u)
    fz_u = mean_fissler_ziegel(losses, va_u, es_u, alpha_cov)
    out: dict[str, Any] = {
        "var_95": var,
        "es_95": es,
        "hit_rate_unscaled": rate,
        "hit_rate": rate,
        "nominal_hit_rate": p_miss,
        "kupiec_lr_unscaled": lr,
        "kupiec_p_unscaled": p,
        "kupiec_lr": lr,
        "kupiec_p": p,
        "christoffersen_ind_lr": float(ind_lr_u),
        "christoffersen_ind_p": float(ind_p_u),
        "christoffersen_cc_lr": float(cc_lr_u),
        "christoffersen_cc_p": float(cc_p_u),
        "christoffersen_ind_lr_unscaled": float(ind_lr_u),
        "christoffersen_ind_p_unscaled": float(ind_p_u),
        "christoffersen_cc_lr_unscaled": float(cc_lr_u),
        "christoffersen_cc_p_unscaled": float(cc_p_u),
        "realized_es": realized_es,
        "acerbi_szekely_z1": float(z1_u),
        "acerbi_szekely_z2": float(z2_u),
        "fissler_ziegel_mean": float(fz_u),
        "es_hit_count": float(n_hits_u),
        "acerbi_szekely_z1_unscaled": float(z1_u),
        "acerbi_szekely_z2_unscaled": float(z2_u),
        "fissler_ziegel_mean_unscaled": float(fz_u),
        "es_hit_count_unscaled": float(n_hits_u),
        "research_only": True,  # no live_pnl_claim key (pnl token forbidden)
    }
    vol = _aligned_col(frame, dates, ids, "vol_20")
    if vol is not None and np.isfinite(vol[tr]).any() and np.isfinite(vol[te]).any():
        med = float(np.nanmedian(vol[np.isfinite(vol)]))
        sc = np.where(np.isfinite(vol), vol, med)
        scaled = ScaledHistoricalTail(alpha_cov).fit(y[tr], sc[tr])
        var_s, es_s = scaled.predict_var_es(sc[te])
        hits_s = (losses >= var_s).astype(float)
        rate_s, lr_s, p_s = kupiec_pof(hits_s, p_miss)
        ind_lr_s, ind_p_s, _ = christoffersen_independence(hits_s)
        cc_lr_s, cc_p_s, _ = christoffersen_cc(hits_s, p_miss)
        z1_s, n_hits_s = acerbi_szekely_z1(losses, var_s, es_s, p_miss)
        z2_s, _ = acerbi_szekely_z2(losses, var_s, es_s)
        fz_s = mean_fissler_ziegel(losses, var_s, es_s, alpha_cov)
        out["var_95_scaled"] = float(np.mean(var_s))
        out["es_95_scaled"] = float(np.mean(es_s))
        out["hit_rate"] = rate_s
        out["kupiec_lr"] = lr_s
        out["kupiec_p"] = p_s
        out["hit_rate_scaled"] = rate_s
        out["kupiec_p_scaled"] = p_s
        out["kupiec_lr_scaled"] = lr_s
        # Primary keys prefer scaled when available (Kupiec pattern).
        out["christoffersen_ind_lr"] = float(ind_lr_s)
        out["christoffersen_ind_p"] = float(ind_p_s)
        out["christoffersen_cc_lr"] = float(cc_lr_s)
        out["christoffersen_cc_p"] = float(cc_p_s)
        out["christoffersen_ind_lr_scaled"] = float(ind_lr_s)
        out["christoffersen_ind_p_scaled"] = float(ind_p_s)
        out["christoffersen_cc_lr_scaled"] = float(cc_lr_s)
        out["christoffersen_cc_p_scaled"] = float(cc_p_s)
        out["acerbi_szekely_z1"] = float(z1_s)
        out["acerbi_szekely_z2"] = float(z2_s)
        out["fissler_ziegel_mean"] = float(fz_s)
        out["es_hit_count"] = float(n_hits_s)
        out["acerbi_szekely_z1_scaled"] = float(z1_s)
        out["acerbi_szekely_z2_scaled"] = float(z2_s)
        out["fissler_ziegel_mean_scaled"] = float(fz_s)
        out["es_hit_count_scaled"] = float(n_hits_s)
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
    ac = almgren_chriss_trajectory(1.0, 5, sigma=0.02, eta=1e-4, gamma=1e-5, risk_aversion=1e-3)
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


__all__ = [
    "bench_distribution",
    "bench_drawdown",
    "bench_liquidity",
    "bench_regime",
    "bench_rl",
    "bench_tail",
    "bench_volatility",
]
