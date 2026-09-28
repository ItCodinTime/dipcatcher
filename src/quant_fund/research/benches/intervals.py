"""Conformal, e-value, jackknife, CV+, and interval benches.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import comb
from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.metrics.conformal import (
    assign_terciles,
    conditional_coverage,
    covered,
    set_metrics,
    worst_slice_coverage,
)
from quant_fund.metrics.cross_section import _date_keys
from quant_fund.metrics.evalues import bench_e_coverage, e_process
from quant_fund.metrics.probability import kupiec_pof
from quant_fund.models.conformal import (
    AdaptiveConformal,
    MondrianACI,
    MondrianCQR,
    SplitCQR,
    SplitOneSided,
)
from quant_fund.models.crc import ConformalRiskControl, bench_crc_var, loss_hit
from quant_fund.models.cv_plus import CVPlus, cv_plus_coverage_level
from quant_fund.models.distribution import (
    GaussianDistribution,
    LinearQuantileDistribution,
    select_scaled_wrappee,
)
from quant_fund.models.jackknife_plus import JackknifePlus, jackknife_plus_coverage_level
from quant_fund.models.quantile_bandit import QuantileThompson
from quant_fund.models.regime import GaussianHMMRegime
from quant_fund.models.weighted_conformal import WeightedSplitCQR
from quant_fund.pipeline.dataset import design_matrix
from quant_fund.pipeline.train import _label_horizon
from quant_fund.portfolio.interval_risk import (
    bench_interval_caps,
    cap_from_interval,
    equal_weight_per_date,
    interval_refs,
)
from quant_fund.validation.cpcv import combinatorial_purged_cv

from .common import (
    _EV_MAX_DATES,
    _JP_MAX_CAL,
    _JP_MAX_TEST,
    _abs_y_labels,
    _aligned_col,
    _even_take,
    _gaussian_interval_split,
    _holdout,
    _interval_label,
    _scaled_fill,
    _tail_date_mask,
    _triple_split,
)
from .ranking import _policy_bandit_metrics, _policy_public_design, _policy_ridge_topk_rewards


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
    cqr_miss = 1.0 - covered(y[te], lo_c, hi_c)
    cqr_miss = cqr_miss[np.isfinite(cqr_miss)]
    if cqr_miss.size >= 10:
        cqr_rate, cqr_lr, cqr_kp = kupiec_pof(cqr_miss, alpha)
    else:
        cqr_rate, cqr_lr, cqr_kp = float("nan"), float("nan"), float("nan")
    qr_m: dict[str, float] | None = None
    if 80 <= int(tr.sum()) <= 4000:
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
    rcols = [
        c for c in ["mkt_ret_1", "mkt_vol_20", "cs_dispersion", "breadth"] if c in frame.columns
    ]
    if len(rcols) >= 2:
        try:
            rsub = (
                frame.select(["event_time", *rcols])
                .unique("event_time")
                .drop_nulls()
                .sort("event_time")
            )
            rx = rsub.select(rcols).to_numpy().astype(float)
            rdates = rsub["event_time"].to_numpy()
            if rx.shape[0] >= 40:
                rtr, _ = _holdout(rdates, 0.3, horizon=_label_horizon(label))
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
            dtr, dcal, dte = _triple_split(d_dates, horizon=_label_horizon(dd_lab))
            vol_dd = _aligned_col(frame, d_dates, d_ids, "vol_20")
            if not (dtr.any() and dcal.any() and dte.any()):
                pass  # purged empty — drawdown lane skipped
            elif vol_dd is not None and np.isfinite(vol_dd).any():
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
            "miss_rate": cqr_rate,
            "kupiec_lr": cqr_lr,
            "kupiec_p": cqr_kp,
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
    """Jackknife+ on vol-scaled residuals. Coverage floor is 1-2α, not 1-α.

    Floor is **marginal under exchangeability** (Barber–Candès 2021), not
    training-conditional (Bian–Barber 2023). Nonempty returns surface
    ``coverage_guarantee_scope="marginal_exchangeable"`` (research-only).
    """
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
    # Residual was fit in y/vol space. predict_interval must convert both the
    # LOO location and the scores back to return units via sc_te. Scaling only
    # the half-width was a unit bug (Jackknife+ coverage ~0.07 vs 1-2α).
    lo, hi = jp.predict_interval(np.zeros_like(y_te), sc_te)
    metrics = set_metrics(y_te, lo, hi)
    hits = 1.0 - covered(y_te, lo, hi)
    hits = hits[np.isfinite(hits)]
    if hits.size >= 10:
        rate, lr, kp = kupiec_pof(hits, 2.0 * alpha)
    else:
        rate, lr, kp = float("nan"), float("nan"), float("nan")
    floor = jackknife_plus_coverage_level(alpha)
    loc = jp.loo_loc_
    return {
        "target": split["label"],
        "alpha": alpha,
        "coverage": metrics.coverage,
        "mean_width": metrics.mean_width,
        "median_width": metrics.median_width,
        "coverage_floor": floor,
        "meets_coverage_floor": bool(
            np.isfinite(metrics.coverage) and metrics.coverage + 1e-12 >= floor
        ),
        "mean_loo_loc": float(np.mean(loc)) if loc is not None and loc.size else float("nan"),
        "mean_abs_y_te": float(np.mean(np.abs(y_te))) if y_te.size else float("nan"),
        "mean_scale_te": float(np.mean(sc_te)) if sc_te.size else float("nan"),
        "coverage_identity": "1-2*alpha",
        "coverage_guarantee_scope": "marginal_exchangeable",
        "coverage_guarantee_claim": (
            "coverage floor is marginal under exchangeability; not training-conditional"
        ),
        "research_only": True,  # no live_pnl_claim key (pnl token forbidden)
        "n": metrics.n,
        "miss_rate": rate,
        "kupiec_lr": lr,
        "kupiec_p": kp,
        "wrappee": split.get("wrappee"),
    }


def _bench_cv_plus_panel(
    frame: pl.DataFrame,
    config: AppConfig,
    *,
    alpha: float = 0.10,
    n_folds: int = 5,
    aggregation: str = "minmax",
) -> dict[str, float | str]:
    """Date-folded CV+ on the lab panel wrappee residuals (not a toy Gaussian).

    Coverage floor is marginal under exchangeability, not training-conditional.
    """
    split = _gaussian_interval_split(frame, config, alpha=alpha)
    if split is None:
        return {}
    y = np.asarray(split["y"], dtype=float)
    dates = np.asarray(split["dates"])
    cal = split["cal"]
    te = split["te"]
    q_cal = np.asarray(split["q_cal"], dtype=float)
    q_te = np.asarray(split["q_te"], dtype=float)
    # Point predictor = wrappee interval midpoint (same units as y). Fill the
    # train slice too: CV+ derives fold means/scales from every fit-row
    # residual, so leaving it uninitialized (np.empty_like) silently fed
    # allocator garbage into the calibration.
    q_tr = np.asarray(split["q_tr"], dtype=float)
    pred = np.full(y.shape, np.nan, dtype=float)
    pred[split["tr"]] = 0.5 * (q_tr[:, 0] + q_tr[:, 1])
    pred[cal] = 0.5 * (q_cal[:, 0] + q_cal[:, 1])
    pred[te] = 0.5 * (q_te[:, 0] + q_te[:, 1])
    # Fit on chronological prefix through end of calibration (never test).
    fit_mask = split["tr"] | cal
    d_fit = dates[fit_mask]
    # CV+ needs unique dates >= n_folds
    if len(np.unique(d_fit)) < n_folds or int(te.sum()) < 8:
        return {}
    model = CVPlus(alpha=alpha, n_folds=n_folds, aggregation=aggregation).fit(
        y[fit_mask], pred[fit_mask], dates=d_fit
    )
    lo, hi = model.predict_interval(pred[te])
    metrics = set_metrics(y[te], lo, hi)
    hits = 1.0 - covered(y[te], lo, hi)
    hits = hits[np.isfinite(hits)]
    # plus/jaw intervals cover at 1-2*alpha (paper bound), so their nominal
    # miss probability is 2*alpha; only minmax targets 1-alpha.
    nominal_miss = alpha if aggregation == "minmax" else 2.0 * alpha
    if hits.size >= 10:
        rate, lr, kp = kupiec_pof(hits, nominal_miss)
    else:
        rate, lr, kp = float("nan"), float("nan"), float("nan")
    return {
        "coverage": metrics.coverage,
        "mean_width": metrics.mean_width,
        "n_dates": float(len(np.unique(dates[te]))),
        "alpha": float(alpha),
        "n_folds": float(n_folds),
        "aggregation": aggregation,
        "coverage_floor": cv_plus_coverage_level(alpha, aggregation),
        "coverage_identity": "1-alpha" if aggregation == "minmax" else "1-2*alpha",
        "miss_rate": rate,
        "kupiec_lr": lr,
        "kupiec_p": kp,
        "target": split["label"],
        "wrappee": split.get("wrappee"),
        "dgp": "panel",
        "claim": "research_metric_only",
        "coverage_guarantee_scope": "marginal_exchangeable",
        "coverage_guarantee_claim": (
            "coverage floor is marginal under exchangeability; not training-conditional"
        ),
        "research_only": True,  # no live_pnl_claim key (pnl token forbidden)
    }


def bench_cv_plus(
    frame: pl.DataFrame | None = None,
    config: AppConfig | None = None,
    alpha: float = 0.10,
    n_folds: int = 5,
    aggregation: str = "minmax",
    seed: int = 23,
) -> dict[str, float | str]:
    """CV+ bench. Prefer lab panel; fixture path is labeled ``dgp=fixture`` only.

    Coverage floor is **marginal under exchangeability** (Barber–Candès 2021;
    agg-specific), not training-conditional (Bian–Barber 2023). Nonempty returns
    surface ``coverage_guarantee_scope="marginal_exchangeable"`` (research-only).
    Fixture rows must not share the research H-table with panel Kupiec tests.
    """
    if frame is not None and config is not None:
        panel_row = _bench_cv_plus_panel(
            frame, config, alpha=alpha, n_folds=n_folds, aggregation=aggregation
        )
        if panel_row:
            return panel_row
        return {}
    rng = np.random.default_rng(seed)
    n_dates, n_names = 120, 8
    dates = np.repeat(np.arange(n_dates), n_names)
    pred = rng.normal(0.0, 0.01, size=dates.size)
    y = pred + rng.normal(0.0, 0.02, size=dates.size)
    cut = int(0.7 * n_dates)
    train = dates < cut
    test = ~train
    model = CVPlus(alpha=alpha, n_folds=n_folds, aggregation=aggregation).fit(
        y[train], pred[train], dates=dates[train].astype(float)
    )
    lo, hi = model.predict_interval(pred[test])
    metrics = set_metrics(y[test], lo, hi)
    return {
        "coverage": metrics.coverage,
        "mean_width": metrics.mean_width,
        "n_dates": float(len(np.unique(dates[test]))),
        "alpha": float(alpha),
        "n_folds": float(n_folds),
        "aggregation": aggregation,
        "coverage_floor": cv_plus_coverage_level(alpha, aggregation),
        "coverage_identity": "1-alpha" if aggregation == "minmax" else "1-2*alpha",
        "seed": float(seed),
        "dgp": "fixture",
        "claim": "research_metric_only",
        "coverage_guarantee_scope": "marginal_exchangeable",
        "coverage_guarantee_claim": (
            "coverage floor is marginal under exchangeability; not training-conditional"
        ),
        "research_only": True,  # no live_pnl_claim key (pnl token forbidden)
    }


def bench_cpcv_audit(
    n_dates: int = 120,
    n_groups: int = 6,
    n_test_groups: int = 2,
    horizon_bars: int = 2,
    embargo_bars: int = 2,
) -> dict[str, float | bool | str]:
    """Audit CPCV date separation and purge/embargo integrity only."""
    dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n_dates)]
    folds = combinatorial_purged_cv(dates, n_groups, n_test_groups, horizon_bars, embargo_bars)
    valid = True
    min_train = n_dates
    min_test = n_dates
    for fold in folds:
        train = set(fold.train_times)
        test = set(fold.test_times)
        valid = valid and train.isdisjoint(test)
        min_train = min(min_train, len(train))
        min_test = min(min_test, len(test))
    expected = int(comb(n_groups, n_test_groups))
    return {
        "n_folds": float(len(folds)),
        "expected_folds": float(expected),
        "min_train_dates": float(min_train),
        "min_test_dates": float(min_test),
        "horizon_bars": float(horizon_bars),
        "embargo_bars": float(embargo_bars),
        "date_level": True,
        "purge_embargo_valid": bool(valid and len(folds) == expected),
        "claim": "validation_integrity_only",
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
        "covariate": str(split.get("cov_name", "vol_20")),
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
    out = bench_interval_caps(lo, hi, weights, max_weight=max_w, width_ref=wr, downside_ref=dr)
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


def bench_localized_from_panel(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    """RBF localized CQR on lab panel vol covariate + wrappee bands."""
    from quant_fund.models.localized_conformal import bench_localized_cqr

    split = _gaussian_interval_split(frame, config)
    if split is None:
        return {}
    cal, te = split["cal"], split["te"]
    y = split["y"]
    q_cal, q_te = split["q_cal"], split["q_te"]
    cov = np.asarray(split["covariate"], dtype=float)
    if int(cal.sum()) < 40 or int(te.sum()) < 20:
        return {}
    row = bench_localized_cqr(
        alpha=float(split["alpha"]),
        y_cal=y[cal],
        lo_cal=q_cal[:, 0],
        hi_cal=q_cal[:, 1],
        x_cal=cov[cal],
        y_test=y[te],
        lo_test=q_te[:, 0],
        hi_test=q_te[:, 1],
        x_test=cov[te],
        dates_test=np.asarray(split["dates"])[te],
        dgp="panel",
    )
    row["target"] = split["label"]
    row["wrappee"] = str(split.get("wrappee", "unknown"))
    row["covariate"] = str(split.get("cov_name", "vol_20"))
    return row


def bench_online_crc_from_panel(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Sequential online CRC on panel wrappee VaR losses (not exponential toy)."""
    from quant_fund.models.online_crc import bench_online_crc

    split = _gaussian_interval_split(frame, config, alpha=0.05)
    if split is None:
        return {}
    y = np.asarray(split["y"], dtype=float)
    dates = split["dates"]
    cal, te = split["cal"], split["te"]
    # Use full chronological path: losses = -y, base = -lower quantile bound
    q_all = np.empty((y.size, 2), dtype=float)
    q_all[cal] = split["q_cal"]
    q_all[te] = split["q_te"]
    # Fill train region with wrappee predict on train covariate
    tr = split["tr"]
    cov = np.asarray(split["covariate"], dtype=float)
    # For train rows, approximate bands from cal wrappee scale at covariate
    if tr.any():
        # reuse scaled wrappee from split via cal endpoints — use raw mid from q where available
        # Fill missing train with nearest available: use scaled gaussian from covariate
        from quant_fund.models.distribution import ScaledGaussianDistribution

        taus = [0.025, 0.975]
        sg = ScaledGaussianDistribution(taus).fit(y[tr], cov[tr])
        q_all[tr] = sg.predict(cov[tr])
    losses = -y
    base = -q_all[:, 0]
    # Map dates to int codes for OnlineCRC
    uniq, inv = np.unique(dates, return_inverse=True)
    date_codes = inv.astype(np.int64)
    keep = split["tr"] | cal | te
    losses = losses[keep]
    base = base[keep]
    date_codes = date_codes[keep]
    warm = int((split["tr"] | cal).sum())  # warm through end of calibration
    if warm < 20 or int(keep.sum()) - warm < 10:
        return {}
    row = bench_online_crc(
        alpha=0.05,
        losses=losses,
        base=base,
        dates=date_codes,
        n_warm=warm,
        dgp="panel",
    )
    row["target"] = split["label"]
    row["wrappee"] = str(split.get("wrappee", "unknown"))
    return row


def bench_conformal_topk_from_panel(
    frame: pl.DataFrame, config: AppConfig, *, k: int = 5, alpha: float = 0.20
) -> dict[str, Any]:
    """Conformal top-k on public-feature ridge scores vs lab ranking labels."""
    from quant_fund.models.conformal_rank import bench_conformal_topk
    from quant_fund.models.ranking import PUBLIC_FEATURES, RidgeRanker, available_features

    label = _interval_label(frame)
    if label is None:
        return {}
    feats = available_features(list(frame.columns), PUBLIC_FEATURES)
    if not feats:
        return {}
    x, y, dates, _fn, _ids = design_matrix(frame, label, feature_names=feats)
    if x.shape[0] < 80:
        return {}
    # Chronological train for scores; conformal_topk does its own cal/test split on dates
    tr, _cal, _te = _triple_split(dates, horizon=_label_horizon(label))
    if int(tr.sum()) < 30:
        return {}
    model = RidgeRanker().fit(x[tr], y[tr])
    scores = model.predict(x)
    # Integer date codes for grouping
    uniq, inv = np.unique(dates, return_inverse=True)
    if int(uniq.size) < 20:
        return {}
    row = bench_conformal_topk(
        k=k,
        alpha=alpha,
        scores=scores,
        labels=y,
        dates=inv.astype(np.int64),
        dgp="panel",
    )
    row["target"] = label
    row["score_model"] = "public_ridge"
    return row


def bench_portfolio_from_panel(
    frame: pl.DataFrame, config: AppConfig, *, alpha: float = 0.10
) -> dict[str, Any]:
    """Equal-weight book conformal sets on lab panel returns (one set per date)."""
    from quant_fund.portfolio.portfolio_conformal import bench_portfolio_cqr

    label = _interval_label(frame)
    if label is None and "future_return_1" in frame.columns:
        label = "future_return_1"
    if label is None:
        return {}
    need = ["event_time", "security_id", label]
    if any(c not in frame.columns for c in need):
        return {}
    sub = frame.select(need).drop_nulls()
    if sub.height < 80:
        return {}
    # Build date -> return vector (aligned name order per date)
    dates = _date_keys(sub["event_time"].to_numpy())
    ids = np.asarray(sub["security_id"].to_numpy()).astype(str)
    rets = sub[label].to_numpy().astype(float)
    by_date: dict[str, list[tuple[str, float]]] = {}
    for d, i, r in zip(dates, ids, rets, strict=True):
        by_date.setdefault(str(d), []).append((i, float(r)))
    if len(by_date) < 40:
        return {}
    ordered = sorted(by_date.keys())
    # Fixed universe = names present on first date with enough overlap; use intersection size
    weights_map: dict[str, np.ndarray] = {}
    returns_map: dict[str, np.ndarray] = {}
    for d in ordered:
        pairs = by_date[d]
        n = len(pairs)
        if n < 2:
            continue
        returns_map[d] = np.array([r for _i, r in pairs], dtype=float)
        weights_map[d] = np.full(n, 1.0 / n, dtype=float)
    if len(returns_map) < 40:
        return {}
    row = bench_portfolio_cqr(weights_map, returns_map, alpha=alpha, dgp="panel")
    row["target"] = label
    row["weight_rule"] = "equal_weight_per_date"
    return row


__all__ = [
    "bench_conformal",
    "bench_conformal_topk_from_panel",
    "bench_cpcv_audit",
    "bench_crc",
    "bench_cv_plus",
    "bench_evalues",
    "bench_interval_risk",
    "bench_jackknife_plus",
    "bench_localized_from_panel",
    "bench_online_crc_from_panel",
    "bench_portfolio_from_panel",
    "bench_quantile_bandit",
    "bench_weighted_conformal",
]
