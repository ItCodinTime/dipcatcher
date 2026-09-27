"""Split and Mondrian conformal sets at a decision date.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.metrics.conformal import assign_terciles
from quant_fund.metrics.cross_section import _date_keys
from quant_fund.models.conformal import MondrianCQR, SplitCQR
from quant_fund.models.distribution import (
    GaussianDistribution,
    ScaledGaussianDistribution,
    ScaledStudentTDistribution,
)
from quant_fund.pipeline.dataset import design_matrix
from quant_fund.schemas.forecast import IntervalMethod

from .history import (
    ForecastIntervals,
    _align_col,
    _date_train_cal,
    _decision_x,
    _distribution_label,
    _horizon_bars,
    _horizon_name,
    history_for_calibration,
    slice_day,
)
from .state import _CONFORMAL_CACHE, INTERVAL_ALPHA
from .wrappee import resolve_wrappee_reselect_cached


def conformal_sets_asof(
    frame: pl.DataFrame,
    asof: datetime,
    config: AppConfig,
    *,
    alpha: float = INTERVAL_ALPHA,
    day_index: dict[str, pl.DataFrame] | None = None,
    assume_sorted: bool = False,
    event_times: list[datetime] | None = None,
) -> ForecastIntervals | None:
    """Fit scaled (t) bands on train dates; calibrate CQR on later dates; apply at ``asof``.

    Decision-bar ``y`` is never in the calibration window. Cross-section sets are
    predicted on the asof date only (dates are not stacked for the set update).
    Homoskedastic Gaussian is used only when vol_20 is missing.
    """
    # The label/horizon drive the fit, so resolve them before the cache lookup:
    # two configs with different distribution targets on the same frame/asof
    # must not share cached intervals.
    label = _distribution_label(frame, config)
    if label is None:
        return None
    horizon_bars = _horizon_bars(label, config)
    horizon = _horizon_name(horizon_bars, config)
    # Cache key uses asof + alpha + label + frame height fingerprint (Phase 18).
    # Never key by ``id(frame)``: Python may reuse an object id after a prior
    # panel is collected, which can return intervals fitted on a different
    # dataset.  A content fingerprint keeps the cache an optimization only.
    row_hash = tuple(int(value) for value in frame.hash_rows().to_list())
    cache_key = (
        str(asof),
        float(alpha),
        str(label),
        int(horizon_bars),
        int(frame.height),
        tuple(frame.columns),
        row_hash,
    )
    hit = _CONFORMAL_CACHE.get(cache_key)
    if hit is not None:
        return hit
    hist = history_for_calibration(
        frame,
        asof,
        horizon_bars,
        day_index=day_index,
        assume_sorted=assume_sorted,
        event_times=event_times,
    )
    if hist.is_empty():
        return None
    x, y, dates, feats, ids = design_matrix(hist, label)
    if x.shape[0] < 24:
        return None
    tr, cal = _date_train_cal(dates)
    if int(tr.sum()) < 8 or int(cal.sum()) < 8:
        return None
    taus = [alpha / 2.0, 1.0 - alpha / 2.0]
    gauss: ScaledGaussianDistribution | ScaledStudentTDistribution | GaussianDistribution
    vol_tr = _align_col(hist, dates[tr], ids[tr], "vol_20")
    vol_cal = _align_col(hist, dates[cal], ids[cal], "vol_20")
    if (
        vol_tr is not None
        and vol_cal is not None
        and bool(np.isfinite(vol_tr).any())
        and bool(np.isfinite(vol_cal).any())
    ):
        med_tr = float(np.nanmedian(vol_tr[np.isfinite(vol_tr)]))
        scale_tr = np.where(np.isfinite(vol_tr), vol_tr, med_tr)
        med = float(np.nanmedian(vol_cal[np.isfinite(vol_cal)]))
        scale_cal = np.where(np.isfinite(vol_cal), vol_cal, med)
        # Wave 7: re-select family on current cal; reuse train fit when
        # (train-primary fp, family) hits. Cal growth can change family without
        # forcing a train MLE when the family is unchanged. Student-t MLE is hot.
        _, gauss, _wrap_meta = resolve_wrappee_reselect_cached(
            taus,
            y[tr],
            scale_tr,
            y[cal],
            scale_cal,
            train_date_keys=tuple(sorted(set(_date_keys(dates[tr])))),
            cal_date_keys=tuple(sorted(set(_date_keys(dates[cal])))),
            alpha=float(alpha),
            label=str(label),
            nominal_coverage=1.0 - alpha,
        )
        _ = _wrap_meta
        q_cal = gauss.predict(scale_cal)
    else:
        gauss = GaussianDistribution(taus).fit(x[tr], y[tr])
        q_cal = gauss.predict(x[cal])
        scale_cal = None
        med = 0.01
    day = slice_day(frame, asof, day_index=day_index)
    if day.is_empty():
        return None
    day_ids = [str(v) for v in day["security_id"].to_list()]
    scaled = isinstance(gauss, ScaledGaussianDistribution | ScaledStudentTDistribution)
    if scaled and "vol_20" in day.columns:
        day_vol_pre = day["vol_20"].to_numpy().astype(float)
        day_vol_pre = np.where(np.isfinite(day_vol_pre), day_vol_pre, med)
        q_day = gauss.predict(day_vol_pre)
    else:
        q_day = gauss.predict(_decision_x(day, feats))
    if scale_cal is not None and bool(np.isfinite(scale_cal).any()):
        lab_cal, cuts = assign_terciles(scale_cal, prefix="vol_20")
        mondrian = MondrianCQR(alpha).calibrate(
            y[cal], q_cal[:, 0], q_cal[:, 1], lab_cal, scale_cal
        )
        if "vol_20" in day.columns:
            day_vol = day["vol_20"].to_numpy().astype(float)
        else:
            day_vol = np.full(day.height, med)
        day_vol = np.where(np.isfinite(day_vol), day_vol, med)
        lab_day, _ = assign_terciles(day_vol, cuts, prefix="vol_20")
        lo, hi = mondrian.predict_sets(q_day[:, 0], q_day[:, 1], lab_day, day_vol)
        method: IntervalMethod = "mondrian_cqr"
    else:
        cqr = SplitCQR(alpha).calibrate(y[cal], q_cal[:, 0], q_cal[:, 1])
        lo, hi = cqr.predict_sets(q_day[:, 0], q_day[:, 1])
        method = "split_cqr"
    result = ForecastIntervals(
        lower={sid: float(lo[i]) for i, sid in enumerate(day_ids)},
        upper={sid: float(hi[i]) for i, sid in enumerate(day_ids)},
        alpha=float(alpha),
        method=method,
        horizon=horizon,
        cal_event_times=tuple(sorted(set(dates[cal].tolist()), key=str)),
    )
    if len(_CONFORMAL_CACHE) > 256:
        _CONFORMAL_CACHE.clear()
    _CONFORMAL_CACHE[cache_key] = result
    return result


__all__ = [
    "conformal_sets_asof",
]
