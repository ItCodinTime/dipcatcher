"""Shared bench split and interval helpers.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.config.models import AppConfig
from quant_fund.metrics.cross_section import _date_keys
from quant_fund.metrics.scoring import pinball_loss
from quant_fund.models.distribution import (
    GaussianDistribution,
    ScaledGaussianDistribution,
    ScaledStudentTDistribution,
    select_scaled_wrappee,
)
from quant_fund.models.ranking import PUBLIC_FEATURES, available_features
from quant_fund.pipeline.dataset import design_matrix
from quant_fund.pipeline.train import _label_horizon
from quant_fund.validation.purging import purge_mask


def _holdout(
    dates: NDArray[Any], frac: float = 0.3, *, horizon: int = 1
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Chronological (train, eval) row masks cut at a unique-date boundary.

    A positional row slice can split one date across train/eval, and a bare
    boundary lets the last ``horizon`` train dates carry labels that realize
    inside the eval window — the leak the fold path purges. Cut on unique
    dates via :func:`purge_mask`; an emptied train side comes back as an
    all-False mask and callers must fail closed (skip the lane).
    """
    keys = np.asarray(dates)
    uniq = np.unique(keys)
    n = uniq.size
    if n < 2:
        return np.zeros(keys.shape[0], dtype=bool), np.zeros(keys.shape[0], dtype=bool)
    cut = max(int(n * (1.0 - frac)), n // 2)
    cut = min(max(cut, 1), n - 1)
    eval_dates = list(uniq[cut:])
    safe = purge_mask(list(uniq), eval_dates[0], eval_dates[-1], max(int(horizon), 0))
    train_dates = uniq[:cut][np.asarray(safe[:cut], dtype=bool)]
    return np.isin(keys, train_dates), np.isin(keys, eval_dates)


def _triple_split(
    dates: NDArray[Any], *, horizon: int = 1
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Chronological train / calibration / test masks, purged at both edges.

    Train is purged against the calibration window and calibration against
    the test window — a calibration row whose label reaches test leaks the
    test outcome into the fitted scaler. Emptied sides are empty masks;
    callers fail closed.
    """
    keys = np.asarray(dates)
    uniq = np.unique(keys)
    n = uniq.size
    empty = np.zeros(keys.shape[0], dtype=bool)
    if n < 3:
        return empty, empty, empty
    if n < 30:
        a, b = max(n // 3, 1), max(2 * n // 3, 2)
    else:
        a = max(int(0.5 * n), 8)
        b = max(int(0.7 * n), a + 5)
    b = min(max(b, a + 1), n - 1)
    h = max(int(horizon), 0)
    cal_dates = uniq[a:b]
    test_dates = uniq[b:]
    tr_safe = purge_mask(list(uniq), cal_dates[0], cal_dates[-1], h)
    train_dates = uniq[:a][np.asarray(tr_safe[:a], dtype=bool)]
    cal_safe = purge_mask(list(uniq), test_dates[0], test_dates[-1], h)
    kept_cal = cal_dates[np.asarray(cal_safe[a:b], dtype=bool)]
    return (
        np.isin(keys, train_dates),
        np.isin(keys, kept_cal),
        np.isin(keys, test_dates),
    )


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


def _crps_from_quantiles_obs(
    y: NDArray[np.float64], quantiles: NDArray[np.float64], taus: list[float] | NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-observation Riemann CRPS (same construction as ``crps_from_quantiles`` mean).

    Research-only series for Diebold–Mariano — not a live capital claim.
    """
    yy = np.asarray(y, dtype=float).reshape(-1)
    q = np.asarray(quantiles, dtype=float)
    t = np.asarray(taus, dtype=float)
    if q.ndim != 2 or q.shape[0] != yy.shape[0] or q.shape[1] != t.size:
        raise ValueError("quantiles must be (n, k) matching y and taus")
    if yy.size == 0:
        return np.asarray([], dtype=float)
    dt = np.diff(np.concatenate([[0.0], t]))
    total = np.zeros(yy.shape[0], dtype=float)
    for k, tau in enumerate(t):
        # Match crps_from_quantiles: CRPS = 2 * integral pinball dtau.
        total += 2.0 * pinball_loss(yy, q[:, k], float(tau)) * float(dt[k])
    return total


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
    q_tr: NDArray[np.float64],
    q_cal: NDArray[np.float64],
    q_te: NDArray[np.float64],
    tr: NDArray[np.bool_],
    cal: NDArray[np.bool_],
    te: NDArray[np.bool_],
    n: int,
) -> tuple[NDArray[np.float64], str]:
    """PIT-safe scale covariate: aligned ``vol_20`` when present.

    Fallback: raw-Gaussian band width on every split (train included — the
    wrappee is fitted on the train slice, so a placeholder scale there would
    make its standardized residual scale explode).
    """
    vol = _aligned_col(frame, dates, ids, "vol_20")
    if vol is not None and np.isfinite(vol).any():
        return _scaled_fill(vol), "vol_20"
    scale = np.full(n, 1e-8)
    for sl, q in ((tr, q_tr), (cal, q_cal), (te, q_te)):
        scale[sl] = np.maximum(q[:, 1] - q[:, 0], 1e-8)
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
    tr, cal, te = _triple_split(dates, horizon=_label_horizon(label))
    if not tr.any() or not cal.any() or not te.any():
        return None
    gauss = GaussianDistribution(taus).fit(x[tr], y[tr])
    q_raw_tr = gauss.predict(x[tr])
    q_raw_cal = gauss.predict(x[cal])
    q_raw_te = gauss.predict(x[te])
    covariate, cov_name = _vol_or_width(
        frame, dates, ids, q_raw_tr, q_raw_cal, q_raw_te, tr, cal, te, y.size
    )
    wrappee_name, wrappee = select_scaled_wrappee(
        taus,
        y[tr],
        covariate[tr],
        y[cal],
        covariate[cal],
        nominal_coverage=1.0 - alpha,
    )
    q_tr = wrappee.predict(covariate[tr])
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
        "q_tr": q_tr,
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
