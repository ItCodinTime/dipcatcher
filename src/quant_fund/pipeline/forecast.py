"""Produce AssetForecasts and target weights for a decision date."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.config.models import AppConfig
from quant_fund.fusion.engine import fuse_signals
from quant_fund.metrics.conformal import assign_terciles
from quant_fund.metrics.cross_section import _date_keys
from quant_fund.models.conformal import MondrianCQR, SplitCQR
from quant_fund.models.covariance import ledoit_wolf_cov, repair_psd, sample_cov
from quant_fund.models.distribution import (
    GaussianDistribution,
    ScaledGaussianDistribution,
    ScaledStudentTDistribution,
    select_scaled_wrappee,
)
from quant_fund.models.ranking import RidgeRanker, available_features
from quant_fund.pipeline.dataset import design_matrix, panel
from quant_fund.portfolio.optimizer import optimize_mean_variance
from quant_fund.schemas.forecast import AssetForecast, IntervalMethod, MarketState

Array = NDArray[np.float64]

INTERVAL_ALPHA = 0.10


@dataclass(frozen=True)
class ForecastIntervals:
    """CQR / Mondrian sets. Separate from raw quantile PIT, pinball, and CRPS."""

    lower: dict[str, float]
    upper: dict[str, float]
    alpha: float
    method: IntervalMethod
    horizon: str
    cal_event_times: tuple[object, ...]


def latest_decision(frame: pl.DataFrame) -> datetime:
    value = frame["event_time"].max()
    if not isinstance(value, datetime):
        raise TypeError("event_time max is not a datetime")
    return value


def _distribution_label(frame: pl.DataFrame, config: AppConfig) -> str | None:
    label = config.train.distribution_target
    if label in frame.columns:
        return label
    for prefix in ("future_log_return", "future_idio_return", "future_return"):
        cands = [c for c in frame.columns if c.startswith(prefix)]
        if cands:
            return cands[0]
    return None


def _horizon_bars(label: str, config: AppConfig) -> int:
    tail = label.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return max(config.horizons.bars)


def _horizon_name(bars: int, config: AppConfig) -> str:
    for b, name in zip(config.horizons.bars, config.horizons.names, strict=True):
        if b == bars:
            return name
    return f"{bars}d"


def history_for_calibration(
    frame: pl.DataFrame, asof: datetime, horizon_bars: int
) -> pl.DataFrame:
    """Rows whose forward labels are realized before ``asof``. Never the decision bar."""
    times = frame["event_time"].unique().sort().to_list()
    if not times:
        return frame.head(0)
    idx = next((i for i, t in enumerate(times) if t >= asof), len(times))
    last = idx - int(horizon_bars) - 1
    if last < 0:
        last = idx - 1
    if last < 0:
        return frame.head(0)
    return frame.filter(pl.col("event_time") <= times[last])


def _align_col(
    frame: pl.DataFrame, dates: np.ndarray, ids: np.ndarray, name: str
) -> Array | None:
    if name not in frame.columns:
        return None
    sub = frame.select(["event_time", "security_id", name]).drop_nulls()
    lookup = {
        (d, str(i)): float(v)
        for d, i, v in zip(
            _date_keys(sub["event_time"].to_numpy()),
            sub["security_id"].to_numpy(),
            sub[name].to_numpy().astype(float),
            strict=False,
        )
    }
    return np.array(
        [
            lookup.get((d, str(i)), np.nan)
            for d, i in zip(_date_keys(dates), ids, strict=True)
        ],
        dtype=float,
    )


def _date_train_cal(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split on unique dates so a session is never half-train, half-cal."""
    keys = _date_keys(dates)
    uniq = sorted(set(keys))
    empty = np.zeros(len(keys), dtype=bool)
    if len(uniq) < 6:
        return empty, empty
    cut = min(max(int(0.7 * len(uniq)), 3), len(uniq) - 2)
    train_keys = set(uniq[:cut])
    cal_keys = set(uniq[cut:])
    tr = np.array([k in train_keys for k in keys], dtype=bool)
    cal = np.array([k in cal_keys for k in keys], dtype=bool)
    return tr, cal


def _decision_x(day: pl.DataFrame, feats: list[str]) -> Array:
    if not feats:
        return np.zeros((day.height, 1))
    use = [c for c in feats if c in day.columns]
    if not use:
        return np.zeros((day.height, len(feats)))
    arr = day.select(use).fill_null(0.0).to_numpy().astype(float)
    if arr.shape[1] == len(feats):
        return arr
    out = np.zeros((day.height, len(feats)))
    for j, col in enumerate(use):
        out[:, feats.index(col)] = arr[:, j]
    return out


def conformal_sets_asof(
    frame: pl.DataFrame,
    asof: datetime,
    config: AppConfig,
    *,
    alpha: float = INTERVAL_ALPHA,
) -> ForecastIntervals | None:
    """Fit scaled (t) bands on train dates; calibrate CQR on later dates; apply at ``asof``.

    Decision-bar ``y`` is never in the calibration window. Cross-section sets are
    predicted on the asof date only (dates are not stacked for the set update).
    Homoskedastic Gaussian is used only when vol_20 is missing.
    """
    label = _distribution_label(frame, config)
    if label is None:
        return None
    horizon_bars = _horizon_bars(label, config)
    horizon = _horizon_name(horizon_bars, config)
    hist = history_for_calibration(frame, asof, horizon_bars)
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
        _, gauss = select_scaled_wrappee(
            taus, y[tr], scale_tr, y[cal], scale_cal, nominal_coverage=1.0 - alpha
        )
        q_cal = gauss.predict(scale_cal)
    else:
        gauss = GaussianDistribution(taus).fit(x[tr], y[tr])
        q_cal = gauss.predict(x[cal])
        scale_cal = None
        med = 0.01
    day = frame.filter(pl.col("event_time") == asof)
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
    return ForecastIntervals(
        lower={sid: float(lo[i]) for i, sid in enumerate(day_ids)},
        upper={sid: float(hi[i]) for i, sid in enumerate(day_ids)},
        alpha=float(alpha),
        method=method,
        horizon=horizon,
        cal_event_times=tuple(sorted(set(dates[cal].tolist()), key=str)),
    )


def forecast_asof(
    config: AppConfig, asof: datetime | None = None, *, interval_alpha: float = INTERVAL_ALPHA
) -> MarketState:
    df = panel(config)
    if asof is None:
        asof = latest_decision(df)
    day = df.filter(pl.col("event_time") == asof)
    if day.is_empty():
        asof = latest_decision(df)
        day = df.filter(pl.col("event_time") == asof)
    feats = available_features(day.columns)
    x = (
        day.select(feats).fill_null(0.0).to_numpy().astype(float)
        if feats
        else np.zeros((day.height, 1))
    )
    rank_path = Path(config.data.root) / "metadata" / "ranker_ridge.joblib"
    if rank_path.exists():
        model = RidgeRanker.load(rank_path)
        scores = model.predict(x)
    else:
        scores = (
            day["cs_pct_mom_20"].fill_null(0.5).to_numpy().astype(float)
            if "cs_pct_mom_20" in day.columns
            else np.zeros(day.height)
        )
    # percentile ranks within the day
    order = scores.argsort().argsort()
    pct = (order + 0.5) / max(len(scores), 1)
    vol = (
        day["vol_20"].fill_null(0.02).to_numpy().astype(float)
        if "vol_20" in day.columns
        else np.full(day.height, 0.02)
    )
    alpha = (pct - 0.5) * 2.0 * config.fusion.alpha_scale
    conf = np.clip(0.5 + np.abs(pct - 0.5), 0.2, 1.0)
    regime = np.ones(day.height)
    tail = np.clip(vol * 0.1, 0, None)
    liq = np.zeros(day.height)
    fused = fuse_signals(alpha, conf, regime, vol, tail, liq, config.fusion)
    notes = []
    if config.data.source == "synthetic":
        notes.append("SYNTHETIC")
    intervals = conformal_sets_asof(df, asof, config, alpha=interval_alpha)
    forecasts = []
    for i, row in enumerate(day.iter_rows(named=True)):
        q05 = float(alpha[i] - 1.65 * vol[i])
        q50 = float(alpha[i])
        q95 = float(alpha[i] + 1.65 * vol[i])
        sid = str(row["security_id"])
        lo_map: dict[str, float] = {}
        hi_map: dict[str, float] = {}
        method: IntervalMethod | None = None
        i_alpha: float | None = None
        if intervals is not None and sid in intervals.lower:
            lo_map = {intervals.horizon: intervals.lower[sid]}
            hi_map = {intervals.horizon: intervals.upper[sid]}
            method = intervals.method
            i_alpha = intervals.alpha
        forecasts.append(
            AssetForecast(
                security_id=sid,
                symbol=row.get("symbol", sid),
                asof=asof,
                model_version="fusion.v1",
                expected_returns={"5d": float(alpha[i])},
                quantiles={"5d": {0.05: q05, 0.5: q50, 0.95: q95}},
                probability_positive={"5d": float(pct[i])},
                alpha={"5d": float(alpha[i])},
                rank_score={"5d": float(scores[i])},
                rank_percentile={"5d": float(pct[i])},
                volatility={"5d": float(vol[i])},
                confidence={"5d": float(conf[i])},
                diagnostics={"fused": float(fused[i])},
                interval_lo=lo_map,
                interval_hi=hi_map,
                interval_alpha=i_alpha,
                interval_method=method,
            )
        )
    return MarketState(asof=asof, forecasts=forecasts, notes=notes)


def optimize_asof(config: AppConfig, asof: datetime | None = None) -> pl.DataFrame:
    state = forecast_asof(config, asof)
    ids = [f.security_id for f in state.forecasts]
    alpha = np.array([f.alpha.get("5d", 0.0) for f in state.forecasts])
    # covariance from trailing returns if present
    hist = panel(config).filter(pl.col("event_time") <= state.asof)
    if "ret_1" in hist.columns and hist.height > 20:
        wide = hist.select(["event_time", "security_id", "ret_1"]).pivot(
            on="security_id", index="event_time", values="ret_1"
        )
        cols = [c for c in ids if c in wide.columns]
        mat = wide.select(cols).to_numpy()
        sig = ledoit_wolf_cov(mat) if mat.shape[0] > mat.shape[1] else sample_cov(mat)
        sig, _ = repair_psd(sig)
        alpha = np.array([dict(zip(ids, alpha, strict=False)).get(c, 0.0) for c in cols])
        ids = cols
    else:
        sig = np.diag(np.ones(len(ids)) * 0.02**2)
    w_prev = np.zeros(len(ids))
    try:
        w, diag = optimize_mean_variance(alpha, sig, w_prev, config)
    except Exception:
        w = np.zeros(len(ids))
        diag = None
    out = pl.DataFrame(
        {
            "event_time": [state.asof] * len(ids),
            "security_id": ids,
            "target_weight": w.tolist(),
            "alpha": alpha.tolist(),
        }
    )
    Path(config.data.root).joinpath("gold").mkdir(parents=True, exist_ok=True)
    out.write_parquet(Path(config.data.root) / "gold" / "target_weights.parquet")
    _ = diag
    return out
