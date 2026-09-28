"""Asset forecasts, target weights, and causal weight panels.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.data.point_in_time import filter_trailing_returns_asof
from quant_fund.fusion.engine import fuse_signals
from quant_fund.models.covariance import (
    IMPLEMENTED_OPTIMIZER_NAMED_SPECS,
    IMPLEMENTED_OPTIMIZER_ONE_STEP_SPECS,
    require_implemented_optimizer_covariance,
)
from quant_fund.models.ranking import available_features
from quant_fund.models.robinhood_plus.constants import ENGINE_NAME, MODEL_VERSION, STATUS_OK
from quant_fund.models.robinhood_plus.engine import (
    RobinhoodPlusNameForecast,
    forecast_robinhood_plus_cross_section,
    kline_columns_present,
)
from quant_fund.pipeline.dataset import panel
from quant_fund.portfolio.interval_risk import apply_interval_caps, interval_refs
from quant_fund.portfolio.optimizer import optimize_mean_variance
from quant_fund.schemas.errors import OptimizationInfeasible
from quant_fund.schemas.forecast import AssetForecast, IntervalMethod, MarketState

from .artifacts import (
    _load_probability_calibrator,
    _load_ranker_cached,
    _load_rl_cached,
    _paper_challenger_stamp,
)
from .conformal import conformal_sets_asof
from .covariance import (
    _market_overlay_diagnostics,
    _market_overlay_note,
    _unmeasured_optimizer_covariance,
    estimate_optimizer_covariance_asof,
)
from .history import (
    _require_assume_sorted_contract,
    build_event_time_day_index,
    history_prefix_upto,
    latest_decision,
    slice_day,
    sort_for_history,
    under_history_sort_contract,
)
from .realized import (
    apply_market_variance_overlay_to_covariance,
    resolve_market_variance_overlay_asof,
)
from .state import INTERVAL_ALPHA, Array, log


def _panel_cached(config: AppConfig) -> pl.DataFrame:
    """Thin wrapper — dataset.panel already caches; keep call site explicit."""
    return panel(config)


def _count_robinhood_plus_ok(
    rh_map: dict[str, RobinhoodPlusNameForecast],
    security_ids: list[str],
) -> int:
    n_ok = 0
    for sid in security_ids:
        hit = rh_map.get(sid)
        if hit is None or hit.forecast.status != STATUS_OK:
            continue
        mu = float(hit.forecast.expected_returns.get("5d", hit.forecast.rank_score))
        if np.isfinite(mu):
            n_ok += 1
    return n_ok


def _robinhood_plus_asof(
    frame: pl.DataFrame,
    asof: datetime,
    config: AppConfig,
    security_ids: list[str],
) -> dict[str, RobinhoodPlusNameForecast]:
    """Causal robinhood+ K-line forecasts. Torch backend is local-weights only."""
    cfg = config.robinhood_plus
    if not cfg.enabled:
        return {}
    if cfg.backend.value == "torch":
        from quant_fund.models.robinhood_plus.torch_backend import forecast_cross_section_torch

        return forecast_cross_section_torch(frame, asof, config, security_ids)
    if not kline_columns_present(frame.columns):
        return {}
    return forecast_robinhood_plus_cross_section(
        frame,
        asof,
        lookback=cfg.lookback,
        pred_len=cfg.pred_len,
        sample_count=cfg.sample_count,
        s1_bits=cfg.s1_bits,
        s2_bits=cfg.s2_bits,
        clip=cfg.clip,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_context=cfg.max_context,
        seed=config.train.random_seed,
        decoder=cfg.decoder.value,
        security_ids=security_ids,
        quantile_levels=tuple(float(q) for q in config.quantiles.levels),
        horizons=tuple(int(h) for h in config.horizons.bars),
    )


def forecast_asof(
    config: AppConfig,
    asof: datetime | None = None,
    *,
    interval_alpha: float = INTERVAL_ALPHA,
    frame: pl.DataFrame | None = None,
    day_index: dict[str, pl.DataFrame] | None = None,
    assume_sorted: bool = False,
    event_times: list[datetime] | None = None,
) -> MarketState:
    df = frame if frame is not None else _panel_cached(config)
    # day_index is optional: build once in causal loops and pass through.
    # Single-asof calls keep a plain filter (building a full index is slower).
    if asof is None:
        asof = latest_decision(df)
    day = slice_day(df, asof, day_index=day_index)
    if day.is_empty():
        # An explicitly requested decision date with no panel rows must fail
        # closed: silently substituting the latest date would persist
        # end-of-sample weights for the wrong as-of (look-ahead).
        raise ValueError(
            f"no panel rows for requested asof={asof!r} (refusing latest-date fallback)"
        )
    model = _load_ranker_cached(config)
    ranker_features = getattr(model, "features", None) if model is not None else None
    if ranker_features is not None:
        if not isinstance(ranker_features, list) or not ranker_features:
            raise ValueError("ranker artifact has an invalid feature contract")
        missing = [feature for feature in ranker_features if feature not in day.columns]
        if missing:
            raise ValueError(f"ranker artifact feature contract missing columns: {missing}")
        feats = [str(feature) for feature in ranker_features]
    else:
        feats = available_features(day.columns)
    x = (
        day.select(feats).fill_null(0.0).to_numpy().astype(float)
        if feats
        else np.zeros((day.height, 1))
    )
    notes: list[str] = []
    if model is not None:
        # A stale joblib / feature-set mismatch must surface: silently degrading
        # to the momentum heuristic would masquerade a fabricated alpha as model
        # output. The heuristic is used only when no ranker is loaded at all.
        scores = model.predict(x)
    else:
        rl_artifact = _load_rl_cached(config)
        if rl_artifact is not None:
            policy, rl_features, policy_note = rl_artifact
            missing = [feature for feature in rl_features if feature not in day.columns]
            if missing:
                raise ValueError(f"RL artifact feature contract missing columns: {missing}")
            rl_x = day.select(rl_features).fill_null(0.0).to_numpy().astype(float)
            score_fn = getattr(policy, "scores", None) or getattr(policy, "predict", None)
            if not callable(score_fn):
                raise ValueError("RL artifact policy does not expose a callable scorer")
            scores = score_fn(rl_x)
            notes = [policy_note]
        else:
            scores = (
                day["cs_pct_mom_20"].fill_null(0.5).to_numpy().astype(float)
                if "cs_pct_mom_20" in day.columns
                else np.zeros(day.height)
            )
    # percentile ranks within the day
    order = scores.argsort().argsort()
    pct = (order + 0.5) / max(len(scores), 1)
    if config.fusion.apply_probability_calibration:
        calibrator = _load_probability_calibrator(config, asof=asof)
        if "cs_pct_mom_20" not in day.columns:
            raise ValueError("calibration requires cs_pct_mom_20 in the forecast panel")
        raw_probability = day["cs_pct_mom_20"].fill_null(0.5).to_numpy().astype(float)
        pct = np.asarray(calibrator.predict(raw_probability), dtype=float)
        notes.append(f"CALIBRATED_{calibrator.method.upper()}")
    vol = (
        day["vol_20"].fill_null(0.02).to_numpy().astype(float)
        if "vol_20" in day.columns
        else np.full(day.height, 0.02)
    )
    alpha = (pct - 0.5) * 2.0 * config.fusion.alpha_scale
    conf = np.clip(0.5 + np.abs(pct - 0.5), 0.2, 1.0)
    security_ids = [str(row["security_id"]) for row in day.iter_rows(named=True)]
    blend = float(config.robinhood_plus.blend_weight) if config.robinhood_plus.enabled else 0.0
    # blend_weight=0: keep the engine as a stamped challenger and do not size
    # the book. Still emit n_ok / n_fallback so a missing OHLC window cannot
    # masquerade as a Kronos path when the engine *is* applied.
    rh_map: dict[str, RobinhoodPlusNameForecast] = {}
    if config.robinhood_plus.enabled and blend > 0.0:
        rh_map = _robinhood_plus_asof(df, asof, config, security_ids)
    rh_ok = _count_robinhood_plus_ok(rh_map, security_ids)
    n_fallback = int(len(security_ids) - rh_ok)
    if rh_map and blend > 0.0:
        for i, sid in enumerate(security_ids):
            hit = rh_map.get(sid)
            if hit is None or hit.forecast.status != STATUS_OK:
                continue
            mu = float(hit.forecast.expected_returns.get("5d", hit.forecast.rank_score))
            if not np.isfinite(mu):
                continue
            scores[i] = (1.0 - blend) * float(scores[i]) + blend * float(hit.forecast.rank_score)
            alpha[i] = (1.0 - blend) * float(alpha[i]) + blend * mu
            conf[i] = (1.0 - blend) * float(conf[i]) + blend * float(hit.forecast.confidence)
        order = scores.argsort().argsort()
        pct = (order + 0.5) / max(len(scores), 1)
        conf = np.clip(conf, 0.2, 1.0)
    regime = np.ones(day.height)
    tail = np.clip(vol * 0.1, 0, None)
    liq = np.zeros(day.height)
    fused = fuse_signals(alpha, conf, regime, vol, tail, liq, config.fusion)
    if config.data.source == "synthetic":
        notes.append("SYNTHETIC")
    notes.append(f"robinhood_plus_n_ok={rh_ok}")
    notes.append(f"robinhood_plus_n_fallback={n_fallback}")
    notes.append(f"robinhood_plus_blend_weight={blend}")
    paper_note, paper_diag = _paper_challenger_stamp(
        config,
        x,
        scores,
        dates=np.asarray([asof] * len(scores), dtype=object),
        ids=np.asarray(security_ids, dtype=object),
    )
    notes.append(paper_note)
    if config.robinhood_plus.enabled and blend <= 0.0:
        notes.append("robinhood_plus_challenger")
    if rh_ok and blend > 0.0:
        notes.append(ENGINE_NAME)
    overlay, overlay_kind = resolve_market_variance_overlay_asof(config, df, asof)
    if overlay is not None and overlay_kind is not None:
        notes.append(_market_overlay_note(overlay_kind))
    # Research-only weight smoke: skip conformal when fusion.skip_intervals is set.
    if bool(getattr(config.fusion, "skip_intervals", False)):
        intervals = None
    else:
        intervals = conformal_sets_asof(
            df,
            asof,
            config,
            alpha=interval_alpha,
            day_index=day_index,
            assume_sorted=assume_sorted,
            event_times=event_times,
        )
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
        diagnostics: dict[str, float | str] = {
            "fused": float(fused[i]),
            "robinhood_plus_n_ok": float(rh_ok),
            "robinhood_plus_n_fallback": float(n_fallback),
            "robinhood_plus_blend_weight": float(blend),
            **paper_diag,
        }
        if overlay is not None and overlay_kind is not None:
            diagnostics.update(_market_overlay_diagnostics(overlay, overlay_kind))
        expected = {"5d": float(alpha[i])}
        q_map: dict[str, dict[float, float]] = {"5d": {0.05: q05, 0.5: q50, 0.95: q95}}
        p_map = {"5d": float(pct[i])}
        hit = rh_map.get(sid)
        if hit is not None and hit.forecast.status == STATUS_OK:
            diagnostics["core_engine"] = ENGINE_NAME
            diagnostics["robinhood_plus_status"] = hit.forecast.status
            diagnostics["robinhood_plus_model_version"] = MODEL_VERSION
            diagnostics["robinhood_plus_backend"] = str(
                hit.forecast.diagnostics.get("backend", "numpy")
            )
            diagnostics["robinhood_plus_decoder"] = str(hit.forecast.diagnostics.get("decoder", ""))
            if hit.forecast.expected_returns:
                expected = dict(hit.forecast.expected_returns)
                expected["5d"] = float(alpha[i])
            if hit.forecast.quantiles:
                q_map = {
                    hz: {float(level): float(val) for level, val in levels.items()}
                    for hz, levels in hit.forecast.quantiles.items()
                }
            if hit.forecast.probability_positive:
                p_map = {
                    hz: float(np.clip(val, 0.0, 1.0))
                    for hz, val in hit.forecast.probability_positive.items()
                }
        elif config.robinhood_plus.enabled:
            diagnostics["core_engine"] = "ridge"
            diagnostics["robinhood_plus_status"] = (
                hit.forecast.status
                if hit is not None
                else ("challenger" if blend <= 0.0 else "absent")
            )
        forecasts.append(
            AssetForecast(
                security_id=sid,
                symbol=row.get("symbol", sid),
                asof=asof,
                model_version="fusion.v1",
                expected_returns=expected,
                quantiles=q_map,
                probability_positive=p_map,
                alpha={"5d": float(alpha[i])},
                rank_score={"5d": float(scores[i])},
                rank_percentile={"5d": float(pct[i])},
                volatility={"5d": float(vol[i])},
                confidence={"5d": float(conf[i])},
                diagnostics=diagnostics,
                interval_lo=lo_map,
                interval_hi=hi_map,
                interval_alpha=i_alpha,
                interval_method=method,
            )
        )
    if overlay is None or overlay_kind is None:
        return MarketState(asof=asof, forecasts=forecasts, notes=notes)
    return MarketState(
        asof=asof,
        forecasts=forecasts,
        notes=notes,
        garch_market_sigma=float(overlay.sigma),
        garch_market_variance=float(overlay.variance),
        garch_cumulative_variance=float(overlay.cumulative_variance),
        garch_horizon=int(overlay.horizon),
        garch_series_scope=overlay.series_scope,
        garch_fit_status=overlay.fit_status,
        garch_n_obs=int(overlay.n_obs),
        market_risk_overlay=overlay_kind,
    )


def _alpha_vector(state: MarketState, config: AppConfig, *, use_fused: bool) -> Array:
    """Raw alpha or fused diagnostics used for mean-variance sizing."""
    if use_fused:
        return np.array(
            [float(f.diagnostics.get("fused", f.alpha.get("5d", 0.0))) for f in state.forecasts],
            dtype=float,
        )
    return np.array([float(f.alpha.get("5d", 0.0)) for f in state.forecasts], dtype=float)


def _align_w_prev(ids: list[str], w_prev: Array | dict[str, float] | None) -> Array:
    if w_prev is None:
        return np.zeros(len(ids), dtype=float)
    if isinstance(w_prev, dict):
        return np.array([float(w_prev.get(i, 0.0)) for i in ids], dtype=float)
    arr = np.asarray(w_prev, dtype=float).reshape(-1)
    if arr.size != len(ids):
        # Shape mismatches must raise, never silently reset the prior book to
        # flat (that would optimize from cash against a different universe).
        raise ValueError(
            f"w_prev length {arr.size} != n_ids {len(ids)}; "
            "pass a security_id-keyed dict when the universe changes"
        )
    return arr


def _apply_forecast_interval_caps(
    w: Array,
    state: MarketState,
    ids: list[str],
    config: AppConfig,
    *,
    w_prev: Array | None = None,
) -> Array:
    """Apply conformal name caps without violating the hard turnover contract."""
    if not config.fusion.apply_interval_caps:
        return w
    by_id = {f.security_id: f for f in state.forecasts}
    lo_list: list[float] = []
    hi_list: list[float] = []
    for sid in ids:
        f = by_id.get(sid)
        if f is None or not f.interval_lo or not f.interval_hi:
            lo_list.append(float("nan"))
            hi_list.append(float("nan"))
            continue
        hz = next(iter(f.interval_lo))
        lo_list.append(float(f.interval_lo[hz]))
        hi_list.append(float(f.interval_hi[hz]))
    lo = np.asarray(lo_list, dtype=float)
    hi = np.asarray(hi_list, dtype=float)
    if not np.isfinite(lo).any() or not np.isfinite(hi).any():
        # A configured risk control silently not applying must be visible.
        log.warning("interval_caps_skipped", reason="no_finite_intervals", n_ids=len(ids))
        return w
    valid = np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
    if int(valid.sum()) == 0:
        log.warning("interval_caps_skipped", reason="no_valid_intervals", n_ids=len(ids))
        return w
    width_ref, downside_ref = interval_refs(lo[valid], hi[valid])
    capped, caps = apply_interval_caps(
        w,
        lo,
        hi,
        max_weight=float(config.constraints.name_max),
        width_ref=width_ref,
        downside_ref=downside_ref,
    )
    if w_prev is None:
        return capped
    prior = np.asarray(w_prev, dtype=float).reshape(-1)
    if prior.shape != capped.shape or not np.isfinite(prior).all():
        raise ValueError("w_prev must be finite and aligned with interval-capped weights")
    turnover_limit = float(config.constraints.turnover_limit)
    capped_turnover = float(np.abs(capped - prior).sum())
    if capped_turnover <= turnover_limit + 1e-12:
        return capped

    # The closest point to the prior book that satisfies the interval boxes is
    # the coordinate-wise projection onto [-caps, caps]. If that minimum move
    # already exceeds the documented hard turnover limit, emitting capped
    # weights would silently violate the optimizer contract; fail closed.
    nearest = np.clip(prior, -caps, caps)
    minimum_turnover = float(np.abs(nearest - prior).sum())
    if minimum_turnover > turnover_limit + 1e-12:
        raise OptimizationInfeasible(
            "interval caps are incompatible with the hard turnover limit: "
            f"minimum turnover {minimum_turnover:.12g} > {turnover_limit:.12g}"
        )

    # Otherwise move from the nearest feasible point toward the capped target
    # until the L1 turnover budget is exactly respected. Both endpoints obey
    # the interval boxes, so the interpolation remains inside them.
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        candidate = nearest + mid * (capped - nearest)
        if float(np.abs(candidate - prior).sum()) <= turnover_limit:
            low = mid
        else:
            high = mid
    return np.asarray(nearest + low * (capped - nearest), dtype=float)


def optimize_asof(
    config: AppConfig,
    asof: datetime | None = None,
    *,
    persist: bool = True,
    w_prev: Array | dict[str, float] | None = None,
    use_fused_alpha: bool | None = None,
    frame: pl.DataFrame | None = None,
    day_index: dict[str, pl.DataFrame] | None = None,
    assume_sorted: bool = False,
    event_times: list[datetime] | None = None,
) -> pl.DataFrame:
    """Mean-variance weights as-of ``asof`` (no look-ahead beyond that decision time).

    OptimizationInfeasible and other genuine failures propagate — never silently
    flatten to zeros (see docs: no silent relaxation / auto-flatten).

    When ``w_prev`` is provided (dict by security_id or aligned array), turnover
    penalties use the prior book. Alpha sizing defaults to fused diagnostics
    (``fusion.use_fused_for_optimize``); set ``use_fused_alpha=False`` to force
    raw alpha. Conformal interval caps apply when intervals exist and
    ``fusion.apply_interval_caps`` is true. Trailing name-covariance uses the
    same ``available_time <= asof`` contract as the GARCH overlay: unpublished
    restatements cannot move Ledoit–Wolf/sample, OAS, or named DCC risk. Null
    availability among usable ``ret_1`` rows fails closed. Default covariance
    is trailing Ledoit-Wolf 2004 plus the GARCH/RGARCH overlay helper shared
    with ``/risk/portfolio``; when T<=N it stays Ledoit-Wolf rather than
    switching to sample. Set ``optimizer.covariance=dcc_gaussian``,
    ``dcc_student_t``, ``adcc``, ``ccc``, ``agdcc``, ``agdcc_full``, or
    ``ewma`` for the named one-step path; those paths do not overlay
    H_{t+1} and use the trailing contiguous complete-case window (holes
    are not concatenated; an incomplete asof row fails closed). Set
    ``optimizer.covariance=oas`` for trailing Chen OAS plus the overlay;
    that path must not silently size as Ledoit–Wolf, sample, EWMA, or DCC,
    and when T<=N it stays OAS. Set
    ``optimizer.covariance=ledoit_wolf_nonlinear`` for trailing analytical
    2020 nonlinear shrinkage plus the overlay; that path must not silently
    size as 2004 linear Ledoit–Wolf, OAS, sample, EWMA, or DCC, and when
    T<=N it stays nonlinear. Set ``optimizer.covariance=sample`` for
    trailing unbiased sample covariance plus the overlay; that path must
    not silently size as Ledoit–Wolf, OAS, EWMA, or DCC, and when T>N it
    stays sample. Named paths fail closed rather than substituting the
    default. Set ``optimizer.covariance=agdcc`` for diagonal CES AG-DCC
    one-step H_{t+1} without overlay; that path must not silently size as
    scalar ADCC or unrestricted AG-DCC. Set
    ``optimizer.covariance=agdcc_full`` for unrestricted CES AG-DCC
    one-step H_{t+1} without overlay; that path must not silently size as
    diagonal AG-DCC. Named CCC must not silently size as Gaussian DCC.
    Factor stays unwired.
    """
    state = forecast_asof(
        config,
        asof,
        frame=frame,
        day_index=day_index,
        assume_sorted=assume_sorted,
        event_times=event_times,
    )
    ids = [f.security_id for f in state.forecasts]
    use_fused = (
        bool(config.fusion.use_fused_for_optimize)
        if use_fused_alpha is None
        else bool(use_fused_alpha)
    )
    alpha = _alpha_vector(state, config, use_fused=use_fused)
    # covariance from trailing returns if present
    base = frame if frame is not None else _panel_cached(config)
    if assume_sorted:
        _require_assume_sorted_contract(base)
        hist = history_prefix_upto(base, state.asof)
    elif under_history_sort_contract(base):
        hist = history_prefix_upto(base, state.asof)
    else:
        hist = base.filter(pl.col("event_time") <= state.asof)
    hist = filter_trailing_returns_asof(hist, state.asof)
    estimator = require_implemented_optimizer_covariance(config.optimizer.covariance)
    if estimator in IMPLEMENTED_OPTIMIZER_NAMED_SPECS:
        estimate = estimate_optimizer_covariance_asof(config, base, state.asof, ids, hist)
        alpha_map = dict(zip(ids, alpha, strict=False))
        ids = estimate.security_ids
        alpha = np.array([alpha_map.get(c, 0.0) for c in ids], dtype=float)
        sig = estimate.sigma
    elif "ret_1" in hist.columns and hist.height > 20:
        estimate = estimate_optimizer_covariance_asof(config, base, state.asof, ids, hist)
        if estimate.unmeasured_reason is None:
            alpha_map = dict(zip(ids, alpha, strict=False))
            ids = estimate.security_ids
            alpha = np.array([alpha_map.get(c, 0.0) for c in ids], dtype=float)
            sig = estimate.sigma
        else:
            log.warning(
                "covariance_fallback",
                reason=estimate.unmeasured_reason,
                finite_rows=int(estimate.n_obs),
                n_bars=int(hist.height),
            )
            sig = np.diag(np.ones(len(ids)) * 0.02**2)
            sig, overlay, overlay_kind = apply_market_variance_overlay_to_covariance(
                config, base, state.asof, sig
            )
            estimate = replace(estimate, sigma=sig, overlay=overlay, market_overlay=overlay_kind)
    else:
        reason = "insufficient_history" if "ret_1" in hist.columns else "no_ret_1"
        log.warning("covariance_fallback", reason=reason, n_bars=int(hist.height))
        sig = np.diag(np.ones(len(ids)) * 0.02**2)
        sig, overlay, overlay_kind = apply_market_variance_overlay_to_covariance(
            config, base, state.asof, sig
        )
        estimate = replace(
            _unmeasured_optimizer_covariance(reason, ids),
            sigma=sig,
            overlay=overlay,
            market_overlay=overlay_kind,
        )
    wp = _align_w_prev(ids, w_prev)
    # Planner cost vector: the same modeled one-way cost the execution engine
    # charges (commission + half spread + per-turnover bps), as a fraction of
    # notional. The scalar ``optimizer.lambda_tc`` stays a dimensionless
    # multiplier; passing unit costs instead of modeled costs made the planner
    # see a 100% penalty and return a dead book.
    one_way_cost = (
        float(config.costs.commission_bps)
        + float(config.costs.half_spread_bps)
        + float(config.costs.bps_per_turnover)
    ) / 1e4
    tc_linear = np.full(len(ids), max(one_way_cost, 0.0), dtype=float)
    w, diag = optimize_mean_variance(alpha, sig, wp, config, tc_linear=tc_linear)
    w = _apply_forecast_interval_caps(w, state, ids, config, w_prev=wp)
    payload: dict[str, object] = {
        "event_time": [state.asof] * len(ids),
        "security_id": ids,
        "target_weight": w.tolist(),
        "alpha": alpha.tolist(),
        "covariance_estimator": [estimate.estimator] * len(ids),
        "covariance_object": [estimate.covariance_object] * len(ids),
        "covariance_spec": [estimate.spec] * len(ids),
    }
    if (
        estimate.unmeasured_reason is None
        and estimate.estimator in IMPLEMENTED_OPTIMIZER_ONE_STEP_SPECS
    ):
        payload["covariance_horizon"] = [1] * len(ids)
    elif (
        estimate.estimator not in IMPLEMENTED_OPTIMIZER_ONE_STEP_SPECS
        and state.garch_market_sigma is not None
    ):
        payload["garch_market_sigma"] = [float(state.garch_market_sigma)] * len(ids)
        payload["garch_market_variance"] = [float(state.garch_market_variance or 0.0)] * len(ids)
        payload["garch_series_scope"] = [str(state.garch_series_scope)] * len(ids)
        payload["market_risk_overlay"] = [str(state.market_risk_overlay)] * len(ids)
    out = pl.DataFrame(payload)
    if persist:
        Path(config.data.root).joinpath("gold").mkdir(parents=True, exist_ok=True)
        out.write_parquet(Path(config.data.root) / "gold" / "target_weights.parquet")
    _ = diag
    return out


def decision_dates(
    config: AppConfig,
    dates: list[datetime] | None = None,
) -> list[datetime]:
    """Subset of ``dates`` that have a row on the causal gold panel.

    The gold panel drops warmup bars (no universe membership yet) and the
    label-horizon tail, so a raw feature-date grid always contains decision
    dates ``optimize_asof`` must refuse. Callers iterating a coarse grid
    intersect it here — the same convention as ``execution-sensitivity``.
    ``dates=None`` returns the panel's own dates.
    """
    on_panel = set(panel(config)["event_time"].unique().to_list())
    if dates is None:
        return sorted(on_panel)
    return [d for d in dates if d in on_panel]


def build_causal_weight_panel(
    config: AppConfig,
    dates: list[datetime] | None = None,
) -> pl.DataFrame:
    """Point-in-time weight panel: ``optimize_asof(asof=d)`` per decision date.

    Does not broadcast end-of-sample weights across history (look-ahead).
    Passes the prior date's target weights as ``w_prev`` so turnover and TC
    terms are causal rather than always-from-cash.
    """
    if dates is None:
        dates = panel(config)["event_time"].unique().sort().to_list()
    if not dates:
        return pl.DataFrame(
            schema={
                "event_time": pl.Datetime,
                "security_id": pl.Utf8,
                "target_weight": pl.Float64,
                "alpha": pl.Float64,
            }
        )
    # Load gold panel + ranker once; reuse across asof dates (Phase 18 hot path).
    # Wave 8: build event_time day index once for exact day slices (PIT-safe).
    # Wave 12: ensure HISTORY_SORT_KEYS once; pass assume_sorted + event_times
    # so calibration history uses prefix slice (not per-asof day-index concat).
    shared = _panel_cached(config)
    if not under_history_sort_contract(shared):
        shared = sort_for_history(shared)
    day_index = build_event_time_day_index(shared)
    event_times = shared["event_time"].unique().sort().to_list()
    _ = _load_ranker_cached(config)
    frames: list[pl.DataFrame] = []
    prev_w: dict[str, float] = {}
    for i, d in enumerate(dates):
        frame = optimize_asof(
            config,
            d,
            persist=(i == len(dates) - 1),
            w_prev=prev_w or None,
            frame=shared,
            day_index=day_index,
            assume_sorted=True,
            event_times=event_times,
        )
        frames.append(frame)
        prev_w = {
            str(r["security_id"]): float(r["target_weight"]) for r in frame.iter_rows(named=True)
        }
    return pl.concat(frames)


__all__ = [
    "build_causal_weight_panel",
    "decision_dates",
    "forecast_asof",
    "optimize_asof",
]
