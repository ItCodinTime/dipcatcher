"""Causal champion/challenger card: ridge-only vs robinhood+.

Same gold panel, same as-ofs. Proper scores only — date-level IC, pinball,
CRPS on path quantiles, Diebold–Mariano. No Sharpe. An engine that cannot
beat ridge on date-level IC must not size the book (ADR-023).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig, RobinhoodPlusBackend
from quant_fund.metrics.cross_section import date_ic_series
from quant_fund.metrics.inference import diebold_mariano
from quant_fund.metrics.scoring import crps_from_quantiles, mean_pinball, pinball_loss
from quant_fund.models.ranking import PUBLIC_FEATURES, available_features, drop_oracle_columns
from quant_fund.models.robinhood_plus.constants import (
    AFFILIATION_DISCLAIMER,
    ENGINE_DISPLAY,
    ENGINE_NAME,
    FAMILY,
    MODEL_VERSION,
)
from quant_fund.pipeline.dataset import panel
from quant_fund.pipeline.forecast import clear_forecast_caches, forecast_asof
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.schemas.forecast import AssetForecast

COMPARE_TAUS = (0.05, 0.50, 0.95)
COMPARE_HORIZON = "5d"


def _q_at(quantiles: dict[float, float] | dict[str, float], tau: float) -> float:
    for key, value in quantiles.items():
        try:
            if abs(float(key) - tau) < 1e-9:
                return float(value)
        except (TypeError, ValueError):
            continue
    return float("nan")


def _horizon_quantiles(row: AssetForecast, horizon: str = COMPARE_HORIZON) -> np.ndarray:
    qmap = row.quantiles.get(horizon) or row.quantiles.get("5d") or {}
    return np.array([_q_at(qmap, tau) for tau in COMPARE_TAUS], dtype=float)


def _row_crps(y: float, quantiles: np.ndarray) -> float:
    if not np.isfinite(y) or not np.isfinite(quantiles).all():
        return float("nan")
    return float(
        crps_from_quantiles(
            np.array([y], dtype=float),
            quantiles.reshape(1, -1),
            np.array(COMPARE_TAUS, dtype=float),
        )
    )


def _finite_label(row: dict[str, Any], name: str) -> float:
    raw = row.get(name)
    if raw is None:
        return float("nan")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _asofs(
    frame: pl.DataFrame,
    lookback: int,
    n_asofs: int,
    *,
    holdout_last_year: bool = False,
) -> list[datetime]:
    times = [t for t in frame["event_time"].unique().sort().to_list() if isinstance(t, datetime)]
    usable = times[max(lookback, 2) :]
    if holdout_last_year and usable:
        year = max(t.year for t in usable)
        held = [t for t in usable if t.year == year]
        usable = held or usable
    if not usable:
        return []
    step = max(1, len(usable) // max(n_asofs, 1))
    picked = usable[::step][-n_asofs:]
    return [t for t in picked if isinstance(t, datetime)]


def _dm_blob(loss_a: np.ndarray, loss_b: np.ndarray, name_a: str, name_b: str) -> dict[str, Any]:
    result = diebold_mariano(loss_a, loss_b, name_a=name_a, name_b=name_b)
    return {
        "mean_loss_diff": float(result.mean_loss_diff),
        "statistic": float(result.statistic),
        "p_value": float(result.p_value),
        "lags": int(result.lags),
        "n": int(result.n),
        "preferred": str(result.preferred),
    }


def _score_pair(
    champion: AssetForecast,
    challenger: AssetForecast,
    y_rank: float,
    y_5d: float,
) -> dict[str, float] | None:
    ridge_score = float(champion.rank_score.get("5d", float("nan")))
    rh_score = float(challenger.rank_score.get("5d", float("nan")))
    ridge_mu = float(champion.expected_returns.get("5d", champion.alpha.get("5d", float("nan"))))
    rh_mu = float(challenger.expected_returns.get("5d", challenger.alpha.get("5d", float("nan"))))
    ridge_q = _horizon_quantiles(champion)
    rh_q = _horizon_quantiles(challenger)
    if not np.isfinite(ridge_score) or not np.isfinite(rh_score):
        return None
    return {
        "y_rank": y_rank,
        "y_5d": y_5d,
        "ridge_score": ridge_score,
        "rh_score": rh_score,
        "ridge_mu": ridge_mu,
        "rh_mu": rh_mu,
        "ridge_crps": _row_crps(y_5d, ridge_q),
        "rh_crps": _row_crps(y_5d, rh_q),
        "ridge_pinball_05": float(pinball_loss(np.array([y_5d]), ridge_q[0:1], 0.05)[0])
        if np.isfinite(y_5d) and np.isfinite(ridge_q[0])
        else float("nan"),
        "rh_pinball_05": float(pinball_loss(np.array([y_5d]), rh_q[0:1], 0.05)[0])
        if np.isfinite(y_5d) and np.isfinite(rh_q[0])
        else float("nan"),
        "ridge_pinball_50": float(pinball_loss(np.array([y_5d]), ridge_q[1:2], 0.50)[0])
        if np.isfinite(y_5d) and np.isfinite(ridge_q[1])
        else float("nan"),
        "rh_pinball_50": float(pinball_loss(np.array([y_5d]), rh_q[1:2], 0.50)[0])
        if np.isfinite(y_5d) and np.isfinite(rh_q[1])
        else float("nan"),
        "ridge_pinball_95": float(pinball_loss(np.array([y_5d]), ridge_q[2:3], 0.95)[0])
        if np.isfinite(y_5d) and np.isfinite(ridge_q[2])
        else float("nan"),
        "rh_pinball_95": float(pinball_loss(np.array([y_5d]), rh_q[2:3], 0.95)[0])
        if np.isfinite(y_5d) and np.isfinite(rh_q[2])
        else float("nan"),
    }


def compare_ridge_vs_robinhood_plus(
    config: AppConfig,
    frame: pl.DataFrame | None = None,
    *,
    n_asofs: int = 8,
    train_ridge: bool = True,
    challenger_backend: str | None = None,
    holdout_last_year: bool = False,
) -> dict[str, Any]:
    """Ridge-only (enabled=false) vs robinhood+ (blend_weight=1) on one panel.

    Book-sizer win is a strictly higher date-level mean Pearson IC of rank_score
    vs the ranking target. Pinball / CRPS / DM are reported, never Sharpe.
    """
    source = "SYNTHETIC" if config.data.source == "synthetic" else str(config.data.source)
    blob: dict[str, Any] = {
        "family": ENGINE_NAME,
        "display": ENGINE_DISPLAY,
        "model_version": MODEL_VERSION,
        "catalog_family": FAMILY,
        "research_only": True,
        "execution_claim": "research_only",
        "claim": "research_metric_only",
        "data_source": source,
        "affiliation_disclaimer": AFFILIATION_DISCLAIMER,
        "champion": "ridge",
        "challenger": ENGINE_NAME,
        "backend": challenger_backend or config.robinhood_plus.backend.value,
        "sizes_book": False,
        "n_asofs": 0,
        "n_scored": 0,
        "status": "pending",
    }
    gold = frame if frame is not None else panel(config)
    if gold.is_empty() or "event_time" not in gold.columns:
        blob["status"] = "empty_panel"
        return _seal(blob)
    cfg_base = config.model_copy(deep=True)
    cfg_base.fusion.skip_intervals = True
    if train_ridge:
        from quant_fund.pipeline.train import train_ranking

        try:
            train_ranking(cfg_base, "ridge")
        except (ValueError, FileNotFoundError):
            blob["ridge_train"] = "unavailable_momentum_fallback"
    rank_label = str(cfg_base.train.ranking_target)
    ret_label = (
        "future_excess_return_5"
        if "future_excess_return_5" in gold.columns
        else "future_log_return_5"
    )
    cfg_ridge = cfg_base.model_copy(deep=True)
    cfg_ridge.robinhood_plus.enabled = False
    cfg_rh = cfg_base.model_copy(deep=True)
    cfg_rh.robinhood_plus.enabled = True
    cfg_rh.robinhood_plus.blend_weight = 1.0
    if challenger_backend is not None:
        cfg_rh.robinhood_plus.backend = RobinhoodPlusBackend(challenger_backend)
    asofs = _asofs(
        gold, cfg_rh.robinhood_plus.lookback, n_asofs, holdout_last_year=holdout_last_year
    )
    blob["n_asofs"] = int(len(asofs))
    blob["ranking_target"] = rank_label
    blob["distribution_label"] = ret_label
    if not asofs:
        blob["status"] = "insufficient_history"
        return _seal(blob)
    rows: list[dict[str, float]] = []
    dates: list[datetime] = []
    n_ok = 0
    n_fallback = 0
    for asof in asofs:
        day = gold.filter(pl.col("event_time") == asof)
        ridge_state = forecast_asof(cfg_ridge, asof, frame=gold)
        rh_state = forecast_asof(cfg_rh, asof, frame=gold)
        ridge_map = {f.security_id: f for f in ridge_state.forecasts}
        rh_map = {f.security_id: f for f in rh_state.forecasts}
        for raw in day.iter_rows(named=True):
            sid = str(raw["security_id"])
            champ = ridge_map.get(sid)
            chal = rh_map.get(sid)
            if champ is None or chal is None:
                n_fallback += 1
                continue
            pair = _score_pair(
                champ,
                chal,
                _finite_label(raw, rank_label),
                _finite_label(raw, ret_label) if ret_label in raw else float("nan"),
            )
            if pair is None:
                n_fallback += 1
                continue
            n_ok += 1
            rows.append(pair)
            dates.append(asof)
    blob["n_scored"] = int(len(rows))
    blob["n_ok"] = int(n_ok)
    blob["n_fallback"] = int(n_fallback)
    if len(rows) < 8:
        blob["status"] = "insufficient_scored"
        return _seal(blob)
    table = {key: np.array([row[key] for row in rows], dtype=float) for key in rows[0]}
    date_arr = np.asarray(dates, dtype=object)
    ic_ridge = date_ic_series(table["ridge_score"], table["y_rank"], date_arr, min_names=3)
    ic_rh = date_ic_series(table["rh_score"], table["y_rank"], date_arr, min_names=3)
    blob["mean_ic_champion"] = float(ic_ridge.mean_pearson)
    blob["mean_ic_challenger"] = float(ic_rh.mean_pearson)
    blob["ic_n_dates"] = int(min(ic_ridge.n_dates, ic_rh.n_dates))
    y5 = table["y_5d"]
    mask_q = np.isfinite(y5)
    if int(mask_q.sum()) >= 8:
        blob["pinball_50_champion"] = mean_pinball(y5[mask_q], table["ridge_mu"][mask_q], 0.50)
        blob["pinball_50_challenger"] = mean_pinball(y5[mask_q], table["rh_mu"][mask_q], 0.50)
        blob["pinball_05_champion"] = float(np.nanmean(table["ridge_pinball_05"][mask_q]))
        blob["pinball_05_challenger"] = float(np.nanmean(table["rh_pinball_05"][mask_q]))
        blob["pinball_95_champion"] = float(np.nanmean(table["ridge_pinball_95"][mask_q]))
        blob["pinball_95_challenger"] = float(np.nanmean(table["rh_pinball_95"][mask_q]))
        blob["crps_champion"] = float(np.nanmean(table["ridge_crps"][mask_q]))
        blob["crps_challenger"] = float(np.nanmean(table["rh_crps"][mask_q]))
        blob["diebold_mariano_crps"] = _dm_blob(
            table["rh_crps"][mask_q],
            table["ridge_crps"][mask_q],
            ENGINE_NAME,
            "ridge",
        )
        blob["diebold_mariano_pinball_50"] = _dm_blob(
            table["rh_pinball_50"][mask_q],
            table["ridge_pinball_50"][mask_q],
            ENGINE_NAME,
            "ridge",
        )
    ic_win = (
        np.isfinite(ic_rh.mean_pearson)
        and np.isfinite(ic_ridge.mean_pearson)
        and int(ic_rh.n_dates) >= 5
        and float(ic_rh.mean_pearson) > float(ic_ridge.mean_pearson)
    )
    blob["sizes_book"] = bool(ic_win)
    blob["default_blend_weight"] = 1.0 if ic_win else 0.0
    blob["status"] = "ok"
    blob["decision"] = (
        "robinhood+ sizes the book"
        if ic_win
        else "robinhood+ remains a stamped challenger (blend_weight=0)"
    )
    return _seal(blob)


def compare_numpy_vs_kronos_mini_vs_ridge(
    config: AppConfig,
    frame: pl.DataFrame | None = None,
    *,
    n_asofs: int = 6,
) -> dict[str, Any]:
    """Same PIT windows: ridge vs numpy Markov vs local Kronos-mini.

    Missing local checkpoints stamp ``skipped_no_local_weights`` — never Hub.
    """
    gold = frame if frame is not None else panel(config)
    numpy_card = compare_ridge_vs_robinhood_plus(
        config, gold, n_asofs=n_asofs, train_ridge=True, challenger_backend="numpy"
    )
    torch_card: dict[str, Any]
    try:
        torch_card = compare_ridge_vs_robinhood_plus(
            config, gold, n_asofs=n_asofs, train_ridge=False, challenger_backend="torch"
        )
    except Exception as exc:  # noqa: BLE001 — stamp the skip; do not size the book
        torch_card = {
            "family": ENGINE_NAME,
            "research_only": True,
            "execution_claim": "research_only",
            "claim": "research_metric_only",
            "backend": "torch",
            "status": "skipped_no_local_weights",
            "reason": str(exc),
            "sizes_book": False,
        }
        if not family_blob_forbidden_metrics_absent(torch_card):
            torch_card = {k: v for k, v in torch_card.items() if "sharpe" not in str(k).lower()}
    blob = {
        "family": ENGINE_NAME,
        "display": ENGINE_DISPLAY,
        "research_only": True,
        "execution_claim": "research_only",
        "claim": "research_metric_only",
        "data_source": numpy_card.get("data_source"),
        "ridge": {
            "mean_ic": numpy_card.get("mean_ic_champion"),
            "crps": numpy_card.get("crps_champion"),
            "pinball_50": numpy_card.get("pinball_50_champion"),
        },
        "numpy_markov": {
            "mean_ic": numpy_card.get("mean_ic_challenger"),
            "crps": numpy_card.get("crps_challenger"),
            "pinball_50": numpy_card.get("pinball_50_challenger"),
            "sizes_book": numpy_card.get("sizes_book"),
            "status": numpy_card.get("status"),
        },
        "kronos_mini": {
            "mean_ic": torch_card.get("mean_ic_challenger"),
            "crps": torch_card.get("crps_challenger"),
            "pinball_50": torch_card.get("pinball_50_challenger"),
            "sizes_book": torch_card.get("sizes_book"),
            "status": torch_card.get("status"),
            "reason": torch_card.get("reason"),
        },
        "numpy_card": numpy_card,
        "torch_card": torch_card,
    }
    return _seal(blob)


def train_public_ridge(config: AppConfig, frame: pl.DataFrame) -> dict[str, Any]:
    """Fit ridge on PUBLIC_FEATURES only. Oracle columns must already be dropped."""
    from quant_fund.pipeline.train import train_ranking

    public = drop_oracle_columns(frame)
    feats = available_features(list(public.columns), list(PUBLIC_FEATURES))
    result = train_ranking(config, "ridge", frame=public, feature_names=feats)
    clear_forecast_caches()
    result["champion_features"] = list(feats)
    result["n_features"] = int(len(result.get("features") or []))
    return result


def compare_g1_public_ridge_vs_kronos(
    config: AppConfig,
    frame: pl.DataFrame | None = None,
    *,
    n_asofs: int = 6,
    include_torch: bool = True,
    holdout_last_year: bool = False,
) -> dict[str, Any]:
    """Fair SYNTHETIC / public-feature card: ridge vs numpy Markov vs Kronos-mini.

    Oracle columns are dropped. Ridge is trained on ``PUBLIC_FEATURES`` only.
    SYNTHETIC evidence cannot size the book or take a champion alias.
    """
    gold = drop_oracle_columns(frame if frame is not None else panel(config))
    source = "SYNTHETIC" if config.data.source == "synthetic" else str(config.data.source)
    blob: dict[str, Any] = {
        "family": ENGINE_NAME,
        "display": ENGINE_DISPLAY,
        "research_only": True,
        "execution_claim": "research_only",
        "claim": "research_metric_only",
        "gate": "G1",
        "data_source": source,
        "champion": "public_ridge",
        "champion_features": list(PUBLIC_FEATURES),
        "oracle_columns_dropped": True,
        "synthetic_not_promotable": source == "SYNTHETIC",
        "champion_alias": False,
        "sizes_book": False,
        "default_blend_weight": 0.0,
        "affiliation_disclaimer": AFFILIATION_DISCLAIMER,
    }
    try:
        train_info = train_public_ridge(config, gold)
        blob["ridge_train"] = {
            "n": train_info.get("n"),
            "features": train_info.get("features"),
            "n_features": train_info.get("n_features"),
            "mean_ic": (train_info.get("metrics") or {}).get("mean_ic"),
        }
    except (ValueError, FileNotFoundError) as exc:
        blob["ridge_train"] = {"status": "unavailable_momentum_fallback", "reason": str(exc)}
    numpy_card = compare_ridge_vs_robinhood_plus(
        config,
        gold,
        n_asofs=n_asofs,
        train_ridge=False,
        challenger_backend="numpy",
        holdout_last_year=holdout_last_year,
    )
    torch_card: dict[str, Any]
    if include_torch:
        try:
            from quant_fund.models.robinhood_plus.torch_backend import bind_local_kronos_weights

            cfg_torch = bind_local_kronos_weights(config)
            torch_card = compare_ridge_vs_robinhood_plus(
                cfg_torch,
                gold,
                n_asofs=n_asofs,
                train_ridge=False,
                challenger_backend="torch",
                holdout_last_year=holdout_last_year,
            )
        except Exception as exc:  # noqa: BLE001 — stamp skip; never Hub; never size the book
            torch_card = {
                "family": ENGINE_NAME,
                "research_only": True,
                "execution_claim": "research_only",
                "claim": "research_metric_only",
                "backend": "torch",
                "status": "skipped_no_local_weights",
                "reason": str(exc),
                "sizes_book": False,
            }
    else:
        torch_card = {
            "family": ENGINE_NAME,
            "research_only": True,
            "execution_claim": "research_only",
            "backend": "torch",
            "status": "skipped_by_request",
            "sizes_book": False,
        }
    blob["ridge"] = {
        "mean_ic": numpy_card.get("mean_ic_champion"),
        "crps": numpy_card.get("crps_champion"),
        "pinball_50": numpy_card.get("pinball_50_champion"),
        "n_scored": numpy_card.get("n_scored"),
        "ic_n_dates": numpy_card.get("ic_n_dates"),
    }
    blob["numpy_markov"] = {
        "mean_ic": numpy_card.get("mean_ic_challenger"),
        "crps": numpy_card.get("crps_challenger"),
        "pinball_50": numpy_card.get("pinball_50_challenger"),
        "sizes_book": False,
        "status": numpy_card.get("status"),
        "diebold_mariano_crps": numpy_card.get("diebold_mariano_crps"),
    }
    blob["kronos_mini"] = {
        "mean_ic": torch_card.get("mean_ic_challenger"),
        "crps": torch_card.get("crps_challenger"),
        "pinball_50": torch_card.get("pinball_50_challenger"),
        "sizes_book": False,
        "status": torch_card.get("status"),
        "reason": torch_card.get("reason"),
        "diebold_mariano_crps": torch_card.get("diebold_mariano_crps"),
        "backend": torch_card.get("backend"),
    }
    blob["numpy_card"] = numpy_card
    blob["torch_card"] = torch_card
    blob["n_asofs"] = numpy_card.get("n_asofs")
    blob["ranking_target"] = numpy_card.get("ranking_target")
    blob["distribution_label"] = numpy_card.get("distribution_label")
    blob["status"] = "ok" if numpy_card.get("status") == "ok" else numpy_card.get("status")
    if source == "SYNTHETIC":
        blob["decision"] = (
            "SYNTHETIC public-feature card is an engine-correctness diagnostic; "
            "it cannot promote, size the book, or take a champion alias"
        )
    else:
        blob["decision"] = (
            "file-tape public-feature card; champion alias still requires "
            "IC win vs public ridge, DM not preferring ridge on CRPS, and calibration floors"
        )
    return _seal(blob)


def _seal(blob: dict[str, Any]) -> dict[str, Any]:
    if not family_blob_forbidden_metrics_absent(blob):
        raise AssertionError("robinhood+ comparison leaked forbidden research keys")
    return blob
