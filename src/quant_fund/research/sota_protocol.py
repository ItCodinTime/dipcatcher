"""Frozen SOTA protocol. Score only after the YAML is hashed.

Two scoreboards on the same as-of windows:
1. Dipcatcher date-level IC / pinball / CRPS / Diebold–Mariano vs public-feature ridge.
2. Kronos-paper path RankIC vs DLinear (in-repo) and the paper TSFM list (skipped
   without local checkpoints).

Calibration is a gate: Jackknife+ coverage ≥ 1−2α, CQR/ACI Kupiec recorded, PIT KS
recorded. An uncalibrated IC win does not size the book.

SYNTHETIC cannot take a champion alias. ``blend_weight`` stays 0 unless a
non-synthetic public-ridge IC win, DM does not prefer ridge on CRPS, and the
calibration gate is green. ``live_pnl_claim`` is always false here.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

from quant_fund.config.models import AppConfig, RobinhoodPlusVariant
from quant_fund.metrics.path_rankic import channel_rankic, mean_path_rankic
from quant_fund.models.dlinear import dlinear_forecast
from quant_fund.models.ranking import PUBLIC_FEATURES
from quant_fund.models.robinhood_plus.compare import (
    _asofs,
    compare_g1_public_ridge_vs_kronos,
)
from quant_fund.models.robinhood_plus.constants import ENGINE_NAME, PRICE_COLS
from quant_fund.models.robinhood_plus.engine import (
    extract_kline,
    forecast_robinhood_plus_cross_section,
)
from quant_fund.research.benches import bench_conformal, bench_distribution, bench_jackknife_plus
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent

KRONOS_PAPER_TSFM = (
    "PatchTST",
    "iTransformer",
    "TimesFM",
    "Chronos",
    "Time-MoE",
    "DLinear",
)


class SotaProtocol(BaseModel):
    """Sealed scoring contract. Extra keys fail closed."""

    model_config = ConfigDict(extra="forbid")

    protocol_id: str = "dipcatcher.sota.v1"
    frozen: bool = True
    ranking_target: str = "future_idio_return_1"
    distribution_target: str = "future_excess_return_5"
    horizons: list[int] = Field(default_factory=lambda: [1, 5])
    embargo_bars: int = 5
    n_asofs: int = 6
    lookback: int = 32
    pred_len: int = 5
    sample_count: int = 4
    variant: str = "mini"
    public_features_only: bool = True
    synthetic_promotable: bool = False
    champion_alias_on_synthetic: bool = False
    holdout_last_year: bool = False
    held_out_names: list[str] = Field(default_factory=list)
    second_market: str = "stooq_uk"
    tsfm_baselines: list[str] = Field(default_factory=lambda: list(KRONOS_PAPER_TSFM))


def load_sota_protocol(path: str | Path) -> SotaProtocol:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("sota protocol YAML must be a mapping")
    inherit = raw.pop("inherit", None)
    if inherit:
        raise ValueError("sota protocol YAML must not inherit; freeze a single file")
    return SotaProtocol.model_validate(raw)


def protocol_sha256(protocol: SotaProtocol) -> str:
    payload = json.dumps(protocol.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_protocol(config: AppConfig, protocol: SotaProtocol) -> AppConfig:
    """Copy frozen lookback / pred_len / sample_count / embargo onto a working config."""
    out = config.model_copy(deep=True)
    out.train.ranking_target = protocol.ranking_target
    out.robinhood_plus.lookback = int(protocol.lookback)
    out.robinhood_plus.pred_len = int(protocol.pred_len)
    out.robinhood_plus.sample_count = int(protocol.sample_count)
    out.robinhood_plus.blend_weight = 0.0
    out.robinhood_plus.variant = RobinhoodPlusVariant(protocol.variant)
    out.validation.embargo_bars = int(protocol.embargo_bars)
    out.fusion.skip_intervals = True
    return out


def _holdout_asofs(times: list[datetime], *, last_year: bool) -> list[datetime]:
    if not last_year or not times:
        return times
    year = max(t.year for t in times)
    held = [t for t in times if t.year == year]
    return held or times


def _future_ohlc(
    frame: pl.DataFrame, security_id: str, asof: datetime, pred_len: int
) -> np.ndarray:
    future = (
        frame.filter((pl.col("security_id") == security_id) & (pl.col("event_time") > asof))
        .sort("event_time")
        .head(pred_len)
    )
    if future.height < pred_len:
        return np.zeros((0, 4), dtype=np.float64)
    return extract_kline(future)[:, :4]


def score_path_rankic(
    config: AppConfig,
    frame: pl.DataFrame,
    *,
    n_asofs: int,
    include_torch: bool = False,
    held_out_names: list[str] | None = None,
    holdout_last_year: bool = False,
) -> dict[str, Any]:
    """Kronos-paper path RankIC on the same as-of windows as the Dipcatcher card."""
    lookback = int(config.robinhood_plus.lookback)
    pred_len = int(config.robinhood_plus.pred_len)
    asofs = _asofs(frame, lookback, n_asofs)
    asofs = _holdout_asofs(asofs, last_year=holdout_last_year)
    blocked = {str(name) for name in (held_out_names or [])}
    dlinear_rows: list[dict[str, float]] = []
    numpy_rows: list[dict[str, float]] = []
    torch_rows: list[dict[str, float]] = []
    n_skip = 0
    for asof in asofs:
        hist = frame.filter(pl.col("event_time") <= asof)
        if "available_time" in hist.columns:
            hist = hist.filter(pl.col("available_time") <= asof)
        if hist.is_empty():
            continue
        ids = [str(s) for s in hist["security_id"].unique().to_list() if str(s) not in blocked]
        numpy_map = forecast_robinhood_plus_cross_section(
            hist,
            asof,
            lookback=lookback,
            pred_len=pred_len,
            sample_count=int(config.robinhood_plus.sample_count),
            security_ids=ids,
            quantile_levels=(0.05, 0.50, 0.95),
            horizons=tuple(int(h) for h in config.horizons.bars),
        )
        torch_map: dict[str, Any] = {}
        if include_torch:
            try:
                from quant_fund.models.robinhood_plus.torch_backend import (
                    bind_local_kronos_weights,
                    forecast_cross_section_torch,
                )

                cfg_t = bind_local_kronos_weights(config)
                torch_map = forecast_cross_section_torch(hist, asof, cfg_t, ids)
            except Exception as exc:  # noqa: BLE001
                torch_map = {"_error": str(exc)}
        for sid in ids:
            name_hist = hist.filter(pl.col("security_id") == sid).sort("event_time").tail(lookback)
            realized = _future_ohlc(frame, sid, asof, pred_len)
            if realized.shape[0] != pred_len or name_hist.height < 2:
                n_skip += 1
                continue
            try:
                past = extract_kline(name_hist)[:, :4]
                pred_dl = dlinear_forecast(past, pred_len)
                dlinear_rows.append(channel_rankic(pred_dl, realized))
            except ValueError:
                n_skip += 1
                continue
            rh = numpy_map.get(sid)
            if rh is not None and rh.forecast.mean_path.shape[0] == pred_len:
                numpy_rows.append(channel_rankic(rh.forecast.mean_path[:, :4], realized))
            if include_torch and sid in torch_map and not str(sid).startswith("_"):
                tf = torch_map[sid]
                mean_path = getattr(getattr(tf, "forecast", None), "mean_path", None)
                if mean_path is not None and getattr(mean_path, "shape", (0,))[0] == pred_len:
                    torch_rows.append(channel_rankic(mean_path[:, :4], realized))
    skipped_tsfm = [name for name in KRONOS_PAPER_TSFM if name != "DLinear"]
    blob: dict[str, Any] = {
        "family": "path_rankic",
        "research_only": True,
        "execution_claim": "research_only",
        "claim": "kronos_paper_path_rankic",
        "n_asofs": int(len(asofs)),
        "n_skip": int(n_skip),
        "lookback": lookback,
        "pred_len": pred_len,
        "dlinear": mean_path_rankic(dlinear_rows),
        "numpy_markov": mean_path_rankic(numpy_rows),
        "kronos_mini": mean_path_rankic(torch_rows) if include_torch else {"status": "skipped"},
        "tsfm_skipped_no_local_weights": skipped_tsfm,
        "note": (
            "path RankIC is Spearman of predicted vs realized OHLC series; "
            "it is not date-level cross-sectional IC and does not size the book"
        ),
    }
    if include_torch and torch_map.get("_error"):
        blob["kronos_mini"] = {
            "status": "skipped_no_local_weights",
            "reason": torch_map["_error"],
        }
    return blob


def calibration_gate(frame: pl.DataFrame, config: AppConfig) -> dict[str, Any]:
    """Jackknife+ floor, CQR/ACI Kupiec, PIT KS. Missing diagnostics fail the gate."""
    jackknife = bench_jackknife_plus(frame, config)
    conformal = bench_conformal(frame, config)
    distribution = bench_distribution(frame, config)
    jp_ok = bool(jackknife.get("meets_coverage_floor")) if jackknife else False
    cqr = conformal.get("cqr") if isinstance(conformal, dict) else None
    aci = conformal.get("aci") if isinstance(conformal, dict) else None
    cqr_p = cqr.get("kupiec_p") if isinstance(cqr, dict) else None
    aci_p = aci.get("kupiec_p") if isinstance(aci, dict) else None
    pit_p = distribution.get("pit_ks_p") if isinstance(distribution, dict) else None
    pit_value = float(cast(Any, pit_p)) if _finite(pit_p) else float("nan")
    pit_uniform = pit_value >= 0.05
    recorded = (
        _finite(jackknife.get("coverage") if jackknife else None)
        and _finite(cqr_p)
        and _finite(aci_p)
        and _finite(pit_p)
    )
    blob = {
        "family": "calibration_gate",
        "research_only": True,
        "execution_claim": "research_only",
        "jackknife_plus": jackknife,
        "cqr": cqr or {},
        "aci": aci or {},
        "pit_ks": distribution.get("pit_ks") if isinstance(distribution, dict) else float("nan"),
        "pit_ks_p": pit_p,
        "meets_jackknife_floor": jp_ok,
        "pit_uniform_5pct": bool(pit_uniform),
        "recorded": bool(recorded),
        "pass": bool(jp_ok and recorded and pit_uniform),
        "note": "an uncalibrated IC win is not SOTA; PIT KS must not reject uniformity at 5%",
    }
    return blob


def _finite(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number))


def _finite_float(value: object) -> float:
    """float(value) narrowed to finite, else NaN. NaN makes later comparisons False."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def promotion_decision(
    *,
    data_source: str,
    g1: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """blend_weight may leave 0 only after a non-synthetic calibrated public-ridge IC win."""
    synthetic = data_source.upper() == "SYNTHETIC" or data_source.lower() == "synthetic"
    ridge_ic = g1.get("ridge", {}).get("mean_ic") if isinstance(g1.get("ridge"), dict) else None
    mini = g1.get("kronos_mini") if isinstance(g1.get("kronos_mini"), dict) else {}
    mini_ic = mini.get("mean_ic") if isinstance(mini, dict) else None
    dm = mini.get("diebold_mariano_crps") if isinstance(mini, dict) else None
    preferred = dm.get("preferred") if isinstance(dm, dict) else None
    ridge_ic_f = _finite_float(ridge_ic)
    mini_ic_f = _finite_float(mini_ic)
    ic_win = bool(
        np.isfinite(ridge_ic_f)
        and np.isfinite(mini_ic_f)
        and mini_ic_f > 0.0
        and mini_ic_f > ridge_ic_f
    )
    dm_not_ridge = preferred == ENGINE_NAME
    calib_ok = bool(calibration.get("pass"))
    move = (not synthetic) and ic_win and dm_not_ridge and calib_ok
    return {
        "synthetic": synthetic,
        "ic_win_vs_public_ridge": bool(ic_win),
        "dm_not_prefer_ridge_crps": bool(dm_not_ridge),
        "calibration_pass": calib_ok,
        "champion_alias": False if synthetic else bool(move),
        "blend_weight": 1.0 if move else 0.0,
        "sizes_book": bool(move),
        "reason": (
            "SYNTHETIC cannot promote"
            if synthetic
            else (
                "kronos-mini may size alpha (blend_weight=1) after public-ridge IC win, "
                "DM, and calibration"
                if move
                else "blend_weight stays 0; missing positive IC, DM, calibration, or file tape"
            )
        ),
    }


def run_sota_protocol(
    config: AppConfig,
    protocol: SotaProtocol,
    frame: pl.DataFrame | None = None,
    *,
    include_torch: bool = True,
    include_path_rankic: bool = True,
    include_calibration: bool = True,
) -> dict[str, Any]:
    """Freeze, then score both boards and the calibration gate on one panel."""
    from quant_fund.models.ranking import drop_oracle_columns
    from quant_fund.pipeline.dataset import panel

    cfg = apply_protocol(config, protocol)
    gold = drop_oracle_columns(frame if frame is not None else panel(cfg))
    if protocol.public_features_only:
        missing = [name for name in PUBLIC_FEATURES if name not in gold.columns]
        if missing:
            raise ValueError(f"public features missing from gold: {missing}")
    g1 = compare_g1_public_ridge_vs_kronos(
        cfg,
        gold,
        n_asofs=protocol.n_asofs,
        include_torch=include_torch,
        holdout_last_year=protocol.holdout_last_year,
    )
    path = (
        score_path_rankic(
            cfg,
            gold,
            n_asofs=protocol.n_asofs,
            include_torch=include_torch,
            held_out_names=protocol.held_out_names,
            holdout_last_year=protocol.holdout_last_year,
        )
        if include_path_rankic
        else {"status": "skipped"}
    )
    calib = calibration_gate(gold, cfg) if include_calibration else {"status": "skipped"}
    source = "SYNTHETIC" if cfg.data.source == "synthetic" else str(cfg.data.source)
    promo = promotion_decision(data_source=source, g1=g1, calibration=calib)
    receipt: dict[str, Any] = {
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol_sha256(protocol),
        "frozen": protocol.frozen,
        "family": ENGINE_NAME,
        "research_only": True,
        "execution_claim": "research_only",
        "claim": "research_metric_only",
        "data_source": source,
        "universe": {
            "source": cfg.data.source,
            "n_names": int(gold["security_id"].n_unique()) if "security_id" in gold.columns else 0,
            "n_rows": int(gold.height),
            "held_out_names": list(protocol.held_out_names),
            "holdout_last_year": protocol.holdout_last_year,
            "second_market": protocol.second_market,
        },
        "horizons": list(protocol.horizons),
        "ranking_target": protocol.ranking_target,
        "distribution_target": protocol.distribution_target,
        "embargo_bars": protocol.embargo_bars,
        "sample_count": protocol.sample_count,
        "variant": protocol.variant,
        "lookback": protocol.lookback,
        "pred_len": protocol.pred_len,
        "n_asofs": protocol.n_asofs,
        "g1": g1,
        "path_rankic": path,
        "calibration": calib,
        "promotion": promo,
        "sizes_book": promo["sizes_book"],
        "blend_weight": promo["blend_weight"],
        "champion_alias": promo["champion_alias"],
        "price_cols": list(PRICE_COLS),
    }
    if not family_blob_forbidden_metrics_absent(receipt):
        raise AssertionError("sota protocol receipt leaked forbidden research keys")
    dest = Path(cfg.data.root) / "metadata" / "sota_receipt.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    receipt["receipt_path"] = str(dest)
    return receipt
