"""Optional torch backend that loads official Kronos weights as robinhood+.

Torch, einops, huggingface_hub, and safetensors are the ``[nn]`` extra.
CI does not download Hub checkpoints. ``allow_network=false`` (default) only
loads local directories. This module never silently falls back to numpy.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.models.robinhood_plus.constants import (
    ENGINE_NAME,
    ENGINE_VERSION,
    FAMILY,
    PRICE_SPACE,
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_MISSING_OHLC,
    VARIANT_HUB,
)
from quant_fund.models.robinhood_plus.engine import (
    RobinhoodPlusNameForecast,
    extract_kline,
    kline_columns_present,
)
from quant_fund.models.robinhood_plus.predictor import empty_forecast, path_forecast_from_paths

_PREDICTOR_CACHE: dict[tuple[Any, ...], Any] = {}


class RobinhoodPlusTorchError(RuntimeError):
    """Torch / pretrained-weight failures. Callers must not swallow these."""


def torch_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def _ensure_kronos_on_path() -> None:
    """Prefer a local Kronos checkout over Hub-installed packages."""
    repo = Path(__file__).resolve().parents[4]
    candidates = (
        repo / "third_party" / "kronos_src",
        repo / "third_party" / "Kronos",
        repo / "third_party" / "kronos",
    )
    for cand in candidates:
        if (cand / "model" / "kronos.py").is_file():
            path = str(cand)
            if path not in sys.path:
                sys.path.insert(0, path)
            return


def local_kronos_weight_dirs(variant: str = "mini") -> tuple[Path, Path] | None:
    """Return local tokenizer/model directories when both exist. Never Hub."""
    if variant not in VARIANT_HUB:
        return None
    repo = Path(__file__).resolve().parents[4]
    root = repo / "third_party" / "kronos_weights"
    spec = VARIANT_HUB[variant]
    tok_name = str(spec["tokenizer"]).rsplit("/", 1)[-1]
    model_name = str(spec["model"]).rsplit("/", 1)[-1]
    tok = root / tok_name
    model = root / model_name
    if (tok / "config.json").is_file() and (model / "config.json").is_file():
        return tok, model
    return None


def bind_local_kronos_weights(config: AppConfig) -> AppConfig:
    """Fill tokenizer_path/model_path from third_party/kronos_weights when unset."""
    cfg = config.robinhood_plus
    if cfg.tokenizer_path and cfg.model_path:
        return config
    dirs = local_kronos_weight_dirs(cfg.variant.value)
    if dirs is None:
        return config
    tok, model = dirs
    out = config.model_copy(deep=True)
    if not out.robinhood_plus.tokenizer_path:
        out.robinhood_plus.tokenizer_path = str(tok)
    if not out.robinhood_plus.model_path:
        out.robinhood_plus.model_path = str(model)
    return out


def load_pretrained_predictor(
    variant: str = "mini",
    *,
    tokenizer_path: str | None = None,
    model_path: str | None = None,
    device: str | None = None,
    max_context: int | None = None,
    allow_network: bool = False,
    clip: float = 5.0,
) -> Any:
    """Load a Kronos predictor and expose it as the robinhood+ torch backend.

    Requires a local checkout of ``model.kronos`` (or installed Kronos) plus
    torch. Official class names stay ``Kronos*`` so Hugging Face ``config.json``
    / state-dict keys load; Dipcatcher public names wrap that object.
    """
    if variant not in VARIANT_HUB:
        raise RobinhoodPlusTorchError(f"unknown robinhood+ variant {variant!r}")
    spec = VARIANT_HUB[variant]
    tok_id = tokenizer_path or str(spec["tokenizer"])
    model_id = model_path or str(spec["model"])
    default_context = {"mini": 2048, "small": 512, "base": 512}[variant]
    context = int(max_context if max_context is not None else default_context)
    if not allow_network and (not Path(tok_id).exists() or not Path(model_id).exists()):
        raise RobinhoodPlusTorchError(
            "torch backend refused Hub download (allow_network=false); "
            f"pass local tokenizer/model directories (got {tok_id!r}, {model_id!r})"
        )
    if not torch_available():
        raise RobinhoodPlusTorchError(
            "robinhood+ torch backend requires the optional [nn] extra (torch)"
        )
    _ensure_kronos_on_path()
    try:
        from model.kronos import Kronos, KronosPredictor, KronosTokenizer
    except ImportError as exc:
        raise RobinhoodPlusTorchError(
            "official Kronos classes are not importable. Install/clone "
            "https://github.com/shiyu-coder/Kronos into third_party/kronos_src "
            "and keep model/ on PYTHONPATH"
        ) from exc
    tokenizer = KronosTokenizer.from_pretrained(tok_id)
    model = Kronos.from_pretrained(model_id)
    return KronosPredictor(model, tokenizer, device=device, max_context=context, clip=clip)


def _cached_predictor(config: AppConfig) -> Any:
    cfg = config.robinhood_plus
    key = (
        cfg.variant.value,
        cfg.tokenizer_path or "",
        cfg.model_path or "",
        int(cfg.max_context),
        bool(cfg.allow_network),
        float(cfg.clip),
    )
    cached = _PREDICTOR_CACHE.get(key)
    if cached is not None:
        return cached
    predictor = load_pretrained_predictor(
        cfg.variant.value,
        tokenizer_path=cfg.tokenizer_path,
        model_path=cfg.model_path,
        max_context=cfg.max_context,
        allow_network=cfg.allow_network,
        clip=cfg.clip,
    )
    _PREDICTOR_CACHE[key] = predictor
    if len(_PREDICTOR_CACHE) > 4:
        oldest = next(iter(_PREDICTOR_CACHE))
        _PREDICTOR_CACHE.pop(oldest, None)
    return predictor


def _future_stamps(times: list[datetime], pred_len: int) -> list[datetime]:
    if len(times) >= 2:
        delta = times[-1] - times[-2]
        if delta.total_seconds() <= 0:
            delta = timedelta(days=1)
    else:
        delta = timedelta(days=1)
    last = times[-1] if times else datetime(2020, 1, 2)
    return [last + delta * (i + 1) for i in range(pred_len)]


def _kline_paths_from_kronos(
    predictor: Any,
    kline: np.ndarray,
    times: list[datetime],
    *,
    pred_len: int,
    sample_count: int,
    temperature: float,
    top_p: float,
) -> np.ndarray:
    """Keep sample paths. Official ``predict`` averages them internally."""
    import pandas as pd

    cols = ["open", "high", "low", "close", "volume", "amount"]
    df = pd.DataFrame(kline, columns=cols)
    x_timestamp = pd.to_datetime(pd.Series(times))
    y_timestamp = pd.to_datetime(pd.Series(_future_stamps(times, pred_len)))
    paths = []
    for _ in range(max(1, sample_count)):
        pred_df = predictor.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=1,
            verbose=False,
        )
        arr = pred_df[cols].to_numpy(dtype=np.float64)
        if arr.shape != (pred_len, 6):
            raise RobinhoodPlusTorchError(
                f"Kronos-mini path shape {arr.shape} != ({pred_len}, 6)"
            )
        paths.append(arr)
    return np.stack(paths, axis=0)


def forecast_cross_section_torch(
    frame: pl.DataFrame,
    asof: datetime,
    config: AppConfig,
    security_ids: list[str],
) -> dict[str, RobinhoodPlusNameForecast]:
    """Causal Kronos-mini (or zoo) forecasts. Fail closed; never fall back to numpy."""
    if not kline_columns_present(frame.columns):
        return {}
    if "event_time" not in frame.columns:
        raise RobinhoodPlusTorchError("torch robinhood+ panel missing event_time")
    predictor = _cached_predictor(config)
    cfg = config.robinhood_plus
    hist = frame.filter(pl.col("event_time") <= asof)
    if "available_time" in hist.columns:
        hist = hist.filter(pl.col("available_time") <= asof)
    if hist.is_empty() or "security_id" not in hist.columns:
        return {}
    wanted = set(security_ids)
    levels = tuple(float(q) for q in config.quantiles.levels) or (0.05, 0.50, 0.95)
    horizons = tuple(int(h) for h in config.horizons.bars)
    out: dict[str, RobinhoodPlusNameForecast] = {}
    for group in hist.partition_by("security_id", maintain_order=True):
        security_id = str(group["security_id"][0])
        if security_id not in wanted:
            continue
        window = group.sort("event_time").tail(cfg.lookback)
        if window.height < 2:
            out[security_id] = RobinhoodPlusNameForecast(
                security_id=security_id,
                forecast=empty_forecast(STATUS_INSUFFICIENT_HISTORY),
            )
            continue
        try:
            kline = extract_kline(window)
        except ValueError:
            out[security_id] = RobinhoodPlusNameForecast(
                security_id=security_id,
                forecast=empty_forecast(STATUS_MISSING_OHLC),
            )
            continue
        times = [t for t in window["event_time"].to_list() if isinstance(t, datetime)]
        if len(times) != kline.shape[0]:
            out[security_id] = RobinhoodPlusNameForecast(
                security_id=security_id,
                forecast=empty_forecast(STATUS_MISSING_OHLC),
            )
            continue
        try:
            paths = _kline_paths_from_kronos(
                predictor,
                kline,
                times,
                pred_len=cfg.pred_len,
                sample_count=cfg.sample_count,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
        except RobinhoodPlusTorchError:
            raise
        except Exception as exc:  # noqa: BLE001 — fail closed, do not numpy-fallback
            raise RobinhoodPlusTorchError(
                f"Kronos-mini inference failed for {security_id!r}: {exc}"
            ) from exc
        forecast = path_forecast_from_paths(
            paths,
            float(kline[-1, 3]),
            quantile_levels=levels,
            horizons=horizons,
            diagnostics={
                "engine": ENGINE_NAME,
                "engine_version": ENGINE_VERSION,
                "family": FAMILY,
                "backend": "torch",
                "decoder": "kronos",
                "price_space": PRICE_SPACE,
                "lookback": float(kline.shape[0]),
                "variant": cfg.variant.value,
            },
        )
        out[security_id] = RobinhoodPlusNameForecast(security_id=security_id, forecast=forecast)
    return out
