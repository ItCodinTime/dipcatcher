"""Fail-closed protocol validation for preserved external crypto panels.

This module deliberately validates candidate evidence only; it does not mint
industry-grade or SOTA claims and performs no network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from quant_fund.data.adapters.binance import BINANCE_SYMBOLS

_REQUIRED_COLUMNS = {
    "security_id",
    "symbol",
    "event_time",
    "available_time",
    "ingested_time",
    "source",
    "revision_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


@dataclass(frozen=True)
class ExternalBenchmarkProtocol:
    """Immutable evaluation metadata for one external candidate run."""

    input_id: str
    input_sha256: str
    symbols: tuple[str, ...] = BINANCE_SYMBOLS
    interval: str = "1d"
    timezone: str = "UTC"
    availability_policy: str = "next-day UTC"
    horizon: int = 5
    candidate_only: bool = True
    proof_status: str = "not_proof"
    proof_eligible: bool = False
    sota_proven: bool = False
    n_dates: int = 0


def build_external_protocol(
    *, input_id: str, input_sha256: str, n_dates: int
) -> ExternalBenchmarkProtocol:
    """Build the frozen candidate-only protocol after basic metadata checks."""
    if (
        not input_id
        or len(input_sha256) != 64
        or any(c not in "0123456789abcdef" for c in input_sha256.lower())
    ):
        raise ValueError("external benchmark input metadata is invalid")
    if isinstance(n_dates, bool) or not isinstance(n_dates, int) or n_dates < 1:
        raise ValueError("n_dates must be a positive integer")
    return ExternalBenchmarkProtocol(
        input_id=input_id, input_sha256=input_sha256.lower(), n_dates=n_dates
    )


def validate_external_panel(
    frame: pl.DataFrame, protocol: ExternalBenchmarkProtocol
) -> dict[str, Any]:
    """Validate a complete five-symbol PIT panel without filling or dropping data."""
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing external panel columns: {sorted(missing)}")
    sources = set(frame.get_column("source").drop_nulls().to_list())
    if not sources or "synthetic" in sources:
        raise ValueError("external benchmark requires non-synthetic source")
    if len(sources) != 1:
        raise ValueError("external panel must use one external source")
    if frame.null_count().to_numpy().sum() > 0:
        raise ValueError("external panel contains null fields")
    for column in ("event_time", "available_time", "ingested_time"):
        dtype = frame.get_column(column).dtype
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != protocol.timezone:
            raise ValueError(f"external panel {column} must use {protocol.timezone} timezone")
    if frame.select((pl.col("event_time") > pl.col("available_time")).any()).to_series().item():
        raise ValueError("available_time must not precede event_time")
    if frame.select((pl.col("available_time") > pl.col("ingested_time")).any()).to_series().item():
        raise ValueError("ingested_time must not precede available_time")
    if frame.select(pl.struct("symbol", "event_time").is_duplicated().any()).to_series().item():
        raise ValueError("duplicate external symbol/event_time observations")
    symbols = set(frame.get_column("symbol").to_list())
    if symbols != set(protocol.symbols):
        raise ValueError(f"external panel symbols must be exactly {protocol.symbols}")
    revisions = set(frame.get_column("revision_id").to_list())
    if len(revisions) != 1:
        raise ValueError("external panel must use one revision_id")
    dates = frame.get_column("event_time").n_unique()
    if dates != protocol.n_dates:
        raise ValueError("external panel date count does not match protocol")
    counts = frame.group_by("symbol").len().get_column("len").to_list()
    if len(counts) != len(protocol.symbols) or len(set(counts)) != 1 or counts[0] != dates:
        raise ValueError("external panel must be complete for every symbol")
    return {
        "ok": True,
        "source": next(iter(sources)),
        "rows": frame.height,
        "symbols": list(protocol.symbols),
        "dates": dates,
        "revision_id": next(iter(revisions)),
        "candidate_only": True,
        "proof_status": "not_proof",
        "proof_eligible": False,
        "sota_proven": False,
    }


def variance_scale_qlike(realized_variance: float, forecast_variance: float) -> float:
    """QLIKE on the variance scale used by TSFM-RV-style evaluations."""
    import math

    if not (realized_variance > 0 and forecast_variance > 0):
        raise ValueError("realized and forecast variances must be positive")
    ratio = realized_variance / forecast_variance
    return ratio - math.log(ratio) - 1.0


def build_external_origins(
    *, n_dates: int, h: int, min_history: int, stride: int, n_validation: int, n_test: int
) -> tuple[Any, Any]:
    """Return deterministic validation and strictly separated test origins."""
    import numpy as np

    for value in (n_dates, h, min_history, stride, n_validation, n_test):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("external origin sizes must be positive integers")
    if stride < h:
        raise ValueError("origin stride must cover the horizon")
    validation = np.arange(min_history, n_dates - h + 1, stride)[:n_validation]
    if validation.size != n_validation:
        raise ValueError("insufficient validation origins")
    first_test = int(validation[-1]) + 2 * h
    test = np.arange(first_test, n_dates - h + 1, stride)
    if test.size < n_test:
        raise ValueError("insufficient test origins")
    return validation, test[:n_test]


def select_external_model(validation_losses: dict[str, Any]) -> str:
    """Select one model using validation losses only, with deterministic ties."""
    if not validation_losses:
        raise ValueError("validation losses must be nonempty")
    import numpy as np

    normalized: dict[str, np.ndarray] = {}
    for name, values in validation_losses.items():
        if not name:
            raise ValueError("validation model names must be nonempty")
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError("validation losses must be finite and nonempty")
        normalized[name] = array
    sizes = {values.size for values in normalized.values()}
    if len(sizes) != 1:
        raise ValueError("validation losses must align")
    return min(sorted(normalized), key=lambda name: float(np.mean(normalized[name])))


def summarize_external_losses(
    validation_losses: dict[str, Any],
    test_losses: dict[str, Any],
    *,
    horizon: int,
    source: str,
) -> dict[str, Any]:
    """Summarize a candidate external comparison without minting proof."""
    import math

    import numpy as np

    if not source or source == "synthetic":
        raise ValueError("external benchmark summary requires a non-synthetic source")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    selected = select_external_model(validation_losses)
    if set(validation_losses) != set(test_losses):
        raise ValueError("validation and test model sets must match")
    normalized_test: dict[str, np.ndarray] = {}
    for name, values in test_losses.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
            raise ValueError("test losses must be finite and nonempty")
        normalized_test[name] = array
    sizes = {values.size for values in normalized_test.values()}
    if len(sizes) != 1:
        raise ValueError("test losses must align")
    from quant_fund.metrics.inference import diebold_mariano, overlap_aware_hac_lags

    competitors = sorted(name for name in normalized_test if name != selected)
    comparisons: dict[str, Any] = {}
    lags = overlap_aware_hac_lags(next(iter(sizes)), horizon)
    for competitor in competitors:
        dm = diebold_mariano(
            normalized_test[selected],
            normalized_test[competitor],
            lags=lags,
            name_a=selected,
            name_b=competitor,
        )
        p_adjusted = min(1.0, dm.p_value * len(competitors)) if math.isfinite(dm.p_value) else None
        comparisons[competitor] = {
            "mean_loss_difference": float(dm.mean_loss_diff),
            "effect_size": float(dm.mean_loss_diff),
            "dm_statistic": float(dm.statistic) if math.isfinite(dm.statistic) else None,
            "p_two_sided": float(dm.p_value) if math.isfinite(dm.p_value) else None,
            "p_two_sided_bonferroni": p_adjusted,
            "hac_lags": int(dm.lags),
            "n": int(dm.n),
            "preferred": dm.preferred,
        }
    return {
        "selected": selected,
        "validation_mean_loss": float(
            np.mean(np.asarray(validation_losses[selected], dtype=float))
        ),
        "test_mean_loss": float(np.mean(normalized_test[selected])),
        "n_test": int(next(iter(sizes))),
        "horizon": horizon,
        "source": source,
        "comparisons": comparisons,
        "candidate_only": True,
        "proof_status": "not_proof",
        "proof_eligible": False,
        "sota_proven": False,
    }
