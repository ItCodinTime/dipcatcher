# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Purged walk-forward and conformal intervals
#
# Data label: tracked real snapshot
#
# Not investment advice. No live-trading claim.
# The US snapshot is scored only after its bytes match `configs/real_benchmark_us_wide.json`.
# If that file is absent, or the bytes are not the sealed snapshot, the example skips.
# Reported scores are pinball, CRPS, and conformal coverage.

# %%
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.config.models import ValidationConfig
from quant_fund.metrics.conformal import conformal_quantile, set_metrics
from quant_fund.metrics.scoring import crps_gaussian, mean_pinball
from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.validation.walk_forward import (
    assert_no_label_overlap,
    session_index,
    walk_forward,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "real_benchmark_us_wide.json"
FEATURES = ("return_1", "return_5", "return_20", "volatility_20")
CONFORMAL_ALPHA = 0.1
TRAIN_BARS = 252
VAL_BARS = 63
TEST_BARS = 63


def _load_object(path: Path) -> dict[str, object]:
    loaded: object = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} must be a JSON object")
    parsed: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise SystemExit(f"{path} has a non-string key")
        parsed[key] = value
    return parsed


def _as_str(payload: dict[str, object], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{key} must be a non-empty string")
    return value


def _as_int(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"{key} must be an integer")
    return value


def _as_float(payload: dict[str, object], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SystemExit(f"{key} must be a number")
    return float(value)


def _as_bool(payload: dict[str, object], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise SystemExit(f"{key} must be a boolean")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _datetimes(frame: pl.DataFrame) -> list[datetime]:
    values = frame["event_time"].unique().sort().to_list()
    times = [value for value in values if isinstance(value, datetime)]
    if len(times) != len(values):
        raise SystemExit("event_time values must be datetimes")
    return times


def _matrix(frame: pl.DataFrame) -> np.ndarray:
    return np.asarray(frame.select(FEATURES).to_numpy(), dtype=np.float64)


def _target(frame: pl.DataFrame) -> np.ndarray:
    return np.asarray(frame["target"].to_numpy(), dtype=np.float64)


def _ridge_predict(
    train: pl.DataFrame,
    features_new: np.ndarray,
    ridge_alpha: float,
) -> np.ndarray:
    design = _matrix(train)
    target = _target(train)
    center = design.mean(axis=0)
    scale = design.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (design - center) / scale
    intercept = float(target.mean())
    gram = standardized.T @ standardized + ridge_alpha * np.eye(len(FEATURES))
    beta = np.linalg.solve(gram, standardized.T @ (target - intercept))
    predicted = intercept + ((features_new - center) / scale) @ beta
    return np.asarray(predicted, dtype=np.float64)


def _features(frame: pl.DataFrame) -> pl.DataFrame:
    ordered = frame.sort(["security_id", "event_time"])
    featured = ordered.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("security_id") - 1.0).alias("return_1"),
        (pl.col("close") / pl.col("close").shift(5).over("security_id") - 1.0).alias("return_5"),
        (pl.col("close") / pl.col("close").shift(20).over("security_id") - 1.0).alias("return_20"),
        (pl.col("close").shift(-1).over("security_id") / pl.col("close") - 1.0).alias("target"),
    ).with_columns(
        pl.col("return_1")
        .rolling_std(window_size=20, min_samples=20)
        .over("security_id")
        .alias("volatility_20")
    )
    return featured.drop_nulls([*FEATURES, "target"])


def _emit(scores: dict[str, object]) -> None:
    if not family_blob_forbidden_metrics_absent(scores):
        raise SystemExit("score dict contains a forbidden headline metric")
    for key in sorted(scores):
        value = scores[key]
        if isinstance(value, float):
            print(f"{key}={value:.6g}")
        else:
            print(f"{key}={value}")


def main() -> None:
    protocol = _load_object(PROTOCOL_PATH)
    dataset = (PROTOCOL_PATH.parent / _as_str(protocol, "dataset_path")).resolve()
    print("data_label=tracked_real_snapshot")
    print("claim=research_only")
    print("not_investment_advice=true")
    print("no_live_trading_claim=true")
    if not dataset.is_file():
        print(f"SKIP: tracked real US snapshot absent ({dataset})")
        return
    actual = _sha256(dataset)
    expected = _as_str(protocol, "dataset_sha256")
    if actual != expected:
        print("SKIP: tracked real US snapshot hash does not match the benchmark config")
        print(f"expected_sha256={expected}")
        print(f"actual_sha256={actual}")
        return

    horizon = _as_int(protocol, "horizon_sessions")
    embargo = _as_int(protocol, "embargo_sessions")
    ridge_alpha = _as_float(protocol, "ridge_alpha")
    frame = pl.read_parquet(
        dataset,
        columns=["security_id", "event_time", "available_time", "close", "source"],
    )
    known = frame.filter(
        pl.col("close").is_finite()
        & (pl.col("close") > 0)
        & pl.col("available_time").is_not_null()
        & (pl.col("available_time") <= pl.col("event_time"))
    )
    work = _features(known)
    if work.is_empty():
        raise SystemExit("sealed snapshot produced no eligible rows")
    times = _datetimes(work)
    folds = walk_forward(
        times,
        ValidationConfig(
            scheme="expanding", train_bars=TRAIN_BARS, val_bars=VAL_BARS, test_bars=TEST_BARS
        ),
        horizon_bars=horizon,
        embargo_bars=embargo,
    )
    if not folds:
        raise SystemExit("walk-forward produced no folds")
    index = session_index(times)
    for fold in folds:
        assert_no_label_overlap(fold, horizon, index)

    observed: list[np.ndarray] = []
    point: list[np.ndarray] = []
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    scale: list[np.ndarray] = []
    for fold in folds:
        train = work.filter(pl.col("event_time").is_in(fold.train_times))
        validation = work.filter(pl.col("event_time").is_in(fold.val_times))
        test = work.filter(pl.col("event_time").is_in(fold.test_times))
        if min(train.height, validation.height, test.height) < 30:
            continue
        validation_hat = _ridge_predict(train, _matrix(validation), ridge_alpha)
        test_hat = _ridge_predict(train, _matrix(test), ridge_alpha)
        validation_y = _target(validation)
        test_y = _target(test)
        if not (
            np.isfinite(validation_hat).all()
            and np.isfinite(test_hat).all()
            and np.isfinite(validation_y).all()
            and np.isfinite(test_y).all()
        ):
            raise SystemExit("non-finite forecast or target")
        residual = validation_y - validation_hat
        half_width = conformal_quantile(np.abs(residual), CONFORMAL_ALPHA)
        sigma = float(np.std(residual, ddof=1))
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise SystemExit("calibration residual scale is not positive")
        observed.append(test_y)
        point.append(test_hat)
        lower.append(test_hat - half_width)
        upper.append(test_hat + half_width)
        scale.append(np.full(test_hat.shape, sigma, dtype=np.float64))
    if not observed:
        raise SystemExit("no walk-forward fold had enough purged rows to score")

    y = np.concatenate(observed)
    mu = np.concatenate(point)
    lo = np.concatenate(lower)
    hi = np.concatenate(upper)
    sigma_hat = np.concatenate(scale)
    coverage = set_metrics(y, lo, hi)
    crps = float(np.nanmean(crps_gaussian(y, mu, sigma_hat)))
    if not np.isfinite(crps):
        raise SystemExit("CRPS was not finite")
    sources = sorted(str(value) for value in known["source"].unique().to_list())
    scores: dict[str, object] = {
        "dataset_sha256": actual,
        "vendor_source": ",".join(sources),
        "survivorship_bias": str(_as_bool(protocol, "survivorship_bias")).lower(),
        "availability_basis": _as_str(protocol, "availability_basis"),
        "holdout_previously_inspected": str(
            _as_bool(protocol, "holdout_previously_inspected")
        ).lower(),
        "price_adjustment": _as_str(protocol, "price_adjustment"),
        "n_names": int(work["security_id"].n_unique()),
        "n_folds": len(observed),
        "n_test_rows": int(y.size),
        "horizon_sessions": horizon,
        "embargo_sessions": embargo,
        "purge_embargo_ok": "true",
        "conformal_alpha": CONFORMAL_ALPHA,
        "pinball_0.1": mean_pinball(y, lo, 0.1),
        "pinball_0.5": mean_pinball(y, mu, 0.5),
        "pinball_0.9": mean_pinball(y, hi, 0.9),
        "crps_gaussian": crps,
        "conformal_coverage": coverage.coverage,
        "conformal_mean_width": coverage.mean_width,
    }
    _emit(scores)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"example_failed={type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
