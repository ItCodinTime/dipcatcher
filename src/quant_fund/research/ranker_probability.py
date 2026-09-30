"""Research-only, date-purged ranker probability comparison.

Run with ``python -m quant_fund.research.ranker_probability --help``. No fitted
model from this experiment is installed in the forecasting pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from quant_fund.config.loader import load_config
from quant_fund.data.parquet import read_derived_parquet
from quant_fund.metrics import inference
from quant_fund.metrics.inference import stationary_bootstrap_indices
from quant_fund.models import calibration, ranking
from quant_fund.models.calibration import ProbabilityCalibrator
from quant_fund.models.ranking import PUBLIC_FEATURES, RidgeRanker
from quant_fund.pipeline.dataset import build_gold, design_frame
from quant_fund.utils.hashing import hash_file


@dataclass(frozen=True)
class ExperimentSpec:
    label: str = "future_idio_return_1"
    label_end: str = "label_end_time_1"
    features: tuple[str, ...] = tuple(PUBLIC_FEATURES)
    train_dates: int = 252
    cal_dates: int = 63
    test_dates: int = 63
    embargo_dates: int = 1
    ridge_alpha: float = 1.0
    n_boot: int = 2000
    mean_block: float = 10.0
    min_test_dates: int = 250
    seed: int = 17

    def validate(self) -> None:
        if self.label != "future_idio_return_1" or self.label_end != "label_end_time_1":
            raise ValueError("this experiment is pinned to the one-session idiosyncratic label")
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("features must be nonempty and unique")
        if "cs_pct_mom_20" not in self.features:
            raise ValueError("the common-row momentum control requires cs_pct_mom_20")
        if min(self.train_dates, self.cal_dates, self.test_dates) < 1:
            raise ValueError("train, calibration, and test date counts must be positive")
        if self.embargo_dates < 0 or self.n_boot < 1 or self.min_test_dates < 1:
            raise ValueError("embargo, bootstrap count, or minimum test dates are invalid")
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be finite and nonnegative")
        if not np.isfinite(self.mean_block) or self.mean_block < 1:
            raise ValueError("mean_block must be finite and at least one")


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _iso(value: datetime) -> str:
    return value.isoformat()


def _validate_frame(frame: pl.DataFrame, spec: ExperimentSpec) -> None:
    required = {
        "event_time",
        "decision_time",
        "max_source_available_time",
        "security_id",
        "feature_set_version",
        spec.label,
        spec.label_end,
        *spec.features,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"probability experiment input lacks columns: {missing}")
    if frame.is_empty() or frame.select("event_time", "security_id").is_duplicated().any():
        raise ValueError("probability experiment requires unique nonempty date/name rows")
    if any(
        frame.schema[name].base_type() != pl.Datetime
        for name in ("event_time", "decision_time", "max_source_available_time", spec.label_end)
    ):
        raise ValueError("experiment timestamps must be Datetime columns")
    invalid_clock = frame.filter(
        pl.any_horizontal(
            pl.col("event_time").is_null(),
            pl.col("decision_time").is_null(),
            pl.col("max_source_available_time").is_null(),
            pl.col("decision_time") < pl.col("event_time"),
            pl.col("max_source_available_time") > pl.col("decision_time"),
            pl.col("security_id").is_null(),
        )
    )
    if invalid_clock.height:
        raise ValueError("a feature or identity is unavailable at its decision time")
    invalid_label = frame.filter(
        pl.col(spec.label).is_not_null()
        & (pl.col(spec.label_end).is_null() | (pl.col(spec.label_end) <= pl.col("decision_time")))
    )
    if invalid_label.height:
        raise ValueError("labeled rows require a strictly later observed label endpoint")
    versions = frame.get_column("feature_set_version").unique().to_list()
    if len(versions) != 1 or versions[0] is None:
        raise ValueError("experiment requires one nonnull feature-set version")


def _losses(
    p: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p = np.asarray(p, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probability predictions must be finite and aligned")
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return (p - y) ** 2, -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def _scale_scores(
    cal: NDArray[np.float64], test: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Use calibration-block moments only so Platt regularization is scale fair."""
    center = float(np.mean(cal))
    scale = float(np.std(cal))
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("calibration scores must have finite nonzero variation")
    return (cal - center) / scale, (test - center) / scale


def _paired_interval(
    losses_a: NDArray[np.float64],
    losses_b: NDArray[np.float64],
    *,
    n_boot: int,
    mean_block: float,
    seed: int,
) -> dict[str, float]:
    """Equal-date paired loss difference and stationary-block percentile interval."""
    delta = np.asarray(losses_a, dtype=float) - np.asarray(losses_b, dtype=float)
    if delta.ndim != 1 or delta.size < 2 or not np.isfinite(delta).all():
        raise ValueError("paired interval needs at least two finite date losses")
    rng = np.random.default_rng(seed)
    draws = stationary_bootstrap_indices(delta.size, n_boot, min(mean_block, delta.size), rng)
    boot = delta[draws].mean(axis=1)
    return {
        "mean_difference": float(delta.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }


def evaluate(frame: pl.DataFrame, spec: ExperimentSpec = ExperimentSpec()) -> dict[str, Any]:
    """Compare three methods on identical untouched date/name test rows.

    A fold fits ridge on train dates, fits both Platt maps on later calibration
    dates, then evaluates a later test block. Whole dates whose latest observed
    label ends at/after the next block are purged before either fit.
    """
    spec.validate()
    _validate_frame(frame, spec)
    ordered = frame.sort("event_time", "security_id")
    extras = ["decision_time", "max_source_available_time", spec.label_end]
    sub = design_frame(
        ordered,
        spec.label,
        list(spec.features),
        extra_columns=[name for name in extras if name not in spec.features],
    )
    numeric = sub.select([*spec.features, spec.label]).to_numpy().astype(float)
    sub = sub.filter(np.isfinite(numeric).all(axis=1))
    if sub.is_empty():
        raise ValueError("no finite common rows remain after feature/label filtering")
    dates = sorted(ordered.get_column("event_time").unique().to_list())
    date_index = {day: i for i, day in enumerate(dates)}
    row_dates = sub.get_column("event_time").to_list()
    row_date_index = np.asarray([date_index[day] for day in row_dates], dtype=np.int32)
    x = sub.select(spec.features).to_numpy().astype(float)
    y = sub.get_column(spec.label).to_numpy().astype(float)
    binary = (y > 0.0).astype(float)
    momentum = sub.get_column("cs_pct_mom_20").to_numpy().astype(float)
    endpoints = (
        ordered.filter(pl.col(spec.label).is_not_null())
        .group_by("event_time")
        .agg(pl.col(spec.label_end).max())
    )
    max_end = {row["event_time"]: row[spec.label_end] for row in endpoints.iter_rows(named=True)}
    all_scores: dict[str, list[tuple[float, float]]] = {
        name: [] for name in ("ranker_platt", "momentum_platt", "train_base_rate")
    }
    folds: list[dict[str, Any]] = []
    cursor = spec.train_dates
    while cursor + spec.cal_dates + spec.test_dates <= len(dates):
        first_cal = dates[cursor]
        test_start_index = cursor + spec.cal_dates
        first_test = dates[test_start_index]
        last_test_index = test_start_index + spec.test_dates
        train_keep = np.asarray(
            [
                i + spec.embargo_dates < cursor and max_end.get(day, first_cal) < first_cal
                for i, day in enumerate(dates[:cursor])
            ],
            dtype=bool,
        )
        cal_keep = np.asarray(
            [
                i + spec.embargo_dates < test_start_index
                and max_end.get(day, first_test) < first_test
                for i, day in enumerate(dates[cursor:test_start_index], start=cursor)
            ],
            dtype=bool,
        )
        tr = np.isin(row_date_index, np.flatnonzero(train_keep))
        ca = np.isin(row_date_index, cursor + np.flatnonzero(cal_keep))
        te = (row_date_index >= test_start_index) & (row_date_index < last_test_index)
        used_train_dates = np.unique(row_date_index[tr])
        used_cal_dates = np.unique(row_date_index[ca])
        fold: dict[str, Any] = {
            "first_cal": _iso(first_cal),
            "first_test": _iso(first_test),
            "last_test": _iso(dates[last_test_index - 1]),
            "n_train_dates": int(used_train_dates.size),
            "n_cal_dates": int(used_cal_dates.size),
            "n_test_dates": int(np.unique(row_date_index[te]).size),
            "n_train_rows": int(tr.sum()),
            "n_cal_rows": int(ca.sum()),
            "n_test_rows": int(te.sum()),
        }
        fold["latest_train_label_end"] = (
            _iso(max(max_end[dates[i]] for i in used_train_dates))
            if used_train_dates.size
            else None
        )
        fold["latest_cal_label_end"] = (
            _iso(max(max_end[dates[i]] for i in used_cal_dates)) if used_cal_dates.size else None
        )
        if tr.sum() < 10 or ca.sum() < 10 or not te.any() or np.unique(binary[ca]).size < 2:
            fold["status"] = "untrainable"
            fold["reason"] = "insufficient common rows or calibration classes"
            folds.append(fold)
            cursor += spec.test_dates
            continue
        ridge = RidgeRanker(alpha=spec.ridge_alpha).fit(
            x[tr], y[tr], dates=np.asarray(row_dates)[tr]
        )
        try:
            ranker_cal, ranker_test = _scale_scores(ridge.predict(x[ca]), ridge.predict(x[te]))
            momentum_cal, momentum_test = _scale_scores(momentum[ca], momentum[te])
        except ValueError as exc:
            fold["status"] = "untrainable"
            fold["reason"] = str(exc)
            folds.append(fold)
            cursor += spec.test_dates
            continue
        ranker_platt = ProbabilityCalibrator("platt").fit(ranker_cal, binary[ca])
        momentum_platt = ProbabilityCalibrator("platt").fit(momentum_cal, binary[ca])
        probabilities = {
            "ranker_platt": ranker_platt.predict(ranker_test),
            "momentum_platt": momentum_platt.predict(momentum_test),
            "train_base_rate": np.full(int(te.sum()), float(binary[tr].mean())),
        }
        test_idx = row_date_index[te]
        test_y = binary[te]
        fold["prediction_sha256"] = _digest(
            {
                "dates": [_iso(row_dates[i]) for i in np.flatnonzero(te)],
                "probabilities": {k: v.tolist() for k, v in probabilities.items()},
            }
        )
        for name, p in probabilities.items():
            brier, log_loss = _losses(p, test_y)
            for date_id in np.unique(test_idx):
                same_date = test_idx == date_id
                all_scores[name].append(
                    (float(brier[same_date].mean()), float(log_loss[same_date].mean()))
                )
        fold["status"] = "scored"
        folds.append(fold)
        cursor += spec.test_dates
    n_scored = len(all_scores["ranker_platt"])
    report: dict[str, Any] = {
        "kind": "ranker_probability_experiment",
        "schema_version": 1,
        "spec": {**asdict(spec), "features": list(spec.features)},
        "feature_set_version": ordered.get_column("feature_set_version").item(0),
        "input_rows": ordered.height,
        "common_rows": sub.height,
        "folds": folds,
        "n_scored_dates": n_scored,
        "n_untrainable_folds": sum(fold["status"] != "scored" for fold in folds),
        "metric_unit": "equal_weight_per_decision_date",
        "research_only": True,
        "synthetic_is_correctness_only": True,
        "production_promotion": False,
        "forward_evidence_accepted": False,
    }
    if n_scored < 2:
        report.update(
            {"status": "unmeasured", "gate_pass": False, "reason": "fewer than two scored dates"}
        )
        return report
    losses = {
        name: {
            "brier": float(np.mean([row[0] for row in values])),
            "log_loss": float(np.mean([row[1] for row in values])),
        }
        for name, values in all_scores.items()
    }
    ranker_brier = np.asarray([row[0] for row in all_scores["ranker_platt"]])
    comparisons = {
        name: _paired_interval(
            ranker_brier,
            np.asarray([row[0] for row in all_scores[name]]),
            n_boot=spec.n_boot,
            mean_block=spec.mean_block,
            seed=spec.seed + i,
        )
        for i, name in enumerate(("momentum_platt", "train_base_rate"))
    }
    gate = (
        n_scored >= spec.min_test_dates
        and report["n_untrainable_folds"] == 0
        and all(comp["ci_high"] < 0.0 for comp in comparisons.values())
        and all(
            losses["ranker_platt"]["log_loss"] <= losses[name]["log_loss"] for name in comparisons
        )
    )
    report.update(
        {
            "status": "measured",
            "losses": losses,
            "paired_brier": comparisons,
            "gate_pass": gate,
            "gate_definition": "min_test_dates; no untrainable folds; paired 95% Brier CI below zero versus both controls; log loss no worse than either control",
        }
    )
    return report


def _load_gold(features: Path, labels: Path, spec: ExperimentSpec) -> pl.DataFrame:
    feat = read_derived_parquet(
        features,
        dataset="ranker gold features",
        required_columns=("event_time", "security_id"),
    )
    lab = read_derived_parquet(
        labels,
        dataset="ranker gold labels",
        required_columns=("event_time", "security_id", spec.label, spec.label_end),
    )
    keys = ["event_time", "security_id"]
    if not set([*keys, spec.label, spec.label_end]) <= set(lab.columns):
        raise ValueError("label artifact lacks the specified return or observed endpoint")
    if feat.select(keys).is_duplicated().any() or lab.select(keys).is_duplicated().any():
        raise ValueError("gold feature/label artifacts contain duplicate date/name keys")
    return feat.join(lab.select([*keys, spec.label, spec.label_end]), on=keys, how="inner")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, help="existing gold/features.parquet")
    parser.add_argument("--labels", type=Path, help="existing gold/labels.parquet")
    parser.add_argument("--bronze-root", type=Path, help="canonical offline bronze file directory")
    parser.add_argument("--lake-root", type=Path, help="derived silver/gold output directory")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-dates", type=int, default=252)
    parser.add_argument("--cal-dates", type=int, default=63)
    parser.add_argument("--test-dates", type=int, default=63)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--feature-columns", nargs="+", default=PUBLIC_FEATURES)
    args = parser.parse_args(argv)
    spec = ExperimentSpec(
        features=tuple(args.feature_columns),
        train_dates=args.train_dates,
        cal_dates=args.cal_dates,
        test_dates=args.test_dates,
        n_boot=args.n_boot,
    )
    spec.validate()
    if args.bronze_root is not None:
        if args.features is not None or args.labels is not None or args.lake_root is None:
            parser.error("--bronze-root requires --lake-root and excludes --features/--labels")
        config = load_config(args.config)
        config.data.source = "file"
        config.data.parquet_path = args.bronze_root.resolve()
        config.data.root = args.lake_root.resolve()
        config.horizons.bars = [1]
        config.horizons.names = ["1d"]
        build_gold(config)
        features = args.lake_root / "gold" / "features.parquet"
        labels = args.lake_root / "gold" / "labels.parquet"
        raw_inputs = {
            name: (
                hash_file(args.bronze_root / f"{name}.parquet")
                if (args.bronze_root / f"{name}.parquet").is_file()
                else None
            )
            for name in ("bars", "corporate_actions", "security_master")
        }
        config_digest = _digest(config.model_dump(mode="json"))
    else:
        if args.features is None or args.labels is None or args.lake_root is not None:
            parser.error("pass both --features and --labels, or --bronze-root with --lake-root")
        features, labels = args.features, args.labels
        raw_inputs = None
        config_digest = None
    result = evaluate(_load_gold(features, labels, spec), spec)
    result["input_sha256"] = {"features": hash_file(features), "labels": hash_file(labels)}
    result["bronze_input_sha256"] = raw_inputs
    result["gold_build_config_sha256"] = config_digest
    result["code_sha256"] = {
        "ranker_probability.py": hash_file(Path(__file__)),
        "ranking.py": hash_file(Path(ranking.__file__)),
        "calibration.py": hash_file(Path(calibration.__file__)),
        "inference.py": hash_file(Path(inference.__file__)),
    }
    result["data_scope"] = "exploratory_previously_inspected_or_unverified"
    result["holdout_previously_inspected_or_unverified"] = True
    result["receipt_sha256"] = _digest(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": result["status"],
                "gate_pass": result["gate_pass"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
