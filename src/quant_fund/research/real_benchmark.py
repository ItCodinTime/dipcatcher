"""Frozen real-data forecast benchmark. No execution or live-return claims.

Run ``python -m quant_fund.research.real_benchmark --help`` for the two-stage
CLI. All models use the same eligible rows and the same fixed training sample.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

FEATURES = ("return_1", "return_5", "return_20", "volatility_20")
MODELS = ("zero", "historical_mean", "rolling_mean_20", "ridge")


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "receipt_sha256": _digest(value)}


def _read_receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    digest = value.pop("receipt_sha256", None)
    if digest != _digest(value):
        raise ValueError(f"receipt hash mismatch: {path.name}")
    return {**value, "receipt_sha256": digest}


def _day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)


@dataclass(frozen=True)
class BenchmarkProtocol:
    dataset_path: str
    dataset_sha256: str
    source_url: str
    usage_basis: str
    price_column: str
    price_adjustment: str
    universe_description: str
    survivorship_bias: bool
    availability_basis: str
    holdout_previously_inspected: bool
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    horizon_sessions: int = 1
    embargo_sessions: int = 1
    decision_delay_seconds: int = 0
    min_train_rows: int = 100
    min_score_dates: int = 30
    ridge_alpha: float = 1.0

    def validate(self) -> None:
        for field in (
            "source_url",
            "usage_basis",
            "price_column",
            "price_adjustment",
            "universe_description",
            "dataset_path",
        ):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f"{field} must be explicitly documented")
        if not self.source_url.startswith(("https://", "http://")):
            raise ValueError("source_url must identify the data source")
        if len(self.dataset_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.dataset_sha256
        ):
            raise ValueError("dataset_sha256 must be a lowercase SHA-256 digest")
        if self.availability_basis not in {"recorded_asof", "reconstructed"}:
            raise ValueError("availability_basis must be recorded_asof or reconstructed")
        for name in ("survivorship_bias", "holdout_previously_inspected"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} requires an explicit boolean disclosure")
        bounds = [
            _day(getattr(self, k))
            for k in (
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
            )
        ]
        if not bounds[0] < bounds[1] < bounds[2] < bounds[3] < bounds[4] < bounds[5]:
            raise ValueError("train, validation and test dates must be strictly ordered")
        for name in ("horizon_sessions", "min_train_rows", "min_score_dates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.embargo_sessions) is not int or self.embargo_sessions < self.horizon_sessions:
            raise ValueError("embargo_sessions must cover the label horizon")
        if (
            type(self.decision_delay_seconds) is not int
            or not 0 <= self.decision_delay_seconds < 86400
        ):
            raise ValueError("decision_delay_seconds must be an integer in [0, 86400)")
        if (
            isinstance(self.ridge_alpha, bool)
            or not np.isfinite(self.ridge_alpha)
            or self.ridge_alpha <= 0
        ):
            raise ValueError("ridge_alpha must be finite and positive")


def _load_bars(protocol: BenchmarkProtocol) -> pl.DataFrame:
    # Read and hash the same bytes that will be parsed; no hash/read race.
    raw = Path(protocol.dataset_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol.dataset_sha256:
        raise ValueError("dataset hash mismatch")
    frame = pl.read_parquet(io.BytesIO(raw))
    required = {
        "security_id",
        "event_time",
        "available_time",
        "ingested_time",
        "source",
        protocol.price_column,
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if frame.is_empty() or any(frame[c].null_count() for c in required):
        raise ValueError("empty dataset or null required values")
    for name in ("event_time", "available_time", "ingested_time"):
        dtype = frame.schema[name]
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
            raise ValueError(f"{name} must be a timezone-aware datetime")
    frame = frame.with_columns(
        pl.col("event_time", "available_time", "ingested_time").dt.convert_time_zone("UTC"),
        pl.col(protocol.price_column).cast(pl.Float64).alias("price"),
        pl.col("security_id").cast(pl.String),
    ).sort(["security_id", "event_time"])
    sources = frame["source"].cast(pl.String).unique().to_list()
    if any(not s.strip() or "synthetic" in s.lower() or "fixture" in s.lower() for s in sources):
        raise ValueError("real-data benchmark rejects synthetic/fixture source labels")
    if frame.select(pl.struct("security_id", "event_time").is_duplicated().any()).item():
        raise ValueError("duplicate security_id/event_time; resolve revisions upstream")
    if frame.filter(~pl.col("price").is_finite() | (pl.col("price") <= 0)).height:
        raise ValueError("prices must be finite and positive")
    if frame.filter(
        (pl.col("available_time") < pl.col("event_time"))
        | (pl.col("ingested_time") < pl.col("available_time"))
    ).height:
        raise ValueError("timestamps must satisfy event <= available <= ingested")
    if frame.select(
        pl.struct("security_id", pl.col("event_time").dt.date()).is_duplicated().any()
    ).item():
        raise ValueError("daily benchmark accepts one bar per security per UTC date")
    sessions = frame.select("event_time").unique()
    if sessions["event_time"].dt.date().n_unique() != sessions.height:
        raise ValueError("all securities must use the same session timestamp per UTC date")
    # An observation available after the declared decision cannot enter features.
    cutoff = pl.col("event_time") + pl.duration(seconds=protocol.decision_delay_seconds)
    return frame.with_columns((pl.col("available_time") <= cutoff).alias("known"))


def _samples(
    frame: pl.DataFrame, protocol: BenchmarkProtocol
) -> tuple[pl.DataFrame, dict[str, Any]]:
    times = frame["event_time"].unique().sort().to_list()
    index = {t: i for i, t in enumerate(times)}
    boundaries = {
        phase: next((j for j, t in enumerate(times) if t.date().isoformat() >= start), len(times))
        for phase, start in (
            ("train", protocol.validation_start),
            ("validation", protocol.test_start),
        )
    }
    rows: list[dict[str, Any]] = []
    missing_windows = 0
    for part in frame.partition_by("security_id", maintain_order=True):
        prices = part["price"].to_numpy()
        events = part["event_time"].to_list()
        positions = np.array([index[t] for t in events])
        known = part["known"].to_numpy()
        available = part["available_time"].to_list()
        returns = np.diff(prices) / prices[:-1]
        for i in range(20, len(prices) - protocol.horizon_sessions):
            end = i + protocol.horizon_sessions
            # Never turn a missing session into a fictitious one-session return.
            if (
                positions[end] - positions[i - 20] != 20 + protocol.horizon_sessions
                or not known[i - 20 : end + 1].all()
            ):
                missing_windows += 1
                continue
            start_date, end_date = events[i].date().isoformat(), events[end].date().isoformat()
            phase = None
            for name, lo, hi in (
                ("train", protocol.train_start, protocol.train_end),
                ("validation", protocol.validation_start, protocol.validation_end),
                ("test", protocol.test_start, protocol.test_end),
            ):
                if lo <= start_date <= end_date <= hi and available[end].date().isoformat() <= hi:
                    phase = name
                    break
            if phase is None:
                continue
            if phase != "test":
                boundary = boundaries[phase]
                if positions[end] + protocol.embargo_sessions >= boundary:
                    continue
            h = protocol.horizon_sessions
            rows.append(
                {
                    "security_id": part["security_id"][0],
                    "event_time": events[i],
                    "label_end": events[end],
                    "phase": phase,
                    "target": float(prices[end] / prices[i] - 1),
                    "return_1": float(returns[i - 1]),
                    "return_5": float(prices[i] / prices[i - 5] - 1),
                    "return_20": float(prices[i] / prices[i - 20] - 1),
                    "volatility_20": float(np.std(returns[i - 20 : i], ddof=1)),
                    "rolling_mean_20": float(np.mean(returns[i - 20 : i]) * h),
                }
            )
    if not rows:
        raise ValueError("no eligible samples for the declared splits")
    samples = pl.DataFrame(rows).sort(["event_time", "security_id"])
    numeric = ["target", *FEATURES, "rolling_mean_20"]
    if not np.isfinite(samples.select(numeric).to_numpy()).all():
        raise ValueError("non-finite derived samples")
    counts = {}
    for phase in ("train", "validation", "test"):
        part = samples.filter(pl.col("phase") == phase)
        counts[phase] = {"rows": part.height, "dates": part["event_time"].n_unique()}
        if counts[phase]["dates"] < protocol.min_score_dates:
            raise ValueError(f"insufficient {phase} dates")
    if counts["train"]["rows"] < protocol.min_train_rows:
        raise ValueError("insufficient training rows")
    return samples, {
        "split_counts": counts,
        "excluded_windows": missing_windows,
        "late_rows": frame.filter(~pl.col("known")).height,
        "n_names": frame["security_id"].n_unique(),
        "rows": frame.height,
        "source_labels": sorted(frame["source"].cast(pl.String).unique().to_list()),
        "first_session": times[0].isoformat(),
        "last_session": times[-1].isoformat(),
    }


def _code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _runtime() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__, "polars": pl.__version__}


def prepare_benchmark(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    raw = json.loads(protocol_path.read_text())
    raw["dataset_path"] = str((protocol_path.parent / raw["dataset_path"]).resolve())
    protocol = BenchmarkProtocol(**raw)
    protocol.validate()
    frame = _load_bars(protocol)
    _, audit = _samples(frame, protocol)
    manifest = _seal(
        {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "protocol": asdict(protocol),
            "code_sha256": _code_sha(),
            "audit": audit,
            "runtime": _runtime(),
            "research_only": True,
            "live_pnl_claim": False,
            "holdout_status": "previously_inspected"
            if protocol.holdout_previously_inspected
            else "uninspected_by_declaration",
            "limitations": [
                "Source, usage rights and historical availability are user declarations, not independently verified.",
                "File hashes detect changes; they do not prove a holdout was never inspected.",
                "Close-to-close forecast scores do not establish executable net returns.",
            ]
            + (["Input universe has survivorship bias."] if protocol.survivorship_bias else [])
            + (
                ["Historical availability timestamps were reconstructed."]
                if protocol.availability_basis == "reconstructed"
                else []
            ),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    return manifest


def _scores(
    samples: pl.DataFrame, train: pl.DataFrame, protocol: BenchmarkProtocol
) -> dict[str, Any]:
    x_train, y_train = train.select(FEATURES).to_numpy(), train["target"].to_numpy()
    x = samples.select(FEATURES).to_numpy()
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z_train, z = (x_train - mean) / scale, (x - mean) / scale
    intercept = float(y_train.mean())
    beta = np.linalg.solve(
        z_train.T @ z_train + protocol.ridge_alpha * np.eye(len(FEATURES)),
        z_train.T @ (y_train - intercept),
    )
    predictions = {
        "zero": np.zeros(len(x)),
        "historical_mean": np.full(len(x), intercept),
        "rolling_mean_20": samples["rolling_mean_20"].to_numpy(),
        "ridge": intercept + z @ beta,
    }
    target = samples["target"].to_numpy()
    result = {}
    for name in MODELS:
        pred = predictions[name]
        if not np.isfinite(pred).all():
            raise ValueError(f"non-finite predictions: {name}")
        scored = (
            samples.select("event_time")
            .with_columns(
                pl.Series("squared_error", (target - pred) ** 2),
                pl.Series("absolute_error", np.abs(target - pred)),
            )
            .group_by("event_time")
            .agg(pl.col("squared_error", "absolute_error").mean())
        )
        result[name] = {
            "date_equal_weight_mse": scored["squared_error"].mean(),
            "date_equal_weight_mae": scored["absolute_error"].mean(),
            "n_rows": samples.height,
            "n_dates": scored.height,
        }
    return result


def score_benchmark(run_dir: Path, phase: str) -> dict[str, Any]:
    if phase not in {"validation", "test"}:
        raise ValueError("phase must be validation or test")
    manifest = _read_receipt(run_dir / "manifest.json")
    if manifest["code_sha256"] != _code_sha():
        raise ValueError("benchmark code changed; prepare a new protocol")
    if manifest["runtime"] != _runtime():
        raise ValueError("benchmark runtime changed; reproduce with the recorded versions")
    protocol = BenchmarkProtocol(**manifest["protocol"])
    protocol.validate()
    if phase == "test":
        validation = _read_receipt(run_dir / "validation.json")
        if (
            validation.get("phase") != "validation"
            or validation.get("manifest_sha256") != manifest["receipt_sha256"]
        ):
            raise ValueError("validation receipt does not match this benchmark")
    destination = run_dir / f"{phase}.json"
    if destination.exists():
        raise FileExistsError("phase already scored; preserve the original result")
    samples, audit = _samples(_load_bars(protocol), protocol)
    if audit != manifest["audit"]:
        raise ValueError("dataset audit changed")
    report = _seal(
        {
            "manifest_sha256": manifest["receipt_sha256"],
            "phase": phase,
            "created_at": datetime.now(UTC).isoformat(),
            "scores": _scores(
                samples.filter(pl.col("phase") == phase),
                samples.filter(pl.col("phase") == "train"),
                protocol,
            ),
            "research_only": True,
            "live_pnl_claim": False,
            "promote": False,
            "claim": "fixed_split_forecast_diagnostic",
            "holdout_status": manifest["holdout_status"],
            "limitations": manifest["limitations"],
        }
    )
    with destination.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="audit data and freeze protocol without publishing scores"
    )
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="score validation first, then the declared test")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--phase", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    try:
        result = (
            prepare_benchmark(args.protocol, args.output)
            if args.command == "prepare"
            else score_benchmark(args.run, args.phase)
        )
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
