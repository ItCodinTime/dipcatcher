"""Generated data tests the benchmark contract; it is not market evidence."""

import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.research.real_benchmark import (
    BenchmarkProtocol,
    _load_bars,
    _samples,
    _scores,
    prepare_benchmark,
    score_benchmark,
)


@pytest.fixture
def inputs(tmp_path):
    dates = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(240)]
    rng = np.random.default_rng(73)
    frames = []
    for name in ("A", "B", "C"):
        frames.append(
            pl.DataFrame(
                {
                    "security_id": [name] * len(dates),
                    "event_time": dates,
                    "available_time": dates,
                    "ingested_time": dates,
                    "source": ["test_contract"] * len(dates),
                    "close": 100 * np.exp(np.cumsum(rng.normal(0.001, 0.015, len(dates)))),
                }
            )
        )
    frame = pl.concat(frames)
    path = tmp_path / "bars.parquet"
    frame.write_parquet(path)
    protocol = BenchmarkProtocol(
        dataset_path=str(path),
        dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_url="https://example.com/test-contract",
        usage_basis="generated test data",
        price_column="close",
        price_adjustment="none in test fixture",
        universe_description="three generated securities",
        survivorship_bias=True,
        availability_basis="reconstructed",
        holdout_previously_inspected=True,
        train_start="2020-01-01",
        train_end="2020-03-31",
        validation_start="2020-04-05",
        validation_end="2020-05-31",
        test_start="2020-06-05",
        test_end="2020-08-20",
        min_train_rows=100,
        min_score_dates=20,
    )
    return frame, protocol


def prepare(tmp_path, protocol, name="run"):
    spec = tmp_path / f"{name}.json"
    spec.write_text(json.dumps(asdict(protocol)))
    run = tmp_path / name
    manifest = prepare_benchmark(spec, run)
    return run, manifest


def with_frame(frame, protocol):
    frame.write_parquet(protocol.dataset_path)
    return replace(
        protocol,
        dataset_sha256=hashlib.sha256(Path(protocol.dataset_path).read_bytes()).hexdigest(),
    )


def test_complete_workflow_requires_validation_and_preserves_disclosures(tmp_path, inputs):
    _, protocol = inputs
    run, manifest = prepare(tmp_path, protocol)
    assert "scores" not in manifest
    assert manifest["holdout_status"] == "previously_inspected"
    assert len(manifest["limitations"]) == 5
    with pytest.raises(FileNotFoundError):
        score_benchmark(run, "test")
    validation = score_benchmark(run, "validation")
    test = score_benchmark(run, "test")
    assert test["promote"] is False
    assert test["live_pnl_claim"] is False
    assert set(test["scores"]) == {"zero", "historical_mean", "rolling_mean_20", "ridge"}
    assert len({x["n_rows"] for x in validation["scores"].values()}) == 1
    with pytest.raises(FileExistsError):
        score_benchmark(run, "test")


def test_frozen_benchmark_can_be_verified_after_checkout_moves(tmp_path, inputs):
    _, protocol = inputs
    run, manifest = prepare(tmp_path, protocol)
    assert not Path(manifest["protocol"]["dataset_path"]).is_absolute()
    moved = tmp_path.parent / f"{tmp_path.name}_moved"
    tmp_path.rename(moved)
    assert score_benchmark(moved / run.name, "validation")["scores"]["zero"]["n_dates"] > 0


def test_holdout_price_changes_cannot_change_validation_scores(tmp_path, inputs):
    frame, protocol = inputs
    run, _ = prepare(tmp_path, protocol)
    original = score_benchmark(run, "validation")["scores"]
    modified = frame.with_columns(
        pl.when(pl.col("event_time") >= datetime(2020, 6, 5, tzinfo=UTC))
        .then(pl.col("close") * 13)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    changed = with_frame(modified, protocol)
    other, _ = prepare(tmp_path, changed, "changed")
    changed_scores = score_benchmark(other, "validation")["scores"]
    assert changed_scores.keys() == original.keys()
    # scores are equal up to float reassociation noise (~1e-17): the contract
    # is that holdout prices cannot move validation scores, not bit-equality.
    for model, metrics in original.items():
        for key, val in metrics.items():
            assert changed_scores[model][key] == pytest.approx(val, rel=1e-12)


def test_hash_and_receipt_tampering_fail(tmp_path, inputs):
    frame, protocol = inputs
    run, _ = prepare(tmp_path, protocol)
    with_frame(frame.with_columns((pl.col("close") + 1).alias("close")), protocol)
    with pytest.raises(ValueError, match="dataset hash"):
        score_benchmark(run, "validation")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["protocol"]["test_start"] = "2020-06-20"
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="receipt hash"):
        score_benchmark(run, "validation")


@pytest.mark.parametrize("violation", ["duplicate", "synthetic", "nan", "time", "naive"])
def test_bad_data_is_rejected(tmp_path, inputs, violation):
    frame, protocol = inputs
    if violation == "duplicate":
        frame = pl.concat([frame, frame.head(1)])
    elif violation == "synthetic":
        frame = frame.with_columns(pl.lit("SYNTHETIC").alias("source"))
    elif violation == "nan":
        frame = frame.with_columns(pl.lit(float("nan")).alias("close"))
    elif violation == "time":
        frame = frame.with_columns(
            (pl.col("event_time") - pl.duration(days=1)).alias("available_time")
        )
    else:
        frame = frame.with_columns(pl.col("event_time").dt.replace_time_zone(None))
    protocol = with_frame(frame, protocol)
    with pytest.raises(ValueError):
        prepare(tmp_path, protocol)


def test_missing_and_late_sessions_do_not_bridge_gaps(tmp_path, inputs):
    frame, protocol = inputs
    gap = datetime(2020, 4, 15, tzinfo=UTC)
    late = datetime(2020, 6, 15, tzinfo=UTC)
    frame = frame.filter(~((pl.col("security_id") == "A") & (pl.col("event_time") == gap)))
    is_late = (pl.col("security_id") == "B") & (pl.col("event_time") == late)
    frame = frame.with_columns(
        pl.when(is_late)
        .then(pl.col("event_time") + pl.duration(days=1))
        .otherwise(pl.col("event_time"))
        .alias("available_time"),
        (pl.col("event_time") + pl.duration(days=2)).alias("ingested_time"),
    )
    protocol = with_frame(frame, protocol)
    samples, audit = _samples(_load_bars(protocol), protocol)
    assert audit["late_rows"] == 1
    assert audit["excluded_windows"] > 0
    a = samples.filter(
        (pl.col("security_id") == "A") & (pl.col("event_time") == gap - timedelta(days=1))
    )
    assert a.is_empty()  # a two-session return cannot be scored as one-session
    b = samples.filter((pl.col("security_id") == "B") & (pl.col("event_time") == late))
    assert b.is_empty()


def test_train_labels_and_embargo_end_before_validation(tmp_path, inputs):
    _, protocol = inputs
    protocol = replace(protocol, train_end="2020-04-04", horizon_sessions=5, embargo_sessions=5)
    samples, _ = _samples(_load_bars(protocol), protocol)
    train = samples.filter(pl.col("phase") == "train")
    assert train["label_end"].max() < datetime(2020, 3, 31, tzinfo=UTC)


def test_scores_weight_dates_equally_with_changing_cross_section(inputs):
    _, protocol = inputs
    samples, _ = _samples(_load_bars(protocol), protocol)
    train = samples.filter(pl.col("phase") == "train")
    scoring = samples.filter(pl.col("phase") == "validation")
    first = scoring["event_time"][0]
    scoring = scoring.filter((pl.col("event_time") == first) | (pl.col("security_id") == "A"))
    observed = _scores(scoring, train, protocol)["zero"]["date_equal_weight_mse"]
    by_date = {}
    for event, target in scoring.select("event_time", "target").iter_rows():
        by_date.setdefault(event, []).append(target**2)
    assert observed == pytest.approx(np.mean([np.mean(values) for values in by_date.values()]))


def test_a_new_protocol_cannot_reuse_another_validation_receipt(tmp_path, inputs):
    _, protocol = inputs
    run, _ = prepare(tmp_path, protocol)
    score_benchmark(run, "validation")
    other, _ = prepare(tmp_path, replace(protocol, ridge_alpha=2.0), "other")
    (other / "validation.json").write_bytes((run / "validation.json").read_bytes())
    with pytest.raises(ValueError, match="does not match"):
        score_benchmark(other, "test")


def test_module_cli_exposes_prepare_and_score():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "quant_fund.research.real_benchmark", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "prepare" in result.stdout
    assert "score" in result.stdout


@pytest.mark.parametrize(
    "changes",
    [
        {"holdout_previously_inspected": "false"},
        {"validation_start": "2020-03-01"},
        {"embargo_sessions": 0},
        {"ridge_alpha": float("nan")},
        {"usage_basis": ""},
    ],
)
def test_invalid_protocol_fails_before_writing(tmp_path, inputs, changes):
    _, protocol = inputs
    with pytest.raises(ValueError):
        prepare(tmp_path, replace(protocol, **changes))
    assert not (tmp_path / "run").exists()
