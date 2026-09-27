"""Causal, common-row ranker calibration experiment checks (SYNTHETIC)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.research.ranker_probability import (
    ExperimentSpec,
    _digest,
    _paired_interval,
    evaluate,
    main,
)


def _fixture(n_dates: int = 240, n_names: int = 8) -> pl.DataFrame:
    rng = np.random.default_rng(29)
    rows = []
    for day in range(n_dates):
        event = datetime(2020, 1, 1, 20, tzinfo=UTC) + timedelta(days=day)
        for name in range(n_names):
            useful = float(rng.uniform(0.0, 1.0))
            momentum = float(rng.uniform(0.0, 1.0))
            residual = 0.05 * (useful - 0.5) + float(rng.normal(0.0, 0.004))
            rows.append(
                {
                    "event_time": event,
                    "decision_time": event,
                    "max_source_available_time": event,
                    "label_end_time_1": event + timedelta(days=1),
                    "security_id": f"N{name}",
                    "feature_set_version": "fixture.public.v1",
                    "cs_pct_mom_20": momentum,
                    "cs_pct_reversal_1": useful,
                    "future_idio_return_1": residual,
                }
            )
    return pl.DataFrame(rows)


def _spec() -> ExperimentSpec:
    return ExperimentSpec(
        features=("cs_pct_mom_20", "cs_pct_reversal_1"),
        train_dates=80,
        cal_dates=25,
        test_dates=25,
        n_boot=200,
        min_test_dates=50,
        mean_block=5,
    )


def test_supervised_platt_scores_common_future_rows_without_promotion() -> None:
    report = evaluate(_fixture(), _spec())
    assert report["status"] == "measured"
    assert report["n_scored_dates"] == 125
    assert report["n_untrainable_folds"] == 0
    assert report["losses"]["ranker_platt"]["brier"] < report["losses"]["momentum_platt"]["brier"]
    assert report["losses"]["ranker_platt"]["brier"] < report["losses"]["train_base_rate"]["brier"]
    assert report["production_promotion"] is False
    assert report["forward_evidence_accepted"] is False
    for fold in report["folds"]:
        assert fold["status"] == "scored"
        assert fold["latest_train_label_end"] < fold["first_cal"]
        assert fold["latest_cal_label_end"] < fold["first_test"]
        assert fold["n_test_rows"] == 25 * 8


def test_late_calibration_labels_are_purged_before_test() -> None:
    frame = _fixture()
    # Delaying one name on a date purges the entire cross-sectional date.
    late = frame.with_columns(
        pl.when(
            (pl.col("event_time") == datetime(2020, 4, 13, 20, tzinfo=UTC))
            & (pl.col("security_id") == "N0")
        )
        .then(pl.col("event_time") + pl.duration(days=30))
        .otherwise(pl.col("label_end_time_1"))
        .alias("label_end_time_1")
    )
    normal = evaluate(frame, _spec())
    delayed = evaluate(late, _spec())
    assert delayed["folds"][0]["n_cal_dates"] < normal["folds"][0]["n_cal_dates"]
    assert delayed["folds"][0]["latest_cal_label_end"] < delayed["folds"][0]["first_test"]


def test_future_test_label_cannot_change_first_fold_probabilities() -> None:
    frame = _fixture()
    altered = frame.with_columns(
        pl.when(pl.col("event_time") == datetime(2020, 4, 15, 20, tzinfo=UTC))
        .then(-pl.col("future_idio_return_1"))
        .otherwise(pl.col("future_idio_return_1"))
        .alias("future_idio_return_1")
    )
    before = evaluate(frame, _spec())
    after = evaluate(altered, _spec())
    assert before["folds"][0]["prediction_sha256"] == after["folds"][0]["prediction_sha256"]


def test_pit_or_label_endpoint_violation_fails_closed() -> None:
    frame = _fixture()
    leaked = frame.with_columns(
        pl.when(pl.col("security_id") == "N0")
        .then(pl.col("decision_time") + pl.duration(seconds=1))
        .otherwise(pl.col("max_source_available_time"))
        .alias("max_source_available_time")
    )
    with pytest.raises(ValueError, match="unavailable"):
        evaluate(leaked, _spec())
    invalid_end = frame.with_columns(pl.col("event_time").alias("label_end_time_1"))
    with pytest.raises(ValueError, match="strictly later"):
        evaluate(invalid_end, _spec())


def test_session_block_interval_is_seeded_and_paired() -> None:
    a = np.full(30, 0.1)
    b = np.full(30, 0.2)
    first = _paired_interval(a, b, n_boot=200, mean_block=5, seed=17)
    assert first == _paired_interval(a, b, n_boot=200, mean_block=5, seed=17)
    assert first["ci_high"] == pytest.approx(-0.1)


def test_standalone_command_seals_a_complete_nonpromoting_receipt(tmp_path: Path) -> None:
    frame = _fixture()
    features = tmp_path / "features.parquet"
    labels = tmp_path / "labels.parquet"
    output = tmp_path / "receipt.json"
    frame.drop("future_idio_return_1", "label_end_time_1").write_parquet(features)
    frame.select(
        "event_time", "security_id", "future_idio_return_1", "label_end_time_1"
    ).write_parquet(labels)
    args = [
        "--features",
        str(features),
        "--labels",
        str(labels),
        "--output",
        str(output),
        "--train-dates",
        "80",
        "--cal-dates",
        "25",
        "--test-dates",
        "25",
        "--n-boot",
        "100",
        "--feature-columns",
        "cs_pct_mom_20",
        "cs_pct_reversal_1",
    ]
    assert main(args) == 0
    receipt = json.loads(output.read_text())
    seal = receipt.pop("receipt_sha256")
    assert seal == _digest(receipt)
    assert receipt["n_scored_dates"] == 125
    assert receipt["holdout_previously_inspected_or_unverified"] is True
    assert receipt["production_promotion"] is False
    with pytest.raises(FileExistsError):
        main(args)
