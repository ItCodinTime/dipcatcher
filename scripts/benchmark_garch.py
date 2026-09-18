#!/usr/bin/env python3
"""Frozen synthetic smoke benchmark; never certifies real-market SOTA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from quant_fund.research.garch_benchmark import (
    BASELINES,
    CANDIDATES,
    evaluate_models,
    select_on_validation,
    summarize_losses,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path, output = args.input.resolve(), args.output.resolve()
    if output.exists() or output.with_suffix(".protocol.json").exists():
        raise ValueError("refusing to overwrite an experiment or reuse its report path")
    frame = pl.read_parquet(source_path)
    required = {"security_id", "event_time", "available_time", "source", "close_total_return"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {required - set(frame.columns)}")
    if frame["source"].null_count() or set(frame["source"].unique()) != {"synthetic"}:
        raise ValueError("this smoke-run protocol is restricted to synthetic data")
    if frame.select(pl.struct("security_id", "event_time").is_duplicated().any()).item():
        raise ValueError("duplicate security/date keys")
    if frame["event_time"].null_count() or frame["available_time"].null_count():
        raise ValueError("missing event/availability timestamps")
    if frame.filter(pl.col("available_time") > pl.col("event_time")).height:
        raise ValueError("late observations require a separate PIT evaluation")
    dates = frame["event_time"].unique().sort().to_list()
    series = {}
    for key, sub in frame.sort(["security_id", "event_time"]).group_by(
        "security_id", maintain_order=True
    ):
        if sub["event_time"].to_list() == dates:
            series[str(key[0])] = sub["close_total_return"].to_numpy()
        if len(series) == 5:
            break
    if len(series) < 5 or len(dates) < 451:
        raise ValueError("need five complete series and at least 451 dates")
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        "source": "synthetic",
        "input": str(source_path),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "securities": list(series),
        "horizon": 5,
        "stride": 5,
        "seed": 42,
        "initial_history": 200,
        "validation_origins": 20,
        "test_origins": 30,
        "candidates": list(CANDIDATES),
        "baselines": list(BASELINES),
        "mean": "Zero",
        "target": "sum of next five squared log returns",
        "aggregation": "equal-weight security QLIKE losses per common date",
        "qlike": "log(forecast) + target / forecast (target-only constants omitted)",
        "validation_start": str(dates[200]),
        "test_start": str(dates[300]),
        "test_last_origin": str(dates[445]),
        "test_last_target": str(dates[450]),
        "selection": "minimum validation mean loss, fixed before test",
        "inference": f"two-sided HAC DM, Bonferroni over {len(BASELINES)} baselines, alpha .05",
        "limitations": [
            "synthetic only",
            "HAR proxy added after inspecting v1; subsequent runs are exploratory",
            "complete-history universe selection",
            "30 test dates",
            "daily-squared-return HAR proxy, not intraday HAR-RV; no realized-GARCH or hybrids",
            "benchmark estimators are not the production GARCHVol pipeline",
        ],
    }
    with output.with_suffix(".protocol.json").open("x") as stream:
        json.dump(protocol, stream, indent=2)

    def phase(start: int, count: int, names: tuple[str, ...]):
        losses: dict[str, list[np.ndarray]] = {name: [] for name in names}
        failures = dict.fromkeys(names, 0)
        for security, prices in series.items():
            result = evaluate_models(
                prices=prices, min_history=start, n_origins=count, candidates=names
            )
            target = result["target"].to_numpy()
            for name in names:
                forecast = result[name].to_numpy()
                loss = np.log(forecast) + target / forecast
                if not np.isfinite(loss).all():
                    raise ValueError("nonfinite QLIKE; experiment invalid")
                losses[name].append(loss)
                failures[name] += sum(
                    s not in {"fitted", "baseline"} for s in result[f"{name}_status"].to_list()
                )
            print(f"phase={start} security={security} complete", flush=True)
        return {name: np.mean(values, axis=0) for name, values in losses.items()}, failures

    validation, val_failures = phase(200, 20, CANDIDATES)
    selected = select_on_validation(validation)
    with output.with_suffix(".selection.json").open("x") as stream:
        json.dump(
            {
                "selected": selected,
                "validation_qlike": {k: float(v.mean()) for k, v in validation.items()},
            },
            stream,
            indent=2,
        )
    test, test_failures = phase(300, 30, (*BASELINES, selected))
    report = summarize_losses(validation, test, source="synthetic")
    report.update(
        protocol=protocol,
        validation_fallback_counts=val_failures,
        test_fallback_counts=test_failures,
        date_losses={k: v.tolist() for k, v in test.items()},
    )
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"protocol", "date_losses"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
