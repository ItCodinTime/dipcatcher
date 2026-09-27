"""Experiment tracking for fx-1 (MLflow, matching the harness's dependency).

Thin wrapper: logs params, metrics, and artifacts if mlflow is importable;
otherwise degrades to a JSON-lines local log so training never crashes on a
missing tracking server. The training *receipt* remains the source of truth —
tracking is observability, not evidence.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


class Tracker:
    """MLflow-if-available, JSONL-always experiment tracker."""

    def __init__(self, experiment: str = "fx-1", fallback_log: str | Path | None = None):
        self._mlflow = None
        self._run = None
        self._fallback = Path(fallback_log) if fallback_log else None
        try:
            import mlflow  # type: ignore

            mlflow.set_experiment(experiment)
            self._run = mlflow.start_run(run_name=f"{experiment}-run")
            self._mlflow = mlflow
        except Exception:  # noqa: BLE001 - tracking must never break training
            self._mlflow = None
        if self._fallback:
            self._fallback.parent.mkdir(parents=True, exist_ok=True)

    def _record(self, kind: str, payload: dict) -> None:
        if self._fallback:
            line = {
                "utc": datetime.now(UTC).isoformat(),
                "kind": kind,
                **payload,
            }
            with self._fallback.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")

    def log_params(self, params: dict) -> None:
        if self._mlflow:
            self._mlflow.log_params({k: str(v) for k, v in params.items()})
        self._record("params", {"params": {k: str(v) for k, v in params.items()}})

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        if self._mlflow:
            self._mlflow.log_metric(key, value, step=step)
        self._record("metric", {"key": key, "value": value, "step": step})

    def log_artifact(self, path: str | Path) -> None:
        if self._mlflow:
            self._mlflow.log_artifact(str(path))
        self._record("artifact", {"path": str(path)})

    def close(self) -> None:
        if self._mlflow and self._run:
            self._mlflow.end_run()
