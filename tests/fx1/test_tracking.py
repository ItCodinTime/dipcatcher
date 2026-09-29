"""Tracker degrade contract: mlflow-if-available, JSONL always, never crash."""

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fx1.train.tracking import Tracker


class _FakeMlflow(ModuleType):
    """Records log calls; no server, no mlruns/ directory."""

    def __init__(self) -> None:
        super().__init__("mlflow")
        self.experiment: str | None = None
        self.run_started = False
        self.run_ended = False
        self.calls: list[tuple[str, Any]] = []

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self, run_name: str | None = None) -> object:
        self.run_started = True
        return object()

    def end_run(self) -> None:
        self.run_ended = True

    def log_params(self, params: dict) -> None:
        self.calls.append(("params", params))

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        self.calls.append(("metric", (key, value, step)))

    def log_artifact(self, path: str) -> None:
        self.calls.append(("artifact", path))


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_fallback_records_every_kind(tmp_path: Path, monkeypatch) -> None:
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    log = tmp_path / "runs" / "track.jsonl"
    tracker = Tracker(experiment="fx-1-test", fallback_log=log)
    tracker.log_params({"lr": 1e-4, "epochs": 2})
    tracker.log_metric("loss", 0.5, step=3)
    tracker.log_artifact("receipts/x.json")
    tracker.close()

    assert fake.experiment == "fx-1-test"
    assert fake.run_started and fake.run_ended
    assert ("metric", ("loss", 0.5, 3)) in fake.calls
    assert ("params", {"lr": "0.0001", "epochs": "2"}) in fake.calls

    lines = _lines(log)
    assert [line["kind"] for line in lines] == ["params", "metric", "artifact"]
    assert lines[0]["params"] == {"lr": "0.0001", "epochs": "2"}
    assert lines[1]["key"] == "loss" and lines[1]["step"] == 3
    assert all("utc" in line for line in lines)


def test_degrades_when_mlflow_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken/missing mlflow must never break training — JSONL still writes."""
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)

    import builtins

    real_import = builtins.__import__

    def _no_mlflow(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mlflow":
            raise ImportError("no mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_mlflow)
    log = tmp_path / "track.jsonl"
    tracker = Tracker(fallback_log=log)
    tracker.log_metric("pinball", 0.012, step=0)
    tracker.close()  # no run to end — must not raise

    lines = _lines(log)
    assert len(lines) == 1 and lines[0]["kind"] == "metric"
    assert lines[0]["value"] == pytest.approx(0.012)


def test_no_fallback_no_file(tmp_path: Path, monkeypatch) -> None:
    """Without a fallback path nothing is written — observability is optional."""
    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    import builtins

    real_import = builtins.__import__

    def _no_mlflow(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mlflow":
            raise ImportError("no mlflow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_mlflow)
    tracker = Tracker()
    tracker.log_params({"a": 1})  # silently no-ops — receipts are the truth
    tracker.close()
