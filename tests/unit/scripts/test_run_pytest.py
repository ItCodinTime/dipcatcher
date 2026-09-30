"""Tests for the portable pytest gate runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _runner_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "run_pytest.py"
    spec = importlib.util.spec_from_file_location("run_pytest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pytest_environment_is_a_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner_module()
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setenv("DIPCATCHER_RUNNER_TEST", "present")

    result = runner.pytest_environment()

    assert result["DIPCATCHER_RUNNER_TEST"] == "present"
    assert result is not runner.os.environ


def test_pytest_environment_exposes_torch_openmp_on_macos(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _runner_module()
    package = tmp_path / "torch"
    library = package / "lib"
    library.mkdir(parents=True)
    (library / "libomp.dylib").touch()
    fake_spec = type("Spec", (), {"submodule_search_locations": [str(package)]})()
    monkeypatch.setattr(runner.sys, "platform", "darwin")
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _name: fake_spec)
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)

    result = runner.pytest_environment()

    assert result["DYLD_LIBRARY_PATH"] == str(library)
