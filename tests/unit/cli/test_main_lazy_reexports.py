"""``quant_fund.cli.main`` lazy re-export helpers + observation-hook opt-in.

Each wrapper must stay a cheap passthrough: importing the CLI cannot eagerly
load config, logging backends, or the doctor pipeline.
"""

from __future__ import annotations

import sys
import types

import pytest

from quant_fund.cli import main

pytestmark = pytest.mark.synthetic


def test_load_config_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    import quant_fund.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: sentinel)
    assert main.load_config("anything") is sentinel


def test_dump_resolved_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.config as config_mod

    monkeypatch.setattr(config_mod, "dump_resolved", lambda *a, **k: ("dumped", a, k))
    assert main.dump_resolved(1, key=2) == ("dumped", (1,), {"key": 2})


def test_configure_logging_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.utils.logging as logging_mod

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(logging_mod, "configure_logging", lambda *a, **k: calls.append((a, k)))
    main.configure_logging(level="INFO")
    assert calls == [((), {"level": "INFO"})]


def test_get_logger_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.utils.logging as logging_mod

    monkeypatch.setattr(logging_mod, "get_logger", lambda name: f"logger:{name}")
    assert main.get_logger("lane") == "logger:lane"


def test_run_doctor_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_fund.pipeline.doctor  # noqa: F401

    doctor_mod = sys.modules["quant_fund.pipeline.doctor"]
    monkeypatch.setattr(doctor_mod, "doctor", lambda *a, **k: {"ok": True})
    assert main.run_doctor() == {"ok": True}


def test_observation_hooks_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
    stub = types.ModuleType("quant_fund.observe.install")
    stub.install_passive_hooks = lambda: pytest.fail("hooks must not install")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "quant_fund.observe.install", stub)
    main._maybe_install_observation_hooks()


@pytest.mark.parametrize("flag", ["0", "no", "off", "bogus"])
def test_observation_hooks_reject_non_flags(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", flag)
    stub = types.ModuleType("quant_fund.observe.install")
    stub.install_passive_hooks = lambda: pytest.fail("hooks must not install")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "quant_fund.observe.install", stub)
    main._maybe_install_observation_hooks()


@pytest.mark.parametrize("flag", ["1", "true", "YES", "on"])
def test_observation_hooks_install_on_opt_in(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", flag)
    calls: list[str] = []
    stub = types.ModuleType("quant_fund.observe.install")
    stub.install_passive_hooks = lambda: calls.append("installed")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "quant_fund.observe.install", stub)
    main._maybe_install_observation_hooks()
    assert calls == ["installed"]
