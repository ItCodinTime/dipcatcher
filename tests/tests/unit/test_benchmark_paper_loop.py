"""Fail-closed reporting policy for the synthetic paper benchmark."""

from pathlib import Path

from scripts import benchmark_paper_loop


def test_benchmark_ledger_validation_requires_complete_artifacts(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_validate(root: Path, *, require_complete: bool = False) -> dict[str, object]:
        calls["root"] = root
        calls["require_complete"] = require_complete
        return {
            "ok": False,
            "errors": ["broker_state.json_missing"],
            "warnings": [],
        }

    monkeypatch.setattr(benchmark_paper_loop, "validate_ledger_schema", fake_validate)

    report = benchmark_paper_loop._validate_benchmark_ledger(Path("run"))

    assert calls == {"root": Path("run"), "require_complete": True}
    assert report == {
        "ok": False,
        "errors": ["broker_state.json_missing"],
        "warnings": [],
    }
