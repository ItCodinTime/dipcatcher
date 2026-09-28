"""The bundled smoke run is a deterministic correctness check."""

from __future__ import annotations

import json

import pytest

from quant_fund.robustness import smoke


def test_run_smoke_returns_proven_linear_report() -> None:
    report = smoke.run_smoke()
    assert report["evidence_class"] == "SYNTHETIC"
    assert report["research_only"] is True
    assert report["live_trading_claim"] is False
    linear = report["linear"]
    assert linear["certified_status"] == "proven"
    assert linear["empirical_status"] == "empirical"
    assert linear["gradient_status"] == "empirical"
    assert linear["radii_comparable"] is True
    assert linear["worst_case_mean"] < 0.4
    assert linear["worst_case_ratio_status"] in {
        "proven_tight",
        "proven_unbounded",
    }
    assert report["schedule_jitter_bars"] == 1.0
    assert report["extension_schema_version"] == 1
    assert report["extension_errors"] == []


def test_main_prints_json_and_exits_zero(capsys) -> None:
    assert smoke.main() == 0
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["linear"]["certified_status"] == "proven"


class _Fail:
    """Monkeypatch helper: every checked card field fails its guard."""

    def __init__(self, **overrides) -> None:
        self._card = {
            "analytic_radius": {"value": 1.0},
            "certified_radius": {"value": 1.0, "status": "proven"},
            "empirical_attack_radius": {"value": 2.0, "status": "empirical"},
            "gradient_attack": {"value": 1.0, "status": "empirical"},
            "distributional_robustness": {
                "worst_case_mean": {"value": 0.3, "status": "proven"},
                "worst_case_ratio": {"value": 0.5, "status": "proven_tight"},
            },
            "sensitivity": {"timing_jitter": {"value": 1.0, "status": "exact_on_path"}},
            "radii_comparable": True,
        }
        self._card.update(overrides)

    def __call__(self, *args, **kwargs) -> dict:
        return self._card


def test_smoke_raises_when_radius_missing(monkeypatch) -> None:
    monkeypatch.setattr(smoke, "certify", _Fail(analytic_radius={"value": None}))
    with pytest.raises(RuntimeError, match="finite radii"):
        smoke.run_smoke()


def test_smoke_raises_when_certificate_not_proven(monkeypatch) -> None:
    monkeypatch.setattr(
        smoke,
        "certify",
        _Fail(certified_radius={"value": 1.0, "status": "high_probability"}),
    )
    with pytest.raises(RuntimeError, match="population certificate"):
        smoke.run_smoke()


def test_smoke_raises_when_certificate_disagrees(monkeypatch) -> None:
    monkeypatch.setattr(
        smoke, "certify", _Fail(certified_radius={"value": 2.0, "status": "proven"})
    )
    with pytest.raises(RuntimeError, match="disagreed with the linear radius"):
        smoke.run_smoke()


def test_smoke_raises_on_unproven_mean_bound(monkeypatch) -> None:
    monkeypatch.setattr(
        smoke,
        "certify",
        _Fail(
            distributional_robustness={
                "worst_case_mean": {"value": 0.3, "status": "empirical"},
                "worst_case_ratio": {"value": 0.5, "status": "proven_tight"},
            }
        ),
    )
    with pytest.raises(RuntimeError, match="worst-case mean"):
        smoke.run_smoke()


def test_smoke_raises_when_mean_bound_wrong(monkeypatch) -> None:
    monkeypatch.setattr(
        smoke,
        "certify",
        _Fail(
            distributional_robustness={
                "worst_case_mean": {"value": 0.9, "status": "proven"},
                "worst_case_ratio": {"value": 0.5, "status": "proven_tight"},
            }
        ),
    )
    with pytest.raises(RuntimeError, match="mean minus radius"):
        smoke.run_smoke()


class _Seq:
    """Return a different card per call (linear check, then schedule check)."""

    def __init__(self, second: dict) -> None:
        self._first = _Fail()._card
        self._second = second
        self._calls = 0

    def __call__(self, *args, **kwargs) -> dict:
        self._calls += 1
        return self._first if self._calls == 1 else self._second


def test_smoke_raises_when_schedule_jitter_wrong(monkeypatch) -> None:
    bad = _Fail(sensitivity={"timing_jitter": {"value": 2.0, "status": "exact_on_path"}})._card
    monkeypatch.setattr(smoke, "certify", _Seq(bad))
    with pytest.raises(RuntimeError, match="one-bar roll"):
        smoke.run_smoke()


def test_smoke_raises_when_stamp_fails_validation(monkeypatch) -> None:
    monkeypatch.setattr(smoke, "certify", _Fail())

    def bad_stamp(notebook, scorecards):
        stamped = dict(notebook)
        stamped["extensions_schema_version"] = 1  # no robustness block
        return stamped

    monkeypatch.setattr(smoke, "stamp_robustness", bad_stamp)
    with pytest.raises(RuntimeError, match="failed validation"):
        smoke.run_smoke()


def test_smoke_raises_when_stamp_mutates_input(monkeypatch) -> None:
    monkeypatch.setattr(smoke, "certify", _Fail())

    def mutating_stamp(notebook, scorecards):
        # Pollute the caller's dict but hand back a clean copy: the smoke
        # guard is supposed to notice the mutation after validation passes.
        notebook["robustness"] = {"polluted": True}
        return {"clean": True}

    monkeypatch.setattr(smoke, "stamp_robustness", mutating_stamp)
    with pytest.raises(RuntimeError, match="mutated the input"):
        smoke.run_smoke()


def test_smoke_raises_on_forbidden_headline(monkeypatch) -> None:
    # The checker is imported inside run_smoke, so patch its source module.
    monkeypatch.setattr(
        "quant_fund.leakage.patterns.find_forbidden_headline",
        lambda text: ["sharpe"],
    )
    with pytest.raises(RuntimeError, match="forbidden tokens"):
        smoke.run_smoke()
