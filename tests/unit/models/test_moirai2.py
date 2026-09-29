"""moirai2 head (P2.1): fail-closed dep gate + causal-window contract.

Every published ``uni2ts`` release pins ``scipy>=1.11.3,<1.12.dev0`` and
``numpy~=1.26.0`` against this repo's ``scipy>=1.14`` / ``numpy>=2.0``, so
the adapter is exercised here against an injected stub uni2ts module — the
availability gate, causality, panel, monotonicity, and honesty contracts
are fully testable offline.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from quant_fund.models.moirai2 import _DEP_ERROR, Moirai2Distribution, availability


class _StubForecast:
    """Sample-path forecast: draws spread around the window's own mean."""

    def __init__(self, window: np.ndarray) -> None:
        center = float(window.mean())
        self.samples = center + np.linspace(-0.1, 0.1, 21)


class _StubPredictor:
    """Deterministic stand-in for the uni2ts lightning predictor."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def predict(self, dataset: Any) -> Any:
        window = np.asarray(dataset[0]["target"], dtype=np.float64)
        self.calls.append(window.copy())
        yield _StubForecast(window)


class _StubMoiraiModule:
    @classmethod
    def from_pretrained(cls, weights_id: str) -> _StubMoiraiModule:
        return cls()


class _StubMoiraiForecast:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def create_predictor(self, batch_size: int = 32) -> _StubPredictor:
        return _StubPredictor()


def _install_stub(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    pkg = types.ModuleType("uni2ts")
    sub = types.ModuleType("uni2ts.model")
    mod = types.ModuleType("uni2ts.model.moirai")
    mod.MoiraiForecast = _StubMoiraiForecast  # type: ignore[attr-defined]
    mod.MoiraiModule = _StubMoiraiModule  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uni2ts", pkg)
    monkeypatch.setitem(sys.modules, "uni2ts.model", sub)
    monkeypatch.setitem(sys.modules, "uni2ts.model.moirai", mod)
    return mod


def _series(n: int = 320, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=n).cumsum() * 0.01


def test_fit_fails_closed_when_dep_absent() -> None:
    head = Moirai2Distribution()
    with pytest.raises(RuntimeError, match="not installed"):
        head.fit(np.zeros(4), _series())


def test_dep_error_names_the_conflict() -> None:
    assert "uni2ts" in _DEP_ERROR
    assert "scipy" in _DEP_ERROR


def test_availability_gate() -> None:
    assert availability() is False


def test_availability_gate_opens_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch)
    assert availability() is True


def test_unfitted_predict_fails_closed() -> None:
    head = Moirai2Distribution()
    with pytest.raises(RuntimeError, match="not been fitted"):
        head.predict(np.zeros((3, 2)))


def test_stubbed_predict_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch)
    head = Moirai2Distribution(lookback=32).fit(np.zeros(4), _series())
    q = head.predict(np.zeros((5, 3)))
    assert q.shape == (5, 3)
    assert np.all(np.diff(q, axis=1) >= 0.0)  # monotone quantile rows


def test_history_windows_are_causal(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch)
    lookback = 32
    head = Moirai2Distribution(lookback=lookback).fit(np.zeros(4), _series())
    hist = _series(120, seed=11)
    out = head.predict_from_history(hist)
    assert out.shape == (hist.size - lookback + 1, 3)
    # Each row comes from its own trailing window only: perturbing a late
    # window's observations cannot change an early row.
    perturbed = hist.copy()
    perturbed[-5:] += 10.0
    out2 = head.predict_from_history(perturbed)
    np.testing.assert_allclose(out[:-5], out2[:-5], atol=1e-9)


def test_output_respects_fitted_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch)
    head = Moirai2Distribution(lookback=32).fit(np.zeros(4), _series())
    meta = head.metadata()
    assert meta.extra["framework"] == "uni2ts"
    assert meta.extra["weights_id"] == "Salesforce/moirai-2.0-R-small"
    assert meta.extra["pretrained"] is True
    assert meta.extra["warmup"] == head.lookback
    assert meta.extra["standardize"] is True
    q = head.predict(np.zeros((2, 1)))
    # Predictions are de-standardized back to the series' own scale.
    assert np.abs(q).max() < 1.0


def test_grid_violations_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _install_stub(monkeypatch)

    class _EmptySamples(_StubForecast):
        def __init__(self, window: np.ndarray) -> None:
            self.samples = np.array([])

    class _BadForecast(_StubMoiraiForecast):
        def create_predictor(self, batch_size: int = 32) -> Any:
            class _P:
                def predict(self, dataset: Any) -> Any:
                    yield _EmptySamples(np.asarray(dataset[0]["target"]))

            return _P()

    mod.MoiraiForecast = _BadForecast  # type: ignore[attr-defined]
    head = Moirai2Distribution(lookback=32).fit(np.zeros(4), _series())
    with pytest.raises(ValueError, match="samples"):
        head.predict(np.zeros((2, 1)))


def test_registry_wires_head() -> None:
    from quant_fund.research.fleet_eval import FLEET_HEAD_REGISTRY

    head = FLEET_HEAD_REGISTRY["moirai2"]([0.1, 0.5, 0.9], 0)
    assert isinstance(head, Moirai2Distribution)
