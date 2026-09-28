"""tabpfn_ts head (P2.5): fail-closed dep gate + causal-window contract.

The real ``tabpfn-time-series`` package cannot resolve in uv.lock (gluonts
``toolz<1`` vs pinned ``exchange-calendars`` ``toolz>=1``), so the adapter is
exercised here against an injected stub predictor module — the causality,
panel, monotonicity, and honesty contracts are fully testable offline.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from quant_fund.models.tabpfn_ts import _DEP_ERROR, TabpfnTsDistribution


class _StubPredictor:
    """Deterministic stand-in for ``TabPFNTimeSeriesPredictor``.

    Emits the requested-order quantiles derived from the window's own mean —
    enough to prove causality (perturbing the window changes the row) and the
    panel contract, without touching weights or network.
    """

    def __init__(self, tabpfn_config: Any = None) -> None:
        self.calls: list[np.ndarray] = []

    def predict(self, window: np.ndarray) -> np.ndarray:
        w = np.asarray(window, dtype=np.float64)
        self.calls.append(w.copy())
        center = float(w.mean())
        # Three taus in the default tests: deterministic fan around the mean.
        return np.array([center - 0.1, center, center + 0.1])


def _install_stub(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    mod = types.ModuleType("tabpfn_time_series")
    mod.TabPFNTimeSeriesPredictor = _StubPredictor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tabpfn_time_series", mod)
    return mod


def _series(n: int = 320, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=n).cumsum() * 0.01


def test_fit_fails_closed_when_dep_absent() -> None:
    head = TabpfnTsDistribution()
    with pytest.raises(RuntimeError, match="not installed"):
        head.fit(np.zeros(4), _series())


def test_dep_error_names_the_conflict() -> None:
    assert "exchange-calendars" in _DEP_ERROR
    assert "toolz" in _DEP_ERROR


def test_unfitted_predict_fails_closed() -> None:
    head = TabpfnTsDistribution()
    with pytest.raises(RuntimeError, match="not been fitted"):
        head.predict(np.zeros((3, 2)))


def test_stubbed_predict_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch)
    head = TabpfnTsDistribution(lookback=32).fit(np.zeros(4), _series())
    q = head.predict(np.zeros((5, 3)))
    assert q.shape == (5, 3)
    assert np.all(np.diff(q, axis=1) >= 0.0)  # monotone quantile rows


def test_history_windows_are_causal(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch)
    lookback = 32
    head = TabpfnTsDistribution(lookback=lookback).fit(np.zeros(4), _series())
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
    head = TabpfnTsDistribution(lookback=32).fit(np.zeros(4), _series())
    meta = head.metadata()
    assert meta.extra["framework"] == "tabpfn-time-series"
    assert meta.extra["pretrained"] is True
    assert meta.extra["warmup"] == head.lookback
    assert meta.extra["standardize"] is True
    q = head.predict(np.zeros((2, 1)))
    # Predictions are de-standardized back to the series' own scale.
    assert np.abs(q).max() < 1.0


def test_grid_violations_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _install_stub(monkeypatch)

    class _BadGrid(_StubPredictor):
        def predict(self, window: np.ndarray) -> np.ndarray:
            return np.array([0.0, 0.0])  # wrong width for a 3-tau grid

    mod.TabPFNTimeSeriesPredictor = _BadGrid  # type: ignore[attr-defined]
    head = TabpfnTsDistribution(lookback=32).fit(np.zeros(4), _series())
    with pytest.raises(ValueError, match="quantiles"):
        head.predict(np.zeros((2, 1)))


def test_registry_wires_head() -> None:
    from quant_fund.research.fleet_eval import FLEET_HEAD_REGISTRY

    head = FLEET_HEAD_REGISTRY["tabpfn_ts"]([0.1, 0.5, 0.9], 0)
    assert isinstance(head, TabpfnTsDistribution)
