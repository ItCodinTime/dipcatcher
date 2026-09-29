"""tirex2 head (P2.2): fail-closed dep gate + causal-window contract.

The real ``tirex-2`` package resolves into uv.lock (unlike tabpfn-ts) but its
~380M checkpoint still needs a HuggingFace download, so the adapter is
exercised here against an injected stub ``tirex2`` module — the causality,
panel, native-grid, and honesty contracts are fully testable offline. One
``@pytest.mark.network`` smoke test exercises the real checkpoint on CPU.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from quant_fund.models.tirex2 import _DEP_ERROR, Tirex2Distribution

_NATIVE = np.linspace(0.1, 0.9, 9)


class _StubTs:
    """Stand-in for ``tirex2.TimeseriesType`` — just records the fields."""

    def __init__(self, target: Any, past_covariates: Any, future_covariates: Any) -> None:
        self.target = target
        self.past_covariates = past_covariates
        self.future_covariates = future_covariates


class _StubModel:
    """Deterministic stand-in for a loaded TiRex-2 ``ForecastModel``.

    Exposes the native 9-point grid and emits, per input window, fan
    quantiles around that window's own mean — enough to prove causality
    (perturbing a window changes only its row) and the panel contract,
    without touching weights or network. Output rows use the upstream
    ``[V_t, Q, H]`` numpy contract.
    """

    quantiles = _NATIVE.copy()

    def __init__(self) -> None:
        self.batches: list[list[Any]] = []

    def forecast(
        self,
        timeseries: list[Any],
        prediction_length: int,
        *,
        output_type: str = "torch",
        **kwargs: Any,
    ) -> list[np.ndarray]:
        self.batches.append(list(timeseries))
        rows = []
        for ts in timeseries:
            w = np.asarray(ts.target, dtype=np.float64).reshape(-1)
            center = float(w.mean())
            q = center + np.linspace(-0.1, 0.1, _NATIVE.size)
            rows.append(q.reshape(1, _NATIVE.size, 1).astype(np.float64))
        return rows


def _install_stub(monkeypatch: pytest.MonkeyPatch) -> _StubModel:
    model = _StubModel()
    mod = types.ModuleType("tirex2")
    mod.load_model = lambda *a, **k: model  # type: ignore[attr-defined]
    mod.TimeseriesType = _StubTs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tirex2", mod)
    return model


def _series(n: int = 320, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=n).cumsum() * 0.01


def test_fit_fails_closed_when_dep_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tirex2", None)
    head = Tirex2Distribution()
    with pytest.raises(RuntimeError, match="not installed"):
        head.fit(np.zeros(4), _series())


def test_dep_error_names_the_extra() -> None:
    assert "nn" in _DEP_ERROR
    assert "uv sync" in _DEP_ERROR


def test_unfitted_predict_fails_closed() -> None:
    head = Tirex2Distribution()
    with pytest.raises(RuntimeError, match="not been fitted"):
        head.predict(np.zeros((3, 2)))


def test_stubbed_predict_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch)
    head = Tirex2Distribution(lookback=32).fit(np.zeros(4), _series())
    q = head.predict(np.zeros((5, 3)))
    assert q.shape == (5, 3)
    assert np.all(np.diff(q, axis=1) >= 0.0)  # monotone quantile rows


def test_windows_passed_raw_not_standardized(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _install_stub(monkeypatch)
    series = _series()
    head = Tirex2Distribution(lookback=32).fit(np.zeros(4), series)
    head.predict(np.zeros((2, 1)))
    sent = np.asarray(model.batches[-1][0].target, dtype=np.float64).reshape(-1)
    np.testing.assert_allclose(sent, series[-32:])  # raw trailing window


def test_history_windows_are_causal(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch)
    lookback = 32
    head = Tirex2Distribution(lookback=lookback).fit(np.zeros(4), _series())
    hist = _series(120, seed=11)
    out = head.predict_from_history(hist)
    assert out.shape == (hist.size - lookback + 1, 3)
    # Each row comes from its own trailing window only: perturbing a late
    # window's observations cannot change an early row.
    perturbed = hist.copy()
    perturbed[-5:] += 10.0
    out2 = head.predict_from_history(perturbed)
    np.testing.assert_allclose(out[:-5], out2[:-5], atol=1e-9)


def test_offgrid_taus_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch)
    # 0.05 / 0.95 are outside the native {0.1..0.9} grid — no interpolation.
    head = Tirex2Distribution(taus=(0.05, 0.5, 0.95), lookback=32)
    with pytest.raises(ValueError, match="native grid"):
        head.fit(np.zeros(4), _series())


def test_metadata_discloses_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub(monkeypatch)
    head = Tirex2Distribution(lookback=32).fit(np.zeros(4), _series())
    meta = head.metadata()
    assert meta.extra["framework"] == "tirex-2"
    assert meta.extra["pretrained"] is True
    assert meta.extra["warmup"] == head.lookback
    assert meta.extra["standardize"] is False
    assert meta.extra["quantile_mapping"] == "native"
    assert "TiRex-2" in meta.extra["model_id"]


def test_registry_wires_head() -> None:
    from quant_fund.research.fleet_eval import FLEET_HEAD_REGISTRY

    head = FLEET_HEAD_REGISTRY["tirex2"]([0.1, 0.5, 0.9], 0)
    assert isinstance(head, Tirex2Distribution)


@pytest.mark.network
def test_real_checkpoint_smoke() -> None:
    """Tiny CPU smoke: loads the real TiRex-2 checkpoint (~380M HF download).

    Skipped by default (``network`` marker) — run explicitly with
    ``pytest -m network tests/unit/models/test_tirex2.py``.
    """
    pytest.importorskip("tirex2")
    head = Tirex2Distribution(lookback=64).fit(np.zeros(4), _series(96))
    q = head.predict(np.zeros((2, 1)))
    assert q.shape == (2, 3)
    assert np.isfinite(q).all()
    assert np.all(np.diff(q, axis=1) >= 0.0)
