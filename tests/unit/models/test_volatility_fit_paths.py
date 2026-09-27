"""Volatility model fit/predict paths: rolling, EWMA metadata, GARCH, HAR, trees.

Complements test_volatility_edges (fail-closed edges) with the fitted paths that
previously had no coverage. Tree backends are optional imports (skip if absent).
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.volatility import (
    EWMAVol,
    GARCHVol,
    HARVol,
    RollingVol,
    TreeVol,
)


def _returns(n: int = 150, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=n)


def test_rolling_vol_fit_metadata_and_predict() -> None:
    model = RollingVol(window=5)
    assert model.fit(np.zeros((3, 1)), np.zeros(3)) is model
    assert model.metadata().name == "rolling"
    out = model.predict(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    assert out.tolist() == [0.3, 0.6]
    r = _returns(30)
    series = model.predict_from_returns(r)
    assert np.isnan(series[:4]).all()
    assert np.isfinite(series[4:]).all()


def test_ewma_metadata_exposes_lambda() -> None:
    model = EWMAVol(lam=0.9)
    meta = model.metadata()
    assert meta.name == "ewma"
    assert meta.extra["lambda"] == pytest.approx(0.9)


def test_garch_long_series_fits_and_predicts_constant_sigma() -> None:
    model = GARCHVol()
    r = _returns(150)
    assert model.fit(np.zeros((r.size, 1)), r, returns=r) is model
    assert model.result is not None
    assert model.last_sigma > 0.0
    pred = model.predict(np.zeros((4, 1)))
    assert pred.shape == (4,)
    assert np.allclose(pred, pred[0])
    assert np.all((pred > 0.0) & np.isfinite(pred))
    assert model.metadata().name == "garch1-1_normal"


def test_garch_short_series_predict_uses_fallback_sigma() -> None:
    model = GARCHVol()
    r = _returns(30)
    model.fit_returns(r)
    assert model.result is None
    pred = model.predict(np.zeros((3, 1)))
    assert np.allclose(pred, model.last_sigma)
    assert np.isfinite(pred).all()


def test_har_consumes_declared_supervised_features() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(40, 5))
    y = 0.0004 + np.abs(rng.normal(0.0, 0.0002, size=40))

    model = HARVol().fit(x, y)

    assert model.fitted is True
    pred = model.predict(x[:6])
    assert pred.shape == (6,)
    assert np.isfinite(pred).all()
    assert np.all(pred > 0.0)


@pytest.mark.parametrize("use_log", [True, False])
def test_har_fit_and_predict_paths(use_log: bool) -> None:
    rng = np.random.default_rng(3)
    rv = 0.0004 + np.abs(rng.normal(0.0, 0.0002, size=90))
    design = HARVol.har_design(rv)
    model = HARVol(use_log=use_log)
    model.fit(design, rv)
    assert model.fitted is True
    clean = design[np.isfinite(design).all(axis=1)]
    assert clean.shape[0] > 0
    pred = model.predict(clean)
    assert np.isfinite(pred).all()
    assert np.all(pred > 0.0)
    assert model.metadata().name == "har_rv"


def test_har_fit_too_few_rows_stays_unfitted() -> None:
    model = HARVol()
    model.fit(np.zeros((5, 1)), np.full(5, 0.0004))
    assert model.fitted is False
    assert np.isnan(model.predict(np.zeros((2, 2)))).all()


@pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
def test_tree_vol_fit_predict_metadata(backend: str) -> None:
    pytest.importorskip(backend)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(80, 3))
    y = 0.0004 + np.abs(rng.normal(0.0, 0.0002, size=80))
    model = TreeVol(backend=backend, seed=11)
    model.fit(x, y)
    pred = model.predict(x[:5])
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
    assert model.metadata().name == f"{backend}_vol"


def test_har_design_uses_only_past_information() -> None:
    rv = np.arange(1.0, 31.0)
    design = HARVol.har_design(rv)
    # Column 1 is the one-bar lag: row t must equal rv[t-1], never rv[t].
    assert np.isnan(design[0, 1])
    assert np.allclose(design[1:, 1], rv[:-1])
    # Trailing means start only once their window holds past-only lags:
    # window 5 needs lagged[1..5] (index 0 is the missing one-bar lag).
    assert np.isnan(design[4, 2])
    assert np.isfinite(design[5, 2])
    assert np.isnan(design[21, 3])
    assert np.isfinite(design[22, 3])
