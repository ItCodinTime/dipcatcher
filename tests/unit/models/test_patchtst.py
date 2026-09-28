"""PatchTST quantile head: deterministic CPU training, fleet contract.

All data here is SYNTHETIC — correctness evidence, never market evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="PatchTST head requires the nn extra")

from quant_fund.models.nbeats import _NN_MIN_WINDOWS  # noqa: E402
from quant_fund.models.patchtst import PatchTSTDistribution  # noqa: E402

TAUS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _seasonal(n: int = 600, seed: int = 0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return (
        np.sin(2.0 * np.pi * t / 24.0)
        + 0.3 * np.sin(2.0 * np.pi * t / 7.0)
        + rng.normal(0.0, 0.05, n)
    )


def test_reconstructs_seasonal_continuation() -> None:
    """Median one-step forecasts on the held-out continuation beat persistence."""
    y = _seasonal()
    x = np.ones((y.size, 1))
    n_train = 512
    model = PatchTSTDistribution(TAUS, lookback=32, epochs=60, seed=0).fit(x[:n_train], y[:n_train])
    hist = model.predict_from_history(y[n_train:])
    # Window ending at i predicts y_i; the final row forecasts one past the end.
    med, actual = hist[:-1, 3], y[n_train + 32 :]
    naive = y[n_train + 31 : -1]
    assert np.abs(med - actual).mean() < 0.75 * np.abs(naive - actual).mean()
    cover80 = float(np.mean((actual >= hist[:-1, 1]) & (actual <= hist[:-1, 5])))
    assert 0.5 <= cover80 <= 1.0


def test_deterministic_across_calls() -> None:
    y = _seasonal(200)
    x = np.ones((y.size, 1))
    a = PatchTSTDistribution(TAUS, lookback=24, epochs=30, seed=7).fit(x, y)
    b = PatchTSTDistribution(TAUS, lookback=24, epochs=30, seed=7).fit(x, y)
    assert np.array_equal(a.predict(x[:11]), b.predict(x[:11]))
    assert np.array_equal(a.predict(x[:11]), a.predict(x[:11]))


def test_deterministic_seed_changes_weights() -> None:
    y = _seasonal(200)
    x = np.ones((y.size, 1))
    a = PatchTSTDistribution(TAUS, lookback=24, epochs=10, seed=1).fit(x, y)
    b = PatchTSTDistribution(TAUS, lookback=24, epochs=10, seed=2).fit(x, y)
    assert not np.array_equal(a.predict(x[:3]), b.predict(x[:3]))


def test_fail_closed_short_series() -> None:
    lookback = 24
    y = _seasonal(lookback + _NN_MIN_WINDOWS - 1)
    with pytest.raises(ValueError, match="observations"):
        PatchTSTDistribution(TAUS, lookback=lookback, epochs=5).fit(np.ones((y.size, 1)), y)
    with pytest.raises(ValueError, match="lookback"):
        PatchTSTDistribution(TAUS, lookback=4)


def test_fail_closed_nonfinite_and_bad_taus() -> None:
    y = _seasonal(120)
    y[60] = np.nan
    with pytest.raises(ValueError, match="all-finite"):
        PatchTSTDistribution(TAUS, lookback=16, epochs=5).fit(np.ones((y.size, 1)), y)
    with pytest.raises(ValueError, match="taus"):
        PatchTSTDistribution((0.5, 0.1), lookback=16)


def test_predict_before_fit_raises() -> None:
    model = PatchTSTDistribution(TAUS, lookback=16, epochs=5)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(np.ones((4, 1)))
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict_from_history(np.zeros(64))


def test_predict_shape_monotone_finite() -> None:
    y = _seasonal(160)
    x = np.ones((y.size, 1))
    model = PatchTSTDistribution(TAUS, lookback=24, epochs=15, seed=3).fit(x, y)
    q = model.predict(np.ones((9, 1)))
    assert q.shape == (9, len(TAUS))
    assert np.isfinite(q).all()
    assert (np.diff(q, axis=1) >= 0.0).all()
    meta = model.metadata()
    assert meta.family == "distribution" and meta.version == "v1"
    assert meta.extra["warmup"] == 24 and meta.extra["lookback"] == 24
    assert meta.extra["n_train_windows"] == 160 - 24
    assert meta.extra["device"] == "cpu" and meta.extra["framework"] == "torch"
    assert meta.extra["architecture"] == "patchtst"
    assert meta.extra["patch_len"] == 8 and meta.extra["n_patches"] == 3
    assert meta.extra["effective_lookback"] == 24


def test_fail_closed_patch_geometry() -> None:
    with pytest.raises(ValueError, match="patch_len"):
        PatchTSTDistribution(TAUS, lookback=16, patch_len=0)
    with pytest.raises(ValueError, match="2 patches"):
        # lookback 16 / patch_len 16 -> single patch: nothing to attend over.
        PatchTSTDistribution(TAUS, lookback=16, patch_len=16)
    with pytest.raises(ValueError, match="n_heads"):
        PatchTSTDistribution(TAUS, lookback=16, hidden=64, n_heads=3)
    with pytest.raises(ValueError, match="n_layers"):
        PatchTSTDistribution(TAUS, lookback=16, n_layers=0)


def test_nondivisible_lookback_drops_leading_remainder() -> None:
    """lookback 25 / patch_len 8 -> 3 patches over the LAST 24 values."""
    y = _seasonal(200)
    x = np.ones((y.size, 1))
    model = PatchTSTDistribution(TAUS, lookback=25, patch_len=8, epochs=5, seed=0)
    model.fit(x, y)
    meta = model.metadata()
    assert meta.extra["n_patches"] == 3
    assert meta.extra["effective_lookback"] == 24
    q = model.predict(x[:5])
    assert q.shape == (5, len(TAUS)) and np.isfinite(q).all()
