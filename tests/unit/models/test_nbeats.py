"""N-BEATS / N-HiTS quantile heads: deterministic CPU training, fleet contract.

All data here is SYNTHETIC — correctness evidence, never market evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="N-BEATS/N-HiTS heads require the nn extra")

from quant_fund.models.nbeats import (  # noqa: E402
    _NN_MIN_WINDOWS,
    NBeatsDistribution,
    NHiTsDistribution,
)
from quant_fund.research.fleet_eval import run_distribution_fleet  # noqa: E402

TAUS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _seasonal(n: int = 600, seed: int = 0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return (
        np.sin(2.0 * np.pi * t / 24.0)
        + 0.3 * np.sin(2.0 * np.pi * t / 7.0)
        + rng.normal(0.0, 0.05, n)
    )


@pytest.mark.parametrize("cls", [NBeatsDistribution, NHiTsDistribution])
def test_reconstructs_seasonal_continuation(cls: type) -> None:
    """Median one-step forecasts on the held-out continuation beat persistence."""
    y = _seasonal()
    x = np.ones((y.size, 1))
    n_train = 512
    model = cls(TAUS, lookback=32, epochs=60, seed=0).fit(x[:n_train], y[:n_train])
    hist = model.predict_from_history(y[n_train:])
    # Window ending at i predicts y_i; the final row forecasts one past the end.
    med, actual = hist[:-1, 3], y[n_train + 32 :]
    naive = y[n_train + 31 : -1]
    assert np.abs(med - actual).mean() < 0.75 * np.abs(naive - actual).mean()
    cover80 = float(np.mean((actual >= hist[:-1, 1]) & (actual <= hist[:-1, 5])))
    assert 0.5 <= cover80 <= 1.0


@pytest.mark.parametrize("cls", [NBeatsDistribution, NHiTsDistribution])
def test_deterministic_across_calls(cls: type) -> None:
    y = _seasonal(200)
    x = np.ones((y.size, 1))
    a = cls(TAUS, lookback=24, epochs=30, seed=7).fit(x, y)
    b = cls(TAUS, lookback=24, epochs=30, seed=7).fit(x, y)
    assert np.array_equal(a.predict(x[:11]), b.predict(x[:11]))
    assert np.array_equal(a.predict(x[:11]), a.predict(x[:11]))


@pytest.mark.parametrize("cls", [NBeatsDistribution, NHiTsDistribution])
def test_deterministic_seed_changes_weights(cls: type) -> None:
    y = _seasonal(200)
    x = np.ones((y.size, 1))
    a = cls(TAUS, lookback=24, epochs=10, seed=1).fit(x, y)
    b = cls(TAUS, lookback=24, epochs=10, seed=2).fit(x, y)
    assert not np.array_equal(a.predict(x[:3]), b.predict(x[:3]))


@pytest.mark.parametrize("cls", [NBeatsDistribution, NHiTsDistribution])
def test_fail_closed_short_series(cls: type) -> None:
    lookback = 24
    y = _seasonal(lookback + _NN_MIN_WINDOWS - 1)
    with pytest.raises(ValueError, match="observations"):
        cls(TAUS, lookback=lookback, epochs=5).fit(np.ones((y.size, 1)), y)
    with pytest.raises(ValueError, match="lookback"):
        cls(TAUS, lookback=4)


@pytest.mark.parametrize("cls", [NBeatsDistribution, NHiTsDistribution])
def test_fail_closed_nonfinite_and_bad_taus(cls: type) -> None:
    y = _seasonal(120)
    y[60] = np.nan
    with pytest.raises(ValueError, match="all-finite"):
        cls(TAUS, lookback=16, epochs=5).fit(np.ones((y.size, 1)), y)
    with pytest.raises(ValueError, match="taus"):
        cls((0.5, 0.1), lookback=16)


@pytest.mark.parametrize("cls", [NBeatsDistribution, NHiTsDistribution])
def test_predict_before_fit_raises(cls: type) -> None:
    model = cls(TAUS, lookback=16, epochs=5)
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict(np.ones((4, 1)))
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict_from_history(np.zeros(64))


@pytest.mark.parametrize("cls", [NBeatsDistribution, NHiTsDistribution])
def test_predict_shape_monotone_finite(cls: type) -> None:
    y = _seasonal(160)
    x = np.ones((y.size, 1))
    model = cls(TAUS, lookback=24, epochs=15, seed=3).fit(x, y)
    q = model.predict(np.ones((9, 1)))
    assert q.shape == (9, len(TAUS))
    assert np.isfinite(q).all()
    assert (np.diff(q, axis=1) >= 0.0).all()
    meta = model.metadata()
    assert meta.family == "distribution" and meta.version == "v1"
    assert meta.extra["warmup"] == 24 and meta.extra["lookback"] == 24
    assert meta.extra["n_train_windows"] == 160 - 24
    assert meta.extra["device"] == "cpu" and meta.extra["framework"] == "torch"


def test_fleet_row_scores_ok() -> None:
    """The heads slot into run_distribution_fleet and produce a scored row."""
    frame, receipt = run_distribution_fleet(
        {
            "nbeats": lambda: NBeatsDistribution(TAUS, lookback=24, epochs=15, seed=0),
            "nhits": lambda: NHiTsDistribution(TAUS, lookback=24, epochs=15, seed=0),
        },
        shards=["iid_gaussian"],
        n_train=160,
        n_eval=48,
        seed=0,
        taus=TAUS,
    )
    assert frame.height == 2
    assert (frame["status"] == "ok").all()
    assert frame["crps"].is_finite().all()
    assert receipt["n_error_rows"] == 0


def test_predict_from_history_alignment() -> None:
    y = _seasonal(140)
    x = np.ones((y.size, 1))
    model = NBeatsDistribution(TAUS, lookback=16, epochs=5, seed=0).fit(x, y)
    hist = model.predict_from_history(y)
    assert hist.shape == (y.size - 16 + 1, len(TAUS))
    assert np.isfinite(hist).all()
    with pytest.raises(ValueError, match="lookback"):
        model.predict_from_history(y[:10])
