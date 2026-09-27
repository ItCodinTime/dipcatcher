"""Tests for quant_fund.models.quantile_forest — QuantileRegressionForest.

Meinshausen (2006, JMLR 7:983-999) sanity checks on synthetic data only:
conditional-median tracking under homoskedastic noise, heteroskedastic
interval-width ordering, non-crossing, level monotonicity, and a loose
sample-size monotonicity on out-of-sample pinball loss at the median. All
randomness is seeded (np.random.default_rng(fixed_int)).
"""

import numpy as np
import pytest
from scipy.stats import spearmanr

from quant_fund.metrics.scoring import mean_pinball
from quant_fund.models.quantile_forest import QuantileRegressionForest

LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
SEED = 20260927


def _make_qrf(**over: object) -> QuantileRegressionForest:
    kwargs: dict[str, object] = dict(n_estimators=50, random_state=SEED)
    kwargs.update(over)
    return QuantileRegressionForest(**kwargs)  # type: ignore[arg-type]


def _homoskedastic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n, 1))
    y = 2.0 * x[:, 0] + rng.normal(0.0, 1.0, size=n)
    return x, y


def test_median_tracks_true_conditional_median() -> None:
    x_tr, y_tr = _homoskedastic(800, seed=1)
    model = _make_qrf().fit(x_tr, y_tr)
    x_te = np.linspace(-3.0, 3.0, 200).reshape(-1, 1)
    grid = model.predict_quantiles(x_te, LEVELS)
    med = grid[:, 2]
    true_med = 2.0 * x_te[:, 0]
    corr = np.corrcoef(med, true_med)[0, 1]
    mae = float(np.mean(np.abs(med - true_med)))
    assert corr > 0.8, f"median correlation {corr:.3f}"
    assert mae < 0.6, f"median MAE {mae:.3f}"


def test_interval_width_tracks_heteroskedastic_sigma() -> None:
    rng = np.random.default_rng(7)
    x_tr = rng.uniform(-2.0, 2.0, size=(800, 1))
    sigma_tr = 0.3 + np.abs(x_tr[:, 0])
    y_tr = x_tr[:, 0] + sigma_tr * rng.standard_normal(800)
    model = _make_qrf().fit(x_tr, y_tr)

    x_te = np.linspace(-2.0, 2.0, 150).reshape(-1, 1)
    grid = model.predict_quantiles(x_te, np.array([0.1, 0.9]))
    width = grid[:, 1] - grid[:, 0]
    true_sigma = 0.3 + np.abs(x_te[:, 0])
    rho = spearmanr(width, true_sigma).statistic
    assert rho > 0.4, f"width-vs-sigma rank correlation {rho:.3f}"


def test_quantiles_non_crossing_at_every_query() -> None:
    x_tr, y_tr = _homoskedastic(600, seed=11)
    model = _make_qrf().fit(x_tr, y_tr)
    rng = np.random.default_rng(12)
    x_te = rng.uniform(-3.0, 3.0, size=(250, 1))
    grid = model.predict_quantiles(x_te, LEVELS)
    assert np.all(np.diff(grid, axis=1) >= 0.0)


def test_higher_levels_give_higher_quantiles() -> None:
    x_tr, y_tr = _homoskedastic(600, seed=21)
    model = _make_qrf().fit(x_tr, y_tr)
    x_te = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
    grid = model.predict_quantiles(x_te, LEVELS)
    lo, hi = int(np.argmin(LEVELS)), int(np.argmax(LEVELS))
    assert np.all(grid[:, hi] >= grid[:, lo])
    assert np.all(np.diff(grid, axis=1) >= 0.0)
    # Median lies between the 10th and 90th percentile at every query.
    mid = int(np.argmin(np.abs(LEVELS - 0.5)))
    assert np.all(grid[:, 0] <= grid[:, mid])
    assert np.all(grid[:, mid] <= grid[:, -1])


def test_more_data_lowers_out_of_sample_median_pinball() -> None:
    small_losses: list[float] = []
    large_losses: list[float] = []
    for split in range(5):
        x_small, y_small = _homoskedastic(120, seed=100 + split)
        x_large, y_large = _homoskedastic(800, seed=100 + split)
        rng = np.random.default_rng(200 + split)
        x_te = rng.uniform(-3.0, 3.0, size=(150, 1))
        y_te = 2.0 * x_te[:, 0] + rng.normal(0.0, 1.0, size=150)
        for x_tr, y_tr, acc in ((x_small, y_small, small_losses), (x_large, y_large, large_losses)):
            model = _make_qrf(n_estimators=30, random_state=SEED + split).fit(x_tr, y_tr)
            med = model.predict_quantiles(x_te, np.array([0.5]))[:, 0]
            acc.append(mean_pinball(y_te, med, 0.5))
    assert float(np.mean(large_losses)) < float(np.mean(small_losses))


def test_predict_before_fit_raises_runtime_error() -> None:
    model = _make_qrf()
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict_quantiles(np.zeros((5, 1)), np.array([0.5]))


@pytest.mark.parametrize(
    "levels",
    [
        np.array([0.5, 0.5]),  # not strictly increasing
        np.array([0.9, 0.1]),  # decreasing
        np.array([0.0, 0.5]),  # boundary outside (0, 1)
        np.array([0.5, 1.0]),  # boundary outside (0, 1)
        np.array([-0.1, 0.5]),  # negative
        np.array([0.5, 1.2]),  # above 1
        np.array([np.nan, 0.5]),  # NaN
    ],
)
def test_invalid_levels_raise_value_error(levels: np.ndarray) -> None:
    x_tr, y_tr = _homoskedastic(120, seed=31)
    model = _make_qrf().fit(x_tr, y_tr)
    with pytest.raises(ValueError):
        model.predict_quantiles(np.zeros((4, 1)), levels)


def test_nan_inf_in_fit_and_predict_raise_value_error() -> None:
    x_tr, y_tr = _homoskedastic(120, seed=41)
    model = _make_qrf()
    x_bad = x_tr.copy()
    x_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.fit(x_bad, y_tr)
    y_bad = y_tr.copy()
    y_bad[0] = np.inf
    with pytest.raises(ValueError):
        model.fit(x_tr, y_bad)
    model.fit(x_tr, y_tr)
    x_pred_bad = np.array([[np.nan], [0.0]])
    with pytest.raises(ValueError):
        model.predict_quantiles(x_pred_bad, np.array([0.5]))


def test_fit_with_n_below_30_raises_value_error() -> None:
    x_tr, y_tr = _homoskedastic(29, seed=51)
    with pytest.raises(ValueError, match="n >="):
        _make_qrf().fit(x_tr, y_tr)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_estimators": 0},
        {"n_estimators": -3},
        {"min_samples_leaf": 0},
        {"max_depth": 0},
        {"max_features": "bogus"},
        {"max_features": -1.5},
    ],
)
def test_invalid_constructor_args_raise_value_error(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _make_qrf(**kwargs)


def test_feature_count_mismatch_raises_value_error() -> None:
    x_tr, y_tr = _homoskedastic(120, seed=61)
    model = _make_qrf().fit(x_tr, y_tr)
    with pytest.raises(ValueError, match="features"):
        model.predict_quantiles(np.zeros((4, 2)), np.array([0.5]))


def test_output_shape_and_determinism() -> None:
    x_tr, y_tr = _homoskedastic(300, seed=71)
    model = _make_qrf().fit(x_tr, y_tr)
    x_te = np.linspace(-2.0, 2.0, 40).reshape(-1, 1)
    g1 = model.predict_quantiles(x_te, LEVELS)
    g2 = model.predict_quantiles(x_te, LEVELS)
    assert g1.shape == (40, LEVELS.size)
    assert np.array_equal(g1, g2)
    model2 = _make_qrf().fit(x_tr, y_tr)
    g3 = model2.predict_quantiles(x_te, LEVELS)
    assert np.array_equal(g1, g3)  # seeded random_state reproduces the forest
