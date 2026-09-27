"""Tests for quant_fund.models.agaci — AggregatedACI (AgACI-EG / FACI-EG)."""

import pathlib

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.metrics.scoring import mean_pinball
from quant_fund.models.agaci import AggregatedACI

LEVELS = np.round(np.arange(0.05, 0.96, 0.05), 2)  # 19 levels, 0.05..0.95
GAMMAS = (0.001, 0.01, 0.05, 0.1)
ALPHA = 0.1
NOMINAL_CENTRAL = 1.0 - 2.0 * ALPHA  # central interval is [alpha, 1 - alpha]
LO_I = int(np.where(np.isclose(LEVELS, ALPHA))[0][0])
HI_I = int(np.where(np.isclose(LEVELS, 1.0 - ALPHA))[0][0])
REGIME_N = 500
TRAIL = 250


def _shifted_stream(seed: int) -> np.ndarray:
    """N(0,1) -> N(0,4) -> 2*Student-t(3), 500 steps per regime."""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [
            rng.normal(0.0, 1.0, REGIME_N),
            rng.normal(0.0, 2.0, REGIME_N),
            rng.standard_t(3, REGIME_N) * 2.0,
        ]
    )


def _forecast_grid() -> np.ndarray:
    """Fixed N(0,1)-calibrated (hence biased once the scale shifts) grid."""
    return norm.ppf(LEVELS)


def _run(stream: np.ndarray, gammas: tuple[float, ...]) -> tuple[np.ndarray, list[np.ndarray]]:
    model = AggregatedACI(quantile_levels=LEVELS, gammas=gammas, alpha=ALPHA)
    grid = _forecast_grid()
    grids = []
    for y in stream:
        out = model.update(y, grid)
        assert out.shape == LEVELS.shape
        assert np.all(np.diff(out) >= -1e-12)  # non-crossing at every step
        w = model.expert_weights_
        assert np.all(w >= 0.0) and np.all(w <= 1.0)  # simplex at every step
        assert np.allclose(w.sum(axis=0), 1.0)
        grids.append(out)
    return np.stack(grids), model.coverage_history


def _mean_pinball_all_levels(stream: np.ndarray, grids: np.ndarray) -> float:
    per_level = [mean_pinball(stream, grids[:, j], float(LEVELS[j])) for j in range(LEVELS.size)]
    return float(np.mean(per_level))


def test_long_run_coverage_under_shift() -> None:
    stream = _shifted_stream(seed=1234)
    grids, _ = _run(stream, GAMMAS)
    covered = (stream >= grids[:, LO_I]) & (stream <= grids[:, HI_I])
    for regime in range(3):
        tail = covered[regime * REGIME_N + (REGIME_N - TRAIL) : (regime + 1) * REGIME_N]
        emp = float(np.mean(tail))
        assert abs(emp - NOMINAL_CENTRAL) <= 0.07, f"regime {regime}: coverage {emp:.3f}"


def test_aggregation_regret_vs_best_fixed_expert() -> None:
    stream = _shifted_stream(seed=99)
    grids_agg, _ = _run(stream, GAMMAS)
    loss_agg = _mean_pinball_all_levels(stream, grids_agg)
    best_fixed = np.inf
    for gamma in GAMMAS:
        grids_e, _ = _run(stream, (gamma,))
        best_fixed = min(best_fixed, _mean_pinball_all_levels(stream, grids_e))
    assert loss_agg <= best_fixed + 0.005


def test_coverage_history_records_per_level_hits() -> None:
    stream = _shifted_stream(seed=5)[:64]
    _, hist = _run(stream, GAMMAS)
    assert len(hist) == 64
    arr = np.stack(hist)
    assert arr.shape == (64, LEVELS.size)
    assert np.all((arr == 0.0) | (arr == 1.0))


def test_first_step_output_is_rearranged_forecast() -> None:
    model = AggregatedACI(quantile_levels=LEVELS, gammas=GAMMAS, alpha=ALPHA)
    out = model.update(0.0, _forecast_grid())
    assert np.allclose(out, np.maximum.accumulate(norm.ppf(LEVELS)))


def test_default_levels_follow_alpha_halves() -> None:
    model = AggregatedACI(alpha=ALPHA)
    assert np.allclose(model.quantile_levels_, [0.05, 0.95])
    assert np.allclose(model.expert_weights_, 0.25)


def test_eta_default_is_sixth_of_mean_gamma_step() -> None:
    g = np.array(GAMMAS)
    model = AggregatedACI(quantile_levels=LEVELS, gammas=g, alpha=ALPHA)
    assert np.isclose(model.eta, 1.0 / (6.0 * float(np.mean(np.diff(g)))))


def test_expert_weights_are_simplex_after_updates() -> None:
    rng = np.random.default_rng(7)
    model = AggregatedACI(quantile_levels=LEVELS, gammas=GAMMAS, alpha=ALPHA)
    grid = _forecast_grid()
    for _ in range(50):
        model.update(float(rng.normal(0.0, 3.0)), grid)
    w = model.expert_weights_
    assert w.shape == (len(GAMMAS), LEVELS.size)
    assert np.all(w >= 0.0)
    assert np.allclose(w.sum(axis=0), 1.0)
    assert model.n_steps_ == 50


@pytest.mark.parametrize(
    "levels",
    [
        np.array([0.5, 0.5]),
        np.array([0.9, 0.1]),
        np.array([0.0, 0.5]),
        np.array([0.5, 1.0]),
        np.array([-0.2, 0.5]),
        np.array([]),
    ],
)
def test_bad_quantile_levels_fail_closed(levels: np.ndarray) -> None:
    with pytest.raises(ValueError, match="quantile_levels"):
        AggregatedACI(quantile_levels=levels, gammas=GAMMAS, alpha=ALPHA)


@pytest.mark.parametrize(
    "gammas",
    [np.array([]), np.array([0.0, 0.1]), np.array([-0.01, 0.1]), np.array([0.1, 0.1])],
)
def test_bad_gammas_fail_closed(gammas: np.ndarray) -> None:
    with pytest.raises(ValueError, match="gammas"):
        AggregatedACI(quantile_levels=LEVELS, gammas=gammas, alpha=ALPHA)


def test_non_eg_aggregation_fail_closed() -> None:
    with pytest.raises(ValueError, match="aggregation"):
        AggregatedACI(quantile_levels=LEVELS, gammas=GAMMAS, aggregation="ml-poe")


def test_update_grid_mismatch_fail_closed() -> None:
    model = AggregatedACI(quantile_levels=LEVELS, gammas=GAMMAS, alpha=ALPHA)
    with pytest.raises(ValueError, match="grid"):
        model.update(0.0, np.zeros(3))
    with pytest.raises(ValueError, match="grid"):
        model.update(0.0, np.full(LEVELS.size, np.nan))
    with pytest.raises(ValueError, match="grid"):
        model.update(0.0, np.zeros((2, LEVELS.size)))


def test_update_bad_y_fail_closed() -> None:
    model = AggregatedACI(quantile_levels=LEVELS, gammas=GAMMAS, alpha=ALPHA)
    with pytest.raises(ValueError, match="y"):
        model.update(np.nan, _forecast_grid())
    with pytest.raises(ValueError, match="y"):
        model.update(np.array([0.0, 1.0]), _forecast_grid())


def test_bad_eta_fail_closed() -> None:
    with pytest.raises(ValueError, match="eta"):
        AggregatedACI(quantile_levels=LEVELS, gammas=GAMMAS, eta=0.0)


def test_module_has_no_forbidden_metrics() -> None:
    import quant_fund.models.agaci as agaci

    text = pathlib.Path(agaci.__file__).read_text(encoding="utf-8").lower()
    for token in ("sharpe", "sortino", "p&l"):
        assert token not in text
