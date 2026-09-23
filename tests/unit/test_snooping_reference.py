"""Compare snooping procedures with arch using identical bootstrap draws."""

from types import SimpleNamespace

import numpy as np
import pytest
from arch.bootstrap import MCS, SPA

from quant_fund.metrics.inference import stationary_bootstrap_indices
from quant_fund.metrics.snooping import model_confidence_set, spa_test


def _bootstrap(data: np.ndarray, indices: np.ndarray) -> SimpleNamespace:
    def bootstrap(reps: int):
        assert reps == len(indices)
        for draw in indices:
            yield (data[draw],), {}

    return SimpleNamespace(bootstrap=bootstrap)


@pytest.mark.parametrize("seed", [3, 19, 41])
def test_spa_centering_matches_arch_on_identical_draws(seed: int) -> None:
    n, reps, block = 160, 399, 5
    rng = np.random.default_rng(seed)
    panel = rng.normal(size=(n, 5)) * [1.0, 0.4, 0.3, 0.7, 2.0]
    panel += [0.04, -0.02, -0.12, -0.7, 0.01]
    indices = stationary_bootstrap_indices(n, reps, block, np.random.default_rng(17))
    boot_means = np.asarray([panel[draw].mean(axis=0) for draw in indices])
    sigma = np.sqrt(n * np.mean((boot_means - panel.mean(axis=0)) ** 2, axis=0))

    # Supply identically studentized inputs and the same estimated variances
    # to isolate Hansen's three null-centering rules from variance estimators.
    standardized = panel / sigma
    reference = SPA(np.zeros(n), -standardized, reps=reps, block_size=block)
    reference.bootstrap = _bootstrap(standardized, indices)
    reference._compute_variance = lambda: setattr(reference, "_loss_diff_var", np.ones(5))
    reference.compute()
    actual = spa_test(panel, n_boot=reps, block=block, seed=17)
    expected = (1.0 + reps * reference.pvalues.to_numpy()) / (reps + 1.0)
    np.testing.assert_allclose(
        [actual.p_lower, actual.p_consistent, actual.p_upper], expected, atol=1e-12
    )
    assert actual.p_lower <= actual.p_consistent <= actual.p_upper


def test_spa_all_negative_means_cannot_reject() -> None:
    panel = np.random.default_rng(8).normal(-0.5, 0.2, size=(80, 3))
    result = spa_test(panel, n_boot=199, block=3)
    assert result.statistic == 0.0
    assert result.p_lower == result.p_consistent == result.p_upper == 1.0


@pytest.mark.parametrize("seed", [1, 7, 29])
def test_mcs_range_matches_arch_on_identical_draws(seed: int) -> None:
    n, reps, block = 160, 399, 5
    rng = np.random.default_rng(seed)
    performance = rng.normal(size=(n, 5)) * [0.05, 0.12, 0.8, 3.0, 0.08]
    performance += [0.0, 0.04, -0.1, -0.3, -0.03]
    indices = stationary_bootstrap_indices(n, reps, block, np.random.default_rng(17))
    reference = MCS(-performance, size=0.10, method="R", reps=reps, block_size=block)
    reference.bootstrap = _bootstrap(np.arange(n), indices)
    reference.compute()
    actual = model_confidence_set(performance, n_boot=reps, block=block, seed=17)
    expected = (1.0 + reps * reference.pvalues.reindex(range(5)).Pvalue.to_numpy()) / (reps + 1.0)
    np.testing.assert_allclose(actual.p_values, expected, atol=1e-12)
    assert list(actual.included) == (expected >= 0.10).tolist()
    assert len(actual.included) == performance.shape[1]


def test_mcs_p_values_do_not_depend_on_requested_alpha() -> None:
    rng = np.random.default_rng(53)
    panel = rng.normal(size=(120, 5)) + [0.0, 0.1, -0.2, -0.3, -0.4]
    a = model_confidence_set(panel, n_boot=199, block=4, alpha=0.05)
    b = model_confidence_set(panel, n_boot=199, block=4, alpha=0.20)
    assert a.p_values == b.p_values
    assert all(not bi or ai for ai, bi in zip(a.included, b.included, strict=True))


def test_mcs_retains_equivalent_models_without_losing_column_identity() -> None:
    x = np.random.default_rng(13).normal(size=100)
    panel = np.column_stack([x, np.zeros(100), x])
    result = model_confidence_set(panel, n_boot=199, block=3)
    assert len(result.included) == 3
    assert result.n_strategies == 3
    assert result.n_dropped == 0
    assert result.included[0] == result.included[2]
    assert all(np.isfinite(result.p_values))


def test_mcs_unstudentizable_nonzero_difference_is_inconclusive() -> None:
    x = np.random.default_rng(23).normal(size=100)
    result = model_confidence_set(np.column_stack([x, x + 1.0]), n_boot=199, block=3)
    assert result.included == (False, False)
    assert all(np.isnan(result.p_values))
