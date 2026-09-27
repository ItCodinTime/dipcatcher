"""Permutation attribution on a fitted synthetic head (SYNTHETIC labels)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.models.ranking import RidgeRanker
from quant_fund.research.explainability import (
    brier,
    crps_quantiles,
    permutation_attribution,
    pinball,
    resolve_score,
    shap_attribution,
)

from .conftest import FEATURES, synthetic_xy


def test_planted_dominant_feature_ranks_top(planted_model) -> None:
    model, x, y = planted_model
    result = permutation_attribution(model.predict, x, y, FEATURES, scoring=pinball(0.5), seed=7)
    assert result.attributions[0].feature == "signal_core"
    assert result.attributions[0].importance_mean > 0.0
    # The planted signal should dominate the remaining features clearly.
    assert result.attributions[0].importance_mean > 5 * result.attributions[1].importance_mean
    assert result.method == "permutation"
    assert np.isfinite(result.baseline_score)


def test_permutation_attribution_is_deterministic(planted_model) -> None:
    model, x, y = planted_model
    first = permutation_attribution(model.predict, x, y, FEATURES, seed=123, n_repeats=4)
    second = permutation_attribution(model.predict, x, y, FEATURES, seed=123, n_repeats=4)
    assert [a.feature for a in first.attributions] == [a.feature for a in second.attributions]
    np.testing.assert_array_equal(
        [a.importance_mean for a in first.attributions],
        [a.importance_mean for a in second.attributions],
    )
    np.testing.assert_array_equal(
        [a.importance_std for a in first.attributions],
        [a.importance_std for a in second.attributions],
    )
    different = permutation_attribution(model.predict, x, y, FEATURES, seed=124, n_repeats=4)
    first_means = {a.feature: a.importance_mean for a in first.attributions}
    different_means = {a.feature: a.importance_mean for a in different.attributions}
    assert not np.array_equal(
        [first_means[f] for f in FEATURES], [different_means[f] for f in FEATURES]
    )


def test_irrelevant_features_have_near_zero_importance(planted_model) -> None:
    model, x, y = planted_model
    result = permutation_attribution(model.predict, x, y, FEATURES, seed=5)
    by_name = {a.feature: a.importance_mean for a in result.attributions}
    # Pure-noise columns the model did not learn should move the score ~0.
    assert abs(by_name["noise_a"]) < 0.1
    assert abs(by_name["noise_b"]) < 0.1


def test_importance_vector_aligns_to_input_order(planted_model) -> None:
    model, x, y = planted_model
    result = permutation_attribution(model.predict, x, y, FEATURES, seed=3)
    vec = result.importance_vector(FEATURES)
    assert vec.shape == (len(FEATURES),)
    assert vec[FEATURES.index("signal_core")] == pytest.approx(
        result.attributions[0].importance_mean
    )


def test_repo_ridge_ranker_recovers_planted_feature() -> None:
    """End-to-end on a real repo head (RidgeRanker) with a planted signal."""
    weights = np.asarray([0.3, 0.0, 0.0, 3.0, 0.0, 0.0])
    x, y = synthetic_xy(99, n_rows=800, weights=weights)
    model = RidgeRanker(alpha=1e-6).fit(x, y)
    result = permutation_attribution(model.predict, x, y, FEATURES, seed=11)
    assert result.attributions[0].feature == "signal_core"


def test_crps_score_spec_handles_quantile_matrix() -> None:
    """A head emitting an (n, k) quantile grid scores under CRPS."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(200, 3))
    y = 2.0 * x[:, 0] + rng.normal(scale=0.2, size=200)
    taus = (0.1, 0.5, 0.9)

    def predict(block: NDArray[np.float64]) -> NDArray[np.float64]:
        center = 2.0 * block[:, 0]
        return np.stack([center - 0.3, center, center + 0.3], axis=1)

    spec = crps_quantiles(taus)
    assert spec.expects_matrix is True
    result = permutation_attribution(predict, x, y, ["a", "b", "c"], scoring=spec, seed=1)
    assert result.attributions[0].feature == "a"


def test_brier_score_spec_flags_probability_feature() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(300, 2))
    prob = 1.0 / (1.0 + np.exp(-3.0 * x[:, 0]))
    y = (rng.uniform(size=300) < prob).astype(float)

    def predict(block: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1.0 / (1.0 + np.exp(-3.0 * block[:, 0]))

    result = permutation_attribution(predict, x, y, ["logit", "idle"], scoring=brier(), seed=2)
    assert result.attributions[0].feature == "logit"
    assert result.baseline_score < 0.2


def test_resolve_score_specs() -> None:
    assert resolve_score(None).name == "pinball_tau0.5"
    assert resolve_score("pinball:0.25").name == "pinball_tau0.25"
    assert resolve_score("crps").expects_matrix is True
    assert resolve_score("crps:0.1,0.9").expects_matrix is True
    assert resolve_score("brier").name == "brier"
    with pytest.raises(ValueError, match="unknown proper score"):
        resolve_score("sharpe")
    with pytest.raises(ValueError, match="tau"):
        resolve_score("pinball:1.5")


def test_attribution_input_validation(planted_model) -> None:
    model, x, y = planted_model
    with pytest.raises(ValueError, match="feature_names length"):
        permutation_attribution(model.predict, x, y, ["only_one"], seed=1)
    with pytest.raises(ValueError, match="unique"):
        permutation_attribution(model.predict, x, y, ["a"] * len(FEATURES), seed=1)
    with pytest.raises(ValueError, match="length mismatch"):
        permutation_attribution(model.predict, x, y[:-1], FEATURES, seed=1)
    with pytest.raises(ValueError, match="n_repeats"):
        permutation_attribution(model.predict, x, y, FEATURES, n_repeats=0, seed=1)


def test_positional_feature_names_when_unspecified(planted_model) -> None:
    _model, x, y = planted_model

    def predict(block: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(block, dtype=float)[:, 0] + np.asarray(block, dtype=float)[:, 1]

    result = permutation_attribution(predict, x[:, :2], y, None, seed=1)
    assert {a.feature for a in result.attributions} == {"x0", "x1"}


def test_shap_backend_available_and_ranks_planted_feature(planted_model) -> None:
    shap = pytest.importorskip("shap")
    del shap
    model, x, y = planted_model
    result = shap_attribution(model.predict, x[:120], y[:120], FEATURES, seed=17, max_rows=60)
    assert result.method == "shap_permutation"
    assert result.attributions[0].feature == "signal_core"


def test_linear_head_conforms_to_metadata_contract(planted_model) -> None:
    model, _x, _y = planted_model
    meta = model.metadata()
    assert isinstance(meta.features, list)
    assert len(meta.features) == len(FEATURES)
    assert meta.name == "linear_head"
