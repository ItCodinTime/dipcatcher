"""Optional neural policy-gradient contextual-bandit checks."""

import numpy as np

from quant_fund.models.deep_rl import PolicyGradientRanker, run_policy_gradient_panel


def test_policy_gradient_panel_is_finite_and_reproducible() -> None:
    rng = np.random.default_rng(9)
    dates = np.repeat(np.arange(8), 6)
    x = rng.normal(size=(dates.size, 3))
    y = x[:, 0] - 0.25 * x[:, 1]
    first = run_policy_gradient_panel(x, y, dates, top_k=2, seed=4, epochs=3)
    second = run_policy_gradient_panel(x, y, dates, top_k=2, seed=4, epochs=3)
    assert first.dates == second.dates
    assert np.allclose(first.policy_reward, second.policy_reward)
    assert np.isfinite(first.cumulative_regret).all()


def test_policy_gradient_rejects_bad_feature_shape() -> None:
    model = PolicyGradientRanker(2, epochs=1)
    with np.testing.assert_raises(ValueError):
        model.predict(np.zeros((3, 1)))
