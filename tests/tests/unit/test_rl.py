import numpy as np

from quant_fund.models.rl import LinUCBRanker, run_linucb_panel, run_thompson_panel


def test_linucb_updates_and_scores() -> None:
    m = LinUCBRanker(3, alpha=1.0)
    x = np.array([1.0, 0.0, 0.0])
    before = m.scores(x.reshape(1, -1))[0]
    m.update(x, 1.0)
    after = m.scores(x.reshape(1, -1))[0]
    assert after > before


def test_linucb_beats_uniform_when_feature_is_reward() -> None:
    rng = np.random.default_rng(0)
    n_dates, n_names, d = 40, 10, 3
    dates = np.repeat(np.arange(n_dates), n_names)
    x = rng.normal(size=(n_dates * n_names, d))
    signal = x[:, 0]
    y = signal + 0.15 * rng.normal(size=signal.size)
    trace = run_linucb_panel(x, y, dates, oracle=signal, top_k=2, alpha=0.5, seed=0)
    assert trace.policy_reward.size >= 20
    assert float(np.mean(trace.policy_reward)) > float(np.mean(trace.uniform_reward))


def test_thompson_panel_is_reproducible_and_finite() -> None:
    rng = np.random.default_rng(3)
    dates = np.repeat(np.arange(12), 6)
    x = rng.normal(size=(dates.size, 2))
    y = x[:, 0] + 0.1 * rng.normal(size=dates.size)
    first = run_thompson_panel(x, y, dates, top_k=2, seed=11)
    second = run_thompson_panel(x, y, dates, top_k=2, seed=11)
    assert np.array_equal(first.dates, second.dates)
    assert np.allclose(first.policy_reward, second.policy_reward)
    assert np.isfinite(first.cumulative_regret).all()
