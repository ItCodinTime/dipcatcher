import numpy as np

from quant_fund.models.rl import LinUCBRanker, run_linucb_panel


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
