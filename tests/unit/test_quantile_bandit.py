from pathlib import Path

import numpy as np

from quant_fund.models.quantile_bandit import QuantileThompson


def test_planted_linear_policy_beats_uniform() -> None:
    rng = np.random.default_rng(0)
    n_dates, n_names, d = 40, 10, 3
    dates = np.repeat(np.arange(n_dates), n_names)
    x = rng.normal(size=(n_dates * n_names, d))
    y = x[:, 0] + 0.15 * rng.normal(size=x.shape[0])
    trace = QuantileThompson(n_quantiles=9, ridge=1.0, seed=0).run_panel(x, y, dates, k=2)
    assert trace.policy_reward.size >= 20
    assert float(np.mean(trace.policy_reward)) > float(np.mean(trace.uniform_reward))


def test_select_honors_topk_size() -> None:
    rng = np.random.default_rng(1)
    bandit = QuantileThompson(n_quantiles=5, ridge=1.0, seed=1)
    x = rng.normal(size=(15, 4))
    idx = bandit.select(x, k=5)
    assert idx.shape == (5,)
    assert len(np.unique(idx)) == 5
    assert set(idx.tolist()).issubset(set(range(15)))
    bandit.update(x[0], 0.4)
    clipped = bandit.select(x[:3], k=10)
    assert clipped.shape == (3,)
    assert len(np.unique(clipped)) == 3


def test_run_panel_updates_without_future_dates() -> None:
    rng = np.random.default_rng(2)
    n_dates, n_names, d = 6, 8, 2
    dates = np.repeat(np.arange(n_dates), n_names)
    x = rng.normal(size=(n_dates * n_names, d))
    name = np.tile(np.arange(n_names), n_dates)
    y = dates.astype(float) * 100.0 + 0.01 * name
    perm = rng.permutation(dates.size)
    x, y, dates = x[perm], y[perm], dates[perm]

    class _Probe(QuantileThompson):
        def __init__(self) -> None:
            super().__init__(n_quantiles=5, ridge=1.0, seed=2)
            self.seen: list[float] = []

        def update(self, context: np.ndarray, reward: float) -> None:
            self.seen.append(float(reward))
            super().update(context, reward)

    bandit = _Probe()
    trace = bandit.run_panel(x, y, dates, k=2)
    assert trace.dates == [str(i) for i in range(n_dates)]
    seen_dates = [int(reward // 100.0) for reward in bandit.seen]
    assert seen_dates == sorted(seen_dates)
    assert all(earlier <= later for earlier, later in zip(seen_dates, seen_dates[1:], strict=False))


def test_module_text_has_no_sharpe() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "src/quant_fund/models/quantile_bandit.py").read_text(encoding="utf-8")
    assert "sharpe" not in text.lower()
