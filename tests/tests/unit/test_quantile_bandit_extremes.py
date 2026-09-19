"""Wave 16: quantile_bandit edge extremes (empty dates, k bounds, planted honesty)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_fund.models.quantile_bandit import QuantileThompson, _topk_indices


def test_n_quantiles_must_be_positive() -> None:
    with pytest.raises(ValueError, match="n_quantiles"):
        QuantileThompson(n_quantiles=0)
    with pytest.raises(ValueError, match="n_quantiles"):
        QuantileThompson(n_quantiles=-3)


def test_select_k_gt_n_clips_to_universe() -> None:
    rng = np.random.default_rng(7)
    bandit = QuantileThompson(n_quantiles=5, ridge=1.0, seed=7)
    x = rng.normal(size=(4, 3))
    idx = bandit.select(x, k=99)
    assert idx.shape == (4,)
    assert len(np.unique(idx)) == 4
    assert set(idx.tolist()) == set(range(4))


def test_select_k_le_zero_clips_to_one() -> None:
    """_topk_indices uses max(1, min(k, n)) — k<=0 becomes a single arm."""
    rng = np.random.default_rng(8)
    bandit = QuantileThompson(n_quantiles=5, ridge=1.0, seed=8)
    x = rng.normal(size=(6, 2))
    idx = bandit.select(x, k=0)
    assert idx.shape == (1,)
    assert 0 <= int(idx[0]) < 6
    idx_neg = bandit.select(x, k=-5)
    assert idx_neg.shape == (1,)


def test_topk_indices_empty_scores_boundary() -> None:
    empty = np.asarray([], dtype=float)
    # max(1, min(k, 0)) → k becomes 0; argsort of empty → empty
    out = _topk_indices(empty, k=3)
    assert out.size == 0


def test_run_panel_empty_dates_yields_empty_trace() -> None:
    bandit = QuantileThompson(n_quantiles=3, ridge=1.0, seed=0)
    x = np.zeros((0, 2), dtype=float)
    y = np.zeros(0, dtype=float)
    dates = np.asarray([], dtype=object)
    trace = bandit.run_panel(x, y, dates, k=2)
    assert trace.dates == []
    assert trace.policy_reward.size == 0
    assert trace.oracle_reward.size == 0
    assert trace.uniform_reward.size == 0
    assert trace.cumulative_regret.size == 0


def test_run_panel_skips_undersized_dates() -> None:
    """Dates with < max(2k, 4) names are skipped (empty kept list)."""
    bandit = QuantileThompson(n_quantiles=3, ridge=1.0, seed=1)
    # 3 dates × 2 names; k=2 → need >=4 finite → all skipped
    dates = np.repeat(np.arange(3), 2)
    x = np.ones((6, 2), dtype=float)
    y = np.linspace(0.0, 1.0, 6)
    trace = bandit.run_panel(x, y, dates, k=2)
    assert trace.dates == []
    assert trace.policy_reward.size == 0


def test_feature_dim_mismatch_raises() -> None:
    bandit = QuantileThompson(n_quantiles=3, ridge=1.0, seed=2)
    bandit.update(np.array([1.0, 2.0]), 0.5)
    with pytest.raises(ValueError, match="expected 2 features"):
        bandit.update(np.array([1.0, 2.0, 3.0]), 0.1)


def test_nonfinite_reward_skipped() -> None:
    bandit = QuantileThompson(n_quantiles=3, ridge=1.0, seed=3)
    bandit.update(np.array([1.0, 0.0]), 1.0)
    before = len(bandit._xs)
    bandit.update(np.array([0.0, 1.0]), float("nan"))
    bandit.update(np.array([0.0, 1.0]), float("inf"))
    assert len(bandit._xs) == before


def test_planted_edge_policy_beats_uniform_honesty() -> None:
    """Strong linear signal → policy mean reward > uniform (research honesty, not Sharpe)."""
    rng = np.random.default_rng(11)
    n_dates, n_names, d = 50, 12, 4
    dates = np.repeat(np.arange(n_dates), n_names)
    x = rng.normal(size=(n_dates * n_names, d))
    y = 2.0 * x[:, 0] - 0.5 * x[:, 1] + 0.05 * rng.normal(size=x.shape[0])
    trace = QuantileThompson(n_quantiles=9, ridge=1.0, seed=11).run_panel(x, y, dates, k=3)
    assert trace.policy_reward.size >= 30
    assert float(np.mean(trace.policy_reward)) > float(np.mean(trace.uniform_reward))
    assert np.isfinite(trace.cumulative_regret).all()
    assert trace.cumulative_regret.size == trace.policy_reward.size
    # No Sharpe keys in trace fields
    assert not hasattr(trace, "sharpe")
    assert not hasattr(trace, "Sharpe")


def test_module_and_trace_have_no_sharpe_keys() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "src/quant_fund/models/quantile_bandit.py").read_text(encoding="utf-8")
    assert "sharpe" not in text.lower()
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(8), 6)
    x = rng.normal(size=(48, 2))
    y = x[:, 0] + 0.1 * rng.normal(size=48)
    trace = QuantileThompson(n_quantiles=5, ridge=1.0, seed=0).run_panel(x, y, dates, k=2)
    keys = set(trace.__dataclass_fields__)
    assert all("sharpe" not in k.lower() for k in keys)
