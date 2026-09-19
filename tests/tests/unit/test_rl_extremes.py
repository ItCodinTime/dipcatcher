"""Wave 29: LinUCB / rl edge extremes (empty panel, dims, k, alpha, no Sharpe)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_fund.models.rl import LinUCBRanker, _topk_mean, run_linucb_panel


def test_n_features_must_be_positive() -> None:
    with pytest.raises(ValueError, match="n_features"):
        LinUCBRanker(0)
    with pytest.raises(ValueError, match="n_features"):
        LinUCBRanker(-2)


def test_bad_alpha_fail_closed() -> None:
    with pytest.raises(ValueError, match="alpha"):
        LinUCBRanker(3, alpha=float("nan"))
    with pytest.raises(ValueError, match="alpha"):
        LinUCBRanker(3, alpha=float("inf"))
    with pytest.raises(ValueError, match="alpha"):
        LinUCBRanker(3, alpha=-0.1)
    # alpha=0 (pure exploit) is allowed
    m = LinUCBRanker(2, alpha=0.0)
    assert m.alpha == 0.0


def test_feature_dim_mismatch_raises() -> None:
    m = LinUCBRanker(3, alpha=1.0)
    with pytest.raises(ValueError, match="expected 3 features"):
        m.update(np.array([1.0, 2.0]), 0.5)
    with pytest.raises(ValueError, match=r"expected \(\*, 3\) features"):
        m.scores(np.ones((4, 2)))


def test_topk_mean_k_gt_n_and_k_le_zero_clip() -> None:
    y = np.asarray([1.0, 3.0, 2.0, 0.0])
    scores = np.asarray([0.1, 0.9, 0.5, 0.2])
    # k>n → all names; top by scores is index 1 (3.0)
    assert _topk_mean(y, scores, k=99) == pytest.approx(float(np.mean(y)))
    # k<=0 → clips to 1 → best score only
    assert _topk_mean(y, scores, k=0) == pytest.approx(3.0)
    assert _topk_mean(y, scores, k=-5) == pytest.approx(3.0)


def test_run_panel_empty_yields_empty_trace() -> None:
    x = np.zeros((0, 3), dtype=float)
    y = np.zeros(0, dtype=float)
    dates = np.asarray([], dtype=object)
    trace = run_linucb_panel(x, y, dates, top_k=2)
    assert trace.dates == []
    assert trace.policy_reward.size == 0
    assert trace.oracle_reward.size == 0
    assert trace.uniform_reward.size == 0
    assert trace.cumulative_regret.size == 0


def test_run_panel_skips_undersized_dates() -> None:
    # 3 dates × 2 names; top_k=2 → need >=4 finite → all skipped
    dates = np.repeat(np.arange(3), 2)
    x = np.ones((6, 2), dtype=float)
    y = np.linspace(0.0, 1.0, 6)
    trace = run_linucb_panel(x, y, dates, top_k=2)
    assert trace.dates == []
    assert trace.policy_reward.size == 0


def test_run_panel_length_mismatch_raises() -> None:
    x = np.ones((6, 2), dtype=float)
    y = np.ones(5, dtype=float)
    dates = np.repeat(np.arange(3), 2)
    with pytest.raises(ValueError, match="y length"):
        run_linucb_panel(x, y, dates)
    with pytest.raises(ValueError, match="dates length"):
        run_linucb_panel(x, np.ones(6), np.arange(5))
    with pytest.raises(ValueError, match="oracle length"):
        run_linucb_panel(x, np.ones(6), dates, oracle=np.ones(4))
    with pytest.raises(ValueError, match="2-d"):
        run_linucb_panel(np.ones(6), np.ones(6), dates)


def test_run_panel_bad_alpha_propagates() -> None:
    dates = np.repeat(np.arange(5), 8)
    x = np.ones((40, 2), dtype=float)
    y = np.linspace(0.0, 1.0, 40)
    with pytest.raises(ValueError, match="alpha"):
        run_linucb_panel(x, y, dates, top_k=2, alpha=float("nan"))


def test_topk_k_gt_n_in_panel_runs() -> None:
    """top_k larger than names clips inside _topk_mean; dates still need size gate."""
    rng = np.random.default_rng(3)
    n_dates, n_names, d = 6, 8, 2
    dates = np.repeat(np.arange(n_dates), n_names)
    x = rng.normal(size=(n_dates * n_names, d))
    y = x[:, 0] + 0.05 * rng.normal(size=x.shape[0])
    # top_k=99 > n_names=8 but max(top_k*2,4)=198 > 8 → all dates skipped
    skipped = run_linucb_panel(x, y, dates, top_k=99)
    assert skipped.dates == []
    # top_k=5 with 8 names → keep (need >=10? max(10,4)=10 > 8 → still skip)
    # need n_names >= max(2*k, 4); for k=3 need >=6 — use k=3 with 8 names
    trace = run_linucb_panel(x, y, dates, top_k=3, alpha=0.5, seed=3)
    assert len(trace.dates) == n_dates
    assert np.isfinite(trace.policy_reward).all()


def test_module_and_trace_have_no_sharpe_keys() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "src/quant_fund/models/rl.py").read_text(encoding="utf-8")
    assert "sharpe" not in text.lower()
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(8), 6)
    x = rng.normal(size=(48, 2))
    y = x[:, 0] + 0.1 * rng.normal(size=48)
    trace = run_linucb_panel(x, y, dates, top_k=2, alpha=0.5, seed=0)
    keys = set(trace.__dataclass_fields__)
    assert all("sharpe" not in k.lower() for k in keys)
    assert all(k.lower() not in {"sortino", "calmar", "pnl", "nav"} for k in keys)
