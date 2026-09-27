"""Attribution drift over time-ordered blocks (SYNTHETIC labels)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.explainability import attribution_drift
from quant_fund.research.explainability.drift import _normalized_shares, js_divergence

from .conftest import LinearHead


def _regime_shift_data(seed: int = 77, n_rows: int = 800):
    """Feature relevance switches halfway: x0 drives early y, x1 drives late y."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, 4))
    half = n_rows // 2
    y = np.empty(n_rows)
    noise = rng.normal(scale=0.3, size=n_rows)
    y[:half] = 3.0 * x[:half, 0] + noise[:half]
    y[half:] = 3.0 * x[half:, 1] + noise[half:]
    model = LinearHead(list("abcd")).fit(x, y)
    return model, x, y


def test_planted_regime_shift_is_flagged() -> None:
    model, x, y = _regime_shift_data()
    names = list("abcd")
    drift = attribution_drift(
        model.predict, x, y, feature_names=names, n_blocks=4, n_repeats=4, seed=5
    )
    assert drift.drift_flagged is True
    assert drift.n_blocks == 4
    assert len(drift.consecutive_spearman) == 3
    assert len(drift.consecutive_js) == 3
    # The relevance switch sits at the block boundary: mid-pair Spearman collapses.
    assert drift.consecutive_spearman[1] < 0.0
    # First two blocks agree on x0; last two on x1.
    block0_top = drift.blocks[0].result.attributions[0].feature
    block3_top = drift.blocks[3].result.attributions[0].feature
    assert block0_top == "a"
    assert block3_top == "b"
    # Drift statistics stay inside their mathematical bounds.
    assert all(0.0 <= v <= np.log(2) + 1e-9 for v in drift.consecutive_js)
    assert all(-1.0 <= v <= 1.0 for v in drift.consecutive_spearman)
    assert 0.0 <= drift.mean_topk_overlap <= 1.0


def test_stationary_attribution_is_not_flagged() -> None:
    """Tiered informative weights give a stable block ranking — no drift."""
    rng = np.random.default_rng(31)
    n_rows = 900
    x = rng.normal(size=(n_rows, 6))
    y = 3.0 * x[:, 0] + 2.0 * x[:, 1] + 1.2 * x[:, 2] + rng.normal(scale=0.3, size=n_rows)
    model = LinearHead(list("abcdef")).fit(x, y)
    drift = attribution_drift(
        model.predict,
        x,
        y,
        feature_names=list("abcdef"),
        n_blocks=3,
        n_repeats=5,
        seed=6,
    )
    assert drift.drift_flagged is False
    assert drift.min_spearman > 0.6


def test_drift_is_deterministic_under_seed() -> None:
    model, x, y = _regime_shift_data()
    names = list("abcd")
    first = attribution_drift(
        model.predict, x, y, feature_names=names, n_blocks=4, n_repeats=3, seed=9
    )
    second = attribution_drift(
        model.predict, x, y, feature_names=names, n_blocks=4, n_repeats=3, seed=9
    )
    assert first.consecutive_spearman == second.consecutive_spearman
    assert first.consecutive_js == second.consecutive_js
    for a, b in zip(first.blocks, second.blocks, strict=True):
        assert [r.feature for r in a.result.attributions] == [
            r.feature for r in b.result.attributions
        ]


def test_drift_respects_time_ordering() -> None:
    """Rows are sorted by `times` before blocking — scrambled input still splits correctly."""
    model, x, y = _regime_shift_data()
    n_rows = y.shape[0]
    rng = np.random.default_rng(3)
    perm = rng.permutation(n_rows)
    times = np.arange(n_rows)  # row identity = chronological rank
    drift = attribution_drift(
        model.predict,
        x[perm],
        y[perm],
        times=times[perm],
        feature_names=list("abcd"),
        n_blocks=4,
        n_repeats=3,
        seed=5,
    )
    # After sorting, block 0 covers early rows (x0-driven), block 3 late rows (x1-driven).
    assert drift.blocks[0].result.attributions[0].feature == "a"
    assert drift.blocks[3].result.attributions[0].feature == "b"
    assert drift.drift_flagged is True


def test_js_divergence_bounds_and_identity() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.uniform(0.0, 10.0, size=8)
        q = rng.uniform(0.0, 10.0, size=8)
        value = js_divergence(p, q)
        assert 0.0 <= value <= np.log(2) + 1e-9
        assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_normalized_shares_sum_to_one() -> None:
    shares = _normalized_shares(np.asarray([3.0, 1.0, 0.0, -2.0]))
    assert shares.sum() == pytest.approx(1.0)
    assert shares[0] > shares[1] > shares[2]
    zero = _normalized_shares(np.zeros(5))
    assert zero.sum() == pytest.approx(1.0)


def test_drift_input_validation() -> None:
    model, x, y = _regime_shift_data()
    with pytest.raises(ValueError, match="n_blocks"):
        attribution_drift(model.predict, x, y, n_blocks=1)
    with pytest.raises(ValueError, match="blocks"):
        attribution_drift(model.predict, x[:3], y[:3], n_blocks=8)
    with pytest.raises(ValueError, match="times length"):
        attribution_drift(model.predict, x, y, times=np.arange(10), n_blocks=2)
    with pytest.raises(ValueError, match="top_k"):
        attribution_drift(model.predict, x, y, n_blocks=2, top_k=0)
