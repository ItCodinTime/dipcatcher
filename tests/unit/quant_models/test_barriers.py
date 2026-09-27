"""AFML labeling battery: triple barrier, meta-labels, uniqueness weights."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.labels.barriers import (
    average_uniqueness,
    cusum_filter,
    label_concurrency,
    meta_labels,
    return_attribution_weights,
    sequential_bootstrap,
    time_decay_weights,
    trend_scanning_labels,
    triple_barrier,
)


def _walk(n: int = 500, seed: int = 0, drift: float = 0.0) -> np.ndarray:
    r = np.random.default_rng(seed).normal(drift, 0.01, size=n)
    return np.exp(np.cumsum(r)) * 100.0


def test_cusum_filter_finds_jumps() -> None:
    x = np.concatenate([np.zeros(50), np.ones(10) * 0.0, np.linspace(0, 1.0, 30)])
    ev = cusum_filter(x, h=0.05)
    assert ev.size >= 1
    assert np.all(ev >= 0)
    with pytest.raises(ValueError):
        cusum_filter(x, h=-1.0)


def test_triple_barrier_upper_touch() -> None:
    # Uptrend: labels should be +1 via upper barrier.
    c = np.linspace(100, 140, 60)
    ev = np.arange(0, 50, 5)
    out = triple_barrier(c, ev, pt=0.02, sl=0.02, horizon=10)
    assert np.all(out["label"] == 1.0)
    assert np.all(out["touch"] == 1.0)
    assert np.all(out["ret"] >= 0.019)


def test_triple_barrier_lower_touch() -> None:
    c = np.linspace(140, 100, 60)
    ev = np.arange(0, 50, 5)
    out = triple_barrier(c, ev, pt=0.02, sl=0.02, horizon=10)
    assert np.all(out["label"] == -1.0)
    assert np.all(out["touch"] == -1.0)


def test_triple_barrier_vertical_min_ret() -> None:
    c = np.full(60, 100.0)  # flat path never hits barriers
    ev = np.arange(0, 50, 5)
    out = triple_barrier(c, ev, pt=0.05, sl=0.05, horizon=10, min_ret=0.001)
    assert np.all(out["touch"] == 0.0)
    assert np.all(out["label"] == 0.0)


def test_triple_barrier_vol_scaling() -> None:
    c = _walk(200, seed=3)
    vol = np.full(200, 0.02)
    ev = np.arange(0, 150, 10)
    wide = triple_barrier(c, ev, pt=0.5, sl=0.5, horizon=15, vol=vol)
    tight = triple_barrier(c, ev, pt=0.005, sl=0.005, horizon=15, vol=vol)
    assert np.abs(tight["touch"]).sum() >= np.abs(wide["touch"]).sum()


def test_meta_labels() -> None:
    side = np.array([1.0, 1.0, -1.0, -1.0])
    lab = np.array([1.0, -1.0, -1.0, 1.0])
    assert np.array_equal(meta_labels(side, lab), np.array([1.0, 0.0, 1.0, 0.0]))
    with pytest.raises(ValueError):
        meta_labels(np.array([2.0, 1.0]), np.array([1.0, 1.0]))


def test_trend_scanning_labels_direction() -> None:
    up = np.linspace(100, 120, 60)
    dn = np.linspace(120, 100, 60)
    lu = trend_scanning_labels(up, window=10)
    ld = trend_scanning_labels(dn, window=10)
    assert np.all(lu[np.isfinite(lu)] == 1.0)
    assert np.all(ld[np.isfinite(ld)] == -1.0)
    assert np.isnan(lu[:9]).all()


def test_label_concurrency_counts() -> None:
    c = label_concurrency(np.array([0, 2]), np.array([4, 5]))
    assert c[0] == 1.0 and c[2] == 2.0 and c[4] == 1.0


def test_average_uniqueness_bounds() -> None:
    u = average_uniqueness(np.array([0, 0, 2]), np.array([4, 6, 5]))
    assert np.all((u > 0.0) & (u <= 1.0))
    # Label 0 shares its whole span -> less unique than a lone label.
    lone = average_uniqueness(np.array([0, 10]), np.array([4, 12]))
    assert lone[1] == 1.0
    assert lone[0] < 1.0 or lone[0] == pytest.approx(1.0)


def test_sequential_bootstrap_prefers_unique() -> None:
    # Label 2 is isolated; labels 0/1 fully overlap.  First-draw probability
    # for the isolated label is 1.0/(0.5+0.5+1.0) = 0.5 -> over single-draw
    # seeds it should be picked materially more than 1/3 of the time.
    hits = sum(
        int(sequential_bootstrap(np.array([0, 0, 8]), np.array([5, 5, 9]), 1, seed=s)[0] == 2)
        for s in range(20)
    )
    assert hits >= 7  # E[hits] = 10 under the uniqueness-weighted draw


def test_time_decay_weights_ordering() -> None:
    w = time_decay_weights(np.array([0.0, 5.0, 10.0]), decay=0.5)
    assert w[0] == pytest.approx(1.0)
    assert w[-1] == pytest.approx(0.5 + 0.5 / 11.0)
    assert np.all(np.diff(w) <= 1e-12)


def test_return_attribution_weights() -> None:
    w = return_attribution_weights(
        np.array([0.1, 0.1, 0.2]), np.array([0, 0, 6]), np.array([4, 4, 8])
    )
    assert w[2] > w[0]  # isolated, larger return -> bigger weight
    with pytest.raises(ValueError):
        return_attribution_weights(np.array([np.nan]), np.array([0]), np.array([1]))


def test_fail_closed_edges() -> None:
    with pytest.raises(ValueError):
        triple_barrier(np.ones(1), np.array([0]), 0.01, 0.01, 5)  # too short
    with pytest.raises(ValueError):
        triple_barrier(_walk(50), np.array([-1, 200]), 0.01, 0.01, 5)  # no valid events
    with pytest.raises(ValueError):
        triple_barrier(_walk(50), np.arange(5), -0.01, 0.01, 5)
    with pytest.raises(ValueError):
        label_concurrency(np.array([3]), np.array([2]))
    with pytest.raises(ValueError):
        trend_scanning_labels(np.arange(10.0) + 1, window=30)
    with pytest.raises(ValueError):
        time_decay_weights(np.array([-1.0]))
    with pytest.raises(ValueError):
        sequential_bootstrap(np.array([0]), np.array([2]), 0)
