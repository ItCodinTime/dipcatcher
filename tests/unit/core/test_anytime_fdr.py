"""Tests for anytime_fdr: e-BH, stopped e-BH, and e-LOND.

All e-values are product likelihood-ratio e-variables for a fair coin
(H0: p = 0.5 vs the fixed alternative p = alt): each flip contributes
e_step = alt/0.5 if heads, (1-alt)/0.5 if tails, with E_H0[e_step] = 1.
Every random draw uses a seeded np.random.default_rng, so runs are
deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from quant_fund.metrics.anytime_fdr import (
    ELond,
    e_bh,
    stopped_e_bh,
)

Array = NDArray[np.float64]

SEED = 20260927


def _coin_e_values(
    rng: np.random.Generator,
    n: int,
    m: int,
    alt: float = 0.75,
    *,
    true_p: float = 0.5,
    n_shared: int = 0,
) -> Array:
    """n product e-values, each from m fair (or true_p) coin flips.

    ``n_shared`` leading flips are shared across all n rows to induce
    arbitrary cross-hypothesis dependence.
    """
    flips = rng.random((n, m)) < true_p
    if n_shared > 0:
        flips[:, :n_shared] = (rng.random(n_shared) < true_p)[None, :]
    log_step = np.where(flips, np.log(alt / 0.5), np.log((1.0 - alt) / 0.5))
    return np.exp(log_step.sum(axis=1))


def _coin_e_paths(
    rng: np.random.Generator,
    n_hyp: int,
    n_time: int,
    alt: float = 0.75,
    *,
    n_shared: int = 0,
) -> Array:
    """(n_hyp, n_time) e-process paths under the global null (true p = 0.5)."""
    flips = rng.random((n_hyp, n_time)) < 0.5
    if n_shared > 0:
        flips[:, :n_shared] = (rng.random(n_shared) < 0.5)[None, :]
    log_step = np.where(flips, np.log(alt / 0.5), np.log((1.0 - alt) / 0.5))
    return np.exp(np.cumsum(log_step, axis=1))


def _fdr_global_null(reject_counts: list[int], reps: int) -> float:
    # Under the global null every rejection is false, so FDR = P(R > 0).
    return float(np.mean([c > 0 for c in reject_counts])) if reps > 0 else 0.0


# ---------------------------------------------------------------- e-BH ----


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_ebh_fdr_global_null_independent(alpha: float) -> None:
    rng = np.random.default_rng(SEED)
    n, m, reps = 40, 25, 300
    counts = [e_bh(_coin_e_values(rng, n, m), alpha).num_rejected for _ in range(reps)]
    assert _fdr_global_null(counts, reps) <= alpha + 0.02


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_ebh_fdr_global_null_dependent(alpha: float) -> None:
    rng = np.random.default_rng(SEED + 1)
    n, m, reps = 40, 25, 300
    counts = [e_bh(_coin_e_values(rng, n, m, n_shared=12), alpha).num_rejected for _ in range(reps)]
    assert _fdr_global_null(counts, reps) <= alpha + 0.02


def test_ebh_deterministic_threshold() -> None:
    # n=4, alpha=0.05: e_(1)=100 >= 4/0.05=80 but e_(2)=1 < 4/(0.05*2)=40.
    res = e_bh(np.array([100.0, 1.0, 1.0, 1.0]), 0.05)
    assert res.num_rejected == 1
    assert res.critical_value == pytest.approx(80.0)
    assert res.rejected.tolist() == [True, False, False, False]
    # No k satisfies e_(k) >= n/(alpha*k): nothing rejected.
    res0 = e_bh(np.array([20.0, 1.0, 1.0, 1.0]), 0.05)
    assert res0.num_rejected == 0
    assert res0.critical_value == float("inf")
    assert not bool(np.any(res0.rejected))


def test_ebh_power_beats_bonferroni() -> None:
    rng = np.random.default_rng(SEED + 2)
    n, m_null, m_sig = 40, 25, 20
    e_null = _coin_e_values(rng, n // 2, m_null)
    e_sig = _coin_e_values(rng, n // 2, m_sig, alt=0.8, true_p=0.8)
    e = np.concatenate([e_null, e_sig])
    alpha = 0.05
    res = e_bh(e, alpha)
    # Bonferroni on inverted e-values: reject iff e_i >= n/alpha (valid at level alpha).
    bonferroni = int(np.sum(e >= n / alpha))
    assert res.num_rejected > bonferroni
    assert res.num_rejected >= bonferroni + 3
    # All rejections land on the signal half.
    assert not bool(np.any(res.rejected[: n // 2]))


@pytest.mark.parametrize(
    "e_values, alpha",
    [
        ([], 0.05),
        ([1.0, 2.0], 0.0),
        ([1.0, 2.0], 1.0),
        ([1.0, 2.0], -0.1),
        ([1.0, 2.0], 1.5),
        ([1.0, 2.0], float("nan")),
        ([1.0, float("nan")], 0.05),
        ([1.0, float("inf")], 0.05),
        ([-1.0, 2.0], 0.05),
    ],
)
def test_ebh_fail_closed(e_values: list[float], alpha: float) -> None:
    with pytest.raises(ValueError):
        e_bh(e_values, alpha)


# --------------------------------------------------------- stopped e-BH ----


def test_stopped_ebh_fdr_global_null_independent() -> None:
    rng = np.random.default_rng(SEED + 3)
    n_hyp, n_time, reps, alpha, boundary = 30, 50, 200, 0.05, 10.0
    counts = []
    for _ in range(reps):
        paths = _coin_e_paths(rng, n_hyp, n_time)
        tau = np.minimum(
            np.argmax(paths >= boundary, axis=1), n_time - 1
        )  # first crossing, else last time
        counts.append(stopped_e_bh(paths, alpha, stop_times=tau).num_rejected)
    assert _fdr_global_null(counts, reps) <= alpha + 0.02


def test_stopped_ebh_fdr_global_null_dependent_stop() -> None:
    rng = np.random.default_rng(SEED + 4)
    n_hyp, n_time, reps, alpha, boundary = 30, 50, 200, 0.05, 10.0
    counts = []
    for _ in range(reps):
        # Shared flips across streams: stop rules depend on common data, so the
        # stopping times are global; only the lifted (correction) rule is valid.
        paths = _coin_e_paths(rng, n_hyp, n_time, n_shared=25)
        tau = np.minimum(np.argmax(paths >= boundary, axis=1), n_time - 1)
        counts.append(stopped_e_bh(paths, alpha, stop_times=tau).num_rejected)
    assert _fdr_global_null(counts, reps) <= alpha + 0.02


def test_stopped_ebh_lifting_and_default_stop() -> None:
    rng = np.random.default_rng(SEED + 5)
    paths = _coin_e_paths(rng, 6, 30)
    res = stopped_e_bh(paths, 0.05)  # default: stop all at the last time
    assert res.stop_times.tolist() == [29] * 6
    # Lifted stopped value = k * (running max)^(1-k) with k = 0.5, from max(., 1).
    expect = 0.5 * np.sqrt(np.maximum(paths.max(axis=1), 1.0))
    np.testing.assert_allclose(res.stopped_e, expect, rtol=1e-12)
    # Consistency with plain e-BH on the lifted terminal values.
    direct = e_bh(res.stopped_e, 0.05)
    assert direct.num_rejected == res.num_rejected
    assert direct.critical_value == pytest.approx(res.critical_value)


def test_stopped_ebh_rejects_strong_stream() -> None:
    rng = np.random.default_rng(SEED + 6)
    paths = _coin_e_paths(rng, 8, 20)
    huge = np.full((1, 20), np.exp(30.0))  # one e-process pinned at e^30
    paths = np.concatenate([paths, huge], axis=0)
    res = stopped_e_bh(paths, 0.05)
    assert bool(res.rejected[-1])
    assert int(res.rejected[:-1].sum()) == 0


@pytest.mark.parametrize(
    "paths, alpha, tau",
    [
        (np.zeros((0, 5)), 0.05, None),  # empty
        (np.zeros(4), 0.05, None),  # 1-D
        (np.array([[1.0, 2.0], [3.0, 4.0]]), 0.0, None),  # bad alpha
        (np.array([[1.0, 2.0], [3.0, 4.0]]), 0.05, [0, 5]),  # tau too large
        (np.array([[1.0, 2.0], [3.0, 4.0]]), 0.05, [-1, 0]),  # negative tau
        (np.array([[1.0, 2.0], [3.0, 4.0]]), 0.05, [0]),  # tau wrong length
        (np.array([[1.0, float("nan")]]), 0.05, None),  # NaN path
        (np.array([[-1.0, 2.0]]), 0.05, None),  # negative e-value
        (np.array([[1.0, 2.0]]), 0.05, [0.5]),  # non-integer tau
    ],
)
def test_stopped_ebh_fail_closed(paths: Array, alpha: float, tau: list[int] | None) -> None:
    with pytest.raises(ValueError):
        stopped_e_bh(paths, alpha, stop_times=tau)


# --------------------------------------------------------------- e-LOND ----


def test_elond_fdr_global_null() -> None:
    rng = np.random.default_rng(SEED + 7)
    stream_len, m, reps, alpha = 60, 15, 300, 0.05
    any_rej = []
    min_wealth = float("inf")
    for _ in range(reps):
        proc = ELond(alpha)
        for _ in range(stream_len):
            proc.submit(float(_coin_e_values(rng, 1, m)[0]))
            min_wealth = min(min_wealth, proc.wealth)
        any_rej.append(proc.num_rejected > 0)
    assert float(np.mean(any_rej)) <= alpha + 0.02
    assert min_wealth >= 0.0


def test_elond_rejects_signal_stream() -> None:
    rng = np.random.default_rng(SEED + 8)
    stream_len, m_null, m_sig = 40, 15, 60
    signal_idx = {i for i in range(stream_len) if i % 4 == 3}
    proc = ELond(0.05)
    rejected_idx: set[int] = set()
    for j in range(stream_len):
        if j in signal_idx:
            e = float(_coin_e_values(rng, 1, m_sig, alt=0.85, true_p=0.85)[0])
            e = max(e, 1e6)  # strong signal: never below the e-LOND threshold
        else:
            e = float(_coin_e_values(rng, 1, m_null)[0])
        if proc.submit(e):
            rejected_idx.add(j)
    assert signal_idx <= rejected_idx
    assert proc.num_rejected >= len(signal_idx)


def test_elond_wealth_grows_and_levels_tracked() -> None:
    proc = ELond(0.05)
    assert proc.wealth == 1.0
    proc.submit(1e9)  # enormous e-value: first hypothesis is rejected
    assert proc.wealth == 2.0
    assert proc.num_rejected == 1
    gamma_1 = 0.07 * np.log(2.0) / (2.0 * np.exp(np.sqrt(np.log(2.0))))
    assert proc.levels[0] == pytest.approx(0.05 * gamma_1)
    assert proc.rejections == (True,)
    assert proc.num_submitted == 1


def test_elond_preloaded_constructor() -> None:
    rng = np.random.default_rng(SEED + 9)
    e = _coin_e_values(rng, 10, 15)
    proc = ELond(0.05, e_values=e)
    assert proc.num_submitted == 10
    assert proc.rejections == tuple(x >= 1.0 / lv for x, lv in zip(e, proc.levels, strict=True))


def test_elond_gamma_sequence_override() -> None:
    rng = np.random.default_rng(SEED + 10)
    e = _coin_e_values(rng, 6, 10)
    proc = ELond(0.05, e_values=e, gamma=[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625])
    assert proc.num_submitted == 6
    assert proc.levels[0] == pytest.approx(0.05 * 0.5)


@pytest.mark.parametrize(
    "alpha, gamma, e_seq",
    [
        (0.0, None, []),  # alpha too small
        (1.0, None, []),  # alpha too large
        (0.05, [1.0, -1.0], [1.0, 1.0]),  # negative gamma entry (raised lazily)
        (0.05, [], []),  # empty gamma sequence
        (0.05, None, [-1.0]),  # negative e-value
        (0.05, None, [float("nan")]),  # NaN e-value
        (0.05, None, [float("inf")]),  # infinite e-value
    ],
)
def test_elond_fail_closed(alpha: float, gamma: list[float] | None, e_seq: list[float]) -> None:
    with pytest.raises(ValueError):
        proc = ELond(alpha, gamma=gamma)
        for e in e_seq:
            proc.submit(e)
