"""P6.2 statistical-core audit: known-answer tests.

Deterministic, offline, SYNTHETIC fixtures only — research-diagnostic
machinery, never a live Sharpe / P&L claim. Each test pins either a
hand-computed value (closed form / reference implementation) or an exact
structural identity of the cited method.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from arch.bootstrap import optimal_block_length as arch_optimal_block_length
from scipy import stats

from quant_fund.metrics.bootstrap import (
    circular_block_indices as cb_indices_matrix,
)
from quant_fund.metrics.bootstrap import (
    pairs_bootstrap,
    sieve_bootstrap_residuals,
    subsample_statistic,
    wild_bootstrap_residuals,
)
from quant_fund.metrics.calibration_tests import spiegelhalter_z
from quant_fund.metrics.inference import (
    benjamini_hochberg,
    bootstrap_mean_ci,
    bootstrap_sharpe_ci,
    circular_block_indices,
    diebold_mariano,
    grouped_mean_tstat,
    jobson_korkie_memmel,
    mean_difference_t,
    mean_tstat,
    newey_west_variance,
    onesided_from_twosided,
    optimal_block_length,
    overlap_aware_hac_lags,
    stationary_bootstrap_indices,
    two_proportion_test,
    two_way_clustered_mean_tstat,
)
from quant_fund.metrics.scoring import (
    coverage,
    crps_empirical,
    crps_from_quantiles,
    crps_gaussian,
    crps_gaussian_mixture,
    crps_student_t,
    gaussian_mixture_quantiles,
    icir,
    interval_width,
    mean_fissler_ziegel,
    mean_pinball,
    pearson_ic,
    pinball_loss,
    pit_values,
    qlike,
    quantile_crossing_rate,
    rank_ic,
    rearrange_quantiles,
)
from quant_fund.metrics.snooping import (
    model_confidence_set,
    reality_check,
    spa_test,
    stepm,
)


def _ar1(phi: float, n: int, seed: int) -> np.ndarray:
    e = np.random.default_rng(seed).normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


# --- HAC / t-stat core -------------------------------------------------------


def test_newey_west_variance_known_answer() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])  # mean 2.5, gamma0 = 1.25, gamma1 = 0.3125
    assert newey_west_variance(x, lags=0) == pytest.approx(0.3125)
    # omega = 1.25 + 2*(1 - 1/2)*0.3125 = 1.5625 -> /n
    assert newey_west_variance(x, lags=1) == pytest.approx(0.390625)


def test_mean_tstat_known_answer() -> None:
    mu, t, p = mean_tstat(np.array([1.0, 2.0, 3.0, 4.0]), lags=0)
    assert mu == pytest.approx(2.5)
    assert t == pytest.approx(2.5 / math.sqrt(0.3125))  # 4.47213595
    assert p == pytest.approx(2.0 * stats.t.sf(2.5 / math.sqrt(0.3125), df=3))


def test_grouped_mean_tstat_known_answer() -> None:
    mu, t, p, g = grouped_mean_tstat(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), np.array([0, 0, 1, 1, 2, 2]), lags=0
    )
    assert g == 3
    assert mu == pytest.approx(3.5)
    # group means [1.5, 3.5, 5.5]: gamma0 = 8/3, var = (8/3)/3 = 8/9
    assert t == pytest.approx(3.5 / math.sqrt(8.0 / 9.0))
    assert p == pytest.approx(2.0 * stats.t.sf(3.5 / math.sqrt(8.0 / 9.0), df=2))


def test_overlap_aware_hac_lags_known_answer() -> None:
    # n = 100 -> default = floor(1.5 * 100^(1/3)) = floor(6.96) = 6; horizon
    # 10 -> h-1 = 9 wins
    assert overlap_aware_hac_lags(100, 1) == 6
    assert overlap_aware_hac_lags(100, 10) == 9
    assert overlap_aware_hac_lags(2, 5) == 4  # short n still floors at h-1


def test_two_way_clustered_mean_tstat_known_answer() -> None:
    # Hand-computed CGM sandwich on a singleton-cell panel:
    # resid = [-2.5,-1.5,-0.5,0.5,1.5,2.5]
    # V_a = (4.5^2+4.5^2)/36 = 1.125; V_b = (2^2+0+2^2)/36 = 2/9;
    # V_white = 17.5/36 -> V = 31/36; df = min(2,3)-1 = 1.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = np.array([0, 0, 0, 1, 1, 1])
    b = np.array([0, 1, 2, 0, 1, 2])
    mu, t, p, n, na, nb = two_way_clustered_mean_tstat(y, a, b)
    assert (mu, n, na, nb) == (pytest.approx(3.5), 6, 2, 3)
    assert t == pytest.approx(3.5 / math.sqrt(31.0 / 36.0))
    assert p == pytest.approx(2.0 * stats.t.sf(3.5 / math.sqrt(31.0 / 36.0), df=1))


def test_diebold_mariano_known_answers() -> None:
    d = np.array([1.0, -1.0, 2.0, 0.0])
    res = diebold_mariano(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]) - d, lags=0
    )
    # d has mean 0.5, gamma0 = 1.25 -> se = sqrt(0.3125)
    assert res.mean_loss_diff == pytest.approx(0.5)
    assert res.statistic == pytest.approx(0.5 / math.sqrt(0.3125))
    assert res.p_value == pytest.approx(2.0 * stats.t.sf(0.5 / math.sqrt(0.3125), df=3))
    assert res.preferred == "b"  # positive diff prefers the second series
    tie = diebold_mariano(np.array([1.0, -1.0, 2.0, -2.0]), np.zeros(4), lags=0)
    assert tie.statistic == 0.0 and tie.p_value == 1.0 and tie.preferred == "tie"
    worse = diebold_mariano(np.full(30, 0.0), np.random.default_rng(0).normal(1.0, 0.2, 30), lags=0)
    assert worse.preferred == "a"


def test_benjamini_hochberg_known_answer() -> None:
    reject, cutoff = benjamini_hochberg(np.array([0.001, 0.02, 0.04, 0.5, 0.9]), alpha=0.05)
    # thresholds alpha*i/5 = [0.01, 0.02, 0.03, 0.04, 0.05]; largest passing rank is 2.
    assert reject.tolist() == [True, True, False, False, False]
    assert cutoff == pytest.approx(0.02)


def test_jobson_korkie_memmel_known_answer() -> None:
    # denom = (1/100)[2(1-0.5) + 0.5(1 + 0.25 - 2*0.25*0.5)] = 0.015
    theta, p = jobson_korkie_memmel(1.0, 0.5, 100, 0.5)
    assert theta == pytest.approx(0.5 / math.sqrt(0.015))
    assert p == pytest.approx(2.0 * stats.norm.sf(abs(theta)))


def test_two_proportion_and_onesided_known_answers() -> None:
    z, p = two_proportion_test(40, 100, 20, 100)
    assert z == pytest.approx(0.2 / math.sqrt(0.3 * 0.7 * 0.02))
    assert p == pytest.approx(2.0 * stats.norm.sf(abs(z)))
    assert onesided_from_twosided(2.0, 0.1, greater=True) == pytest.approx(0.05)
    assert onesided_from_twosided(-2.0, 0.1, greater=True) == pytest.approx(0.95)
    assert onesided_from_twosided(-2.0, 0.1, greater=False) == pytest.approx(0.05)


def test_mean_difference_t_known_answer() -> None:
    t, p = mean_difference_t(1.0, 100, sd=2.0)
    assert t == pytest.approx(5.0)
    assert p == pytest.approx(2.0 * stats.t.sf(5.0, df=99))


# --- Block-length selection (PW2004) and bootstrap resampling ----------------


def test_optimal_block_length_matches_pw2004_reference() -> None:
    """Regression test for the c_i=2 stationary-bootstrap constant and the
    K_N/m_max/B_max rules: must match ``arch.bootstrap.optimal_block_length``
    (the canonical PW2004 implementation) whenever the value is >= 1."""
    for seed in range(3):
        for n, phi in [(500, 0.5), (2000, 0.2), (4000, 0.85), (4000, -0.3)]:
            x = _ar1(phi, n, seed)
            ref = float(arch_optimal_block_length(x)["stationary"].iloc[0])
            got = optimal_block_length(x)
            if ref >= 1.0:
                assert got == pytest.approx(ref, rel=1e-9, abs=1e-9)


def test_optimal_block_length_negative_dependence_not_floored_to_one() -> None:
    """Negative autocorrelation still implies dependent data: the PW2004
    optimum uses G^2, so a dominant negative rho must NOT collapse to 1.0."""
    x = _ar1(-0.3, 4000, 5)
    ref = float(arch_optimal_block_length(x)["stationary"].iloc[0])
    assert ref > 3.0  # sanity on the reference itself
    assert optimal_block_length(x) == pytest.approx(ref, rel=1e-9, abs=1e-9)


def test_optimal_block_length_edges_unchanged() -> None:
    assert optimal_block_length(np.ones(500)) == 1.0
    assert np.isnan(optimal_block_length(np.arange(5.0)))
    assert np.isnan(optimal_block_length(np.array([])))
    # Bounded by B_max = ceil(min(3 sqrt n, n/3))
    x = _ar1(0.98, 500, 9)
    assert optimal_block_length(x) <= math.ceil(min(3.0 * math.sqrt(500), 500 / 3.0))


def test_circular_block_indices_known_answer() -> None:
    # block == n -> each draw is a cyclic shift containing every index once
    idx = circular_block_indices(6, 6, np.random.default_rng(3))
    assert np.array_equal(np.sort(idx), np.arange(6))
    mat = cb_indices_matrix(6, 6, 3, seed=3)
    assert mat.shape == (3, 6)
    for row in mat:
        assert np.array_equal(np.sort(row), np.arange(6))


def test_stationary_bootstrap_unbiased_and_uniform_marginal() -> None:
    # Stationarity: E[f(bar)*] = f(bar) exactly; block=1 -> iid resampling.
    n, n_boot = 400, 2000
    rng = np.random.default_rng(11)
    x = rng.normal(2.0, 1.0, size=n)
    idx = stationary_bootstrap_indices(n, n_boot, 1.0, np.random.default_rng(5))
    assert idx.shape == (n_boot, n)
    resampled_mean = x[idx].mean(axis=1).mean()
    mc_se = float(x.std(ddof=1) / math.sqrt(n * n_boot))
    assert abs(float(resampled_mean) - float(x.mean())) < 5.0 * mc_se
    # marginal frequency of any index -> 1/n
    freq = float(np.mean(idx == 0))
    assert abs(freq - 1.0 / n) < 5.0 * math.sqrt(1.0 / n / (n * n_boot))


def test_bootstrap_mean_ci_known_answer() -> None:
    lo, hi, mu = bootstrap_mean_ci(np.full(10, 2.5), n_boot=50, block=2, seed=0)
    assert (lo, hi, mu) == (2.5, 2.5, 2.5)
    x = np.arange(50.0)
    lo2, hi2, mu2 = bootstrap_mean_ci(x, n_boot=400, block=1, seed=3)
    assert lo2 <= mu2 <= hi2 and mu2 == pytest.approx(float(x.mean()))
    # iid resample (block=1) CI width matches the CLT scale 2*sd/sqrt(n)
    width = hi2 - lo2
    assert width == pytest.approx(2.0 * 1.96 * float(x.std(ddof=1)) / math.sqrt(50), rel=0.35)


def test_bootstrap_sharpe_ci_known_answer() -> None:
    # Constant series: Sharpe is undefined; must fail closed to NaN, never
    # mint an astronomical value from float noise in np.std.
    lo, hi, pt = bootstrap_sharpe_ci(np.full(20, 0.01), n_boot=50, block=2, seed=0)
    assert np.isnan(lo) and np.isnan(hi) and np.isnan(pt)
    x = np.random.default_rng(0).normal(0.5, 1.0, size=200)
    _, _, pt2 = bootstrap_sharpe_ci(x, n_boot=100, seed=4)
    assert pt2 == pytest.approx(float(x.mean() / x.std(ddof=1) * math.sqrt(252)))


def test_subsample_statistic_known_answer() -> None:
    out = subsample_statistic(np.arange(10.0), np.mean, 5, overlapping=True)
    assert np.array_equal(out["sub_stats"], np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    assert out["full_stat"] == pytest.approx(4.5)
    assert out["q025"] == pytest.approx(2.125)
    assert out["q975"] == pytest.approx(6.875)
    assert out["n_blocks"] == 6.0


def test_wild_bootstrap_mammen_support_and_moments() -> None:
    e = np.random.default_rng(0).normal(size=400)
    wb = wild_bootstrap_residuals(e, n_boot=2000, seed=1)
    a = (1.0 + math.sqrt(5.0)) / 2.0
    b = (1.0 - math.sqrt(5.0)) / 2.0
    mask = np.abs(e) > 1e-12
    v = wb[:, mask] / e[None, mask]
    # every multiplier is within float noise of {a, b}
    dist = np.min(np.abs(v[..., None] - np.array([a, b])), axis=-1)
    assert float(dist.max()) < 1e-12
    assert v.mean() == pytest.approx(0.0, abs=0.03)
    assert float((v**2).mean()) == pytest.approx(1.0, abs=0.05)
    assert float((v**3).mean()) == pytest.approx(1.0, abs=0.08)


def test_sieve_bootstrap_recovers_ar1_coefficient() -> None:
    e = _ar1(0.7, 300, 0)
    draws = sieve_bootstrap_residuals(e, order=1, n_boot=40, seed=2)
    assert draws.shape == (40, 300)
    assert np.isfinite(draws).all()
    lag1 = np.array([np.corrcoef(row[:-1], row[1:])[0, 1] for row in draws])
    assert float(lag1.mean()) == pytest.approx(0.7, abs=0.08)


def test_pairs_bootstrap_exact_solution() -> None:
    x = np.arange(20.0)
    X = np.column_stack([np.ones(20), x])
    out = pairs_bootstrap(1.0 + 2.0 * x, X, n_boot=50, seed=0)
    np.testing.assert_allclose(out, np.tile([1.0, 2.0], (50, 1)), atol=1e-8)


# --- Snooping (White RC / Hansen SPA / Romano-Wolf StepM / HLN MCS) -----------


def test_reality_check_constant_panel_known_answer() -> None:
    # Recentered bootstrap means are all 0 -> no exceedance -> p = 1/(B+1).
    res = reality_check(np.ones((20, 3)), n_boot=50, block=2, seed=7)
    assert res.statistic == pytest.approx(math.sqrt(20))
    assert res.p_value == pytest.approx(1.0 / 51.0)
    assert res.best_index == 0 and res.best_mean == pytest.approx(1.0)


def test_reality_check_strong_winner_min_pvalue() -> None:
    w = np.random.default_rng(0).normal(0.0, 0.05, size=(80, 3))
    w[:, 1] += 0.2
    res = reality_check(w, n_boot=199, block=2, seed=1)
    assert res.best_index == 1
    assert res.p_value <= 2.0 / 200.0


def test_spa_ordering_and_statistic_identities() -> None:
    rng = np.random.default_rng(2)
    f = rng.normal(0.0, 0.4, size=(200, 5)) + np.array([0.02, -0.01, -0.15, -0.6, 0.01])
    res = spa_test(f, n_boot=199, block=3, seed=9)
    assert res.statistic == pytest.approx(max(0.0, max(res.studentized)))
    assert res.p_lower <= res.p_consistent <= res.p_upper
    assert res.best_index == int(np.argmax(res.studentized))
    again = spa_test(f, n_boot=199, block=3, seed=9)
    assert again == res  # deterministic under seed


def test_spa_constant_columns_fail_closed() -> None:
    res = spa_test(np.full((30, 2), 1.0), n_boot=50, block=2, seed=7)
    assert res.n_dropped == 2 and res.n_strategies == 0
    assert np.isnan(res.p_consistent)


def test_spa_all_negative_means_known_answer() -> None:
    f = np.random.default_rng(8).normal(-0.5, 0.2, size=(80, 3))
    res = spa_test(f, n_boot=199, block=3, seed=4)
    assert res.statistic == 0.0
    assert res.p_lower == res.p_consistent == res.p_upper == 1.0


def test_stepm_known_answers() -> None:
    rng = np.random.default_rng(3)
    f = rng.normal(0.0, 0.05, size=(120, 4))
    f[:, 2] += 0.02
    st = stepm(f, n_boot=99, block=3, alpha=0.05, seed=7)
    assert st.order[0] == 2  # most significant first
    assert st.rejected[2] is True
    assert st.n_rejected == sum(st.rejected)
    # Step-down: adjusted p-values are nondecreasing along the ordering and
    # rejections are exactly the prefix with adjusted_p <= alpha.
    adj = np.asarray(st.adjusted_p)
    assert np.all(np.diff(adj[np.asarray(st.order)]) >= -1e-12)
    for i, flag in enumerate(st.rejected):
        assert flag == bool(st.adjusted_p[i] <= st.alpha)


def test_mcs_identical_columns_known_answer() -> None:
    base = np.random.default_rng(3).normal(0.0, 0.05, size=(120, 1))
    f = np.column_stack([base[:, 0], base[:, 0], base[:, 0]])
    res = model_confidence_set(f, n_boot=99, block=3, alpha=0.10, seed=7)
    assert res.p_values == (1.0, 1.0, 1.0)
    assert res.included == (True, True, True)
    assert res.elimination_order == ()
    assert res.n_included == 3


def test_mcs_dominated_column_known_answer() -> None:
    base = np.random.default_rng(3).normal(0.0, 0.05, size=(120, 3))
    f = np.column_stack([base[:, 0] - 1.0, base[:, 1], base[:, 2]])
    res = model_confidence_set(f, n_boot=99, block=3, alpha=0.10, seed=7)
    # The bootstrap max statistic never reaches the huge observed range stat.
    assert res.p_values[0] == pytest.approx(1.0 / 100.0)
    assert res.p_values[1:] == (1.0, 1.0)
    assert res.included == (False, True, True)
    assert res.elimination_order == (0,)


# --- Proper scores: pinball / CRPS / QLIKE / PIT / FZ -------------------------


def test_pinball_known_answers() -> None:
    assert pinball_loss(np.array([1.0]), np.array([0.0]), 0.9)[0] == pytest.approx(0.9)
    assert pinball_loss(np.array([0.0]), np.array([1.0]), 0.9)[0] == pytest.approx(0.1)
    # tau = 0.5 -> half the absolute error
    losses = pinball_loss(np.array([1.0, 2.0, 3.0]), np.zeros(3), 0.5)
    assert np.array_equal(losses, np.array([0.5, 1.0, 1.5]))
    assert mean_pinball(np.array([1.0, 2.0, 3.0]), np.zeros(3), 0.5) == pytest.approx(1.0)


def test_crps_from_quantiles_left_sum_known_answer() -> None:
    # Frozen MATH_SPEC convention: 2 * sum_k pinball(tau_k) * (tau_k - tau_{k-1})
    # with tau_0 = 0 (a left Riemann sum, not the true CRPS integral).
    # pinball_tau(1, 0) = tau -> 2*(0.25+0.5+0.75)*0.25 = 0.75
    got = crps_from_quantiles(
        np.array([1.0]), np.array([[0.0, 0.0, 0.0]]), np.array([0.25, 0.5, 0.75])
    )
    assert got == pytest.approx(0.75)
    # y below the quantile point-mass: pinball_tau(0, 1) = 1 - tau -> also 0.75
    got2 = crps_from_quantiles(
        np.array([0.0]), np.array([[1.0, 1.0, 1.0]]), np.array([0.25, 0.5, 0.75])
    )
    assert got2 == pytest.approx(0.75)


def test_crps_gaussian_known_answer_and_scale_identity() -> None:
    # CRPS(N(0,1), 0) = 2*phi(0) - 1/sqrt(pi) = (sqrt(2)-1)/sqrt(pi)
    expected = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
    assert crps_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))[0] == pytest.approx(
        expected
    )
    # sigma-multiplicativity: CRPS(y; mu, sigma) = sigma * CRPS(z; 0, 1)
    a = crps_gaussian(np.array([3.0]), np.array([1.0]), np.array([2.0]))[0]
    b = crps_gaussian(np.array([1.0]), np.array([0.0]), np.array([1.0]))[0]
    assert a == pytest.approx(2.0 * b)


def test_crps_student_t_gaussian_limit() -> None:
    # t_nu -> N as nu -> infty: crps_student_t(nu=500) ~= crps_gaussian
    y, mu, sig = np.array([0.7, -1.2]), np.array([0.5, -1.0]), np.array([1.0, 2.0])
    got = crps_student_t(y, mu, sig, 500.0)
    ref = crps_gaussian(y, mu, sig)
    np.testing.assert_allclose(got, ref, rtol=5e-3)
    assert np.isnan(crps_student_t(y, mu, sig, 2.0)).all()  # nu <= 2 fails closed


def test_crps_gaussian_mixture_reduces_to_single_component() -> None:
    y = np.array([0.3, -1.1])
    mix = crps_gaussian_mixture(y, np.array([1.0]), np.array([0.5]), np.array([2.0]))
    ref = crps_gaussian(y, np.full(2, 0.5), np.full(2, 2.0))
    np.testing.assert_allclose(mix, ref, rtol=1e-12)
    # two-component symmetric mixture stays finite and nonnegative
    two = crps_gaussian_mixture(
        y, np.array([0.5, 0.5]), np.array([-1.0, 1.0]), np.array([0.5, 0.5])
    )
    assert np.isfinite(two).all() and np.all(two >= 0.0)


def test_gaussian_mixture_quantiles_known_answer() -> None:
    q = gaussian_mixture_quantiles(
        np.array([1.0]), np.array([2.0]), np.array([0.5]), np.array([0.5, 0.975])
    )
    assert q[0] == pytest.approx(2.0, abs=1e-6)
    assert q[1] == pytest.approx(2.0 + 0.5 * stats.norm.ppf(0.975), abs=1e-6)


def test_crps_empirical_known_answers() -> None:
    # single-point sample: |x - y| with no spread term
    assert crps_empirical(2.0, np.array([5.0])) == pytest.approx(3.0)
    # [0, 2] at y=1: mean|Xi-y| = 1, mean|Xi-Xj| = (0+2+2+0)/4 = 1 -> 1 - 0.5
    assert crps_empirical(1.0, np.array([0.0, 2.0])) == pytest.approx(0.5)


def test_qlike_known_answers() -> None:
    assert qlike(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)
    # ratio r = 2 -> 2 - ln 2 - 1 = 1 - ln 2
    assert qlike(np.array([2.0]), np.array([1.0])) == pytest.approx(1.0 - math.log(2.0))


def test_pit_values_known_answers() -> None:
    q = np.array([[1.0, 2.0, 3.0]])
    taus = np.array([0.25, 0.5, 0.75])
    pits = pit_values(np.array([2.0, 1.5, 0.5, 5.0]), np.repeat(q, 4, axis=0), taus)
    assert pits[0] == pytest.approx(0.5)  # exactly on a knot
    assert pits[1] == pytest.approx(0.375)  # linear interpolation midpoint
    assert pits[2] == pytest.approx(0.0)  # below the grid
    assert pits[3] == pytest.approx(1.0)  # above the grid


def test_coverage_masks_nonfinite() -> None:
    """Regression: non-finite observations are masked, not counted as misses."""
    assert coverage(
        np.array([0.5, np.nan]), np.array([0.0, 0.0]), np.array([1.0, 1.0])
    ) == pytest.approx(1.0)
    assert np.isnan(coverage(np.array([np.nan]), np.array([0.0]), np.array([1.0])))
    assert coverage(
        np.array([0.0, 5.0]), np.array([-1.0, 0.0]), np.array([0.5, 2.0])
    ) == pytest.approx(0.5)


def test_interval_width_and_crossing_known_answers() -> None:
    assert interval_width(np.array([-1.0, 2.0]), np.array([1.0, 5.0])) == pytest.approx(2.5)
    assert np.isnan(interval_width(np.array([1.0]), np.array([0.0])))  # upper < lower
    crossed = np.array([[1.0, 3.0, 2.0], [1.0, 2.0, 3.0]])
    assert quantile_crossing_rate(crossed, np.array([0.25, 0.5, 0.75])) == pytest.approx(0.5)
    assert quantile_crossing_rate(
        rearrange_quantiles(crossed), np.array([0.25, 0.5, 0.75])
    ) == pytest.approx(0.0)


def test_ic_family_known_answers() -> None:
    assert pearson_ic(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 6.0, 8.0])
    ) == pytest.approx(1.0)
    assert pearson_ic(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 3.0, 2.0, 1.0])
    ) == pytest.approx(-1.0)
    # rank IC with average ties: pred [0,0,1,2] ranks [1.5,1.5,3,4]
    assert rank_ic(np.array([0.0, 0.0, 1.0, 2.0]), np.array([0.0, 0.0, 1.0, 2.0])) == pytest.approx(
        1.0
    )
    assert icir(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(0.0)
    assert np.isnan(icir(np.full(4, 0.3)))


def test_fissler_ziegel_proper_at_true_pair() -> None:
    # Uniform losses on a grid: VaR_0.9 = 0.9, ES_0.9 = 0.95 must beat
    # perturbed (v, e) pairs for the joint FZ0 score to be elicitable.
    grid = np.linspace(0.0, 1.0, 501)
    alpha = 0.9
    best = mean_fissler_ziegel(grid, np.full(501, 0.9), np.full(501, 0.95), alpha)
    assert best < mean_fissler_ziegel(grid, np.full(501, 0.85), np.full(501, 0.95), alpha)
    assert best < mean_fissler_ziegel(grid, np.full(501, 0.95), np.full(501, 0.95), alpha)
    assert best < mean_fissler_ziegel(grid, np.full(501, 0.9), np.full(501, 0.9), alpha)
    assert best < mean_fissler_ziegel(grid, np.full(501, 0.9), np.full(501, 1.0), alpha)


def test_spiegelhalter_known_answers() -> None:
    # p = [0.2, 0.6, 0.2, 0.6, 0.5] satisfies sum p(1-2p) = 0 -> z = 0 -> p = 1
    out = spiegelhalter_z(np.array([0.2, 0.6, 0.2, 0.6, 0.5]), np.zeros(5))
    assert out["z"] == pytest.approx(0.0, abs=1e-12)
    assert out["pvalue"] == pytest.approx(1.0, abs=1e-12)
    # prob 0.8 always hit: num = 5*0.2*(-0.6) = -0.6, var = 5*0.36*0.16 = 0.288
    out2 = spiegelhalter_z(np.full(5, 0.8), np.ones(5))
    assert out2["z"] == pytest.approx(-0.6 / math.sqrt(0.288))
    assert out2["pvalue"] == pytest.approx(2.0 * stats.norm.sf(0.6 / math.sqrt(0.288)))
