"""Known answers for CPCV paths, CSCV/PBO, DSR, PSR, and MinTRL.

Published anchors, cited in the tests that lock them:

* Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection
  Bias, Backtest Overfitting, and Non-Normality", Journal of Portfolio
  Management 40(5), 2014, section "A Numerical Example". Annualized selection
  score 2.5, T=1250 daily observations (250/year), N=100, variance of the
  non-annualized trial scores V=1/(2*250), skewness -3, raw kurtosis 10.
  The paper's figures are SR0 ≈ 0.1132, DSR ≈ 0.9004; DSR ≈ 0.9505 at N=46;
  and DSR ≈ 0.9505 under Normal moments (skew 0, raw kurtosis 3) at N=88.

* Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier", Journal of
  Risk 15(2), 2012, section 5. An annualized score of 2 against a benchmark
  of 1 at 95% confidence needs 2.73 / 2.83 / 3.24 years of daily (252),
  weekly (52), and monthly (12) IID Normal observations. The same monthly
  comparison with skewness -0.72 and raw kurtosis 5.78 (the HFR moment pair
  that recovers the paper's 4.99-year figure; also the inputs in
  PerformanceAnalytics::MinTrackRecord) is 4.99 years.

* Bailey & López de Prado (2012), section 3. A monthly score of 1.59/sqrt(12)
  over 24 observations, Normal moments, has PSR(0) ≈ 0.982.

* López de Prado, Advances in Financial Machine Learning (2018), ch. 12.
  N groups and k test groups produce C(N, k) splits and C(N-1, k-1) paths.

* Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest
  Overfitting", Journal of Computational Finance 20(4), 2017. PBO is the
  fraction of CSCV splits in which the in-sample best is below the
  out-of-sample median. Under pure noise that fraction is about one half,
  and the deflated probability rejects the best noise trial.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from itertools import combinations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.metrics.overfitting import (
    CORRELATED_TRIAL_MIN_RHO,
    choose_cscv_slices,
    cscv_performance,
    deflated_sharpe,
    effective_n_trials,
    expected_max_sharpe,
    min_track_record_length,
    overfitting_diagnostics,
    probabilistic_sharpe,
    probability_of_backtest_overfitting,
)
from quant_fund.validation.cpcv import (
    combinatorial_purged_cv,
    combinatorial_purged_indices,
    cpcv_n_paths,
    cpcv_n_splits,
    cpcv_path_assignments,
    stitch_group_paths,
)

# Bailey & López de Prado (2014), "A Numerical Example".
_PAPER_VARIANCE = 1.0 / (2.0 * 250.0)
_PAPER_SCORE = 2.5 / math.sqrt(250.0)
_PAPER_LENGTH = 1250


def test_deflated_probability_matches_bailey_lopezdeprado_2014() -> None:
    """JPM 2014 numerical example: DSR 0.9004, and 0.9505 at N=46 and N=88."""
    sr0 = expected_max_sharpe(100, _PAPER_VARIANCE)
    assert sr0 == pytest.approx(0.1132, abs=5e-5)
    dsr = deflated_sharpe(_PAPER_SCORE, _PAPER_LENGTH, -3.0, 10.0, 100, _PAPER_VARIANCE)
    assert dsr == pytest.approx(0.9004, abs=5e-4)
    assert dsr < 0.95
    dsr_46 = deflated_sharpe(_PAPER_SCORE, _PAPER_LENGTH, -3.0, 10.0, 46, _PAPER_VARIANCE)
    assert dsr_46 == pytest.approx(0.9505, abs=5e-4)
    dsr_normal = deflated_sharpe(_PAPER_SCORE, _PAPER_LENGTH, 0.0, 3.0, 88, _PAPER_VARIANCE)
    assert dsr_normal == pytest.approx(0.9505, abs=5e-4)


def test_min_trl_matches_bailey_lopezdeprado_2012_frequency_table() -> None:
    """Journal of Risk 2012 §5: 2.73 / 2.83 / 3.24 years, and 4.99 with HFR moments."""

    def years(periods: int, skew: float = 0.0, kurtosis: float = 3.0) -> float:
        observed = 2.0 / math.sqrt(periods)
        benchmark = 1.0 / math.sqrt(periods)
        length = min_track_record_length(observed, skew, kurtosis, conf=0.95, sr_star=benchmark)
        return length / periods

    assert years(252) == pytest.approx(2.73, abs=0.005)
    assert years(52) == pytest.approx(2.83, abs=0.005)
    assert years(12) == pytest.approx(3.24, abs=0.005)
    assert years(12, skew=-0.72, kurtosis=5.78) == pytest.approx(4.99, abs=0.01)


def test_psr_normal_two_year_monthly_track_record() -> None:
    """Bailey & López de Prado (2012) §3: annualized 1.59 over 24 months, PSR(0)≈0.982."""
    monthly = 1.59 / math.sqrt(12.0)
    psr = probabilistic_sharpe(monthly, 0.0, 24, 0.0, 3.0)
    assert psr == pytest.approx(0.982, abs=0.001)
    # The deflated probability cannot exceed the undeflated one when the
    # hurdle is a non-negative expected maximum.
    dsr = deflated_sharpe(monthly, 24, 0.0, 3.0, 10, 0.01)
    assert dsr <= psr


def test_cpcv_path_count_is_the_published_binomial() -> None:
    """AFML ch. 12: φ = C(N-1, k-1) = C(N, k) * k / N."""
    for n_groups, n_test in ((6, 2), (8, 3), (5, 1), (7, 3)):
        splits = cpcv_n_splits(n_groups, n_test)
        paths = cpcv_n_paths(n_groups, n_test)
        assert splits == math.comb(n_groups, n_test)
        assert paths == math.comb(n_groups - 1, n_test - 1)
        assert paths * n_groups == splits * n_test
        assignments = cpcv_path_assignments(n_groups, n_test)
        assert assignments.shape == (paths, n_groups)
        combos = list(combinations(range(n_groups), n_test))
        seen: set[tuple[int, int]] = set()
        for path in assignments:
            assert len(set(path.tolist())) <= n_groups
            for group, split_id in enumerate(path.tolist()):
                assert group in combos[split_id]
                incidence = (split_id, group)
                assert incidence not in seen
                seen.add(incidence)
        for split_id, combo in enumerate(combos):
            for group in combo:
                assert (split_id, group) in seen


def test_stitched_paths_read_only_the_assigned_split() -> None:
    assignments = cpcv_path_assignments(4, 2)
    n_splits = cpcv_n_splits(4, 2)
    n_groups = 4
    scores = np.arange(n_splits * n_groups, dtype=float).reshape(n_splits, n_groups)
    stitched = stitch_group_paths(scores, assignments)
    for path_id in range(stitched.shape[0]):
        for group in range(n_groups):
            split_id = int(assignments[path_id, group])
            assert stitched[path_id, group] == scores[split_id, group]


def test_index_cpcv_matches_datetime_cpcv_on_a_daily_grid() -> None:
    n = 48
    origin = datetime(2020, 1, 1, tzinfo=UTC)
    times = [origin + timedelta(days=i) for i in range(n)]
    dated = combinatorial_purged_cv(times, 6, 2, horizon_bars=2, embargo_bars=1)
    indexed = combinatorial_purged_indices(n, 6, 2, label_horizon=2, embargo=1)
    assert len(dated) == len(indexed)
    index = {stamp: i for i, stamp in enumerate(times)}
    for fold, (train, test) in zip(dated, indexed, strict=True):
        assert [index[stamp] for stamp in fold.train_times] == train.tolist()
        assert [index[stamp] for stamp in fold.test_times] == test.tolist()


def test_cscv_hand_counted_overfit_and_clean_fractions() -> None:
    """Two slices, two configs. Each slice crowns a different config that then loses OOS.

    IS slice 0 selects A (mean 1 > 0); OOS A is 0, below the median 0.5.
    IS slice 1 selects B; OOS B is 0, below the median. PBO = 1.
    A config that leads in both slices has PBO = 0.
    """
    overfit = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    is_perf, oos_perf = cscv_performance(overfit, 2)
    assert probability_of_backtest_overfitting(is_perf, oos_perf) == 1.0
    clean = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    is_clean, oos_clean = cscv_performance(clean, 2)
    assert probability_of_backtest_overfitting(is_clean, oos_clean) == 0.0
    assert choose_cscv_slices(3) == 0
    assert choose_cscv_slices(32) == 16


def test_duplicate_trials_collapse_and_orthogonal_trials_do_not() -> None:
    rng = np.random.default_rng(11)
    base = rng.normal(size=80)
    duplicates = np.column_stack([base, base.copy(), base + 1e-12])
    n_eff, labels = effective_n_trials(duplicates)
    assert n_eff == 1
    assert set(labels.tolist()) == {0}
    orthogonal = rng.normal(size=(400, 4))
    n_orth, _labels = effective_n_trials(orthogonal, min_rho=CORRELATED_TRIAL_MIN_RHO)
    assert n_orth == 4
    mixed = np.column_stack([base, base.copy(), rng.normal(size=80), rng.normal(size=80)])
    n_mixed, mixed_labels = effective_n_trials(mixed)
    assert n_mixed == 3
    assert mixed_labels[0] == mixed_labels[1]
    assert mixed_labels[2] != mixed_labels[0]
    assert mixed_labels[3] != mixed_labels[0]


def test_pure_noise_strategies_have_pbo_near_one_half_and_rejected_dsr() -> None:
    """IID noise configs: mean CSCV PBO is near 1/2 and every DSR rejects.

    Bailey et al. (2017): with no edge, the in-sample winner is exchangeable
    with the other configs, so each split contributes 1/2 and E[PBO] = 1/2.
    Overlapping CSCV combinations make one matrix's PBO much noisier than a
    binomial on C(16, 8), so the assertion is the mean over independent
    noise matrices. The best column of each matrix fails the deflated 95%
    bar from Bailey & López de Prado (2014).
    """
    pbos: list[float] = []
    for seed in range(1000, 1040):
        scores = np.random.default_rng(seed).normal(size=(128, 16))
        report = overfitting_diagnostics(scores, n_trials=16, horizon_bars=1, embargo_bars=1)
        assert report["metrics_status"] == "computed"
        assert report["n_trials"] == 16
        assert report["cscv_combinations"] == math.comb(16, 8)
        assert report["research_only"] is True
        assert report["purge_disjoint"] is True
        assert report["cpcv_n_paths"] == math.comb(5, 1)
        pbo = float(report["pbo"])
        dsr = float(report["dsr"])
        assert 0.0 < pbo < 1.0
        assert dsr < 0.95
        assert float(report["dsr_counted_trials"]) < 0.95
        assert dsr <= float(report["psr"]) + 1e-12
        pbos.append(pbo)
    assert sum(pbos) / len(pbos) == pytest.approx(0.5, abs=0.05)


def test_diagnostics_do_not_invent_a_probability_without_trials() -> None:
    report = overfitting_diagnostics(None, n_trials=0)
    assert report["pbo"] is None
    assert report["dsr"] is None
    assert report["psr"] is None
    assert report["min_trl"] is None
    assert report["n_trials_effective"] == 0
    assert report["metrics_status"] == "unavailable"


def test_normal_psr_formula_matches_a_hand_evaluation() -> None:
    """PSR = Φ((SR-SR*)√(T-1) / √(1 - γ3 SR + (γ4-1)/4 SR²)), Gaussian moments."""
    observed, benchmark, length = 0.2, 0.0, 50
    inside = 1.0 + 0.5 * observed**2
    expected = float(norm.cdf((observed - benchmark) * math.sqrt(length - 1) / math.sqrt(inside)))
    assert probabilistic_sharpe(observed, benchmark, length, 0.0, 3.0) == pytest.approx(expected)
