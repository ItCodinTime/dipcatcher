"""P6.2 validation-layer audit KATs: pin the verified-correct values.

Companion to docs/AUDIT_P62B_VALIDATION.md — each test is a known-answer or
identity check against the cited reference semantics.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from quant_fund.validation.cpcv import (
    combinatorial_purged_indices,
    cpcv_n_paths,
    cpcv_n_splits,
    cpcv_path_assignments,
    stitch_group_paths,
)
from quant_fund.validation.embargo import embargo_mask
from quant_fund.validation.fdr import (
    benjamini_hochberg,
    benjamini_yekutieli,
    hochberg,
    holm,
    simes,
    storey_pi0,
)
from quant_fund.validation.purging import purge_mask
from quant_fund.validation.regime_eval import (
    RegimeSummary,
    regime_eval_gate,
)
from quant_fund.validation.walk_forward import Fold, assert_no_label_overlap

T0 = datetime(2026, 1, 1, tzinfo=UTC)
DAY = timedelta(days=1)


def _times(n: int) -> list[datetime]:
    return [T0 + i * DAY for i in range(n)]


def _idx(times: list[datetime]) -> dict[datetime, int]:
    return {t: i for i, t in enumerate(sorted(set(times)))}


class TestFDRKATs:
    P = np.array([0.01, 0.04, 0.03])

    def test_holm(self):
        # asc [0.01,0.03,0.04] -> [3*0.01, 2*0.03, 1*0.04] -> running max
        adj = holm(self.P)["adjusted"]
        assert np.allclose(adj, [0.03, 0.06, 0.06])

    def test_hochberg(self):
        # desc [0.04,0.03,0.01] -> [1*.04, 2*.03, 3*.01]=[.04,.06,.03] ->
        # running min [.04,.04,.03] -> restore onto original positions.
        adj = hochberg(self.P)["adjusted"]
        assert np.allclose(adj, [0.03, 0.04, 0.04])

    def test_bh(self):
        # asc raw = [.03, .045, .04]; suffix-min -> [.03, .04, .04]
        adj = benjamini_hochberg(self.P)["adjusted"]
        assert np.allclose(adj, [0.03, 0.04, 0.04])

    def test_by(self):
        # BH * c(3) = *11/6; p_(2) row: 0.04*11/6 = 0.0733...
        adj = benjamini_yekutieli(self.P)["adjusted"]
        assert np.allclose(adj, [0.055, 0.04 * 11 / 6, 0.04 * 11 / 6])

    def test_simes(self):
        # min_i 3*p_(i)/i = min(.03, .045, .04) = .03
        assert simes(self.P) == pytest.approx(0.03)

    def test_storey_pi0_all_null(self):
        # Uniform p-values => pi0 ~ 1 (conservative).
        rng = np.random.default_rng(0)
        assert storey_pi0(rng.uniform(size=500)) > 0.8

    def test_pvals_reject_oob(self):
        with pytest.raises(ValueError):
            benjamini_hochberg(np.array([-0.01, 0.5]))


class TestPurgeEmbargoKATs:
    def test_purge_label_reaching_test(self):
        times = _times(10)
        idx = _idx(times)
        # Decisions at sessions 0..7; test block [5,6]; horizon 3 bars.
        # Sessions 2,3,4 have labels ending at 5,6,7 -> reach holdout.
        mask = purge_mask(times[:8], times[5], times[6], 3, session_index=idx)
        # safe: sessions 0,1 (labels end 3,4 < 5) and 7 (post-holdout: its
        # label runs forward). unsafe: 2,3,4 (labels reach 5,6,7) and 5,6
        # (inside the holdout).
        assert mask == [True, True, False, False, False, False, False, True]

    def test_purge_no_overlap_when_horizon_short(self):
        times = _times(10)
        idx = _idx(times)
        mask = purge_mask(times[:4], times[5], times[6], 1, session_index=idx)
        assert mask == [True] * 4

    def test_embargo_drops_only_post_block(self):
        times = _times(8)
        idx = _idx(times)
        mask = embargo_mask(times, times[4], 2, idx)
        # block ends at session 4 -> drop sessions 5,6 only.
        assert mask == [True] * 5 + [False, False] + [True]

    def test_assert_no_label_overlap_catches_reach(self):
        times = _times(10)
        idx = _idx(times)
        fold = Fold(train_times=[times[3]], val_times=[times[5]], test_times=[times[6]])
        with pytest.raises(AssertionError, match="label overlap"):
            assert_no_label_overlap(fold, 3, idx)
        # horizon 1 ends at 4 < 5 -> clean.
        assert_no_label_overlap(fold, 1, idx)


class TestCPCVIdentities:
    def test_counts(self):
        assert cpcv_n_splits(6, 2) == 15
        assert cpcv_n_paths(6, 2) == 5

    def test_path_assignments_cover_once(self):
        a = cpcv_path_assignments(6, 2)
        assert a.shape == (5, 6)
        # each row covers all groups; each (split, group) incidence used once.
        used = set()
        for p in range(5):
            for g in range(6):
                s = int(a[p, g])
                assert g in list(__import__("itertools").combinations(range(6), 2))[s]
                assert (s, g) not in used
                used.add((s, g))
        assert len(used) == 30  # C(6,2)*k = 15*2

    def test_stitch_uses_testing_split(self):
        a = cpcv_path_assignments(4, 1)
        scores = np.arange(16, dtype=float).reshape(4, 4)
        out = stitch_group_paths(scores, a)
        for g in range(4):
            # path entries for group g must equal scores[s, g] where s tests g.
            assert np.allclose(out[:, g], scores[a[:, g], g])

    def test_index_space_purges_label_reach(self):
        folds = combinatorial_purged_indices(12, 4, 1, label_horizon=2, embargo=1)
        assert len(folds) == 4
        for train, test in folds:
            assert np.intersect1d(train, test).size == 0
            lo, hi = int(test[0]), int(test[-1])
            # no train index whose (i, i+2] reaches the test block
            assert not np.any((train < lo) & (train + 2 >= lo))
            # embargo: none within 1 before/after the block
            assert not np.any((train == lo - 1) | (train == hi + 1))


class TestRegimeGateKATs:
    def _summary(self, counts, means, e_report=None):

        counts = np.asarray(counts, dtype=np.intp)
        n = int(counts.sum())
        return RegimeSummary(
            names=tuple(f"r{i}" for i in range(len(counts))),
            counts=counts,
            means=np.asarray(means, dtype=float),
            stderrs=np.full(len(counts), 0.1),
            pooled_mean=float(np.mean(means)),
            pooled_stderr=0.1,
            pooled_n=n,
            diffs=np.zeros(len(counts)),
            diff_ci_low=np.zeros(len(counts)),
            diff_ci_high=np.zeros(len(counts)),
            diff_ci_level=0.9,
            n_boot=200,
            mean_block=2.0,
            e_value_report=e_report,
        )

    def test_clean_pass(self):
        assert regime_eval_gate(self._summary([500, 500], [0.5, 0.55])).passed

    def test_tiny_diluted_regime_fails(self):
        # share 30/1030 ≈ 0.029 < 0.05 with n=30 → insufficient_regime_share.
        res = regime_eval_gate(self._summary([1000, 30], [0.5, 0.5]))
        assert not res.passed
        assert any("insufficient_regime_share" in r for r in res.reasons)

    def test_tiny_share_small_n_ignored(self):
        # share < min but n < 30 → statistically ignorable, still passes.
        res = regime_eval_gate(self._summary([1000, 10], [0.5, 0.5]))
        assert res.passed

    def test_heterogeneity_cv_fails(self):
        res = regime_eval_gate(self._summary([100, 100], [0.01, 5.0]), max_regime_cv=1.0)
        assert not res.passed
        assert any("regime_mean_cv_exceeded" in r for r in res.reasons)

    def test_regime_contradicts_pooled(self):
        e = {
            "pooled": {"crossed": False},
            "regimes": {"r0": {"crossed": True}, "r1": {"crossed": False}},
        }
        res = regime_eval_gate(self._summary([500, 500], [0.5, 0.5], e_report=e))
        assert not res.passed
        assert any("regime_contradicts_pooled:r0" in r for r in res.reasons)

    def test_regime_cross_agreeing_pooled_passes(self):
        e = {
            "pooled": {"crossed": True},
            "regimes": {"r0": {"crossed": True}, "r1": {"crossed": False}},
        }
        assert regime_eval_gate(self._summary([500, 500], [0.5, 0.5], e_report=e)).passed

    def test_invalid_thresholds_fail_closed(self):
        with pytest.raises(ValueError):
            regime_eval_gate(self._summary([1, 1], [0.0, 0.0]), min_regime_share=1.5)
        with pytest.raises(ValueError):
            regime_eval_gate(self._summary([1, 1], [0.0, 0.0]), max_regime_cv=0.0)
