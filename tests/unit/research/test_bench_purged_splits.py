"""Regression pins for the purged date-aware bench splits.

Pre-fix, ``_holdout``/``_triple_split`` cut by *row position*: a date straddling
the cut landed in both train and eval, and labels whose horizon reached the eval
window stayed in train. The masks below pin the contract: unique-date boundary,
``purge_mask`` gap on every edge, fail-closed empties.
"""

from __future__ import annotations

import numpy as np

from quant_fund.research.benches.common import _holdout, _triple_split


def _date_sets(mask_triplet, keys):
    return [np.unique(keys[m]) for m in mask_triplet]


def test_holdout_cuts_on_unique_dates() -> None:
    keys = np.repeat(np.arange(40), 3)  # dup rows share a date
    tr, te = _holdout(keys, 0.3, horizon=1)
    trd, ted = np.unique(keys[tr]), np.unique(keys[te])
    assert trd.size and ted.size
    assert not set(trd) & set(ted)
    assert int(min(ted) - max(trd)) >= 1


def test_holdout_purges_horizon_labels() -> None:
    keys = np.repeat(np.arange(40), 2)
    tr, te = _holdout(keys, 0.3, horizon=4)
    trd, ted = np.unique(keys[tr]), np.unique(keys[te])
    assert int(min(ted) - max(trd)) >= 4  # last train label ends before eval starts


def test_triple_split_ordering_and_purged_edges() -> None:
    keys = np.arange(100)
    tr, cal, te = _triple_split(keys, horizon=5)
    trd, cald, ted = _date_sets((tr, cal, te), keys)
    assert max(trd) < min(cald) < min(ted) and max(cald) < min(ted)
    assert int(min(cald) - max(trd)) >= 5  # train labels purged vs cal
    assert int(min(ted) - max(cald)) >= 5  # cal labels purged vs test


def test_triple_split_never_splits_a_date() -> None:
    keys = np.repeat(np.arange(60), 3)
    tr, cal, te = _triple_split(keys, horizon=2)
    trd, cald, ted = _date_sets((tr, cal, te), keys)
    assert not set(trd) & set(cald)
    assert not set(cald) & set(ted)
    assert not set(trd) & set(ted)


def test_tiny_panels_fail_closed_to_empty_masks() -> None:
    tr, cal, te = _triple_split(np.arange(2), horizon=1)
    assert not (tr.any() or cal.any() or te.any())
    tr, te = _holdout(np.arange(1), horizon=1)
    assert not (tr.any() or te.any())


def test_masks_are_boolean_row_masks_over_input() -> None:
    keys = np.repeat(np.arange(50), 2)
    tr, cal, te = _triple_split(keys, horizon=3)
    for m in (tr, cal, te):
        assert m.dtype == bool and m.shape == keys.shape
    # no row is in two sides
    assert not (tr & cal).any() and not (cal & te).any() and not (tr & te).any()


def test_eval_perturbation_cannot_reach_train() -> None:
    """Causality of the cut: dates inside the eval window stay eval-only."""
    keys = np.repeat(np.arange(50), 2)
    tr, te = _holdout(keys, 0.3, horizon=3)
    trd, ted = np.unique(keys[tr]), np.unique(keys[te])
    assert trd.size < np.unique(keys).size  # purge actually removed some dates
    assert all(d < min(ted) for d in trd)


# ---------------------------------------------------------------------------
# Exhaustive invariance: enumerate every date-multiset pattern and verify the
# mask algebra holds on each — not sampled properties, the full small space.


def _multisets(n_unique: int, max_mult: int = 2):
    """All row arrays: n_unique dates, each repeated 1..max_mult times."""
    import itertools

    uniq = np.arange(n_unique)
    for mults in itertools.product(range(1, max_mult + 1), repeat=n_unique):
        yield np.repeat(uniq, mults)


def _assert_disjoint_order_purged(tr, cal, te, keys, h_edge) -> None:
    uniq = np.unique(keys)
    pos = {d: i for i, d in enumerate(uniq.tolist())}
    sides = [np.unique(keys[m]) for m in (tr, cal, te)]
    trd, cald, ted = sides
    assert not set(trd) & set(cald)
    assert not set(cald) & set(ted)
    assert not set(trd) & set(ted)
    if trd.size and cald.size:
        assert max(trd) < min(cald)
        for d in trd:  # label window (i, i+h] must end before cal starts
            assert pos[d] + h_edge < pos[min(cald)]
    if cald.size and ted.size:
        assert max(cald) < min(ted)
        for d in cald:
            assert pos[d] + h_edge < pos[min(ted)]


def test_triple_split_exhaustive_small_panels() -> None:
    n_checked = 0
    for keys in _multisets(4):
        for h in (0, 1, 3):
            tr, cal, te = _triple_split(keys, horizon=h)
            assert [m.dtype for m in (tr, cal, te)] == [bool] * 3
            _assert_disjoint_order_purged(tr, cal, te, keys, h)
            n_checked += 1
    assert n_checked == 2**4 * 3  # every multiset × every horizon


def test_holdout_exhaustive_small_panels() -> None:
    for keys in _multisets(3):
        for h in (0, 1, 2, 4):
            tr, te = _holdout(keys, 0.3, horizon=h)
            trd, ted = np.unique(keys[tr]), np.unique(keys[te])
            assert not set(trd) & set(ted)
            if trd.size and ted.size:
                pos = {d: i for i, d in enumerate(np.unique(keys).tolist())}
                assert max(trd) < min(ted)
                for d in trd:
                    assert pos[d] + h < pos[min(ted)]


def test_split_masks_idempotent_under_row_shuffle() -> None:
    """Membership masks must not depend on input row order."""
    keys = np.repeat(np.arange(30), 2)
    rng = np.random.default_rng(0)
    tr0, cal0, te0 = _triple_split(keys, horizon=2)
    for _ in range(5):
        perm = rng.permutation(keys.size)
        tr, cal, te = _triple_split(keys[perm], horizon=2)
        # same date sets regardless of row order
        for a, b in ((tr0, tr), (cal0, cal), (te0, te)):
            assert set(np.unique(keys[a])) == set(np.unique(keys[perm][b]))
