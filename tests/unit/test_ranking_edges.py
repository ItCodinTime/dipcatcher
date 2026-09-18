"""Wave 18: ranking group_sizes / CompositeRanker / LambdaRank edge fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ranking import (
    CompositeRanker,
    LGBMLambdaRanker,
    _to_relevance,
    available_features,
    group_sizes,
)


def test_group_sizes_empty_all_same_all_unique() -> None:
    assert group_sizes(np.array([])).tolist() == []
    assert group_sizes(np.array([7, 7, 7, 7])).tolist() == [4]
    assert group_sizes(np.array([1, 2, 3, 4])).tolist() == [1, 1, 1, 1]
    # Consecutive runs only — non-contiguous equal dates split
    assert group_sizes(np.array([1, 1, 2, 1, 1])).tolist() == [2, 1, 2]


def test_group_sizes_sum_equals_n() -> None:
    dates = np.array([10, 10, 11, 11, 11, 12])
    g = group_sizes(dates)
    assert int(g.sum()) == len(dates)


def test_available_features_filters_missing() -> None:
    cols = ["cs_z_mom_20", "noise", "cs_z_vol_20"]
    assert available_features(cols, ["cs_z_mom_20", "missing", "cs_z_vol_20"]) == [
        "cs_z_mom_20",
        "cs_z_vol_20",
    ]
    assert available_features(["a"], ["b"]) == []


def test_composite_ranker_empty_and_constant_y() -> None:
    ranker = CompositeRanker(cols=["f0", "f1"], signs={"f0": 1.0, "f1": -1.0})
    # Empty design: 0 rows
    empty = np.zeros((0, 2), dtype=float)
    out = ranker.fit(empty, np.array([])).predict(empty)
    assert out.shape == (0,)

    # Constant y is ignored (baseline composite); finite scores from features
    x = np.array([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]], dtype=float)
    y = np.ones(3)
    pred = ranker.fit(x, y).predict(x)
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))
    # signs: f0 * 1 + f1 * -1 → mean of [0.2, -0.8] etc.
    expected = np.nanmean(x * np.array([1.0, -1.0]), axis=1)
    assert pred == pytest.approx(expected)
    meta = ranker.metadata()
    assert meta.name == "composite"
    assert "sharpe" not in (meta.extra or {})


def test_composite_ranker_nan_column_uses_nanmean() -> None:
    ranker = CompositeRanker(cols=["a", "b"], signs={"a": 1.0, "b": 1.0})
    x = np.array([[1.0, np.nan], [2.0, 4.0]], dtype=float)
    pred = ranker.fit(x, np.zeros(2)).predict(x)
    assert pred[0] == pytest.approx(1.0)  # nanmean skips nan
    assert pred[1] == pytest.approx(3.0)


def test_lambdarank_requires_group() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 3))
    y = rng.normal(size=16)
    with pytest.raises(ValueError, match="group"):
        LGBMLambdaRanker(n_estimators=5, seed=0).fit(x, y)


def test_lambdarank_single_group_and_constant_y() -> None:
    rng = np.random.default_rng(1)
    n = 12
    x = rng.normal(size=(n, 2))
    y = np.full(n, 0.42)  # constant within group → relevance all 0 via digitize edge
    g = np.array([n], dtype=np.int32)
    m = LGBMLambdaRanker(n_estimators=10, seed=1).fit(x, y, group=g)
    pred = m.predict(x)
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))
    assert m.metadata().name == "lambdarank"


def test_lambdarank_two_date_groups_edge() -> None:
    rng = np.random.default_rng(2)
    dates = np.array([0, 0, 0, 1, 1, 1, 1])
    x = rng.normal(size=(7, 3))
    y = x[:, 0] + rng.normal(size=7) * 0.05
    g = group_sizes(dates)
    assert g.tolist() == [3, 4]
    m = LGBMLambdaRanker(n_estimators=15, seed=2).fit(x, y, group=g)
    pred = m.predict(x[:3])
    assert pred.shape == (3,)


def test_to_relevance_quintiles_and_constant_group() -> None:
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    g = np.array([5], dtype=np.int32)
    rel = _to_relevance(y, g)
    assert rel.shape == (5,)
    assert rel.min() >= 0
    assert rel.max() <= 4
    # Constant within-group → digitize lands on a single grade (no crash)
    y2 = np.full(6, 0.5, dtype=float)
    rel2 = _to_relevance(y2, np.array([3, 3], dtype=np.int32))
    assert rel2.shape == (6,)
    assert set(rel2.tolist()) <= {0, 1, 2, 3, 4}


def test_to_relevance_nan_member_never_top_graded() -> None:
    """A missing return must not be minted as top-quintile relevance.

    NaN sorts to the end under ``np.digitize``, so the pre-fix path silently
    handed NaN members grade 4 (highest relevance) inside mixed groups —
    corrupting LambdaRank training signal. Fail-closed: non-finite members
    carry no signal (grade 0) while finite members keep their quintiles.
    """
    y = np.array([np.nan, 0.0, 1.0, 2.0], dtype=float)
    rel = _to_relevance(y, np.array([4], dtype=np.int32))
    assert rel[0] == 0
    assert rel[1:].max() >= 1
    assert set(rel[1:].tolist()) <= {0, 1, 2, 3, 4}


def test_to_relevance_all_nan_group_fail_closed() -> None:
    """All-NaN group cannot fabricate quintiles; no crash, no fake grades."""
    y = np.full(4, np.nan, dtype=float)
    rel = _to_relevance(y, np.array([4], dtype=np.int32))
    assert rel.tolist() == [0, 0, 0, 0]
