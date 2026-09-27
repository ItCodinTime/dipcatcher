"""Tests for metrics/cluster_validity.py — internal validity indices."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.cluster_validity import (
    calinski_harabasz,
    davies_bouldin,
    silhouette_score,
)


def _blobs(sep: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((50, 2)) * 0.3
    b = rng.standard_normal((50, 2)) * 0.3 + np.array([sep, sep])
    return np.vstack([a, b]), np.array([0] * 50 + [1] * 50)


def test_good_clustering_scores() -> None:
    x, labels = _blobs(sep=8.0)
    assert silhouette_score(x, labels) > 0.6
    assert calinski_harabasz(x, labels) > 100.0
    assert davies_bouldin(x, labels) < 0.5


def test_separation_improves_indices() -> None:
    x_close, lab = _blobs(sep=1.0, seed=1)
    x_far, _ = _blobs(sep=10.0, seed=1)
    assert silhouette_score(x_far, lab) > silhouette_score(x_close, lab)
    assert davies_bouldin(x_far, lab) < davies_bouldin(x_close, lab)


def test_random_labels_low_silhouette() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((100, 2))
    labels = rng.integers(0, 2, size=100)
    assert silhouette_score(x, labels) < 0.1


def test_fail_closed() -> None:
    x = np.random.default_rng(0).standard_normal((10, 2))
    with pytest.raises(ValueError):
        silhouette_score(x, np.zeros(10, dtype=int))  # single cluster
    with pytest.raises(ValueError):
        calinski_harabasz(x, np.zeros(5, dtype=int))  # misaligned
