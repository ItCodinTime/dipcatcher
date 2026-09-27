"""Tests for models/spectral_clustering.py — Ng-Jordan-Weiss clustering."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.spectral_clustering import spectral_clustering


def _two_blobs(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((40, 2)) * 0.3 + np.array([0.0, 0.0])
    b = rng.standard_normal((40, 2)) * 0.3 + np.array([6.0, 6.0])
    x = np.vstack([a, b])
    truth = np.array([0] * 40 + [1] * 40)
    return x, truth


def test_recovers_two_separated_blobs() -> None:
    x, truth = _two_blobs()
    out = spectral_clustering(x, n_clusters=2, seed=0)
    labels = np.asarray(out["labels"])
    # accuracy up to a label swap
    acc = max((labels == truth).mean(), (labels != truth).mean())
    assert acc > 0.95


def test_labels_shape_and_values() -> None:
    x, _ = _two_blobs(seed=3)
    out = spectral_clustering(x, n_clusters=2, seed=1)
    labels = np.asarray(out["labels"])
    assert labels.shape == (80,)
    assert set(np.unique(labels).tolist()) <= {0, 1}


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        spectral_clustering(np.ones((3, 2)), n_clusters=1)
    with pytest.raises(ValueError):
        spectral_clustering(np.ones((2, 2)), n_clusters=5)  # n < clusters
