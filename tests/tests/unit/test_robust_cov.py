"""Tests for models/robust_cov.py — FastMCD, OGK, Stahel-Donoho."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.robust_cov import (
    fast_mcd,
    ogk_cov,
    stahel_donoho_outlyingness,
)


def _contaminated(n: int = 200, p: int = 3, frac: float = 0.1, seed: int = 7):
    rng = np.random.default_rng(seed)
    cov_true = np.array([[1.0, 0.4, 0.0], [0.4, 1.5, 0.2], [0.0, 0.2, 1.0]])[:p, :p]
    x = rng.multivariate_normal(np.zeros(p), cov_true, size=n)
    n_bad = int(frac * n)
    x[:n_bad] += np.array([8.0, -8.0, 8.0])[:p]  # gross contamination
    return x, cov_true


def test_ogk_closer_than_sample() -> None:
    x, cov_true = _contaminated()
    sample = np.cov(x.T)
    ogk = ogk_cov(x)
    err_sample = np.linalg.norm(sample - cov_true)
    err_ogk = np.linalg.norm(ogk["cov"] - cov_true)
    assert err_ogk < err_sample


def test_mcd_flags_outliers() -> None:
    x, cov_true = _contaminated(n=250, frac=0.12, seed=3)
    out = fast_mcd(x, rng=np.random.default_rng(0))
    d = np.asarray(out["distances"])
    n_bad = int(0.12 * 250)
    # the contaminated rows should have the largest robust distances
    top = np.argsort(d)[-n_bad:]
    assert np.mean(top < n_bad) > 0.7
    err = np.linalg.norm(np.asarray(out["cov"]) - cov_true)
    assert err < np.linalg.norm(np.cov(x.T) - cov_true)


def test_sd_outlyingness_detects() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 4))
    x[5] = np.array([8.0, 8.0, 8.0, 8.0])
    x[9] = np.array([-7.0, -7.0, 0.0, 0.0])
    o = stahel_donoho_outlyingness(x, n_dirs=200, rng=np.random.default_rng(0))
    top2 = np.argsort(o)[-2:]
    assert 5 in top2 or 9 in top2
    assert o[5] > np.median(o) * 3


def test_mcd_h_bound() -> None:
    x = np.random.default_rng(0).standard_normal((60, 3))
    with pytest.raises(ValueError):
        fast_mcd(x, h=2)
    out = fast_mcd(x, h=55, n_starts=20, rng=np.random.default_rng(0))
    assert out["n_kept"] >= 55 - 10


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        ogk_cov(np.random.default_rng(0).standard_normal((6, 5)))  # n <= 2p
    with pytest.raises(ValueError):
        ogk_cov(np.full((50, 3), np.nan))
    with pytest.raises(ValueError):
        ogk_cov(
            np.column_stack(
                [np.ones(50), np.random.default_rng(0).standard_normal(50), np.ones(50)]
            )
        )  # zero MAD col
    with pytest.raises(ValueError):
        stahel_donoho_outlyingness(np.random.default_rng(0).standard_normal((50, 3)), n_dirs=5)


def test_ogk_clean_data_close() -> None:
    rng = np.random.default_rng(12)
    cov_true = np.array([[1.0, 0.5], [0.5, 2.0]])
    x = rng.multivariate_normal(np.zeros(2), cov_true, size=1000)
    ogk = ogk_cov(x)
    assert np.linalg.norm(ogk["cov"] - cov_true) < 0.35
