"""Tests for metrics/ledoit_wolf_sharpe.py — Sharpe difference test."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.ledoit_wolf_sharpe import sharpe_difference_test


def test_equal_sharpe_not_rejected() -> None:
    rng = np.random.default_rng(0)
    r1 = 0.001 + 0.02 * rng.standard_normal(2000)
    r2 = 0.001 + 0.02 * rng.standard_normal(2000)
    out = sharpe_difference_test(r1, r2)
    assert out["pvalue"] > 0.1
    assert abs(out["diff"]) < 0.2


def test_different_sharpe_rejected() -> None:
    rng = np.random.default_rng(1)
    r1 = 0.004 + 0.01 * rng.standard_normal(3000)  # high Sharpe
    r2 = 0.000 + 0.03 * rng.standard_normal(3000)  # low Sharpe
    out = sharpe_difference_test(r1, r2)
    assert out["sr1"] > out["sr2"]
    assert out["pvalue"] < 0.05


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        sharpe_difference_test(np.ones(10), np.ones(10))  # too short
    with pytest.raises(ValueError):
        sharpe_difference_test(np.zeros(50), np.zeros(50))  # degenerate variance


@pytest.mark.parametrize("lag", [-1, 50, 1.5, True])
def test_invalid_hac_lag_rejected(lag: object) -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=50)
    b = rng.normal(size=50)
    with pytest.raises(ValueError, match="lag"):
        sharpe_difference_test(a, b, lag=lag)  # type: ignore[arg-type]
