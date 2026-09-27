"""Unit tests for the SPA / reality-check driver (PROOFCORE W4 §7.4)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.snooping import SpaResult, spa_test
from quant_fund.proofcore.contracts import RealityFilterError
from quant_fund.reality.spa import (
    reality_check_from_trials,
    spa_from_trials,
    stepm_from_trials,
)


def _panel(seed: int = 5, n: int = 60, k: int = 4, mu: float = 0.001) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {f"t{k_}": rng.normal(mu, 0.01, size=n) for k_ in range(k)}


def test_spa_driver_matches_direct_call() -> None:
    panel = _panel()
    mat = np.column_stack([panel[f"t{k_}"] for k_ in range(4)])
    direct = spa_test(mat, n_boot=100, seed=7)
    driven = spa_from_trials(panel, n_boot=100, seed=7)
    assert isinstance(driven, SpaResult)
    assert driven.p_consistent == pytest.approx(direct.p_consistent, abs=1e-12)
    assert driven.p_lower == pytest.approx(direct.p_lower, abs=1e-12)
    assert driven.p_upper == pytest.approx(direct.p_upper, abs=1e-12)


def test_benchmark_differentials_drop_benchmark_column() -> None:
    panel = _panel()
    driven = spa_from_trials(panel, benchmark="t0", n_boot=50, seed=7)
    assert driven.n_strategies == 3


def test_driver_fail_closed_inputs() -> None:
    with pytest.raises(RealityFilterError):
        spa_from_trials({}, n_boot=10)
    with pytest.raises(RealityFilterError, match="length mismatch"):
        spa_from_trials({"a": np.zeros(30), "b": np.zeros(31)}, n_boot=10)
    with pytest.raises(RealityFilterError, match="benchmark"):
        spa_from_trials(_panel(), benchmark="nope", n_boot=10)


def test_reality_check_and_stepm_drivers() -> None:
    panel = _panel()
    rc = reality_check_from_trials(panel, n_boot=50, seed=1)
    sm = stepm_from_trials(panel, n_boot=50, seed=1)
    assert 0.0 <= rc.p_value <= 1.0
    assert sm.n_strategies == 4
