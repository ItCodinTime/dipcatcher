"""Hypothesis identity tests for the reality filter (PROOFCORE W4 §7.8, A3 #5).

All tests run under a derandomized profile: identities are deterministic
mathematical facts, not sampling statements.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.metrics.overfitting import probabilistic_sharpe
from quant_fund.proofcore.contracts import TrialLedgerRow
from quant_fund.reality.cscv import pbo_from_performance
from quant_fund.reality.dsr import dsr_from_ledger, effective_trials
from quant_fund.reality.fdr import bh_fdr
from quant_fund.reality.psr import min_trl_from_returns, psr_from_returns

DERANDOMIZED = settings(derandomize=True, max_examples=60, deadline=None)

_HEX = "ab" * 32

returns_st = st.lists(
    st.floats(-0.2, 0.2, allow_nan=False, allow_infinity=False),
    min_size=8,
    max_size=400,
).map(np.asarray)


def _nonconstant(r: np.ndarray) -> np.ndarray:
    # keep moments well-formed: nonzero sample std
    return r if np.std(r, ddof=1) > 1e-12 else np.linspace(-0.01, 0.01, r.size)


def _make_row(i: int, *, sr: float, cluster: str | None = None, n_obs: int = 500,
              family: str = "discovery") -> TrialLedgerRow:
    return TrialLedgerRow(
        trial_id=f"{i:064x}",
        bundle_hash=_HEX,
        family=family,  # type: ignore[arg-type]
        strategy=f"s{i}",
        cluster_id=cluster or f"c{i}",
        created_utc="2026-01-01T00:00:00+00:00",
        n_obs=n_obs,
        periods_per_year=252.0,
        sharpe_periodic=sr,
        skew=0.0,
        kurtosis_raw=3.0,
        returns_sha256="cd" * 32,
    )


# 1. PSR identity: sr_star = sr_periodic(r) -> PSR == 0.5 exactly.
@DERANDOMIZED
@given(returns_st)
def test_psr_identity_at_own_sharpe(r: np.ndarray) -> None:
    r = _nonconstant(r)
    info = psr_from_returns(r, sr_star=0.0, periods_per_year=252.0)
    at_self = psr_from_returns(r, sr_star=info["sr_periodic"], periods_per_year=252.0)
    assert at_self["psr"] == pytest.approx(0.5, abs=1e-12)


# 2. DSR monotonicity: more effective trials -> DSR non-increasing (fixed data).
@DERANDOMIZED
@given(
    st.lists(
        st.floats(-0.05, 0.15, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    )
)
def test_dsr_non_increasing_in_effective_trials(srs: list[float]) -> None:
    flat = [_make_row(i, sr=s, cluster="one") for i, s in enumerate(srs)]
    split = [_make_row(i, sr=s) for i, s in enumerate(srs)]
    assert dsr_from_ledger(split) <= dsr_from_ledger(flat) + 1e-12


# 3. DSR <-> PSR degeneracy: a single trial gives DSR == PSR vs sr_star=0.
@DERANDOMIZED
@given(
    st.floats(-0.1, 0.2, allow_nan=False, allow_infinity=False),
    st.integers(4, 2000),
)
def test_dsr_psr_degeneracy_single_trial(sr: float, n_obs: int) -> None:
    row = _make_row(1, sr=sr, n_obs=n_obs)
    assert dsr_from_ledger([row]) == pytest.approx(
        probabilistic_sharpe(sr, 0.0, n_obs, 0.0, 3.0), abs=1e-12
    )


# 4+5. PBO symmetry and bounds on full-rank inputs.
@DERANDOMIZED
@given(
    st.integers(4, 8),
    st.integers(2, 5).map(lambda x: 2 * x),  # even n_trials: lambda == 0 impossible
    st.integers(0, 10_000),
)
def test_pbo_symmetry_and_bounds(n_comb: int, n_trials: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    is_perf = rng.normal(size=(n_comb, n_trials))
    oos_perf = rng.normal(size=(n_comb, n_trials))
    # full-rank (distinct values) almost surely; enforce exactly
    is_perf += np.arange(n_trials) * 1e-9
    oos_perf += np.arange(n_trials) * 1e-9
    fwd = pbo_from_performance(is_perf, oos_perf)
    # §7.8 identity 4, corrected: the exact complementarity is under OOS-rank
    # inversion (negating the OOS ordering leaves the IS-optimal trial fixed
    # and mirrors its rank percentile: omega -> 1 - omega, lambda -> -lambda).
    # Negating BOTH matrices changes which trial is IS-optimal and has no
    # complementary relationship — verified numerically false.
    rev = pbo_from_performance(is_perf, -oos_perf)
    assert 0.0 <= fwd["pbo"] <= 1.0
    assert all(math.isfinite(x) for x in fwd["logits"])
    if n_comb >= 4:
        assert rev["pbo"] == pytest.approx(1.0 - fwd["pbo"], abs=1e-12)
        np.testing.assert_allclose(
            np.asarray(fwd["logits"]), -np.asarray(rev["logits"]), atol=1e-12
        )


# 6. MinTRL sign: positive-SR series -> positive MinTRL; SR <= sr_star -> NaN.
@DERANDOMIZED
@given(returns_st)
def test_mintrl_sign_identity(r: np.ndarray) -> None:
    r = _nonconstant(r)
    mtrl = min_trl_from_returns(r, sr_star=0.0, periods_per_year=252.0)
    sr = float(np.mean(r) / np.std(r, ddof=1))
    if sr > 1e-9:
        assert np.isfinite(mtrl) and mtrl > 1.0
    elif sr < -1e-9:
        assert np.isnan(mtrl)


# 7. Units regression (A1 F1): zero-edge iid book -> PSR ~ 0.5; and the
# returns-only API has no `sr` parameter, so the annualized-SR bug is
# statically unexpressible.
@DERANDOMIZED
@given(st.integers(0, 10_000))
def test_units_regression_zero_edge_and_api_shape(seed: int) -> None:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.01, size=252)
    r = r - float(np.mean(r))  # exactly zero-edge: PSR must be 0.5, NOT ~1.0
    psr = psr_from_returns(r, sr_star=0.0, periods_per_year=252.0)["psr"]
    assert psr == pytest.approx(0.5, abs=1e-9)
    params = inspect.signature(psr_from_returns).parameters
    assert "sr" not in params and "sharpe" not in params
    params_mtrl = inspect.signature(min_trl_from_returns).parameters
    assert "sr" not in params_mtrl and "sharpe" not in params_mtrl


# 8. BH-FDR sanity: closed-form cutoffs — k strong + m weak p-values reject
# exactly the strong ones at q=0.05 when 0.001 <= 0.05 * k / (k + m).
@DERANDOMIZED
@given(st.integers(1, 6), st.integers(1, 6))
def test_bh_fdr_closed_form(k: int, m: int) -> None:
    from scipy.stats import norm

    strong = float(norm.isf(0.001) / np.sqrt(499))
    weak = float(norm.isf(0.9) / np.sqrt(499))
    rows = [_make_row(i + 1, sr=strong) for i in range(k)] + [
        _make_row(100 + i, sr=weak) for i in range(m)
    ]
    out = bh_fdr(rows, q=0.05)["discovery"]
    assert set(out) == {r.trial_id for r in rows[:k]}


def test_effective_trials_is_cluster_count() -> None:
    # sanity anchor (non-hypothesis): eff = #clusters regardless of sizes
    assert effective_trials([10, 10, 10]) == effective_trials([1, 1, 1]) == 3.0
