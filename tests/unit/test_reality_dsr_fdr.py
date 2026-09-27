"""Unit pins for effective-trials DSR and BH-FDR (PROOFCORE W4 §7.2/§7.5)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.metrics.overfitting import probabilistic_sharpe
from quant_fund.proofcore.contracts import RealityFilterError, TrialLedgerRow
from quant_fund.reality.dsr import dsr_from_ledger, effective_trials
from quant_fund.reality.fdr import bh_fdr, trial_pvalue

_HEX = "ab" * 32


def make_row(
    i: int,
    *,
    family: str = "discovery",
    cluster: str | None = None,
    sr: float = 0.05,
    n_obs: int = 500,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> TrialLedgerRow:
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
        skew=skew,
        kurtosis_raw=kurt,
        returns_sha256="cd" * 32,
    )


def test_effective_trials_counts_clusters() -> None:
    assert effective_trials([5, 3, 1]) == 3.0
    assert effective_trials([1]) == 1.0


def test_effective_trials_fail_closed() -> None:
    with pytest.raises(RealityFilterError):
        effective_trials([])
    with pytest.raises(RealityFilterError):
        effective_trials([2, 0])


def test_dsr_single_trial_equals_psr() -> None:
    row = make_row(1, sr=0.08, n_obs=400)
    dsr = dsr_from_ledger([row])
    psr = probabilistic_sharpe(0.08, 0.0, 400, 0.0, 3.0)
    assert dsr == pytest.approx(psr, abs=1e-12)


def test_dsr_more_clusters_deflates() -> None:
    # Same per-trial stats, but spread across many independent clusters ->
    # the expected max SR under the null rises -> DSR drops.
    few = [make_row(i, sr=float(s), cluster="one") for i, s in enumerate(np.linspace(0.03, 0.07, 5), start=1)]
    many = [make_row(i, sr=float(s)) for i, s in enumerate(np.linspace(0.03, 0.07, 5), start=1)]
    assert dsr_from_ledger(many) < dsr_from_ledger(few)


def test_dsr_hand_computed_pin() -> None:
    rows = [make_row(1, sr=0.10), make_row(2, sr=0.05), make_row(3, sr=0.00)]
    srs = np.array([0.10, 0.05, 0.00])
    var_sr = float(np.var(srs, ddof=1))
    gamma = 0.5772156649015329
    term = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / 3.0) + gamma * norm.ppf(1.0 - 1.0 / (3.0 * np.e))
    sr_star = np.sqrt(var_sr) * term
    se = float(np.sqrt(1.0 - 0.0 * 0.10 + ((3.0 - 1.0) / 4.0) * 0.10**2))  # Lo 2002
    expected = float(norm.cdf((0.10 - sr_star) * np.sqrt(500 - 1) / se))
    assert dsr_from_ledger(rows) == pytest.approx(expected, abs=1e-12)


def test_dsr_empty_ledger_fail_closed() -> None:
    with pytest.raises(RealityFilterError):
        dsr_from_ledger([])


def test_bh_fdr_closed_form_cutoffs() -> None:
    """p = [0.001] * k + [0.9] * m: BH at q=0.05 rejects exactly the first k.

    k=2, m=3 -> m_tot=5; cutoffs 0.01, 0.02, ... — 0.001 <= 0.01 and
    0.001 <= 0.02 pass, 0.9 <= 0.03 fails, so exactly the two strong trials
    reject (pinned from the BH step-up rule).
    """
    # sr per trial tuned so 1 - PSR ~= 0.001 (strong) / 0.9 (null-ish).
    strong = norm.isf(0.001) / np.sqrt(499)  # sr s.t. Phi(sr*sqrt(n-1)) = 0.999
    weak = norm.isf(0.9) / np.sqrt(499)
    rows = [make_row(1, sr=float(strong)), make_row(2, sr=float(strong))] + [
        make_row(i, sr=float(weak)) for i in range(3, 6)
    ]
    out = bh_fdr(rows, q=0.05)
    assert sorted(out["discovery"]) == sorted([rows[0].trial_id, rows[1].trial_id])
    assert out["calibration"] == []


def test_bh_fdr_families_never_pooled_and_bound_excluded() -> None:
    strong = float(norm.isf(0.001) / np.sqrt(499))
    cal = [make_row(i, family="calibration", sr=strong) for i in range(1, 4)]
    bound = [make_row(i, family="bound", sr=strong) for i in range(4, 6)]
    out = bh_fdr(cal + bound, q=0.05)
    assert sorted(out["calibration"]) == sorted(r.trial_id for r in cal)
    assert all(r.trial_id not in out["calibration"] + out["discovery"] for r in bound)


def test_bh_fdr_nan_pvalue_never_rejects() -> None:
    poisoned = make_row(1, sr=0.5, n_obs=1)  # n_obs < 2 -> PSR NaN -> p NaN
    strong = make_row(2, sr=float(norm.isf(0.001) / np.sqrt(499)))
    out = bh_fdr([poisoned, strong], q=0.05)
    assert poisoned.trial_id not in out["discovery"]
    assert np.isnan(trial_pvalue(poisoned))


def test_bh_fdr_monotone_in_q() -> None:
    srs = np.linspace(-0.02, 0.15, 12)
    rows = [make_row(i + 1, sr=float(s)) for i, s in enumerate(srs)]
    lo = bh_fdr(rows, q=0.01)["discovery"]
    hi = bh_fdr(rows, q=0.10)["discovery"]
    assert set(lo) <= set(hi)


def test_bh_fdr_rejects_bad_q() -> None:
    with pytest.raises(ValueError, match="q must"):
        bh_fdr([make_row(1)], q=1.5)
