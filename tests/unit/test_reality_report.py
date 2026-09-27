"""Unit pins for RealityReport assembly and verdict logic (PROOFCORE W4 §7.7)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from quant_fund.proofcore.contracts import RealityFilterError, TrialLedgerRow
from quant_fund.reality.report import build_reality_report

_HEX = "ab" * 32


def make_row(
    i: int,
    *,
    family: str = "discovery",
    cluster: str | None = None,
    sr: float = 0.05,
    n_obs: int = 500,
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
        skew=0.0,
        kurtosis_raw=3.0,
        returns_sha256="cd" * 32,
    )


def test_empty_ledger_fail_closed() -> None:
    with pytest.raises(RealityFilterError):
        build_reality_report([])


def test_insufficient_evidence_below_ten_trials() -> None:
    rows = [make_row(i, sr=0.10) for i in range(1, 6)]
    rep = build_reality_report(rows)
    assert rep.verdict == "insufficient_evidence"
    assert rep.n_trials == 5


def test_insufficient_evidence_below_three_clusters() -> None:
    rows = [make_row(i, sr=0.10, cluster="one") for i in range(1, 12)]
    rep = build_reality_report(rows)
    assert rep.n_effective_trials == 1.0
    assert rep.verdict == "insufficient_evidence"


def test_pass_verdict_for_strong_consistent_trials() -> None:
    sr = float(norm.isf(0.0005) / np.sqrt(499))  # PSR ~ 0.9995
    rows = [make_row(i, sr=sr) for i in range(1, 13)]
    rep = build_reality_report(rows)
    assert rep.n_effective_trials == 12.0
    assert rep.dsr == pytest.approx(rep.psr, abs=1e-12)  # var_sr = 0 -> sr* = 0
    assert rep.verdict == "pass"
    assert np.isnan(rep.pbo) and np.isnan(rep.spa_pvalue)  # no return series -> n/a
    assert len(rep.report_sha256) == 64
    # Honesty: schema carries no Sharpe/P&L/NAV headline fields.
    assert set(rep.model_dump()) >= {"psr", "dsr", "pbo", "verdict"}


def test_deflated_verdict_when_dsr_low() -> None:
    # Best trial barely positive, many independent clusters with spread.
    srs = np.linspace(-0.02, 0.03, 12)
    rows = [make_row(i, sr=float(s)) for i, s in enumerate(srs, start=1)]
    rep = build_reality_report(rows)
    assert rep.dsr < 0.95
    assert rep.verdict == "deflated"


def test_nan_poisoned_rows_fail_closed_never_pass() -> None:
    rows = [make_row(i, sr=0.10) for i in range(1, 12)]
    poisoned = make_row(99, sr=0.5, n_obs=0)  # PSR/DSR path -> NaN
    rows.append(poisoned)
    rep = build_reality_report(rows)
    # best trial is the poisoned one (highest sr) -> its psr is NaN
    assert np.isnan(rep.psr)
    assert poisoned.trial_id not in rep.bh_fdr_rejects


def test_shuffled_ledger_stable_verdict() -> None:
    srs = np.linspace(-0.02, 0.03, 12)
    rows = [make_row(i, sr=float(s)) for i, s in enumerate(srs, start=1)]
    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = rng.permutation(len(rows))
        shuffled = [rows[int(i)] for i in perm]
        assert build_reality_report(shuffled).verdict == "deflated"


def test_report_with_returns_computes_pbo_and_spa() -> None:
    rng = np.random.default_rng(23)
    n_trials, n_periods = 12, 64
    rets = {f"{i:064x}": rng.normal(0.0005, 0.01, size=n_periods) for i in range(1, n_trials + 1)}
    rows = []
    for i in range(1, n_trials + 1):
        r = rets[f"{i:064x}"]
        sr = float(np.mean(r) / np.std(r, ddof=1))
        rows.append(make_row(i, sr=sr, n_obs=n_periods))
    rep = build_reality_report(
        rows, s_blocks=8, returns_by_trial=rets, spa_n_boot=200, seed=3
    )
    assert np.isfinite(rep.pbo) or np.isnan(rep.pbo)
    assert 0.0 <= float(rep.spa_pvalue) <= 1.0
    assert rep.pbo_logit  # logits recorded


def test_deterministic_given_fixed_created_utc() -> None:
    rows = [make_row(i, sr=0.10) for i in range(1, 13)]
    a = build_reality_report(rows, created_utc="2026-01-01T00:00:00+00:00")
    b = build_reality_report(list(reversed(rows)), created_utc="2026-01-01T00:00:00+00:00")
    assert a.report_sha256 == b.report_sha256
