"""Tests for quant_fund.validation.regime_eval — regime-conditional evaluation gate."""

from dataclasses import replace

import numpy as np
import pytest

from quant_fund.validation.regime_eval import (
    RegimeSummary,
    regime_conditional_summary,
    regime_eval_gate,
    regime_stratified_evalue,
)

N_BOOT = 200  # spec-fixed bootstrap replicates


def _dates(n: int) -> np.ndarray:
    return np.datetime64("2020-01-01") + np.arange(n).astype("timedelta64[D]")


def _blocked_regimes(sizes: list[int], labels: list[str]) -> np.ndarray:
    return np.concatenate(
        [np.full(s, lab, dtype=object) for s, lab in zip(sizes, labels, strict=True)]
    )


def _blocked_scores(sizes: list[int], means: list[float], std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(m, std, s) for s, m in zip(sizes, means, strict=True)])


def _contradiction_report() -> dict:
    """A strictly better in regime 'bull' only, placed LAST so the pooled
    predictable bound is already inflated by the heavy-noise 'bear' regime
    when the bull segment starts (pooled path stays below threshold)."""
    rng = np.random.default_rng(21)
    n_bear, n_bull = 540, 60
    d_bear = rng.normal(0.0, 10.0, n_bear)  # null, large scale inflates pooled bound
    d_bull = np.full(n_bull, 2.0)  # A better by 2 units of loss
    d = np.concatenate([d_bear, d_bull])
    regimes = _blocked_regimes([n_bear, n_bull], ["bear", "bull"])
    return regime_stratified_evalue(
        _dates(d.size), regimes, np.zeros(d.size), d, rng=np.random.default_rng(22)
    )


# ---------------------------------------------------------------------------
# (a) known regime means are recovered
# ---------------------------------------------------------------------------


def test_summary_recovers_known_regime_means() -> None:
    sizes = [200, 200, 200]
    means = [0.5, 1.5, 2.5]
    scores = _blocked_scores(sizes, means, 1.0, seed=101)
    regimes = _blocked_regimes(sizes, ["calm", "stress", "crisis"])
    summary = regime_conditional_summary(
        _dates(sum(sizes)), regimes, scores, regime_names={"calm": "Calm"}
    )
    assert summary.names == ("Calm", "stress", "crisis")
    assert summary.pooled_n == 600
    assert np.all(summary.counts == np.array([200, 200, 200]))
    for r, mean in enumerate(means):
        assert abs(summary.means[r] - mean) < 0.2
        expected_se = 1.0 / np.sqrt(200)
        assert abs(summary.stderrs[r] - expected_se) < 0.03
    assert abs(summary.pooled_mean - 1.5) < 0.1
    assert abs(summary.diffs[0] - (0.5 - 1.5)) < 0.05
    assert summary.n_boot == N_BOOT
    assert np.all(summary.diff_ci_high > summary.diff_ci_low)


def test_summary_is_deterministic_run_to_run() -> None:
    sizes = [150, 150]
    scores = _blocked_scores(sizes, [1.0, 2.0], 1.0, seed=7)
    regimes = _blocked_regimes(sizes, ["a", "b"])
    s1 = regime_conditional_summary(_dates(300), regimes, scores)
    s2 = regime_conditional_summary(_dates(300), regimes, scores)
    np.testing.assert_allclose(s1.diff_ci_low, s2.diff_ci_low)
    np.testing.assert_allclose(s1.diff_ci_high, s2.diff_ci_high)


# ---------------------------------------------------------------------------
# (b) bootstrap CIs cover the true per-regime-vs-pooled difference at 90%
# ---------------------------------------------------------------------------


def test_bootstrap_ci_coverage_at_nominal_90pct() -> None:
    """Per-regime coverage of the true mean-minus-pooled difference at
    nominal 90%, pooled over regime-replicate pairs (diff statistics are
    linearly dependent across regimes, so the criterion is per-regime)."""
    n_mc = 200
    sizes = [100, 100, 100]
    means = np.array([0.5, 1.5, 2.0])
    true_pooled = float(np.mean(means))  # equal shares
    true_diffs = means - true_pooled
    hits = 0
    pairs = 0
    for i in range(n_mc):
        scores = _blocked_scores(sizes, list(means), 1.0, seed=7000 + i)
        regimes = _blocked_regimes(sizes, ["r0", "r1", "r2"])
        summary = regime_conditional_summary(
            _dates(300),
            regimes,
            scores,
            rng=np.random.default_rng(9000 + i),
        )
        covered = np.logical_and(
            summary.diff_ci_low <= true_diffs, true_diffs <= summary.diff_ci_high
        )
        hits += int(np.sum(covered))
        pairs += covered.size
    rate = hits / pairs
    assert rate >= 0.90 - 0.05, f"pooled coverage {rate:.3f} below 0.85"


# ---------------------------------------------------------------------------
# (c) anytime-validity smoke: no-difference e-paths rarely cross 1/alpha
# ---------------------------------------------------------------------------


def test_eprocess_no_difference_respects_ville_threshold() -> None:
    n_rep = 200
    n = 400
    threshold = 20.0  # 1 / 0.05
    pooled_crossings = 0
    regime_crossings = 0
    regime_paths = 0
    for i in range(n_rep):
        rng = np.random.default_rng(5000 + i)
        d = rng.normal(0.0, 1.0, n)
        regimes = _blocked_regimes([n // 2, n // 2], ["x", "y"])
        report = regime_stratified_evalue(
            _dates(n), regimes, np.zeros(n), d, rng=np.random.default_rng(6000 + i)
        )
        assert report["threshold"] == threshold
        pooled_path = np.asarray(report["pooled"]["e"], dtype=float)
        assert pooled_path.ndim == 1 and pooled_path.size == n
        assert bool(np.all(pooled_path > 0.0))
        if bool(np.any(pooled_path >= threshold)):
            pooled_crossings += 1
        for info in report["regimes"].values():
            regime_paths += 1
            path = np.asarray(info["e"], dtype=float)
            assert bool(np.all(path > 0.0))
            if bool(np.any(path >= threshold)):
                regime_crossings += 1
    pooled_rate = pooled_crossings / n_rep
    regime_rate = regime_crossings / regime_paths
    assert pooled_rate <= 0.10, f"pooled Ville violation rate {pooled_rate:.3f}"
    assert regime_rate <= 0.15, f"per-regime Ville violation rate {regime_rate:.3f}"


def test_eprocess_recovers_regime_localized_advantage() -> None:
    report = _contradiction_report()
    assert report["regimes"]["bull"]["crossed"] is True
    assert report["regimes"]["bull"]["mean_diff"] == pytest.approx(2.0)
    assert report["pooled"]["crossed"] is False
    lo, hi = report["regimes"]["bull"]["diff_ci90"]
    assert lo <= 2.0 <= hi


# ---------------------------------------------------------------------------
# (d) gate: fails on heterogeneous / contradicted regimes, passes homogeneous
# ---------------------------------------------------------------------------


def _summary_for_gate(sizes: list[int], means: list[float], seed: int) -> RegimeSummary:
    scores = _blocked_scores(sizes, means, 0.5, seed=seed)
    regimes = _blocked_regimes(sizes, [f"g{i}" for i in range(len(sizes))])
    return regime_conditional_summary(_dates(sum(sizes)), regimes, scores)


def test_gate_fails_on_heterogeneous_regime_means() -> None:
    summary = _summary_for_gate([400, 400, 400], [3.0, -3.0, 3.0], seed=11)
    result = regime_eval_gate(summary)
    assert result.passed is False
    assert any(r.startswith("regime_mean_cv_exceeded") for r in result.reasons)


def test_gate_passes_on_homogeneous_regime_means() -> None:
    summary = _summary_for_gate([400, 400, 400], [1.5, 1.5, 1.5], seed=12)
    result = regime_eval_gate(summary)
    assert result.passed is True
    assert result.reasons == ()


def test_gate_fails_on_small_regime_share_with_evidence() -> None:
    summary = _summary_for_gate([2900, 100], [1.5, 1.5], seed=13)
    result = regime_eval_gate(summary)
    assert result.passed is False
    assert any(r.startswith("insufficient_regime_share") for r in result.reasons)


def test_gate_fails_when_regime_evalue_contradicts_pooled() -> None:
    report = _contradiction_report()
    assert report["pooled"]["crossed"] is False
    # Homogeneous, balanced summary => only the contradiction can fire.
    summary = _summary_for_gate([300, 300], [1.5, 1.5], seed=23)
    result = regime_eval_gate(replace(summary, e_value_report=report))
    assert result.passed is False
    assert "regime_contradicts_pooled:bull" in result.reasons


def test_gate_passes_when_pooled_also_crosses() -> None:
    """If the pooled e-value crosses too, a crossing regime is no contradiction."""
    n = 400
    d = np.full(n, 1.0)  # A uniformly better: both pooled and per-regime cross
    regimes = _blocked_regimes([n // 2, n // 2], ["u", "v"])
    report = regime_stratified_evalue(
        _dates(n), regimes, np.zeros(n), d, rng=np.random.default_rng(31)
    )
    assert report["pooled"]["crossed"] is True
    summary = _summary_for_gate([200, 200], [1.5, 1.5], seed=32)
    result = regime_eval_gate(replace(summary, e_value_report=report))
    assert result.passed is True


# ---------------------------------------------------------------------------
# (e) fail-closed edges
# ---------------------------------------------------------------------------


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        regime_conditional_summary(_dates(10), np.full(10, "a", dtype=object), np.zeros(9))


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        regime_conditional_summary(
            np.array([], dtype="datetime64[D]"),
            np.array([], dtype=object),
            np.zeros(0),
        )


def test_single_regime_raises() -> None:
    with pytest.raises(ValueError, match="single unique"):
        regime_conditional_summary(_dates(50), np.full(50, "only", dtype=object), np.zeros(50))


def test_nonfinite_scores_raise() -> None:
    scores = np.zeros(50)
    scores[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        regime_conditional_summary(_dates(50), _blocked_regimes([25, 25], ["a", "b"]), scores)


def test_unsortable_dates_raise() -> None:
    with pytest.raises(ValueError, match="sortable"):
        regime_conditional_summary(
            np.asarray(["x", 1, "y", 2], dtype=object),
            np.asarray(["a", "b", "a", "b"], dtype=object),
            np.zeros(4),
        )


def test_evalue_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        regime_stratified_evalue(
            _dates(20), _blocked_regimes([10, 10], ["a", "b"]), np.zeros(20), np.zeros(19)
        )


def test_evalue_single_regime_raises() -> None:
    with pytest.raises(ValueError, match="single unique"):
        regime_stratified_evalue(
            _dates(20), np.full(20, "only", dtype=object), np.zeros(20), np.zeros(20)
        )


def test_gate_invalid_thresholds_raise() -> None:
    summary = _summary_for_gate([50, 50], [1.0, 1.0], seed=61)
    for bad_share in (0.0, 1.0, 1.5, np.nan):
        with pytest.raises(ValueError):
            regime_eval_gate(summary, min_regime_share=bad_share)
    for bad_cv in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError):
            regime_eval_gate(summary, max_regime_cv=bad_cv)


def test_gate_ignores_absent_evalue_report() -> None:
    summary = _summary_for_gate([100, 100], [1.5, 1.5], seed=71)
    result = regime_eval_gate(summary)  # no e_value_report attached
    assert result.passed is True
