"""Tests for research/benches_w810.py — SOTA canon wave 8-10 scorecard families."""

from __future__ import annotations

import numpy as np

from quant_fund.research.benches_w810 import (
    bench_anytime_valid,
    bench_distributional_ml,
    bench_energy_score,
    bench_leakage_redteam,
    bench_regime_eval,
    bench_ts_conformal,
)
from quant_fund.research.catalog import (
    OPTIONAL_BENCHMARK_FAMILIES,
    family_blob_forbidden_metrics_absent,
    family_blob_has_finite_observation,
)

_BENCHES = (
    bench_anytime_valid,
    bench_energy_score,
    bench_ts_conformal,
    bench_regime_eval,
    bench_leakage_redteam,
    bench_distributional_ml,
)


def test_families_registered_as_optional() -> None:
    for fam in (
        "anytime_valid",
        "energy_score",
        "ts_conformal",
        "regime_eval",
        "leakage_redteam",
        "distributional_ml",
    ):
        assert fam in OPTIONAL_BENCHMARK_FAMILIES


def test_benches_return_clean_finite_blobs() -> None:
    for bench in _BENCHES:
        blob = bench()
        assert isinstance(blob, dict) and blob, bench.__name__
        assert family_blob_has_finite_observation(blob), bench.__name__
        assert family_blob_forbidden_metrics_absent(blob), bench.__name__
        assert all(np.isfinite(v) for v in blob.values()), bench.__name__


def test_anytime_valid_controls_fdr_and_detects() -> None:
    blob = bench_anytime_valid()
    assert blob["null_fdr_empirical"] <= 0.10
    assert blob["detector_null_fa_rate"] <= 0.10
    assert blob["detector_alarm_time"] > 600.0
    assert blob["e_bh_rejected"] >= 8.0


def test_energy_score_propriety_gap_positive() -> None:
    blob = bench_energy_score()
    assert blob["propriety_gap"] > 0.0


def test_ts_conformal_coverage_and_watch_alarm() -> None:
    blob = bench_ts_conformal()
    assert blob["enbpi_abs_coverage_error"] < 0.15
    assert blob["watch_detected"] == 1.0


def test_regime_eval_gate_directions() -> None:
    blob = bench_regime_eval()
    assert blob["homo_gate_passed"] == 1.0
    assert blob["hetero_gate_failed"] == 1.0


def test_leakage_redteam_blindspot_locked() -> None:
    blob = bench_leakage_redteam()
    assert blob["audit_clean_passed"] == 1.0
    assert blob["audit_planted_flagged"] == 1.0
    assert blob["blindspot_contrast"] > 0.5
    assert blob["deflated_alpha_trials10"] < 0.05


def test_distributional_ml_beats_baselines() -> None:
    blob = bench_distributional_ml()
    assert blob["crps_improvement"] > 0.0
    assert blob["qrf_median_corr"] > 0.8


def test_benches_are_deterministic() -> None:
    for bench in _BENCHES:
        first, second = bench(), bench()
        assert first == second, bench.__name__
