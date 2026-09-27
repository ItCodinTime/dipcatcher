"""Worker-count bit identity, checkpoint resume, and honest tail reports."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from quant_fund.mc_engine.engine import EngineConfig, resume_simulation, run_simulation
from quant_fund.mc_engine.scenario import GbmPortfolioGenerator, IdentityShockGenerator
from quant_fund.research.catalog.constants import FORBIDDEN_RESEARCH_METRIC_KEYS


def _walk_keys(document: object) -> set[str]:
    found: set[str] = set()
    if isinstance(document, dict):
        for key, value in document.items():
            found.add(str(key))
            found |= _walk_keys(value)
    elif isinstance(document, list):
        for item in document:
            found |= _walk_keys(item)
    return found


def _identity_config(**overrides: object) -> EngineConfig:
    base = dict(
        n_paths=256,
        chunk_size=64,
        seed=11,
        workers=1,
        backend="serial",
        shock_mode="crude",
        memory_mode="exact",
        measure_variance_reduction=True,
        ruin_level=0.5,
    )
    base.update(overrides)
    return EngineConfig(**base)  # type: ignore[arg-type]


def test_worker_count_does_not_change_a_single_bit() -> None:
    generator = IdentityShockGenerator(n_steps=4, n_factors=2)
    config = _identity_config(n_paths=512, chunk_size=128)
    serial = run_simulation(generator, config)
    process = run_simulation(generator, replace(config, workers=3, backend="process"))
    assert serial["fingerprint"] == process["fingerprint"]
    assert (
        serial["expected_shortfall"]["0.975"]["estimate"]
        == process["expected_shortfall"]["0.975"]["estimate"]
    )
    assert serial["moments"]["mean_loss"] == process["moments"]["mean_loss"]
    assert serial["probability_of_ruin"]["n_ruined"] == process["probability_of_ruin"]["n_ruined"]
    assert serial["research_only"] is True
    assert serial["live_pnl_claim"] is False
    assert serial["market_evidence"] is False


def test_report_keys_avoid_forbidden_performance_headlines() -> None:
    generator = GbmPortfolioGenerator(
        mu=[0.01],
        covariance=[[0.04]],
        weights=[1.0],
        n_steps=8,
    )
    report = run_simulation(generator, _identity_config(n_paths=2_000, chunk_size=500, seed=2))
    assert _walk_keys(report).isdisjoint(FORBIDDEN_RESEARCH_METRIC_KEYS)
    assert report["data_source"] == "SYNTHETIC"
    es_975 = report["expected_shortfall"]["0.975"]["estimate"]
    es_99 = report["expected_shortfall"]["0.99"]["estimate"]
    assert es_99 >= es_975
    assert report["expected_shortfall"]["0.975"]["ci_low"] is not None
    assert report["probability_of_ruin"]["ci_method"] == "wilson"
    assert 0.0 <= report["probability_of_ruin"]["estimate"] <= 1.0
    assert report["evt"]["available"] is True
    assert report["time_to_recovery"]["censored_excluded_from_mean"] is True


def test_antithetic_identity_mean_is_exactly_zero() -> None:
    generator = IdentityShockGenerator(n_steps=1, n_factors=1)
    report = run_simulation(
        generator,
        _identity_config(n_paths=128, chunk_size=32, shock_mode="antithetic", seed=4),
    )
    assert report["moments"]["mean_loss"] == 0.0
    antithetic = report["variance_reduction"]["antithetic"]
    # Pair averages are ~0. A few ulps of numeraire arithmetic can leave a
    # tiny residual variance, which is a finite astronomical factor, not a
    # fabricated moderate one.
    if antithetic["variance_reduction_factor_infinite"]:
        assert antithetic["variance_reduced_estimator"] == 0.0
    else:
        assert float(antithetic["variance_reduction_factor"]) > 1e12


def test_control_variate_reports_a_finite_factor_without_clamping() -> None:
    generator = IdentityShockGenerator(n_steps=1, n_factors=1)
    report = run_simulation(
        generator,
        _identity_config(
            n_paths=500,
            chunk_size=100,
            control_variate=True,
            pilot_every=5,
            seed=6,
        ),
    )
    block = report["variance_reduction"]["control_variate"]
    # Loss is 1 - (1 + shock), so the population coefficient is near -1 and
    # not exactly -1 in float64. The factor is still enormous.
    assert abs(float(block["coefficient"]) + 1.0) < 0.02
    if block["variance_reduction_factor_infinite"]:
        assert block["variance_reduction_factor"] is None
    else:
        assert float(block["variance_reduction_factor"]) > 1_000.0


def test_importance_sampling_reports_ess_and_a_measured_factor() -> None:
    generator = IdentityShockGenerator(n_steps=4, n_factors=1)
    report = run_simulation(
        generator,
        _identity_config(
            n_paths=2_000,
            chunk_size=400,
            shock_mode="importance",
            importance_shift=-0.5,
            seed=8,
        ),
    )
    importance = report["importance"]
    assert importance["effective_sample_size_fraction"] > 0.05
    assert abs(importance["mean_weight"] - 1.0) < 0.1
    factor = report["variance_reduction"]["importance_sampling"]
    assert (
        factor["variance_reduction_factor"] is not None
        or factor["variance_reduction_factor_infinite"]
    )
    assert report["evt"]["available"] is False
    assert "weighted" in report["evt"]["reason"]


def test_qmc_is_deterministic_across_workers_and_does_not_invent_a_single_scramble_factor() -> None:
    generator = IdentityShockGenerator(n_steps=2, n_factors=1)
    config = _identity_config(
        n_paths=128,
        chunk_size=32,
        shock_mode="qmc_sobol",
        n_scrambles=1,
        seed=10,
    )
    serial = run_simulation(generator, config)
    process = run_simulation(generator, replace(config, workers=2, backend="process"))
    assert serial["fingerprint"] == process["fingerprint"]
    qmc = serial["variance_reduction"]["quasi_monte_carlo"]
    assert qmc["variance_reduction_factor"] is None
    assert "n_scrambles" in qmc["reason"]
    assert serial["expected_shortfall"]["0.975"]["ci_low"] is None
    assert serial["sobol"]["balance_power_of_two"] is True


def test_qmc_replicates_measure_a_factor_from_real_scrambles() -> None:
    generator = IdentityShockGenerator(n_steps=1, n_factors=1)
    report = run_simulation(
        generator,
        _identity_config(
            n_paths=64,
            chunk_size=32,
            shock_mode="qmc_sobol",
            n_scrambles=4,
            seed=12,
        ),
    )
    qmc = report["variance_reduction"]["quasi_monte_carlo"]
    assert len(qmc["scramble_means"]) == 4
    assert qmc["scramble_means"][0] == pytest.approx(report["moments"]["mean_loss"])
    assert qmc["variance_reduction_factor"] is not None or qmc["variance_reduction_factor_infinite"]
    assert qmc["rqmc_mean_standard_error"] is not None


def test_sketch_mode_is_labeled_approximate_and_still_worker_stable() -> None:
    generator = GbmPortfolioGenerator(mu=[0.0], covariance=[[0.09]], weights=[1.0], n_steps=5)
    config = _identity_config(n_paths=300, chunk_size=100, memory_mode="sketch", seed=13)
    serial = run_simulation(generator, config)
    process = run_simulation(generator, replace(config, workers=3, backend="process"))
    assert serial["fingerprint"] == process["fingerprint"]
    assert serial["expected_shortfall"]["0.975"]["approximate"] is True
    assert serial["expected_shortfall"]["0.975"]["ci_low"] is None
    assert serial["value_at_risk"]["0.975"]["psquare"] is None
    assert serial["evt"]["available"] is False


def test_checkpoint_resume_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    generator = GbmPortfolioGenerator(
        mu=[0.0, 0.01], covariance=[[0.04, 0.0], [0.0, 0.01]], weights=[0.5, 0.5], n_steps=5
    )
    full = run_simulation(
        generator,
        _identity_config(n_paths=240, chunk_size=60, seed=15, checkpoint_dir=None),
    )
    partial = run_simulation(
        generator,
        _identity_config(
            n_paths=240,
            chunk_size=60,
            seed=15,
            checkpoint_dir=str(tmp_path),
            stop_after_new_chunks=2,
        ),
    )
    assert partial["status"] == "incomplete"
    assert partial["chunks_completed"] == 2
    # A mismatched generator is refused.
    other = IdentityShockGenerator(n_steps=5, n_factors=1)
    with pytest.raises(ValueError, match="fingerprint"):
        resume_simulation(tmp_path, other)
    resumed = resume_simulation(tmp_path, generator, workers=2, backend="process")
    assert resumed["status"] == "complete"
    assert resumed["fingerprint"] == full["fingerprint"]
    # Resuming a finished checkpoint does not change the bits.
    again = resume_simulation(tmp_path, generator, workers=1, backend="serial")
    assert again["fingerprint"] == full["fingerprint"]


def test_progress_callback_is_monotonic() -> None:
    seen: list[int] = []

    def progress(event: dict[str, float | int]) -> None:
        seen.append(int(event["chunks_completed"]))

    generator = IdentityShockGenerator(n_steps=2, n_factors=1)
    run_simulation(
        generator, _identity_config(n_paths=90, chunk_size=30, seed=16), progress=progress
    )
    assert seen == [1, 2, 3]


def test_ray_backend_without_ray_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "ray":
            raise ImportError("ray missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    generator = IdentityShockGenerator(n_steps=1, n_factors=1)
    with pytest.raises(ImportError, match="ray is not installed"):
        run_simulation(generator, _identity_config(n_paths=8, chunk_size=4, backend="ray"))


def test_validation_rejects_inconsistent_designs() -> None:
    generator = IdentityShockGenerator(n_steps=2, n_factors=1)
    with pytest.raises(ValueError):
        run_simulation(generator, _identity_config(n_paths=0))
    with pytest.raises(ValueError):
        run_simulation(
            generator, _identity_config(shock_mode="antithetic", n_paths=5, chunk_size=2)
        )
    with pytest.raises(ValueError):
        run_simulation(generator, _identity_config(shock_mode="crude", n_scrambles=3))


@pytest.mark.slow
def test_one_hundred_thousand_paths_match_across_worker_counts() -> None:
    generator = GbmPortfolioGenerator(mu=[0.0], covariance=[[0.04]], weights=[1.0], n_steps=8)
    config = EngineConfig(
        n_paths=100_000,
        chunk_size=10_000,
        seed=21,
        workers=1,
        backend="serial",
        shock_mode="crude",
        memory_mode="exact",
        measure_variance_reduction=False,
    )
    serial = run_simulation(generator, config)
    parallel = run_simulation(generator, replace(config, workers=2, backend="process"))
    assert serial["fingerprint"] == parallel["fingerprint"]
    assert serial["n_paths"] == 100_000
    assert (
        serial["expected_shortfall"]["0.99"]["estimate"]
        >= serial["expected_shortfall"]["0.975"]["estimate"]
    )
    # The document round-trips as strict JSON.
    json.dumps(serial, allow_nan=False)


def test_merge_order_of_completed_chunks_does_not_matter(tmp_path: Path) -> None:
    """Chunks saved out of completion order reload into the same fingerprint."""
    generator = IdentityShockGenerator(n_steps=3, n_factors=1)
    first = run_simulation(
        generator,
        _identity_config(n_paths=160, chunk_size=40, seed=18, checkpoint_dir=str(tmp_path / "a")),
    )
    second = run_simulation(
        generator,
        _identity_config(
            n_paths=160,
            chunk_size=40,
            seed=18,
            checkpoint_dir=str(tmp_path / "b"),
            backend="process",
            workers=4,
        ),
    )
    assert first["fingerprint"] == second["fingerprint"]
