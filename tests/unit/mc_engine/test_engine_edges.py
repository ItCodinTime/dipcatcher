"""mc_engine/engine edge paths: config validation matrix, simulate_chunk
guards, checkpoint/resume rejections, and the missing-ray import path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_fund.mc_engine.engine import (
    ChunkTask,
    EngineConfig,
    _load_manifest,
    resume_simulation,
    run_simulation,
    simulate_chunk,
)
from quant_fund.mc_engine.scenario import GbmPortfolioGenerator, IdentityShockGenerator

pytestmark = pytest.mark.synthetic


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


class TestValidate:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"n_paths": 0}, "n_paths"),
            ({"n_paths": True}, "n_paths"),
            ({"chunk_size": 0}, "chunk_size"),
            ({"chunk_size": True}, "chunk_size"),
            ({"seed": -1}, "seed"),
            ({"seed": True}, "seed"),
            ({"workers": 0}, "workers"),
            ({"workers": True}, "workers"),
            ({"backend": "x"}, "backend"),
            ({"shock_mode": "x"}, "shock_mode"),
            ({"memory_mode": "x"}, "memory_mode"),
            ({"shock_mode": "antithetic", "n_paths": 255}, "antithetic"),
            ({"shock_mode": "antithetic", "chunk_size": 65}, "antithetic"),
            ({"n_scrambles": 2}, "n_scrambles"),
            ({"n_scrambles": 0}, "n_scrambles"),
            ({"pilot_every": 1}, "pilot_every"),
            ({"ruin_level": float("nan")}, "ruin_level"),
            ({"es_levels": ()}, "es_levels"),
            ({"es_levels": (1.5,)}, "es_levels"),
            ({"ci_level": 1.0}, "ci_level"),
            ({"tdigest_compression": 5.0}, "tdigest_compression"),
            ({"importance_shift": float("inf")}, "importance_shift"),
            ({"evt_threshold": float("nan")}, "evt_threshold"),
            ({"max_exceedances_per_chunk": 0}, "max_exceedances"),
        ],
    )
    def test_invalid_config_rejected(self, overrides, match) -> None:
        generator = IdentityShockGenerator(n_steps=4, n_factors=2)
        with pytest.raises(ValueError, match=match):
            run_simulation(generator, _identity_config(**overrides))

    def test_external_shock_generator_required(self) -> None:
        generator = IdentityShockGenerator(n_steps=4, n_factors=2, accepts_external_shocks=False)
        with pytest.raises(ValueError, match="external shocks"):
            run_simulation(
                generator, _identity_config(shock_mode="importance", importance_shift=0.1)
            )


class _BadShapeGenerator:
    """Generator that returns a wrongly-shaped returns matrix."""

    name = "bad_shape"
    n_steps = 4
    n_factors = 2
    accepts_external_shocks = True
    data_source = "SYNTHETIC"

    def spec_dict(self) -> dict[str, Any]:
        return {"type": self.name}

    def generate(self, indices, shocks=None, seed=0):
        class Batch:
            returns = np.zeros((len(indices), 3))  # wrong n_steps
            log_importance_weight = None

        return Batch()


class _WeightedGenerator:
    """Generator emitting log importance weights the engine must fold in."""

    name = "weighted"
    n_steps = 4
    n_factors = 1
    accepts_external_shocks = True
    data_source = "SYNTHETIC"

    def spec_dict(self) -> dict[str, Any]:
        return {"type": self.name}

    def generate(self, indices, shocks=None, seed=0):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0, 0.05, size=(len(indices), self.n_steps))

        class Batch:
            log_importance_weight = np.zeros(len(indices))
            control = None
            control_mean = None

        batch = Batch()
        batch.returns = returns
        return batch


def _task(generator, **overrides) -> ChunkTask:
    base = dict(
        chunk_id=0,
        n_paths=32,
        chunk_size=32,
        seed=3,
        n_steps=generator.n_steps,
        n_factors=generator.n_factors,
        shock_mode="crude",
        importance_shift=None,
        qmc_scramble=True,
        qmc_seed=0,
        memory_mode="exact",
        ruin_level=0.5,
        es_levels=(0.95,),
        compression=100.0,
        pilot_every=2,
        evt_threshold=None,
        max_exceedances=10,
        measure_reference_crude=False,
        accepts_external_shocks=generator.accepts_external_shocks,
        generator=generator,
    )
    base.update(overrides)
    return ChunkTask(**base)  # type: ignore[arg-type]


class TestSimulateChunk:
    def test_external_shock_mode_rejected(self) -> None:
        task = _task(
            _BadShapeGenerator(),
            shock_mode="importance",
            accepts_external_shocks=False,
        )
        with pytest.raises(ValueError, match="external shocks"):
            simulate_chunk(task)

    def test_bad_return_shape(self) -> None:
        task = _task(_BadShapeGenerator())
        with pytest.raises(ValueError, match="generator returned shape"):
            simulate_chunk(task)

    def test_log_importance_weights_fold_in(self) -> None:
        summary = simulate_chunk(_task(_WeightedGenerator()))
        assert summary.chunk_id == 0


class TestManifest:
    def test_manifest_not_object(self, tmp_path) -> None:
        (tmp_path / "manifest.json").write_text("[1]")
        with pytest.raises(ValueError, match="not an object"):
            _load_manifest(tmp_path)

    def test_manifest_missing(self, tmp_path) -> None:
        assert _load_manifest(tmp_path) is None


class TestResume:
    def test_no_manifest(self, tmp_path) -> None:
        generator = IdentityShockGenerator(n_steps=4, n_factors=1)
        with pytest.raises(ValueError, match="no checkpoint manifest"):
            resume_simulation(tmp_path, generator)

    def _partial_checkpoint(self, tmp_path: Path) -> GbmPortfolioGenerator:
        generator = GbmPortfolioGenerator(mu=[0.0], covariance=[[0.09]], weights=[1.0], n_steps=5)
        run_simulation(
            generator,
            _identity_config(
                n_paths=240,
                chunk_size=60,
                seed=15,
                checkpoint_dir=str(tmp_path),
                stop_after_new_chunks=2,
            ),
        )
        return generator

    def test_payload_missing_and_bad_backend(self, tmp_path) -> None:
        generator = self._partial_checkpoint(tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        manifest.pop("payload")
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="payload"):
            resume_simulation(tmp_path, generator)

    def test_bad_backend(self, tmp_path) -> None:
        generator = self._partial_checkpoint(tmp_path)
        with pytest.raises(ValueError, match="backend"):
            resume_simulation(tmp_path, generator, backend="weird")

    def test_generator_fingerprint_mismatch(self, tmp_path) -> None:
        self._partial_checkpoint(tmp_path)
        other = IdentityShockGenerator(n_steps=5, n_factors=1)
        with pytest.raises(ValueError, match="fingerprint"):
            resume_simulation(tmp_path, other)

    def test_chunk_missing_on_resume(self, tmp_path) -> None:
        generator = self._partial_checkpoint(tmp_path)
        chunks = sorted(tmp_path.glob("chunk_*.npz"))
        assert chunks
        chunks[-1].unlink()
        with pytest.raises(ValueError, match="missing"):
            resume_simulation(tmp_path, generator)


class TestRayBackend:
    def test_ray_missing(self, monkeypatch) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ray":
                raise ImportError("no ray")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        generator = IdentityShockGenerator(n_steps=4, n_factors=1)
        with pytest.raises(ImportError, match="ray"):
            run_simulation(generator, _identity_config(backend="ray", workers=2))
