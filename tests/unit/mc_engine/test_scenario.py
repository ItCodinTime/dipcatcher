"""Built-in generators and the plug-in contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from quant_fund.mc_engine.philox import USER_STREAM_ID_MIN, philox_normals
from quant_fund.mc_engine.scenario import (
    GbmPortfolioGenerator,
    IdentityShockGenerator,
    VolTargetStrategyGenerator,
    generator_from_spec,
)


def test_gbm_portfolio_is_long_only_and_finite() -> None:
    generator = GbmPortfolioGenerator(
        mu=[0.02, 0.05],
        covariance=[[0.04, 0.01], [0.01, 0.09]],
        weights=[0.25, 0.75],
        n_steps=6,
    )
    shocks = philox_normals(1, np.arange(32), 12, stream_id=1).reshape(32, 6, 2)
    batch = generator.generate(np.arange(32), shocks=shocks, seed=1)
    assert batch.returns.shape == (32, 6)
    assert np.all(batch.returns > -1.0)
    assert batch.control_mean == 0.0
    rebuilt = generator_from_spec(generator.spec_dict())
    again = rebuilt.generate(np.arange(32), shocks=shocks, seed=1)
    assert np.array_equal(batch.returns, again.returns)
    with pytest.raises(ValueError):
        GbmPortfolioGenerator(
            mu=[0.0, 0.0],
            covariance=[[0.04, 0.0], [0.0, 0.04]],
            weights=[0.2, 0.2],
            n_steps=2,
        )
    with pytest.raises(ValueError):
        GbmPortfolioGenerator(
            mu=[0.0],
            covariance=[[0.04]],
            weights=[-0.2],
            n_steps=2,
        )


def test_vol_target_position_uses_only_the_past() -> None:
    generator = VolTargetStrategyGenerator(
        mu=0.0, sigma=0.2, n_steps=30, lookback=5, target_vol=0.1
    )
    base = philox_normals(2, np.arange(4), 30, stream_id=1).reshape(4, 30, 1)
    mutated = base.copy()
    mutated[:, 20:, :] += 3.0
    original = generator.generate(np.arange(4), shocks=base, seed=2)
    changed = generator.generate(np.arange(4), shocks=mutated, seed=2)
    # Returns before the mutated shocks stay identical. The return on the first
    # mutated step changes because the asset move itself changed.
    assert np.array_equal(original.returns[:, :20], changed.returns[:, :20])
    assert not np.array_equal(original.returns[:, 20:], changed.returns[:, 20:])
    assert generator.spec_dict()["type"] == "vol_target_strategy"
    rebuilt = generator_from_spec(generator.spec_dict())
    assert rebuilt.lookback == 5


@dataclass(frozen=True)
class _IndexedPlugIn:
    """Stand-in for a bootstrap / copula / HMM engine that owns its draws."""

    history: np.ndarray
    name: str = "indexed_plugin"
    data_source: str = "USER_SUPPLIED"
    accepts_external_shocks: bool = False

    @property
    def n_steps(self) -> int:
        return int(self.history.shape[0])

    @property
    def n_factors(self) -> int:
        return 1

    def spec_dict(self) -> dict[str, object]:
        return {"type": "indexed_plugin", "n_steps": self.n_steps}

    def generate(self, path_indices, *, shocks, seed):
        from quant_fund.mc_engine.scenario import ScenarioBatch

        assert shocks is None
        assert USER_STREAM_ID_MIN >= 16
        draw = philox_uniforms_row(seed, path_indices)
        starts = np.floor(draw * self.history.shape[0]).astype(np.int64) % self.history.shape[0]
        # A one-step scenario picked from the library. Enough to show the contract.
        picked = self.history[starts]
        returns = np.zeros((path_indices.size, self.n_steps), dtype=np.float64)
        returns[:, 0] = picked
        return ScenarioBatch(
            returns=returns, control=picked, control_mean=float(self.history.mean())
        )


def philox_uniforms_row(seed: int, path_indices: np.ndarray) -> np.ndarray:
    from quant_fund.mc_engine.philox import philox_uniforms

    return philox_uniforms(seed, path_indices, 1, stream_id=USER_STREAM_ID_MIN)[:, 0]


def test_plugin_draw_depends_on_path_index_not_call_order() -> None:
    history = np.linspace(-0.02, 0.02, 50)
    plugin = _IndexedPlugIn(history=history)
    first = plugin.generate(np.array([1, 4, 9]), shocks=None, seed=3)
    second = plugin.generate(np.array([9, 1]), shocks=None, seed=3)
    assert np.array_equal(first.returns[0], second.returns[1])
    assert np.array_equal(first.returns[2], second.returns[0])


def test_identity_generator_round_trip() -> None:
    generator = IdentityShockGenerator(n_steps=2, n_factors=1)
    shocks = np.zeros((3, 2, 1))
    shocks[:, 0, 0] = [0.1, -0.2, 0.3]
    batch = generator.generate(np.arange(3), shocks=shocks, seed=0)
    assert np.allclose(batch.returns[:, 0], [0.1, -0.2, 0.3])
    rebuilt = generator_from_spec(generator.spec_dict())
    assert rebuilt.n_steps == 2
