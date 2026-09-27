"""Reproducible Monte Carlo engine for strategy and portfolio scenario risk.

Research simulation only. Built-in generators are synthetic. Nothing in this
package submits orders or claims live profit.
"""

from quant_fund.mc_engine.aggregate import P2Quantile, TDigest, Welford
from quant_fund.mc_engine.engine import (
    MC_ENGINE_VERSION,
    EngineConfig,
    resume_simulation,
    run_simulation,
)
from quant_fund.mc_engine.philox import (
    USER_STREAM_ID_MIN,
    philox_normals,
    philox_uniforms,
)
from quant_fund.mc_engine.scenario import (
    GbmPortfolioGenerator,
    IdentityShockGenerator,
    ScenarioBatch,
    ScenarioGenerator,
    VolTargetStrategyGenerator,
)

__all__ = [
    "MC_ENGINE_VERSION",
    "USER_STREAM_ID_MIN",
    "EngineConfig",
    "GbmPortfolioGenerator",
    "IdentityShockGenerator",
    "P2Quantile",
    "ScenarioBatch",
    "ScenarioGenerator",
    "TDigest",
    "VolTargetStrategyGenerator",
    "Welford",
    "philox_normals",
    "philox_uniforms",
    "resume_simulation",
    "run_simulation",
]
