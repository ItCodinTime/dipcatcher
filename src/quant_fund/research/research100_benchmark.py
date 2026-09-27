"""Small reproducible integration bench; synthetic data cannot demonstrate alpha."""

from __future__ import annotations

import hashlib
import inspect
from importlib.metadata import version
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.research_evaluation import (
    compare_research_paths,
    walk_forward_allocations,
)
from quant_fund.models.covariance import ledoit_wolf_cov
from quant_fund.portfolio.allocators import inverse_volatility
from quant_fund.portfolio.research_allocators import (
    multi_period_target,
    risk_constrained_kelly,
    volatility_managed_weights,
)

Array = NDArray[np.float64]


def synthetic_benchmark(seed: int = 100) -> dict[str, Any]:
    """Frozen five-candidate comparison, 80 train + 60 walk-forward observations.

    The first 40 observations calibrate inverse-variance scaling. Allocation
    fitting uses the last 40 observations, excluding the current return.
    Expected returns in MPO repeat a shrunk historical mean for three periods.
    Kelly uses equally weighted historical scenarios. These are bench policies,
    not the original authors' empirical strategies. No hyperparameter search.
    """
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=(140, 3))
    returns = 0.0002 + innovations * np.array([0.012, 0.008, 0.016])
    # A fixed regime shock tests adaptation; not calibrated to favor a candidate.
    returns[100:] *= 1.7
    scale = float(np.var(returns[:40].mean(axis=1)))
    heldout_input = returns[40:]

    def equal(history: Array, previous: Array) -> Array:
        return np.full(3, 1 / 3)

    def ivp(history: Array, previous: Array) -> Array:
        return inverse_volatility(ledoit_wolf_cov(history))

    def mpo(history: Array, previous: Array) -> Array:
        result = multi_period_target(
            np.tile(0.25 * history.mean(axis=0), (3, 1)),
            ledoit_wolf_cov(history),
            previous,
            risk_aversion=5,
            linear_cost=0.001,
            quadratic_cost=0.002,
        )
        # Solver-tolerance roundoff only; the method already checked feasibility.
        w: Array = np.asarray(np.maximum(result.weights[0], 0), dtype=float)
        return w / max(1.0, float(w.sum()))

    def kelly(history: Array, previous: Array) -> Array:
        result = risk_constrained_kelly(1 + history, np.full(len(history), 1 / len(history)))
        w = np.maximum(result.weights[:-1], 0)
        return w / max(1.0, float(w.sum()))

    def managed(history: Array, previous: Array) -> Array:
        # Dummy future return contributes no information to the last exposure.
        history_with_dummy = np.append(history.mean(axis=1), 0.0)
        exposure = volatility_managed_weights(history_with_dummy, training_scale=scale, window=21)[
            -1
        ]
        return np.full(3, exposure / 3)

    policies = {
        "equal_weight": equal,
        "inverse_volatility": ivp,
        "multi_period": mpo,
        "risk_constrained_kelly": kelly,
        "volatility_managed": managed,
    }
    paths = {
        name: walk_forward_allocations(heldout_input, policy, lookback=40, fee_rate=0.001)
        for name, policy in policies.items()
    }
    baseline = paths["equal_weight"].net_returns
    names = [name for name in policies if name != "equal_weight"]
    comparison = compare_research_paths(
        baseline,
        np.column_stack([paths[n].net_returns for n in names]),
        n_boot=199,
        block=5,
        seed=seed,
    )
    summaries = {}
    for name, path in paths.items():
        summaries[name] = {
            "observations": int(path.net_returns.size),
            "sum_cost_fraction": float(path.cost_fraction.sum()),
            "total_one_way_turnover": float(path.turnover.sum()),
        }
    module_hashes = {}
    for function in [
        synthetic_benchmark,
        multi_period_target,
        walk_forward_allocations,
        ledoit_wolf_cov,
        inverse_volatility,
    ]:
        module = inspect.getmodule(function)
        if module is not None:
            module_hashes[module.__name__] = hashlib.sha256(
                inspect.getsource(module).encode()
            ).hexdigest()
    return {
        "schema_version": 1,
        "data_label": "SYNTHETIC",
        "claim": "integration_evidence_only",
        "promotable": False,
        "seed": seed,
        "data_sha256": hashlib.sha256(returns.tobytes()).hexdigest(),
        "parameters": {
            "scale_train": [0, 40],
            "lookback": 40,
            "oos_rows": [80, 140],
            "fee_rate": 0.001,
            "block": 5,
            "n_boot": 199,
            "training_scale": scale,
        },
        "versions": {name: version(name) for name in ["numpy", "scipy", "cvxpy", "clarabel"]},
        "module_sha256": module_hashes,
        "candidate_order": names,
        "summaries": summaries,
        "comparison": comparison,
    }
