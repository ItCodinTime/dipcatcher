"""Aggregated adaptive conformal inference: AgACI-EG / FACI-EG.

Online expert aggregation over ACI (Gibbs & Candès, 2021, NeurIPS 34,
pp. 1660-1672) instances run at different learning rates γ. Removes the
single-γ choice that governs ACI's validity/efficiency trade-off
(Zaffran, Dieuleveut, Féron, Goude, Josse, 2022, ICML, arXiv:2202.07282 —
AgACI, Algorithm 1; Zaffran, Fermanian, Letzer, Ndiaye, Tayeh, Bazerque,
Cugliari, Josse, Gazin, Asgarwal, Archimbaud, Cheifa, Guillory, Ichola,
Kegel, Lutzeyer, Marin, Mozharovskyi, Munsch, Nguyen, Ozhegov, Picioli,
Romano, Sakr, Schmid, Schulze, Thube, Tomasini, Vial, Vosseler, Yang,
2022, NeurIPS — "Adaptive Conformal Predictions Under Distribution
Shift", FACI-EG/FACI-OGD).

Each expert e is one ACI loop per quantile level τ: a per-level conformal
offset, the (1 − α_t)-quantile of past residual scores (shared calibration
set across experts, as in the papers), with the one-sided ACI recursion
α_{t+1} = clip(α_t + γ_e · (α_τ − 1{y > q̂}), ε, 1 − ε) where α_τ = 1 − τ.
Experts are mixed per level with weights on the probability simplex,
updated by EXPONENTIATED GRADIENT (Hedge) on the per-expert pinball
losses ℓ_t^e(τ) = ρ_τ(y_t, q̂_t^{e,τ}) (Koenker & Bassett, 1978);
Cesa-Bianchi & Lugosi, 2006, ch. 2–3). Because ρ_τ is convex in its
second argument, the aggregated loss satisfies ρ_τ(y, q̃) ≤ Σ_e w_e
ρ_τ(y, q̂^e) ≤ min_e ρ_τ(y, q̂^e) + O(sqrt(log E / T)) — the empirical
regret form tested in tests/unit/test_agaci.py. AgACI's original rule is
BOA/ML-OGD with the gradient trick (Gaillard, Stoltz, Van Erven, 2014,
COLT); EG on per-expert pinball losses is the FACI-EG variant and the
only rule implemented here ('ml-poe' is deliberately omitted).

Default learning rate η = 1 / (6 · mean γ-grid step), the EG scale used in
the AgACI/FACI experiments (arXiv:2202.07282, Sec. 4 and supplement); the
constant absorbs the unit Lipschitz constant of ρ_τ in its quantile
argument. Only 'eg' aggregation is available; any other value fails
closed.

Output grids are made non-crossing by a cumulative maximum followed by
row-wise rearrangement (Chernozhukov, Fernández-Val, Galichon, 2010, Ann.
Statist. 38; quant_fund.metrics.scoring.rearrange_quantiles). Honesty:
only coverage and pinball diagnostics are exposed (see the AGENTS.md
honesty contract).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.conformal import conformal_quantile
from quant_fund.metrics.scoring import pinball_loss, rearrange_quantiles

Array = NDArray[np.float64]

__all__ = ["AggregatedACI"]

_ALPHA_CLIP: tuple[float, float] = (1e-3, 1.0 - 1e-3)


class AggregatedACI:
    """Aggregated ACI: E = len(gammas) ACI experts per quantile level, EG-mixed.

    Parameters
    ----------
    quantile_levels:
        Strictly increasing levels in (0, 1). When None, defaults to the
        two-sided AgACI grid (alpha/2, 1 - alpha/2).
    gammas:
        Strictly increasing, positive ACI learning rates; one expert each.
    alpha:
        Target miscoverage; used only to derive the default levels.
    eta:
        EG learning rate. None → 1 / (6 · mean(diff(gammas))).
    aggregation:
        'eg' only. Anything else raises ValueError.

    Notes
    -----
    ``update`` returns the conformalized grid *before* incorporating ``y``
    (the prediction issued at the current step); weights, per-expert
    α-state, and the score store are then updated. ``coverage_history``
    records, per step, the per-level one-sided coverage indicators
    1{y ≤ q̃_j} of the aggregated grid.
    """

    def __init__(
        self,
        quantile_levels: Array | None = None,
        gammas: Array | Iterable[float] = (0.001, 0.01, 0.05, 0.1),
        alpha: float = 0.10,
        eta: float | None = None,
        aggregation: str = "eg",
    ) -> None:
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        if aggregation != "eg":
            raise ValueError("only aggregation='eg' is implemented ('ml-poe' omitted)")
        if quantile_levels is None:
            levels = np.array([self.alpha / 2.0, 1.0 - self.alpha / 2.0], dtype=float)
        else:
            levels = np.asarray(quantile_levels, dtype=float).reshape(-1)
        if levels.size == 0:
            raise ValueError("quantile_levels must be non-empty")
        if not np.all(np.isfinite(levels)) or not np.all((levels > 0.0) & (levels < 1.0)):
            raise ValueError("quantile_levels must lie in (0, 1)")
        if np.any(np.diff(levels) <= 0.0):
            raise ValueError("quantile_levels must be strictly increasing")
        g = np.asarray(gammas, dtype=float).reshape(-1)
        if g.size == 0:
            raise ValueError("gammas must be non-empty")
        if not np.all(np.isfinite(g)) or np.any(g <= 0.0):
            raise ValueError("gammas must be positive and finite")
        if g.size > 1 and np.any(np.diff(g) <= 0.0):
            raise ValueError("gammas must be strictly increasing")
        if eta is None:
            eta = 1.0 / (6.0 * float(np.mean(np.diff(g)))) if g.size > 1 else 1.0
        else:
            if not np.isfinite(float(eta)) or float(eta) <= 0.0:
                raise ValueError("eta must be positive and finite")
            eta = float(eta)
        self.eta = float(eta)
        self._levels = levels
        self._alpha_levels = 1.0 - levels
        self._gammas = g
        n_exp = int(g.size)
        n_lvl = int(levels.size)
        self._alpha_t = np.tile(self._alpha_levels, (n_exp, 1))
        self._weights = np.full((n_exp, n_lvl), 1.0 / n_exp, dtype=float)
        self._scores: list[list[float]] = [[] for _ in range(n_lvl)]
        self._n = 0
        self.coverage_history: list[Array] = []

    @property
    def expert_weights_(self) -> Array:
        """Current EG weights, shape (n_experts, n_levels); a copy."""
        return self._weights.copy()

    @property
    def quantile_levels_(self) -> Array:
        return self._levels.copy()

    @property
    def gammas_(self) -> Array:
        return self._gammas.copy()

    @property
    def n_steps_(self) -> int:
        return self._n

    def _expert_offsets(self) -> Array:
        n_exp, n_lvl = self._weights.shape
        offsets = np.zeros((n_exp, n_lvl), dtype=float)
        for j in range(n_lvl):
            hist = self._scores[j]
            if not hist:
                continue
            col = np.asarray(hist, dtype=float)
            for i in range(n_exp):
                offsets[i, j] = conformal_quantile(col, float(self._alpha_t[i, j]))
        return offsets

    def update(self, y: float, quantile_forecast_grid: Array) -> Array:
        """Issue the aggregated conformalized grid, then learn from ``y``.

        Parameters
        ----------
        y:
            Scalar realized value, finite.
        quantile_forecast_grid:
            1-d raw quantile forecasts, one per level, same length as
            ``quantile_levels``, all finite.

        Returns
        -------
        np.ndarray
            Non-crossing conformalized quantile grid for the current step.
        """
        y_arr = np.asarray(y, dtype=float)
        if y_arr.size != 1:
            raise ValueError("y must be a scalar")
        yv = float(y_arr.ravel()[0])
        if not np.isfinite(yv):
            raise ValueError("y must be finite")
        grid = np.asarray(quantile_forecast_grid, dtype=float)
        n_lvl = int(self._levels.size)
        if grid.ndim != 1 or grid.size != n_lvl:
            raise ValueError(f"grid must be 1-d with {n_lvl} entries (one per quantile level)")
        if not np.all(np.isfinite(grid)):
            raise ValueError("grid must contain only finite values")
        n_exp = int(self._gammas.size)

        expert_q = grid[None, :] + self._expert_offsets()
        q_tilde = (self._weights * expert_q).sum(axis=0)
        q_out = np.maximum.accumulate(q_tilde)
        q_out = rearrange_quantiles(q_out.reshape(1, -1)).ravel()
        self.coverage_history.append((yv <= q_out).astype(float))

        losses = np.empty((n_exp, n_lvl), dtype=float)
        for j in range(n_lvl):
            losses[:, j] = pinball_loss(np.full(n_exp, yv), expert_q[:, j], float(self._levels[j]))
        logw = np.log(np.clip(self._weights, 1e-300, None)) - self.eta * losses
        logw -= logw.max(axis=0, keepdims=True)
        unnorm = np.exp(logw)
        self._weights = unnorm / unnorm.sum(axis=0, keepdims=True)

        miss = (yv > expert_q).astype(float)
        self._alpha_t = np.clip(
            self._alpha_t + self._gammas[:, None] * (self._alpha_levels[None, :] - miss),
            _ALPHA_CLIP[0],
            _ALPHA_CLIP[1],
        )
        for j in range(n_lvl):
            self._scores[j].append(yv - float(grid[j]))
        self._n += 1
        return q_out
