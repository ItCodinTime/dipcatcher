"""Two-state volatility-regime mixture distribution head (dip_regime).

A Gaussian HMM infers a low/high-volatility state sequence over the fit
window; per-state empirical CDFs are mixed with the one-step state
probabilities (last filtered posterior propagated through the fitted
transition matrix). Each tau's quantile is the generalized inverse of
that mixture CDF. Falls back to a |y| threshold split with a persistence
heuristic when the HMM cannot be fitted, and to a single-state empirical
distribution when one state is empty.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.regime import GaussianHMMRegime, VolThresholdRegime

MIN_OBS = 60


class RegimeDistribution(JoblibMixin):
    """Unconditional 2-state vol-regime mixture head.

    Rows must be one security in strict time order; the pipeline rejects
    pooled panels because HMM transitions must not cross securities.
    ``x`` is ignored: the fit-boundary mixed quantile row is tiled for
    every test observation. State labels are reordered so column 1 is
    always the high-vol state.
    """

    def __init__(self, taus: list[float], seed: int = 42, vol_quantile: float = 0.7) -> None:
        self.taus = taus
        self.seed = seed
        self.vol_quantile = vol_quantile
        self.q_: NDArray[np.float64] | None = None
        self.state_quantiles_: NDArray[np.float64] | None = None
        self.mix_weights_: NDArray[np.float64] | None = None
        self.n_states_effective_ = 0
        self.state_estimator_ = ""
        self.transition_ = np.eye(2, dtype=float)
        self.last_filtered_weights_: NDArray[np.float64] | None = None

    def _state_probs(self, yy: NDArray[np.float64]) -> NDArray[np.float64]:
        """Filtered posterior state probs (n, 2) on |y|, col 1 = high-vol."""
        feat = np.abs(yy)[:, None]
        try:
            hmm = GaussianHMMRegime(n_states=2, seed=self.seed)
            hmm.fit(feat)
            probs = np.asarray(hmm.predict_proba(feat), dtype=float)
            trans = np.asarray(hmm.model.transmat_, dtype=float)
            if (
                trans.shape != (2, 2)
                or not np.isfinite(trans).all()
                or np.any(trans < 0.0)
                or np.any(trans.sum(axis=1) <= 0.0)
            ):
                raise ValueError("invalid HMM transition matrix")
            self.transition_ = trans / trans.sum(axis=1, keepdims=True)
            self.state_estimator_ = "hmm"
        except (AttributeError, ValueError, RuntimeError):
            self.transition_ = np.eye(2, dtype=float)
            thr = VolThresholdRegime(q=self.vol_quantile)
            thr.fit(feat)
            probs = np.asarray(thr.predict_proba(feat), dtype=float)
            self.state_estimator_ = "vol_threshold"
        denom = np.maximum(probs.sum(axis=0), 1e-12)
        state_abs_mean = (probs * np.abs(yy)[:, None]).sum(axis=0) / denom
        if state_abs_mean[0] > state_abs_mean[1]:
            probs = probs[:, ::-1]
            self.transition_ = self.transition_[::-1, ::-1]
        return probs

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> RegimeDistribution:
        yy = np.asarray(y, dtype=float).reshape(-1)
        yy = yy[np.isfinite(yy)]
        if yy.size < MIN_OBS:
            raise ValueError("RegimeDistribution requires >= 60 finite observations")
        probs = self._state_probs(yy)
        self.last_filtered_weights_ = np.asarray(probs[-1], dtype=float)
        w = self.last_filtered_weights_ @ self.transition_
        self.mix_weights_ = np.asarray(w / max(float(w.sum()), 1e-12), dtype=float)
        assign = np.argmax(probs, axis=1)
        states = [yy[assign == s] for s in range(2)]
        tt = np.asarray(self.taus, dtype=float)
        if states[0].size == 0 or states[1].size == 0:
            self.q_ = np.quantile(yy, tt).astype(float)
            self.state_quantiles_ = np.tile(self.q_, (2, 1))
            self.n_states_effective_ = 1
            return self
        self.state_quantiles_ = np.stack([np.quantile(states[s], tt) for s in range(2)], axis=0)
        grid = np.unique(yy)
        cdf = np.zeros(grid.size)
        for s in range(2):
            cdf += self.mix_weights_[s] * (
                np.searchsorted(np.sort(states[s]), grid, side="right") / states[s].size
            )
        # Generalized inverse of the stepwise mixture CDF (monotone by
        # construction): q_tau = inf{grid : F_mix >= tau}.
        idx = np.clip(np.searchsorted(cdf, tt, side="left"), 0, grid.size - 1)
        self.q_ = np.asarray(grid[idx], dtype=float)
        self.n_states_effective_ = 2
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.q_ is None:
            raise RuntimeError("distribution model has not been fitted")
        return np.tile(self.q_, (x.shape[0], 1))

    def metadata(self) -> ModelMeta:
        p_high_next = float(self.mix_weights_[1]) if self.mix_weights_ is not None else 0.0
        p_high_last = (
            float(self.last_filtered_weights_[1])
            if self.last_filtered_weights_ is not None
            else 0.0
        )
        return ModelMeta(
            family="distribution",
            name="regime",
            version="v1",
            extra={
                "n_states_effective": self.n_states_effective_,
                "state_estimator": self.state_estimator_,
                "p_high_vol_next": p_high_next,
                "p_high_vol_last_filtered": p_high_last,
            },
        )
