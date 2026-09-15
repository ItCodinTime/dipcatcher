"""Probabilistic regime inference. States are unlabeled at fit time."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from quant_fund.models.base import JoblibMixin, ModelMeta


class VolThresholdRegime(JoblibMixin):
    """Heuristic 2-state baseline: high vs low realized vol."""

    def __init__(self, q: float = 0.7) -> None:
        self.q = q
        self.cut = 0.0

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64] | None = None, **kwargs: Any
    ) -> VolThresholdRegime:
        vol = x[:, 0]
        self.cut = float(np.nanquantile(vol, self.q))
        return self

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        high = (x[:, 0] >= self.cut).astype(float)
        return np.column_stack([1.0 - high, high])

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.predict_proba(x)[:, 1]

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="regime", name="vol_threshold", version="v1")


class GaussianHMMRegime(JoblibMixin):
    def __init__(self, n_states: int = 3, seed: int = 42) -> None:
        self.n_states = n_states
        self.seed = seed
        self.scaler = StandardScaler()
        self.model: Any = None
        self.labels: dict[int, str] = {}

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64] | None = None, **kwargs: Any
    ) -> GaussianHMMRegime:
        from hmmlearn.hmm import GaussianHMM

        xx = self.scaler.fit_transform(np.where(np.isfinite(x), x, 0.0))
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=50,
            random_state=self.seed,
        )
        self.model.fit(xx)
        means = self.model.means_
        # interpret: sort by first feature (typically market return) and vol (2nd)
        order = np.argsort(means[:, 0]) if means.shape[1] else np.arange(self.n_states)
        names = ["stress", "risk_off", "neutral", "risk_on", "low_vol_trend"]
        for rank, state in enumerate(order):
            self.labels[int(state)] = names[min(rank, len(names) - 1)]
        return self

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return online filtered probabilities, never future-smoothed states.

        ``hmmlearn.predict_proba`` uses the complete sequence and therefore
        conditions state probabilities at time *t* on observations after *t*.
        This method deliberately implements the forward filter instead. Use
        :meth:`predict_smoothed_proba` only for retrospective diagnostics.
        """
        if self.model is None:
            raise RuntimeError("GaussianHMMRegime must be fitted before prediction")
        from scipy.special import logsumexp

        xx = self.scaler.transform(np.where(np.isfinite(x), x, 0.0))
        means = np.asarray(self.model.means_, dtype=float)
        covars = np.asarray(self.model.covars_, dtype=float)
        log_emission = np.empty((xx.shape[0], self.n_states), dtype=float)
        for state in range(self.n_states):
            var = covars[state]
            if var.ndim == 2:
                var = np.diag(var)
            var = np.clip(var, 1e-12, None)
            log_emission[:, state] = -0.5 * np.sum(
                np.log(2.0 * np.pi * var) + (xx - means[state]) ** 2 / var,
                axis=1,
            )
        log_trans = np.log(np.clip(np.asarray(self.model.transmat_), 1e-300, None))
        log_prob = np.log(np.clip(np.asarray(self.model.startprob_), 1e-300, None))
        filtered = np.empty_like(log_emission)
        for t in range(xx.shape[0]):
            if t:
                log_prob = logsumexp(log_prob[:, None] + log_trans, axis=0)
            log_prob = log_prob + log_emission[t]
            log_prob -= logsumexp(log_prob)
            filtered[t] = log_prob
        return np.exp(filtered)

    def predict_smoothed_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return full-sequence smoothed probabilities for retrospective analysis."""
        if self.model is None:
            raise RuntimeError("GaussianHMMRegime must be fitted before prediction")
        xx = self.scaler.transform(np.where(np.isfinite(x), x, 0.0))
        p = self.model.predict_proba(xx)
        return p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.predict_proba(x)

    def aic_bic(self, x: NDArray[np.float64]) -> dict[str, float]:
        xx = self.scaler.transform(np.where(np.isfinite(x), x, 0.0))
        n = xx.shape[0]
        k = self.n_states
        d = xx.shape[1]
        n_params = k * d + k * d + k * (k - 1) + (k - 1)  # means, diag cov, trans, start
        ll = float(self.model.score(xx))
        aic = 2 * n_params - 2 * ll
        bic = float(np.log(n) * n_params - 2 * ll)
        return {"aic": float(aic), "bic": bic, "avg_ll": ll / max(n, 1), "n_params": float(n_params)}

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="regime",
            name="gaussian_hmm",
            version="v1",
            extra={"n_states": self.n_states, "labels": self.labels},
        )


class SingleStateRegime(JoblibMixin):
    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64] | None = None, **kwargs: Any
    ) -> SingleStateRegime:
        return self

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones((x.shape[0], 1))

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones(x.shape[0])

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="regime", name="single_state", version="v1")
