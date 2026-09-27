"""NGBoost-lite: natural-gradient boosting for a Gaussian predictive distribution.

Duan, Anand, Ding, Basu, Ng & Schuler (2020, ICML, PMLR 119, pp. 2690-2700,
"NGBoost: Natural Gradient Boosting for Probabilistic Prediction",
arXiv:1910.03225). Each boosting round fits one base learner per
distribution parameter to the NATURAL gradient of the proper scoring rule,
i.e. the ordinary gradient pre-multiplied by the inverse Riemannian metric
(Amari 1998). For the Gaussian in (mu, log sigma) coordinates under the
log score the Fisher information is diagonal, diag(1/sigma^2, 2), so

    nat_grad_mu        = -(y - mu)                    (ordinary: -(y-mu)/sigma^2)
    nat_grad_log_sigma = (1 - z^2) / 2,  z = (y - mu)/sigma   (ordinary: 1 - z^2)

Under CRPS (Gneiting & Raftery 2007) the metric is the Hessian of the CRPS
divergence, diag(1/(sqrt(pi) sigma), sigma/(2 sqrt(pi))); both scores are
supported and the exact expressions live in ``_gradients``. Updates are

    theta_{m} = theta_{m-1} - lr * rho_m * f_m(x),

with ``rho_m`` a scalar line-search scale (NGBoost Algorithm 1, step 8)
chosen by golden-section on the training score. Base learners are shallow
sklearn ``DecisionTreeRegressor`` (paper default depth 3), one per
parameter. ``fit`` supports early stopping on a validation set.

Predictions expose mu/sigma, quantiles, PIT, log score and CRPS — proper
scores only (AGENTS.md honesty contract). Fail-closed: non-finite inputs,
unfitted predict, invalid hyperparameters, and non-positive sigma raise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor

from quant_fund.metrics.scoring import crps_gaussian, log_score_gaussian

Array = NDArray[np.float64]

__all__ = ["NGBoostGaussian", "NGBoostFitInfo"]

_LOG_SIGMA_CLIP: tuple[float, float] = (-12.0, 12.0)


@dataclass(frozen=True)
class NGBoostFitInfo:
    """Boosting trace. Scores are the minimised loss (NLL or CRPS; lower is better).

    ``train_score`` and ``val_score`` have one entry per round that was fit
    (``searched_rounds``). Early stopping keeps learners through ``best_round``
    and drops the patience tail, so ``n_rounds`` can be smaller than
    ``searched_rounds``. ``best_round`` is 1-based.
    """

    n_rounds: int
    searched_rounds: int
    train_score: list[float]
    val_score: list[float]
    best_round: int


def _gradients(y: Array, mu: Array, log_sigma: Array, score: str) -> tuple[Array, Array]:
    sigma = np.exp(log_sigma)
    z = (y - mu) / sigma
    if score == "logscore":
        return -(y - mu), 0.5 * (1.0 - z * z)
    # CRPS(N(mu, sigma), y) = sigma * (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)).
    # Metric = Hessian of the CRPS divergence at the truth (Duan et al. 2020,
    # Sec. 3.2): diag(1/(sqrt(pi) sigma), sigma/(2 sqrt(pi))) in (mu, log sigma).
    phi = norm.pdf(z)
    cdf = norm.cdf(z)
    d_mu = -(2.0 * cdf - 1.0)
    d_log_sigma = (2.0 * phi - 1.0 / np.sqrt(np.pi)) * sigma
    metric_mu = 1.0 / (np.sqrt(np.pi) * sigma)
    metric_log_sigma = sigma / (2.0 * np.sqrt(np.pi))
    return d_mu / metric_mu, d_log_sigma / metric_log_sigma


def _score(y: Array, mu: Array, log_sigma: Array, score: str) -> float:
    """Mean loss to MINIMISE: negative log score (NLL) or CRPS."""
    sigma = np.exp(log_sigma)
    if score == "logscore":
        return float(-np.mean(log_score_gaussian(y, mu, sigma)))
    return float(np.mean(crps_gaussian(y, mu, sigma)))


def _line_search(y: Array, mu: Array, ls: Array, d_mu: Array, d_ls: Array, score: str) -> float:
    """Bounded step in ``(0, 2]``. Returns 0 when no positive step lowers the loss.

    A failed or boundary-zero search used to fall back to a unit step, which
    can climb the loss. A zero step leaves the parameters unchanged.
    """
    base = _score(y, mu, ls, score)

    def obj(r: float) -> float:
        return _score(y, mu - r * d_mu, np.clip(ls - r * d_ls, *_LOG_SIGMA_CLIP), score)

    res = minimize_scalar(obj, bounds=(0.0, 2.0), method="bounded")
    if not bool(res.success) or not np.isfinite(res.x):
        return 0.0
    rho = float(res.x)
    if rho <= 0.0 or obj(rho) > base + 1e-12:
        return 0.0
    return rho


def _replay(
    init: tuple[float, float],
    learners: list[tuple[DecisionTreeRegressor, DecisionTreeRegressor, float]],
    X: Array,
    learning_rate: float,
) -> tuple[Array, Array]:
    mu = np.full(X.shape[0], init[0])
    ls = np.full(X.shape[0], init[1])
    for t_mu, t_ls, rho in learners:
        step = learning_rate * rho
        mu = mu - step * np.asarray(t_mu.predict(X), dtype=float)
        ls = ls - step * np.asarray(t_ls.predict(X), dtype=float)
    return mu, np.clip(ls, *_LOG_SIGMA_CLIP)


class NGBoostGaussian:
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        score: str = "logscore",
        line_search: bool = True,
        early_stopping_rounds: int | None = None,
        seed: int = 0,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if score not in ("logscore", "crps"):
            raise ValueError("score must be 'logscore' or 'crps'")
        if early_stopping_rounds is not None and early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be >= 1 or None")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.score = score
        self.line_search = bool(line_search)
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = int(seed)
        self._init: tuple[float, float] | None = None
        self._learners: list[tuple[DecisionTreeRegressor, DecisionTreeRegressor, float]] = []
        self._n_features = 0
        self.fit_info: NGBoostFitInfo | None = None

    def _check_xy(self, X: Array, y: Array | None) -> tuple[Array, Array | None]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n, p)")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must be finite")
        if y is None:
            return X, None
        y = np.asarray(y, dtype=float).ravel()
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y length mismatch")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must be finite")
        return X, y

    def _tree(self) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            criterion="squared_error",
        )

    def _raw_params(self, X: Array, n_rounds: int | None = None) -> tuple[Array, Array]:
        if self._init is None:
            raise RuntimeError("NGBoostGaussian is not fitted")
        learners = self._learners if n_rounds is None else self._learners[:n_rounds]
        return _replay(self._init, learners, X, self.learning_rate)

    def fit(
        self,
        X: Array,
        y: Array,
        X_val: Array | None = None,
        y_val: Array | None = None,
    ) -> NGBoostGaussian:
        X, y_ = self._check_xy(X, y)
        assert y_ is not None
        y = y_
        if X.shape[0] < 2 * self.min_samples_leaf:
            raise ValueError("too few samples")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be given together")
        if self.early_stopping_rounds is not None and X_val is None:
            raise ValueError("early stopping needs a validation set")
        val: tuple[Array, Array] | None = None
        if X_val is not None and y_val is not None:
            Xv, yv = self._check_xy(X_val, y_val)
            assert yv is not None
            if Xv.shape[1] != X.shape[1]:
                raise ValueError("X_val feature count mismatch")
            val = (Xv, yv)

        sd0 = float(np.std(y))
        if sd0 <= 0.0:
            raise ValueError("y has zero variance; nothing to fit")
        init = (float(np.mean(y)), float(np.log(sd0)))
        learners: list[tuple[DecisionTreeRegressor, DecisionTreeRegressor, float]] = []
        mu = np.full(y.shape[0], init[0])
        ls = np.full(y.shape[0], init[1])
        train_hist: list[float] = []
        val_hist: list[float] = []
        best_val = np.inf
        best_round = 0
        stale = 0

        for m in range(self.n_estimators):
            g_mu, g_ls = _gradients(y, mu, ls, self.score)
            if not (np.all(np.isfinite(g_mu)) and np.all(np.isfinite(g_ls))):
                raise ValueError("natural gradient is not finite")
            t_mu = self._tree().fit(X, g_mu)
            t_ls = self._tree().fit(X, g_ls)
            d_mu = np.asarray(t_mu.predict(X), dtype=float)
            d_ls = np.asarray(t_ls.predict(X), dtype=float)
            rho = _line_search(y, mu, ls, d_mu, d_ls, self.score) if self.line_search else 1.0
            step = self.learning_rate * rho
            mu = mu - step * d_mu
            ls = np.clip(ls - step * d_ls, *_LOG_SIGMA_CLIP)
            learners.append((t_mu, t_ls, rho))
            train_hist.append(_score(y, mu, ls, self.score))
            if val is not None:
                vmu, vls = _replay(init, learners, val[0], self.learning_rate)
                vs = _score(val[1], vmu, vls, self.score)
                if not np.isfinite(vs):
                    raise ValueError("validation loss is not finite")
                val_hist.append(vs)
                if vs < best_val - 1e-12:
                    best_val = vs
                    best_round = m + 1
                    stale = 0
                else:
                    stale += 1
                    if (
                        self.early_stopping_rounds is not None
                        and stale >= self.early_stopping_rounds
                    ):
                        break
            else:
                best_round = m + 1

        if val is not None and self.early_stopping_rounds is not None:
            if best_round < 1:
                raise ValueError("validation loss never improved; nothing to keep")
            learners = learners[:best_round]
        self._n_features = int(X.shape[1])
        self._init = init
        self._learners = learners
        self.fit_info = NGBoostFitInfo(
            n_rounds=len(learners),
            searched_rounds=len(train_hist),
            train_score=train_hist,
            val_score=val_hist,
            best_round=best_round,
        )
        return self

    def predict_params(self, X: Array) -> tuple[Array, Array]:
        if self._init is None:
            raise RuntimeError("NGBoostGaussian is not fitted")
        X, _ = self._check_xy(X, None)
        if X.shape[1] != self._n_features:
            raise ValueError("feature count mismatch")
        mu, ls = self._raw_params(X)
        sigma = np.exp(ls)
        if np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
            raise ValueError("predicted sigma is not positive finite")
        return mu, sigma

    def predict(self, X: Array) -> Array:
        return self.predict_params(X)[0]

    def predict_quantiles(self, X: Array, taus: Array) -> Array:
        taus = np.asarray(taus, dtype=float).ravel()
        if (
            taus.size == 0
            or not np.all(np.isfinite(taus))
            or np.any(taus <= 0.0)
            or np.any(taus >= 1.0)
        ):
            raise ValueError("taus must be non-empty and in (0, 1)")
        mu, sigma = self.predict_params(X)
        return np.asarray(mu[:, None] + sigma[:, None] * norm.ppf(taus)[None, :], dtype=float)

    def pit(self, X: Array, y: Array) -> Array:
        X, y_ = self._check_xy(X, y)
        assert y_ is not None
        mu, sigma = self.predict_params(X)
        return np.asarray(norm.cdf((y_ - mu) / sigma), dtype=float)

    def crps(self, X: Array, y: Array) -> float:
        X, y_ = self._check_xy(X, y)
        assert y_ is not None
        mu, sigma = self.predict_params(X)
        return float(np.mean(crps_gaussian(y_, mu, sigma)))

    def log_score(self, X: Array, y: Array) -> float:
        """Mean Gaussian log score (higher is better)."""
        X, y_ = self._check_xy(X, y)
        assert y_ is not None
        mu, sigma = self.predict_params(X)
        return float(np.mean(log_score_gaussian(y_, mu, sigma)))
