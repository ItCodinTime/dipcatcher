r"""Causal cross-sectional engines from heavy asset-pricing papers.

Kelly–Malamud–Zhou (JoF 2024 / NBER w30217): random Fourier features plus
ridge in the paper's z parameterization, including the dual/ridgeless
form when P>T.

Kozak–Nagel–Santosh (JFE 2020 / NBER w24070): SDF ridge
b = (Sigma + z I)^{-1} mu on characteristic-managed portfolios. Extra
shrinkage on low-eigenvalue PCs is the Sigma+zI prior, not a second
knob.

Kelly–Pruitt–Su (JFE 2019 / NBER w24540): IPCA ALS with Gamma' Gamma = I_K.

These are ranking challengers. They do not size the book, do not mint
Sharpe, and do not promote SYNTHETIC cards.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.ranking import _finite


def voc_ridge(signals: NDArray[np.float64], y: NDArray[np.float64], z: float) -> NDArray[np.float64]:
    r"""Kelly–Malamud–Zhou ridge: beta(z) = (z I + T^{-1} S'S)^{-1} T^{-1} S' R.

    Dual form beta = S'(SS' + z T I)^{-1} y when P>T. z=0 is ridgeless
    (minimum-norm interpolator via least squares).
    """
    s = np.asarray(signals, dtype=float)
    r = np.asarray(y, dtype=float).reshape(-1)
    n_obs, n_features = s.shape
    if n_obs == 0:
        return np.zeros(n_features, dtype=float)
    if r.shape[0] != n_obs:
        raise ValueError("voc_ridge y length must match S rows")
    if not np.isfinite(z) or z < 0:
        raise ValueError("voc_ridge z must be finite and non-negative")
    sample_t = float(n_obs)
    if n_features <= n_obs:
        return _voc_ridge_primal(s, r, z, sample_t)
    return _voc_ridge_dual(s, r, z, sample_t)


def _voc_ridge_primal(
    signals: NDArray[np.float64], y: NDArray[np.float64], z: float, sample_t: float
) -> NDArray[np.float64]:
    n_features = signals.shape[1]
    gram = (signals.T @ signals) / sample_t
    rhs = (signals.T @ y) / sample_t
    if z > 0:
        return np.linalg.solve(gram + z * np.eye(n_features), rhs)
    return np.linalg.lstsq(gram, rhs, rcond=None)[0]


def _voc_ridge_dual(
    signals: NDArray[np.float64], y: NDArray[np.float64], z: float, sample_t: float
) -> NDArray[np.float64]:
    n_obs = signals.shape[0]
    dual = signals @ signals.T
    if z > 0:
        dual = dual + (z * sample_t) * np.eye(n_obs)
        alpha = np.linalg.solve(dual, y)
    else:
        alpha = np.linalg.lstsq(dual, y, rcond=None)[0]
    return signals.T @ alpha


def random_fourier_features(
    x: NDArray[np.float64],
    omega: NDArray[np.float64],
    bandwidth: float,
) -> NDArray[np.float64]:
    r"""JoF (20) / NBER (21) pair: S_i = [sin(gamma omega_i' G), cos(gamma omega_i' G)]'.

    Column standardization of S is applied by the ranker (NBER fn. 36),
    so the optional P^{-1/2} scale is omitted here.
    """
    proj = bandwidth * (np.asarray(x, dtype=float) @ np.asarray(omega, dtype=float).T)
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=1)


def date_groups(dates: NDArray[Any]) -> list[NDArray[np.intp]]:
    """Row indices per distinct date, first-seen order. Dates need not be sorted."""
    buckets: dict[Any, list[int]] = {}
    order: list[Any] = []
    for i, stamp in enumerate(dates):
        key = stamp
        if isinstance(stamp, np.generic):
            key = stamp.item()
        elif isinstance(stamp, np.ndarray):
            key = stamp.tobytes()
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(i)
    return [np.asarray(buckets[key], dtype=np.intp) for key in order]


def characteristic_managed_portfolios(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dates: NDArray[Any],
) -> NDArray[np.float64]:
    """Date-level F_t = n_t^{-1} Z_t' r_t (KNS / IPCA managed portfolios)."""
    groups = date_groups(dates)
    n_char = x.shape[1]
    out = np.zeros((len(groups), n_char), dtype=float)
    for t, idx in enumerate(groups):
        z_t = x[idx]
        r_t = y[idx]
        finite = np.isfinite(z_t).all(axis=1) & np.isfinite(r_t)
        if not finite.any():
            continue
        z_f = z_t[finite]
        r_f = r_t[finite]
        out[t] = (z_f.T @ r_f) / float(z_f.shape[0])
    return out


def sdf_ridge_loadings(
    managed: NDArray[np.float64], z: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """KNS (22): b = (Sigma + z I)^{-1} mu. Returns b, mu, eigenvalues of Sigma."""
    f = np.asarray(managed, dtype=float)
    if f.ndim != 2 or f.shape[0] < 2:
        raise ValueError("SDF ridge needs at least two dates of managed portfolios")
    if not np.isfinite(z) or z < 0:
        raise ValueError("SDF ridge z must be finite and non-negative")
    mu = np.mean(f, axis=0)
    cov = np.cov(f, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    n_char = cov.shape[0]
    rhs = cov + z * np.eye(n_char)
    if z > 0:
        b = np.linalg.solve(rhs, mu)
    else:
        b = np.linalg.lstsq(cov, mu, rcond=None)[0]
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    return b, mu, evals


def _identify_ipca(
    gamma: NDArray[np.float64], factors: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gamma'T Gamma = I_K, diagonal descending factor covariance, non-negative mean f."""
    q, r = np.linalg.qr(gamma, mode="reduced")
    gamma_q = q
    factors_q = factors @ r.T
    if factors_q.shape[0] > 1 and gamma_q.shape[1] > 0:
        cov = np.cov(factors_q, rowvar=False, ddof=1)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=float)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        gamma_q = gamma_q @ evecs
        factors_q = factors_q @ evecs
    mu = np.mean(factors_q, axis=0) if factors_q.size else np.zeros(gamma_q.shape[1])
    flip = np.where(mu < 0.0, -1.0, 1.0)
    return gamma_q * flip, factors_q * flip


def fit_ipca_als(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dates: NDArray[Any],
    n_factors: int,
    max_iter: int = 50,
    tol: float = 1e-6,
    unrestricted: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, NDArray[np.float64]]:
    """ALS for IPCA FOCs (6)–(7). Returns Gamma, {f_t}, iterations, Gamma_alpha.

    Restricted model sets Gamma_alpha = 0. Unrestricted jointly estimates
    (Gamma_alpha, Gamma) with F_aug,t = (1, f_t)', which is Kelly–Pruitt–Su
    unrestricted IPCA, not a one-shot residual projection.
    """
    if n_factors < 1:
        raise ValueError("IPCA n_factors must be positive")
    xx, yy, mask = _finite(x, y)
    if xx.shape[0] == 0:
        raise ValueError("IPCA fit has no finite rows")
    d_ok = np.asarray(dates)[mask]
    groups = date_groups(d_ok)
    panels: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = []
    managed_rows: list[NDArray[np.float64]] = []
    n_char = xx.shape[1]
    for idx in groups:
        z_t = xx[idx]
        r_t = yy[idx]
        if z_t.shape[0] < n_factors:
            continue
        w_t = z_t.T @ z_t
        x_t = z_t.T @ r_t
        panels.append((z_t, r_t, w_t))
        managed_rows.append(x_t)
    if len(panels) < 2:
        raise ValueError("IPCA needs at least two dates with enough names")
    managed = np.stack(managed_rows, axis=0)  # X_t = Z_t' r_t, shape (T, n_char)
    w_all = np.stack([w_t for _, _, w_t in panels], axis=0)  # W_t = Z_t' Z_t, (T, n_char, n_char)
    gram = managed.T @ managed
    evals, evecs = np.linalg.eigh(gram)
    k = min(int(n_factors), n_char, len(panels))
    gamma = np.asarray(evecs[:, -k:][:, ::-1], dtype=float)
    gamma, _ = _identify_ipca(gamma, np.zeros((len(panels), k)))
    n_iter = 0
    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        factors = _ipca_factor_step(gamma, w_all, managed)
        vec_g = _ipca_gamma_step(factors, w_all, managed)
        gamma_new = np.asarray(vec_g.reshape((n_char, k), order="F"), dtype=float)
        gamma_new, factors = _identify_ipca(gamma_new, factors)
        delta = np.max(np.abs(gamma_new - gamma))
        gamma = gamma_new
        if delta < tol:
            break
    gamma_alpha = np.zeros(n_char, dtype=float)
    if not unrestricted:
        return gamma, factors, n_iter, gamma_alpha
    n_aug = k + 1
    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        # Z_t' (r_t - Z_t Gamma_alpha) = X_t - W_t Gamma_alpha, batched over dates.
        managed_resid = managed - w_all @ gamma_alpha
        factors = _ipca_factor_step(gamma, w_all, managed_resid)
        f_aug = np.concatenate([np.ones((factors.shape[0], 1)), factors], axis=1)
        vec_g = _ipca_gamma_step(f_aug, w_all, managed)
        packed = np.asarray(vec_g.reshape((n_char, n_aug), order="F"), dtype=float)
        gamma_alpha_new = packed[:, 0]
        gamma_new = packed[:, 1:]
        gamma_new, factors = _identify_ipca(gamma_new, factors)
        delta = max(
            float(np.max(np.abs(gamma_new - gamma))),
            float(np.max(np.abs(gamma_alpha_new - gamma_alpha))),
        )
        gamma = gamma_new
        gamma_alpha = gamma_alpha_new
        if delta < tol:
            break
    return gamma, factors, n_iter, gamma_alpha


def _ipca_factor_step(
    gamma: NDArray[np.float64],
    w_all: NDArray[np.float64],
    managed: NDArray[np.float64],
) -> NDArray[np.float64]:
    """FOC (6) for every date at once: f_t = (Gamma' W_t Gamma)^{-1} Gamma' X_t."""
    k = gamma.shape[1]
    # Batched BLAS: (W_t Gamma) then Gamma' (.) for every date; no 3-operand einsum.
    gwg = np.matmul(gamma.T, np.matmul(w_all, gamma))  # (T, k, k)
    rhs = managed @ gamma  # (T, k) == Gamma' X_t
    eye_k = np.eye(k)
    try:
        return np.asarray(np.linalg.solve(gwg + 1e-12 * eye_k, rhs[..., None])[..., 0], dtype=float)
    except np.linalg.LinAlgError:
        out = np.zeros((w_all.shape[0], k), dtype=float)
        for t in range(w_all.shape[0]):
            out[t] = np.linalg.lstsq(gwg[t], rhs[t], rcond=None)[0]
        return out


def _ipca_gamma_step(
    factors: NDArray[np.float64],
    w_all: NDArray[np.float64],
    managed: NDArray[np.float64],
) -> NDArray[np.float64]:
    """FOC (7): vec(Gamma) = (sum_t f_t f_t' kron W_t)^{-1} sum_t (f_t kron X_t).

    One GEMM over the stacked dates gives the same sum as the per-date
    ``np.kron`` loop, in column-major (Fortran) vec order.
    """
    n_dates, n_char, _ = w_all.shape
    k = factors.shape[1]
    ff = (factors[:, :, None] * factors[:, None, :]).reshape(n_dates, k * k)  # (T, k*k)
    lhs4 = (ff.T @ w_all.reshape(n_dates, n_char * n_char)).reshape(k, k, n_char, n_char)
    # lhs[(i,a),(j,b)] = sum_t f_ti f_tj W_t[a,b]  ==  kron(f f', W) in Fortran vec order.
    lhs = lhs4.transpose(0, 2, 1, 3).reshape(n_char * k, n_char * k)
    rhs_vec = (factors.T @ managed).reshape(n_char * k)  # rhs[(i,a)] = sum_t f_ti X_t[a]
    try:
        return np.asarray(np.linalg.solve(lhs + 1e-10 * np.eye(n_char * k), rhs_vec), dtype=float)
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.lstsq(lhs, rhs_vec, rcond=None)[0], dtype=float)


class RandomFourierRanker(JoblibMixin):
    """Stacked-panel RFF ridge (Kelly–Malamud–Zhou 2024). Public CS features only."""

    def __init__(
        self,
        n_features: int = 256,
        bandwidth: float = 2.0,
        z: float = 1.0,
        seed: int = 42,
    ) -> None:
        if n_features < 2 or n_features % 2 != 0:
            raise ValueError("RFF n_features must be an even integer >= 2")
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("RFF bandwidth gamma must be finite and positive")
        if not np.isfinite(z) or z < 0:
            raise ValueError("RFF ridge z must be finite and non-negative")
        self.n_features = int(n_features)
        self.bandwidth = float(bandwidth)
        self.z = float(z)
        self.seed = int(seed)
        self.scaler_x = StandardScaler()
        self.scaler_s = StandardScaler()
        self.omega: NDArray[np.float64] | None = None
        self.beta: NDArray[np.float64] | None = None
        self.y_mean: float = 0.0
        self.n_train: int = 0

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> RandomFourierRanker:
        xx, yy, _ = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("RFF ranker fit has no finite rows")
        self.scaler_x.fit(xx)
        xs = self.scaler_x.transform(xx)
        n_pairs = self.n_features // 2
        rng = np.random.default_rng(self.seed)
        self.omega = rng.normal(size=(n_pairs, xs.shape[1]))
        signals = random_fourier_features(xs, self.omega, self.bandwidth)
        self.scaler_s.fit(signals)
        s_std = self.scaler_s.transform(signals)
        self.y_mean = float(np.mean(yy))
        self.n_train = int(xx.shape[0])
        self.beta = voc_ridge(s_std, yy - self.y_mean, self.z)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.omega is None or self.beta is None:
            raise ValueError("RandomFourierRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        if x.shape[0] == 0:
            return np.zeros(0, dtype=float)
        xs = self.scaler_x.transform(x)
        signals = random_fourier_features(xs, self.omega, self.bandwidth)
        s_std = self.scaler_s.transform(signals)
        return self.y_mean + s_std @ self.beta

    def metadata(self) -> ModelMeta:
        extra: dict[str, Any] = {
            "n_features": self.n_features,
            "bandwidth_gamma": self.bandwidth,
            "z": self.z,
            "n_train": self.n_train,
            "paper": "Kelly, Malamud, Zhou, JoF 2024",
        }
        if self.n_train > 0:
            extra["p_over_t"] = float(self.n_features) / float(self.n_train)
        name = "rff_ridgeless" if self.z == 0.0 else "rff"
        return ModelMeta(family="ranking", name=name, version="v1", extra=extra)


class SDFRidgeRanker(JoblibMixin):
    """Characteristic-managed SDF ridge (Kozak–Nagel–Santosh 2020)."""

    def __init__(self, z: float = 1.0) -> None:
        if not np.isfinite(z) or z < 0:
            raise ValueError("SDF ridge z must be finite and non-negative")
        self.z = float(z)
        self.b: NDArray[np.float64] | None = None
        self.mu: NDArray[np.float64] | None = None
        self.eigenvalues: NDArray[np.float64] | None = None
        self.n_dates: int = 0

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> SDFRidgeRanker:
        dates = kwargs.get("dates")
        if dates is None:
            raise ValueError("SDF ridge requires per-row dates")
        xx, yy, mask = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("SDF ridge fit has no finite rows")
        d_ok = np.asarray(dates)[mask]
        managed = characteristic_managed_portfolios(xx, yy, d_ok)
        self.b, self.mu, self.eigenvalues = sdf_ridge_loadings(managed, self.z)
        self.n_dates = int(managed.shape[0])
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.b is None:
            raise ValueError("SDFRidgeRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return x @ self.b

    def metadata(self) -> ModelMeta:
        extra: dict[str, Any] = {
            "z": self.z,
            "n_dates": self.n_dates,
            "paper": "Kozak, Nagel, Santosh, JFE 2020",
        }
        if self.eigenvalues is not None:
            extra["n_eigenvalues"] = int(self.eigenvalues.shape[0])
        return ModelMeta(family="ranking", name="sdf_ridge", version="v1", extra=extra)


class IPCARanker(JoblibMixin):
    """Instrumented PCA ALS (Kelly–Pruitt–Su 2019). Scores are Z (Gamma_alpha + Gamma mu_f)."""

    def __init__(
        self,
        n_factors: int = 3,
        max_iter: int = 50,
        tol: float = 1e-6,
        unrestricted: bool = False,
    ) -> None:
        if n_factors < 1:
            raise ValueError("IPCA n_factors must be positive")
        if max_iter < 1:
            raise ValueError("IPCA max_iter must be positive")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("IPCA tol must be finite and positive")
        self.n_factors = int(n_factors)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.unrestricted = bool(unrestricted)
        self.gamma: NDArray[np.float64] | None = None
        self.gamma_alpha: NDArray[np.float64] | None = None
        self.mu_f: NDArray[np.float64] | None = None
        self.n_iter: int = 0

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> IPCARanker:
        dates = kwargs.get("dates")
        if dates is None:
            raise ValueError("IPCA requires per-row dates")
        self.gamma, factors, self.n_iter, self.gamma_alpha = fit_ipca_als(
            x,
            y,
            np.asarray(dates),
            self.n_factors,
            self.max_iter,
            self.tol,
            unrestricted=self.unrestricted,
        )
        self.mu_f = np.mean(factors, axis=0)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.gamma is None or self.mu_f is None or self.gamma_alpha is None:
            raise ValueError("IPCARanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return x @ (self.gamma_alpha + self.gamma @ self.mu_f)

    def metadata(self) -> ModelMeta:
        extra: dict[str, Any] = {
            "n_factors": self.n_factors,
            "n_iter": self.n_iter,
            "unrestricted": self.unrestricted,
            "paper": "Kelly, Pruitt, Su, JFE 2019",
        }
        if self.gamma is not None:
            gram = self.gamma.T @ self.gamma
            extra["gamma_gram_offdiag_abs_max"] = float(
                np.max(np.abs(gram - np.eye(gram.shape[0])))
            )
        name = "ipca_alpha" if self.unrestricted else "ipca"
        return ModelMeta(family="ranking", name=name, version="v1", extra=extra)
