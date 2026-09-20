"""Further causal rankers from landmark cross-sectional papers.

Lettau–Pelger (RFS 2020): RP-PCA on characteristic-managed portfolios.
Giglio–Xiu (JPE 2021 / NBER w23527): three-pass risk premia with omitted factors.
Freyberger–Neuhierl–Weber (RFS 2020 / NBER w23227): adaptive group LASSO on
quadratic splines of rank-transformed characteristics, then OLS on selected
groups.
Feng–Giglio–Xiu (JoF 2020): post-double-selection LASSO then OLS.
Fama–MacBeth (JPE 1973): date-level CS slopes, averaged.
Gu–Kelly–Xiu (RFS 2020 / NBER w25398): PCR, PLS, GBRT (trees; no NN).
Kelly–Pruitt (JoE 2015): automatic-proxy three-pass regression filter.
Kelly–Malamud–Pedersen (JoF 2023 / NBER w27388): principal portfolios.

Kozak–Nagel–Santosh elastic-net SDF (eq. 28) and unrestricted IPCA live in
asset_pricing.py. None of these size the book. Metadata must not carry Sharpe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression

from quant_fund.models.asset_pricing import (
    characteristic_managed_portfolios,
    date_groups,
)
from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.ranking import _finite

PAPER_RANKER_NAMES = (
    "rff",
    "rff_ridgeless",
    "sdf_ridge",
    "sdf_en",
    "ipca",
    "ipca_alpha",
    "rp_pca",
    "fnw",
    "gx3pass",
    "ds_lasso",
    "fm",
    "pcr",
    "pls",
    "tprf",
    "gbrt",
    "pp",
)

DATED_FIT_RANKERS = frozenset(
    {
        "sdf_ridge",
        "sdf_en",
        "ipca",
        "ipca_alpha",
        "rp_pca",
        "gx3pass",
        "fnw",
        "fm",
        "pp",
    }
)
DATED_PREDICT_RANKERS = frozenset({"fnw", "pp"})
ID_FIT_RANKERS = frozenset({"pp"})
ID_PREDICT_RANKERS = frozenset({"pp"})


def rp_pca_loadings(
    managed: NDArray[np.float64],
    gamma: float,
    n_factors: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Lettau–Pelger matrix (5): (1/T) X'X + gamma mu mu'.

    gamma = -1 is covariance PCA. gamma > -1 over-weights means. Returns
    eigenvectors Lambda (L x K), factor means, and eigenvalues of the RP matrix.
    """
    x = np.asarray(managed, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("RP-PCA needs at least two dates of managed portfolios")
    if not np.isfinite(gamma) or gamma < -1.0:
        raise ValueError("RP-PCA gamma must be finite and >= -1")
    n_time, n_char = x.shape
    k = min(int(n_factors), n_char, n_time)
    if k < 1:
        raise ValueError("RP-PCA n_factors must be positive")
    mu = np.mean(x, axis=0)
    second = (x.T @ x) / float(n_time)
    rp = second + float(gamma) * np.outer(mu, mu)
    evals, evecs = np.linalg.eigh(rp)
    order = np.argsort(evals)[::-1]
    lam = np.asarray(evecs[:, order[:k]], dtype=float)
    factors = x @ lam
    mu_f = np.mean(factors, axis=0)
    return lam, mu_f, evals[order]


def gx_three_pass_premia(
    managed: NDArray[np.float64],
    n_factors: int,
) -> NDArray[np.float64]:
    """Giglio–Xiu three-pass characteristic premia on managed portfolios.

    Pass 1: PCA of test-asset (managed) returns. Pass 2: CS regression of
    average returns on PCA loadings. Pass 3: TS regression of each managed
    column on the PCs. Characteristic premium is eta' lambda_pca.
    """
    x = np.asarray(managed, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("Giglio–Xiu three-pass needs at least three dates")
    n_time, n_char = x.shape
    k = min(int(n_factors), n_char, n_time - 1)
    if k < 1:
        raise ValueError("Giglio–Xiu n_factors must be positive")
    mu = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    evals, evecs = np.linalg.eigh(cov)
    v_k = np.asarray(evecs[:, np.argsort(evals)[::-1][:k]], dtype=float)
    pcs = x @ v_k
    lambda_pca = v_k.T @ mu
    gram = pcs.T @ pcs
    premia = np.zeros(n_char, dtype=float)
    jitter = 1e-10 * np.eye(k)
    for j in range(n_char):
        eta = np.linalg.lstsq(gram + jitter, pcs.T @ x[:, j], rcond=None)[0]
        premia[j] = float(eta @ lambda_pca)
    return premia


def sdf_elastic_net_loadings(
    managed: NDArray[np.float64],
    l2: float,
    l1: float,
    max_iter: int = 400,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """KNS (28): HJ-distance elastic net on managed-portfolio means.

    Smooth objective b'Sigma b - 2 b'mu + l2 ||b||^2, plus l1 ||b||_1.
    """
    f = np.asarray(managed, dtype=float)
    if f.ndim != 2 or f.shape[0] < 2:
        raise ValueError("SDF elastic net needs at least two dates")
    if not np.isfinite(l1) or l1 < 0 or not np.isfinite(l2) or l2 < 0:
        raise ValueError("SDF elastic-net penalties must be finite and non-negative")
    mu = np.mean(f, axis=0)
    cov = np.cov(f, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    n_char = cov.shape[0]
    eigmax = float(np.max(np.linalg.eigvalsh(cov))) if n_char else 1.0
    lipschitz = 2.0 * (max(eigmax, 0.0) + l2) + 1e-8
    step = 1.0 / lipschitz
    b = np.zeros(n_char, dtype=float)
    for _ in range(max_iter):
        grad = 2.0 * (cov @ b) - 2.0 * mu + 2.0 * l2 * b
        z = b - step * grad
        thresh = step * l1
        b_new = np.sign(z) * np.maximum(np.abs(z) - thresh, 0.0)
        if float(np.max(np.abs(b_new - b))) < tol:
            b = b_new
            break
        b = b_new
    return b


def cs_rank_unit_interval(
    x: NDArray[np.float64], dates: NDArray[Any] | None
) -> NDArray[np.float64]:
    """FNW rank transform: cross-sectional rank on each date, mapped to (0, 1)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    if dates is None:
        groups = [np.arange(x.shape[0], dtype=np.intp)]
    else:
        groups = date_groups(np.asarray(dates))
    for idx in groups:
        block = x[idx]
        ranks = np.zeros_like(block, dtype=float)
        for col in range(block.shape[1]):
            values = block[:, col]
            finite = np.isfinite(values)
            n_ok = int(finite.sum())
            if n_ok == 0:
                continue
            order = np.argsort(values[finite], kind="mergesort")
            placed = np.empty(n_ok, dtype=float)
            placed[order] = np.arange(1, n_ok + 1, dtype=float)
            ranks[finite, col] = (placed - 0.5) / float(n_ok)
        out[idx] = ranks
    return out


def quadratic_spline_basis(unit: NDArray[np.float64], n_intervals: int) -> NDArray[np.float64]:
    """FNW (4): 1, c, c^2, and max(c - t_l, 0)^2 at equally spaced knots."""
    if n_intervals < 2:
        raise ValueError("FNW n_intervals must be >= 2")
    c = np.clip(np.asarray(unit, dtype=float), 0.0, 1.0)
    cols = [np.ones(c.shape[0]), c, c * c]
    for knot_i in range(1, n_intervals):
        knot = float(knot_i) / float(n_intervals)
        cols.append(np.maximum(c - knot, 0.0) ** 2)
    return np.column_stack(cols)


def _group_lasso(
    design: NDArray[np.float64],
    y: NDArray[np.float64],
    groups: list[NDArray[np.intp]],
    lam: float,
    weights: NDArray[np.float64] | None = None,
    max_iter: int = 400,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    n_obs, n_col = design.shape
    if n_obs == 0:
        return np.zeros(n_col, dtype=float)
    spectral = float(np.linalg.norm(design, ord=2))
    step = 1.0 / ((spectral * spectral / float(n_obs)) + 1e-12)
    coef = np.zeros(n_col, dtype=float)
    y_c = y - float(np.mean(y))
    w = np.ones(len(groups), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    for _ in range(max_iter):
        grad = design.T @ (design @ coef - y_c) / float(n_obs)
        z = coef - step * grad
        updated = np.zeros_like(coef)
        for g_i, idx in enumerate(groups):
            zg = z[idx]
            norm = float(np.linalg.norm(zg))
            thresh = step * lam * float(w[g_i])
            if norm > thresh:
                updated[idx] = zg * (1.0 - thresh / norm)
        if float(np.max(np.abs(updated - coef))) < tol:
            coef = updated
            break
        coef = updated
    return coef


def fnw_adaptive_group_lasso(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dates: NDArray[Any],
    n_intervals: int = 4,
    lam: float = 0.05,
) -> tuple[NDArray[np.float64], float, list[int]]:
    """Two-step adaptive group LASSO + OLS on selected characteristics.

    Intercept is global and unpenalized (deviation: FNW puts p1=1 inside each
    group). Characteristics already dropped in step 1 stay dropped (w_s = inf).
    """
    ranks = cs_rank_unit_interval(x, dates)
    n_obs, n_char = ranks.shape
    pieces = [quadratic_spline_basis(ranks[:, j], n_intervals)[:, 1:] for j in range(n_char)]
    design = np.hstack(pieces)
    groups = []
    width = pieces[0].shape[1]
    for j in range(n_char):
        groups.append(np.arange(j * width, (j + 1) * width, dtype=np.intp))
    first = _group_lasso(design, y, groups, lam)
    weights = np.zeros(n_char, dtype=float)
    for j, idx in enumerate(groups):
        nrm = float(np.linalg.norm(first[idx]))
        weights[j] = 1.0 / nrm if nrm > 1e-12 else np.inf
    finite_w = np.where(np.isfinite(weights), weights, 1e12)
    second = _group_lasso(design, y, groups, lam, weights=finite_w)
    selected: list[int] = []
    for j, idx in enumerate(groups):
        if np.isfinite(weights[j]) and float(np.linalg.norm(second[idx])) > 1e-10:
            selected.append(j)
    intercept = float(np.mean(y))
    spline_coef = np.zeros(0, dtype=float)
    if not selected:
        # Empty AG-LASSO would score every name with the intercept (NaN IC).
        # Fall back to OLS on every spline group rather than a constant.
        selected = list(range(n_char))
    cols = np.hstack([pieces[j] for j in selected])
    fit = LinearRegression(fit_intercept=True).fit(cols, y)
    intercept = float(fit.intercept_)
    spline_coef = np.asarray(fit.coef_, dtype=float)
    return spline_coef, intercept, selected


class SDFElasticNetRanker(JoblibMixin):
    """KNS (28) HJ-distance elastic net on managed portfolios."""

    def __init__(self, l2: float = 1.0, l1: float = 0.05) -> None:
        if not np.isfinite(l2) or l2 < 0 or not np.isfinite(l1) or l1 < 0:
            raise ValueError("SDF elastic-net penalties must be finite and non-negative")
        self.l2 = float(l2)
        self.l1 = float(l1)
        self.l1_effective = float(l1)
        self.b: NDArray[np.float64] | None = None
        self.n_dates: int = 0

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> SDFElasticNetRanker:
        dates = kwargs.get("dates")
        if dates is None:
            raise ValueError("SDF elastic net requires per-row dates")
        xx, yy, mask = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("SDF elastic net fit has no finite rows")
        managed = characteristic_managed_portfolios(xx, yy, np.asarray(dates)[mask])
        self.b = sdf_elastic_net_loadings(managed, self.l2, self.l1)
        if float(np.max(np.abs(self.b))) < 1e-14:
            # Fixed l1 can zero every loading on a small-mean panel; KNS then
            # collapses to a constant score (NaN IC). Fall back to the l1=0
            # path of eq. 28 rather than emit a constant.
            self.b = sdf_elastic_net_loadings(managed, self.l2, 0.0)
            self.l1_effective = 0.0
        else:
            self.l1_effective = self.l1
        self.n_dates = int(managed.shape[0])
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.b is None:
            raise ValueError("SDFElasticNetRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return x @ self.b

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="sdf_en",
            version="v1",
            extra={
                "l1": self.l1,
                "l1_effective": float(getattr(self, "l1_effective", self.l1)),
                "l2": self.l2,
                "n_dates": self.n_dates,
                "paper": "Kozak, Nagel, Santosh, JFE 2020 eq.28",
            },
        )


class RPPCARanker(JoblibMixin):
    """Lettau–Pelger RP-PCA on characteristic-managed portfolios."""

    def __init__(self, n_factors: int = 3, gamma: float = 10.0) -> None:
        if n_factors < 1:
            raise ValueError("RP-PCA n_factors must be positive")
        if not np.isfinite(gamma) or gamma < -1.0:
            raise ValueError("RP-PCA gamma must be finite and >= -1")
        self.n_factors = int(n_factors)
        self.gamma = float(gamma)
        self.lambda_load: NDArray[np.float64] | None = None
        self.mu_f: NDArray[np.float64] | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> RPPCARanker:
        dates = kwargs.get("dates")
        if dates is None:
            raise ValueError("RP-PCA requires per-row dates")
        xx, yy, mask = _finite(x, y)
        managed = characteristic_managed_portfolios(xx, yy, np.asarray(dates)[mask])
        self.lambda_load, self.mu_f, _ = rp_pca_loadings(managed, self.gamma, self.n_factors)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.lambda_load is None or self.mu_f is None:
            raise ValueError("RPPCARanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return x @ (self.lambda_load @ self.mu_f)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="rp_pca",
            version="v1",
            extra={"n_factors": self.n_factors, "rp_gamma": self.gamma, "paper": "Lettau, Pelger, RFS 2020"},
        )


class GXThreePassRanker(JoblibMixin):
    """Giglio–Xiu three-pass characteristic premia mapped back through Z."""

    def __init__(self, n_factors: int = 3) -> None:
        if n_factors < 1:
            raise ValueError("Giglio–Xiu n_factors must be positive")
        self.n_factors = int(n_factors)
        self.premia: NDArray[np.float64] | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> GXThreePassRanker:
        dates = kwargs.get("dates")
        if dates is None:
            raise ValueError("Giglio–Xiu three-pass requires per-row dates")
        xx, yy, mask = _finite(x, y)
        managed = characteristic_managed_portfolios(xx, yy, np.asarray(dates)[mask])
        self.premia = gx_three_pass_premia(managed, self.n_factors)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.premia is None:
            raise ValueError("GXThreePassRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return x @ self.premia

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="gx3pass",
            version="v1",
            extra={"n_factors": self.n_factors, "paper": "Giglio, Xiu, JPE 2021 / NBER w23527"},
        )


class FNWRanker(JoblibMixin):
    """Freyberger–Neuhierl–Weber adaptive group LASSO on quadratic splines."""

    def __init__(self, n_intervals: int = 4, lam: float = 0.05) -> None:
        if n_intervals < 2:
            raise ValueError("FNW n_intervals must be >= 2")
        if not np.isfinite(lam) or lam < 0:
            raise ValueError("FNW lam must be finite and non-negative")
        self.n_intervals = int(n_intervals)
        self.lam = float(lam)
        self.spline_coef: NDArray[np.float64] | None = None
        self.intercept: float = 0.0
        self.selected: list[int] = []

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> FNWRanker:
        dates = kwargs.get("dates")
        if dates is None:
            raise ValueError("FNW requires per-row dates")
        xx, yy, mask = _finite(x, y)
        spline_coef, intercept, selected = fnw_adaptive_group_lasso(
            xx, yy, np.asarray(dates)[mask], self.n_intervals, self.lam
        )
        self.spline_coef = spline_coef
        self.intercept = float(intercept)
        self.selected = selected
        return self

    def predict(self, x: NDArray[np.float64], dates: NDArray[Any] | None = None) -> NDArray[np.float64]:
        if self.spline_coef is None:
            raise ValueError("FNWRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        if not self.selected:
            return np.full(x.shape[0], self.intercept, dtype=float)
        ranks = cs_rank_unit_interval(x, dates)
        cols = np.hstack(
            [
                quadratic_spline_basis(ranks[:, j], self.n_intervals)[:, 1:]
                for j in self.selected
            ]
        )
        return self.intercept + cols @ self.spline_coef

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="fnw",
            version="v1",
            extra={
                "n_intervals": self.n_intervals,
                "lam": self.lam,
                "n_selected": len(self.selected),
                "paper": "Freyberger, Neuhierl, Weber, RFS 2020 / NBER w23227",
            },
        )


class DoubleSelectionRanker(JoblibMixin):
    """Feng–Giglio–Xiu / Belloni–Chernozhukov–Hansen post-double-selection OLS."""

    def __init__(self, alpha: float = 0.01) -> None:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("double-selection alpha must be finite and non-negative")
        self.alpha = float(alpha)
        self.coef: NDArray[np.float64] | None = None
        self.intercept: float = 0.0
        self.selected: list[int] = []

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> DoubleSelectionRanker:
        xx, yy, _ = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("double-selection fit has no finite rows")
        n_char = xx.shape[1]
        selected: set[int] = set()
        if self.alpha == 0.0:
            selected = set(range(n_char))
        else:
            first = Lasso(alpha=self.alpha, max_iter=8000).fit(xx, yy)
            selected = {i for i, c in enumerate(first.coef_) if abs(float(c)) > 1e-12}
            for j in list(selected):
                others = [k for k in range(n_char) if k != j]
                if not others:
                    continue
                second = Lasso(alpha=self.alpha, max_iter=8000).fit(xx[:, others], xx[:, j])
                for loc, coef in enumerate(second.coef_):
                    if abs(float(coef)) > 1e-12:
                        selected.add(others[loc])
        if not selected:
            selected = set(range(n_char))
        cols = sorted(selected)
        fit = LinearRegression(fit_intercept=True).fit(xx[:, cols], yy)
        self.intercept = float(fit.intercept_)
        self.coef = np.zeros(n_char, dtype=float)
        for loc, j in enumerate(cols):
            self.coef[j] = float(fit.coef_[loc])
        self.selected = cols
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coef is None:
            raise ValueError("DoubleSelectionRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return self.intercept + x @ self.coef

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="ds_lasso",
            version="v1",
            extra={
                "alpha": self.alpha,
                "n_selected": len(self.selected),
                "paper": "Feng, Giglio, Xiu, JoF 2020; Belloni, Chernozhukov, Hansen",
            },
        )


def _column_standardize(
    x: NDArray[np.float64],
    mean: NDArray[np.float64] | None = None,
    scale: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(x, dtype=float)
    if mean is None:
        mean = np.mean(x, axis=0)
    if scale is None:
        scale = np.std(x, axis=0, ddof=0)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return (x - mean) / scale, np.asarray(mean, dtype=float), np.asarray(scale, dtype=float)


def _ols_intercept(design: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    y = np.asarray(y, dtype=float).reshape(-1)
    if design.size == 0:
        return float(np.mean(y)) if y.size else 0.0, np.zeros(0, dtype=float)
    mat = np.column_stack([np.ones(y.shape[0]), np.asarray(design, dtype=float)])
    coef = np.linalg.lstsq(mat, y, rcond=None)[0]
    return float(coef[0]), np.asarray(coef[1:], dtype=float)


def fama_macbeth_slopes(
    x: NDArray[np.float64], y: NDArray[np.float64], dates: NDArray[Any]
) -> NDArray[np.float64]:
    """Fama–MacBeth (1973): mean of date-level CS OLS slopes (no intercept)."""
    groups = date_groups(dates)
    n_char = x.shape[1]
    slopes: list[NDArray[np.float64]] = []
    for idx in groups:
        z_t = x[idx]
        r_t = y[idx]
        finite = np.isfinite(z_t).all(axis=1) & np.isfinite(r_t)
        if int(finite.sum()) < n_char + 2:
            continue
        _, beta = _ols_intercept(z_t[finite], r_t[finite])
        if beta.shape[0] == n_char and np.all(np.isfinite(beta)):
            slopes.append(beta)
    if not slopes:
        raise ValueError("Fama–MacBeth needs at least one date with enough names")
    return np.mean(np.stack(slopes, axis=0), axis=0)


def pcr_loadings(
    x: NDArray[np.float64], y: NDArray[np.float64], n_factors: int
) -> tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """GKX PCR: leading PCs of standardized Z, then OLS of y on the scores."""
    xs, mean, scale = _column_standardize(x)
    n_obs, n_char = xs.shape
    k = min(int(n_factors), n_char, n_obs)
    if k < 1:
        raise ValueError("PCR n_factors must be positive")
    _, _, vt = np.linalg.svd(xs, full_matrices=False)
    omega = np.asarray(vt[:k].T, dtype=float)
    scores = xs @ omega
    intercept, theta = _ols_intercept(scores, y)
    return omega, intercept, theta, mean, scale


def tprf_fit(
    x: NDArray[np.float64], y: NDArray[np.float64], n_factors: int
) -> tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Kelly–Pruitt automatic-proxy 3PRF (JoE 2015 Table 1 + Table 2).

    Predictors are column-standardized. Each automatic proxy is the residual
    of the previous 3PRF forecast (proxy 1 = y). Pass 1/2 include intercepts.
    """
    xs, mean, scale = _column_standardize(x)
    n_obs, n_pred = xs.shape
    k = min(int(n_factors), n_pred, max(n_obs - 2, 1))
    if k < 1:
        raise ValueError("3PRF n_factors must be positive")
    residual = np.asarray(y, dtype=float).reshape(-1).copy()
    proxies = np.zeros((n_obs, 0), dtype=float)
    phi = np.zeros((n_pred, 0), dtype=float)
    intercept = float(np.mean(y))
    beta = np.zeros(0, dtype=float)
    for _ in range(k):
        proxies = np.column_stack([proxies, residual]) if proxies.size else residual.reshape(-1, 1)
        n_prox = proxies.shape[1]
        proxy_design = np.column_stack([np.ones(n_obs), proxies])
        gram = proxy_design.T @ proxy_design
        phi_aug = np.linalg.lstsq(gram, proxy_design.T @ xs, rcond=None)[0]
        phi = np.asarray(phi_aug[1:].T, dtype=float)
        pass2 = np.column_stack([np.ones(n_pred), phi])
        gram2 = pass2.T @ pass2
        factors_aug = np.linalg.lstsq(gram2, pass2.T @ xs.T, rcond=None)[0].T
        factors = np.asarray(factors_aug[:, 1:], dtype=float)
        intercept, beta = _ols_intercept(factors, y)
        fitted = intercept + factors @ beta
        residual = np.asarray(y, dtype=float).reshape(-1) - fitted
        if n_prox >= k:
            break
    return phi, intercept, beta, mean, scale


def tprf_predict(
    x: NDArray[np.float64],
    phi: NDArray[np.float64],
    intercept: float,
    beta: NDArray[np.float64],
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    xs, _, _ = _column_standardize(x, mean=mean, scale=scale)
    n_pred = xs.shape[1]
    pass2 = np.column_stack([np.ones(n_pred), phi])
    factors_aug = np.linalg.lstsq(pass2, xs.T, rcond=None)[0].T
    factors = np.asarray(factors_aug[:, 1:], dtype=float)
    return intercept + factors @ beta


def prediction_matrix(
    signals: NDArray[np.float64],
    returns: NDArray[np.float64],
    dates: NDArray[Any],
    ids: NDArray[Any],
) -> tuple[NDArray[np.float64], list[Any]]:
    """Kelly–Malamud–Pedersen Pi = E[R_{t+1} S_t'] on an unbalanced name panel.

    y is already the forward ranking label aligned to date t, so the outer
    product on each date is R_{t+h} S_t'.
    """
    names: list[Any] = []
    index: dict[Any, int] = {}
    for sid in ids:
        key = sid.item() if isinstance(sid, np.generic) else sid
        if key not in index:
            index[key] = len(names)
            names.append(key)
    n_names = len(names)
    if n_names < 2:
        raise ValueError("principal portfolios need at least two names")
    pi = np.zeros((n_names, n_names), dtype=float)
    count = np.zeros((n_names, n_names), dtype=float)
    for idx in date_groups(dates):
        loc = []
        rows = []
        for row in idx:
            key = ids[row]
            if isinstance(key, np.generic):
                key = key.item()
            loc.append(index[key])
            rows.append(int(row))
        loc_a = np.asarray(loc, dtype=np.intp)
        r = returns[rows]
        s = signals[rows]
        finite = np.isfinite(r) & np.isfinite(s)
        if int(finite.sum()) < 2:
            continue
        loc_a = loc_a[finite]
        r = r[finite]
        s = s[finite]
        pi[np.ix_(loc_a, loc_a)] += np.outer(r, s)
        count[np.ix_(loc_a, loc_a)] += 1.0
    if float(np.max(count)) < 1.0:
        raise ValueError("principal portfolios fit has no finite date blocks")
    pi = pi / np.maximum(count, 1.0)
    return pi, names


class FamaMacBethRanker(JoblibMixin):
    """Fama–MacBeth (1973) average CS slope mapped through Z."""

    def __init__(self) -> None:
        self.lambda_bar: NDArray[np.float64] | None = None
        self.n_dates: int = 0

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> FamaMacBethRanker:
        dates = kwargs.get("dates")
        if dates is None:
            raise ValueError("Fama–MacBeth requires per-row dates")
        xx, yy, mask = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("Fama–MacBeth fit has no finite rows")
        self.lambda_bar = fama_macbeth_slopes(xx, yy, np.asarray(dates)[mask])
        self.n_dates = len(date_groups(np.asarray(dates)[mask]))
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.lambda_bar is None:
            raise ValueError("FamaMacBethRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return x @ self.lambda_bar

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="fm",
            version="v1",
            extra={"n_dates": self.n_dates, "paper": "Fama, MacBeth, JPE 1973"},
        )


class PCRRanker(JoblibMixin):
    """Gu–Kelly–Xiu principal-component regression on public CS features."""

    def __init__(self, n_factors: int = 3) -> None:
        if n_factors < 1:
            raise ValueError("PCR n_factors must be positive")
        self.n_factors = int(n_factors)
        self.omega: NDArray[np.float64] | None = None
        self.theta: NDArray[np.float64] | None = None
        self.intercept: float = 0.0
        self.mean: NDArray[np.float64] | None = None
        self.scale: NDArray[np.float64] | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> PCRRanker:
        xx, yy, _ = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("PCR fit has no finite rows")
        self.omega, self.intercept, self.theta, self.mean, self.scale = pcr_loadings(
            xx, yy, self.n_factors
        )
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.omega is None or self.theta is None or self.mean is None or self.scale is None:
            raise ValueError("PCRRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        xs, _, _ = _column_standardize(x, mean=self.mean, scale=self.scale)
        return self.intercept + (xs @ self.omega) @ self.theta

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="pcr",
            version="v1",
            extra={"n_factors": self.n_factors, "paper": "Gu, Kelly, Xiu, RFS 2020 PCR"},
        )


class PLSRanker(JoblibMixin):
    """Gu–Kelly–Xiu / Kelly–Pruitt PLS (SIMPLS) on public CS features."""

    def __init__(self, n_factors: int = 3) -> None:
        if n_factors < 1:
            raise ValueError("PLS n_factors must be positive")
        self.n_factors = int(n_factors)
        self.model: PLSRegression | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> PLSRanker:
        xx, yy, _ = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("PLS fit has no finite rows")
        k = min(self.n_factors, xx.shape[1], xx.shape[0])
        self.model = PLSRegression(n_components=k, scale=True)
        self.model.fit(xx, yy)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model is None:
            raise ValueError("PLSRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return np.asarray(self.model.predict(x), dtype=float).reshape(-1)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="pls",
            version="v1",
            extra={"n_factors": self.n_factors, "paper": "Gu, Kelly, Xiu, RFS 2020 PLS; Kelly, Pruitt, JoE 2015"},
        )


class ThreePassFilterRanker(JoblibMixin):
    """Kelly–Pruitt automatic-proxy three-pass regression filter."""

    def __init__(self, n_factors: int = 3) -> None:
        if n_factors < 1:
            raise ValueError("3PRF n_factors must be positive")
        self.n_factors = int(n_factors)
        self.phi: NDArray[np.float64] | None = None
        self.intercept: float = 0.0
        self.beta: NDArray[np.float64] | None = None
        self.mean: NDArray[np.float64] | None = None
        self.scale: NDArray[np.float64] | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> ThreePassFilterRanker:
        xx, yy, _ = _finite(x, y)
        if xx.shape[0] < 3:
            raise ValueError("3PRF fit needs at least three finite rows")
        self.phi, self.intercept, self.beta, self.mean, self.scale = tprf_fit(
            xx, yy, self.n_factors
        )
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.phi is None or self.beta is None or self.mean is None or self.scale is None:
            raise ValueError("ThreePassFilterRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return tprf_predict(x, self.phi, self.intercept, self.beta, self.mean, self.scale)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="tprf",
            version="v1",
            extra={"n_factors": self.n_factors, "paper": "Kelly, Pruitt, Journal of Econometrics 2015"},
        )


class GBRTRanker(JoblibMixin):
    """Gu–Kelly–Xiu gradient boosted regression trees (Huber, shallow)."""

    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int = 2,
        learning_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        if n_estimators < 1 or max_depth < 1:
            raise ValueError("GBRT n_estimators and max_depth must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("GBRT learning_rate must be finite and positive")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.model: GradientBoostingRegressor | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> GBRTRanker:
        xx, yy, _ = _finite(x, y)
        if xx.shape[0] < 2:
            raise ValueError("GBRT fit has no finite rows")
        self.model = GradientBoostingRegressor(
            loss="huber",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            random_state=self.seed,
        )
        self.model.fit(xx, yy)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model is None:
            raise ValueError("GBRTRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return np.asarray(self.model.predict(x), dtype=float)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="gbrt",
            version="v1",
            extra={
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "loss": "huber",
                "paper": "Gu, Kelly, Xiu, RFS 2020 GBRT+H",
            },
        )


class PrincipalPortfolioRanker(JoblibMixin):
    """Kelly–Malamud–Pedersen prediction-matrix ranker (truncated SVD of Pi)."""

    def __init__(self, n_factors: int = 3) -> None:
        if n_factors < 1:
            raise ValueError("principal-portfolio n_factors must be positive")
        self.n_factors = int(n_factors)
        self.beta: NDArray[np.float64] | None = None
        self.pi_k: NDArray[np.float64] | None = None
        self.names: list[Any] = []
        self.index: dict[Any, int] = {}

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> PrincipalPortfolioRanker:
        dates = kwargs.get("dates")
        ids = kwargs.get("ids")
        if dates is None or ids is None:
            raise ValueError("principal portfolios require per-row dates and ids")
        xx, yy, mask = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("principal portfolios fit has no finite rows")
        d_ok = np.asarray(dates)[mask]
        id_ok = np.asarray(ids)[mask]
        _, self.beta = _ols_intercept(xx, yy)
        signal = xx @ self.beta
        pi, names = prediction_matrix(signal, yy, d_ok, id_ok)
        k = min(self.n_factors, pi.shape[0])
        u, sv, vt = np.linalg.svd(pi, full_matrices=False)
        self.pi_k = (u[:, :k] * sv[:k]) @ vt[:k]
        self.names = names
        self.index = {name: i for i, name in enumerate(names)}
        return self

    def predict(
        self, x: NDArray[np.float64], dates: NDArray[Any] | None = None, ids: NDArray[Any] | None = None
    ) -> NDArray[np.float64]:
        if self.beta is None or self.pi_k is None:
            raise ValueError("PrincipalPortfolioRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        own = x @ self.beta
        if dates is None or ids is None:
            return own
        ids = np.asarray(ids)
        out = own.copy()
        n_names = self.pi_k.shape[0]
        for idx in date_groups(np.asarray(dates)):
            s_full = np.zeros(n_names, dtype=float)
            present: list[tuple[int, int]] = []
            for row in idx:
                key = ids[row]
                if isinstance(key, np.generic):
                    key = key.item()
                loc = self.index.get(key)
                if loc is None:
                    continue
                s_full[loc] = own[row]
                present.append((int(row), loc))
            scored = self.pi_k @ s_full
            for row, loc in present:
                out[row] = scored[loc]
        return out

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="pp",
            version="v1",
            extra={
                "n_factors": self.n_factors,
                "n_names": len(self.names),
                "paper": "Kelly, Malamud, Pedersen, JoF 2023 / NBER w27388",
            },
        )
