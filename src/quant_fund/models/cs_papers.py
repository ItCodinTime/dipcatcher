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
Rapach–Strauss–Zhou (RFS 2010): equal-weight combination of univariate CS OLS,
an IC-weighted variant (non-negative train date-IC weights), and discounted
nested-MSFE weights (``combo_msfe``).
Zou (JASA 2006): adaptive LASSO on public characteristics.
Fama–MacBeth with per-date ridge (small-N CS; OLS FM needs N>p+2).
A priori signed classic characteristics (Jegadeesh, JT skip, residual momentum,
Bali MAX, Ang idio-vol, Amihud, George–Hwang 52w). Daily short-horizon subset
(``classic_st`` / ``ridge_st`` / ``fm_st`` / ``combo_ic_st``): Jegadeesh daily
reversal, Lehmann weekly reversal, JT skip, residual momentum, Bali MAX,
Ang ivol. Single-characteristic Jegadeesh baseline: ``reversal``.

Kozak–Nagel–Santosh elastic-net SDF (eq. 28) and unrestricted IPCA live in
asset_pricing.py. None of these size the book. Metadata must not carry Sharpe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from quant_fund.models.asset_pricing import (
    characteristic_managed_portfolios,
    date_groups,
)
from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.ranking import RidgeRanker, _finite

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
    "combo",
    "alasso",
    "classic",
    "fm_ridge",
    "combo_ic",
    "reversal",
    "classic_st",
    "ridge_st",
    "fm_st",
    "combo_ic_st",
    "combo_msfe",
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
        "fm_ridge",
        "pp",
        "combo_ic",
        "classic",
        "ridge",
        "reversal",
        "classic_st",
        "ridge_st",
        "fm_st",
        "combo_ic_st",
        "combo_msfe",
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


def _target_scaled_penalty(penalty: float, target: NDArray[np.float64]) -> float:
    """An l1 penalty quoted in target-standard-deviation units, made absolute.

    sklearn ``Lasso`` and the in-house group LASSO penalize in the units of
    ``y``. A fixed absolute ``alpha`` that is mild on a unit-variance test
    target zeroes every coefficient on a 5-day return with sd 0.03. Scaling
    by ``sd(y)`` keeps one config knob meaningful across labels.
    """
    scale = float(np.std(np.asarray(target, dtype=float)))
    if not np.isfinite(scale) or scale <= 0.0:
        return float(penalty)
    return float(penalty) * scale


def _l1_fit(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    alpha: float,
    *,
    max_iter: int = 4000,
    tol: float = 1e-4,
) -> tuple[float, NDArray[np.float64]]:
    """sklearn LASSO with Gram precompute.

    Naive coordinate descent is ``O(n p n_iter)``. Expanding horse-race
    windows have ``p≈40`` and ``n`` in the 1e5–1e6 range, so the n-path
    stalls (adaptive LASSO could not finish a 5-day tape). The Gram is
    ``p×p``; one ``X'X`` multiply then cheap CD. Coefficients match the
    naive path up to solver tolerance.
    """
    model = Lasso(
        alpha=float(alpha),
        max_iter=int(max_iter),
        tol=float(tol),
        precompute=True,
        copy_X=True,
        selection="cyclic",
        fit_intercept=True,
    )
    model.fit(x, y)
    return float(model.intercept_), np.asarray(model.coef_, dtype=float)


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
    # ``lam`` is in units of the target's standard deviation so the same knob
    # means the same shrinkage on a 1-day and a 20-day return label.
    lam = _target_scaled_penalty(lam, y)
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
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> DoubleSelectionRanker:
        xx, yy, _ = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("double-selection fit has no finite rows")
        xs, mean, scale = _column_standardize(xx)
        self.mean_ = mean
        self.scale_ = scale
        n_char = xs.shape[1]
        selected: set[int] = set()
        if self.alpha == 0.0:
            selected = set(range(n_char))
        else:
            # Penalties are quoted in units of the regressand's sd (label for the
            # outcome LASSO, characteristic j for each treatment LASSO). Columns
            # are standardized first: sklearn L1 is not scale-invariant, and a
            # single mixed-scale CS column makes post-selection OLS a constant.
            _, first_coef = _l1_fit(xs, yy, _target_scaled_penalty(self.alpha, yy))
            selected = {i for i, c in enumerate(first_coef) if abs(float(c)) > 1e-12}
            for j in list(selected):
                others = [k for k in range(n_char) if k != j]
                if not others:
                    continue
                _, second_coef = _l1_fit(
                    xs[:, others],
                    xs[:, j],
                    _target_scaled_penalty(self.alpha, xs[:, j]),
                )
                for loc, coef in enumerate(second_coef):
                    if abs(float(coef)) > 1e-12:
                        selected.add(others[loc])
        if not selected:
            selected = set(range(n_char))
        cols = sorted(selected)
        fit = LinearRegression(fit_intercept=True).fit(xs[:, cols], yy)
        self.intercept = float(fit.intercept_)
        self.coef = np.zeros(n_char, dtype=float)
        for loc, j in enumerate(cols):
            self.coef[j] = float(fit.coef_[loc])
        self.selected = cols
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coef is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("DoubleSelectionRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        xs = (x - self.mean_) / self.scale_
        return self.intercept + xs @ self.coef

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


def fama_macbeth_ridge_slopes(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    dates: NDArray[Any],
    alpha: float = 1.0,
    min_names: int = 8,
) -> NDArray[np.float64]:
    """Date-level CS ridge, then average λ. Identified when N ≈ p (OLS FM is not)."""
    groups = date_groups(dates)
    n_char = x.shape[1]
    slopes: list[NDArray[np.float64]] = []
    ridge = Ridge(alpha=float(alpha), fit_intercept=True)
    for idx in groups:
        z_t = x[idx]
        r_t = y[idx]
        finite = np.isfinite(z_t).all(axis=1) & np.isfinite(r_t)
        if int(finite.sum()) < int(min_names):
            continue
        ridge.fit(z_t[finite], r_t[finite])
        beta = np.asarray(ridge.coef_, dtype=float)
        if beta.shape[0] == n_char and np.all(np.isfinite(beta)):
            slopes.append(beta)
    if not slopes:
        raise ValueError("ridge Fama–MacBeth needs at least one date with enough names")
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


class FamaMacBethRidgeRanker(JoblibMixin):
    """Per-date ridge CS slopes, averaged. The 54-name OLS FM object is unidentified."""

    def __init__(self, alpha: float = 1.0) -> None:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("fm_ridge alpha must be finite and non-negative")
        self.alpha = float(alpha)
        self.lambda_bar: NDArray[np.float64] | None = None
        self.n_dates: int = 0

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> FamaMacBethRidgeRanker:
        dates = kwargs.get("dates")
        if dates is None:
            raise ValueError("ridge Fama–MacBeth requires per-row dates")
        xx, yy, mask = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("ridge Fama–MacBeth fit has no finite rows")
        self.lambda_bar = fama_macbeth_ridge_slopes(
            xx, yy, np.asarray(dates)[mask], alpha=self.alpha
        )
        self.n_dates = len(date_groups(np.asarray(dates)[mask]))
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.lambda_bar is None:
            raise ValueError("FamaMacBethRidgeRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return x @ self.lambda_bar

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="fm_ridge",
            version="v1",
            extra={
                "alpha": self.alpha,
                "n_dates": self.n_dates,
                "paper": "Fama, MacBeth, JPE 1973; date-level ridge",
            },
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


class CombinationRanker(JoblibMixin):
    """Rapach–Strauss–Zhou equal-weight combination of univariate CS OLS."""

    def __init__(self) -> None:
        self.intercepts: NDArray[np.float64] | None = None
        self.slopes: NDArray[np.float64] | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> CombinationRanker:
        xx, yy, _ = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("combination fit has no finite rows")
        intercepts = np.zeros(xx.shape[1], dtype=float)
        slopes = np.zeros(xx.shape[1], dtype=float)
        for j in range(xx.shape[1]):
            intercepts[j], beta = _ols_intercept(xx[:, [j]], yy)
            slopes[j] = float(beta[0]) if beta.size else 0.0
        self.intercepts = intercepts
        self.slopes = slopes
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.intercepts is None or self.slopes is None:
            raise ValueError("CombinationRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        stacked = self.intercepts + x * self.slopes
        return np.mean(stacked, axis=1)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="combo",
            version="v1",
            extra={"paper": "Rapach, Strauss, Zhou, RFS 2010"},
        )


CLASSIC_SIGNS: dict[str, float] = {
    "cs_z_reversal_1": 1.0,
    "cs_z_mom_skip_5_20": 1.0,
    "cs_z_idio_mom_20": 1.0,
    "cs_z_mom_12_1": 1.0,
    "cs_z_high_52w_prox": 1.0,
    "cs_z_max_ret_20": -1.0,
    "cs_z_idio_vol_60": -1.0,
    "cs_z_amihud": 1.0,
}


class ClassicRanker(JoblibMixin):
    """A priori signed public characteristics. No estimated slopes, no OOS peek."""

    def __init__(
        self,
        signs: dict[str, float] | None = None,
        name: str = "classic",
        paper: str = (
            "Jegadeesh 1990; Jegadeesh–Titman skip; Blitz residual mom; "
            "Bali MAX; Ang ivol; Amihud; George–Hwang"
        ),
    ) -> None:
        self.signs = dict(signs) if signs is not None else dict(CLASSIC_SIGNS)
        self._name = str(name)
        self._paper = str(paper)
        self.weights: NDArray[np.float64] | None = None
        self.n_active: int = 0

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> ClassicRanker:
        xx, _, _ = _finite(x, y)
        n_char = int(xx.shape[1]) if xx.size else int(np.asarray(x).shape[1])
        names = kwargs.get("features")
        if names is None or len(list(names)) != n_char:
            weights = np.ones(n_char, dtype=float)
        else:
            weights = np.array([float(self.signs.get(str(n), 0.0)) for n in names], dtype=float)
            if float(np.max(np.abs(weights))) < 1e-15:
                raise ValueError(f"{self._name} matched no signed characteristics")
        self.weights = weights
        self.n_active = int(np.sum(np.abs(weights) > 0.0))
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.weights is None:
            raise ValueError(f"{self._name} is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return x @ self.weights

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name=self._name,
            version="v1",
            extra={"n_active": self.n_active, "paper": self._paper},
        )


def _mean_date_ic(
    x_col: NDArray[np.float64], y: NDArray[np.float64], dates: NDArray[Any]
) -> float:
    ics: list[float] = []
    for idx in date_groups(dates):
        a = x_col[idx]
        b = y[idx]
        finite = np.isfinite(a) & np.isfinite(b)
        if int(finite.sum()) < 5:
            continue
        if float(np.std(a[finite])) < 1e-12 or float(np.std(b[finite])) < 1e-12:
            continue
        ics.append(float(np.corrcoef(a[finite], b[finite])[0, 1]))
    return float(np.mean(ics)) if ics else 0.0


class ICWeightedCombinationRanker(JoblibMixin):
    """Rapach combination with non-negative train date-IC weights (not OOS-tuned)."""

    def __init__(self) -> None:
        self.intercepts: NDArray[np.float64] | None = None
        self.slopes: NDArray[np.float64] | None = None
        self.weights: NDArray[np.float64] | None = None

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> ICWeightedCombinationRanker:
        dates = kwargs.get("dates")
        if dates is None:
            raise ValueError("IC-weighted combination requires per-row dates")
        xx, yy, mask = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("IC-weighted combination fit has no finite rows")
        d = np.asarray(dates)[mask]
        intercepts = np.zeros(xx.shape[1], dtype=float)
        slopes = np.zeros(xx.shape[1], dtype=float)
        ics = np.zeros(xx.shape[1], dtype=float)
        for j in range(xx.shape[1]):
            intercepts[j], beta = _ols_intercept(xx[:, [j]], yy)
            slopes[j] = float(beta[0]) if beta.size else 0.0
            ics[j] = _mean_date_ic(xx[:, j], yy, d)
        w = np.maximum(ics, 0.0)
        if float(np.sum(w)) <= 0.0:
            w = np.ones_like(w)
        self.intercepts = intercepts
        self.slopes = slopes
        self.weights = w / float(np.sum(w))
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.intercepts is None or self.slopes is None or self.weights is None:
            raise ValueError("ICWeightedCombinationRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        stacked = self.intercepts + x * self.slopes
        return stacked @ self.weights

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="combo_ic",
            version="v1",
            extra={"paper": "Rapach, Strauss, Zhou, RFS 2010 IC-weighted combination"},
        )


# Daily CS core (Jegadeesh / Lehmann / JT skip / residual mom / MAX / ivol).
# Rank-space duplicates of the same characteristic are omitted so z and pct
# do not double-count. Signs are a priori, not OOS-tuned.
SHORT_HORIZON_FEATURES: tuple[str, ...] = (
    "cs_z_reversal_1",
    "cs_z_ret_5",
    "cs_z_mom_5",
    "cs_z_ret_overnight",
    "cs_z_ret_open_close",
    "cs_z_mom_skip_5_20",
    "cs_z_idio_mom_20",
    "cs_z_max_ret_20",
    "cs_z_idio_vol_60",
    "cs_z_skew_20",
    "cs_z_ret_overnight_20",
    "cs_z_ret_intraday_20",
)
CLASSIC_ST_SIGNS: dict[str, float] = {
    "cs_z_reversal_1": 1.0,
    "cs_z_ret_5": -1.0,
    "cs_z_mom_skip_5_20": 1.0,
    "cs_z_idio_mom_20": 1.0,
    "cs_z_max_ret_20": -1.0,
    "cs_z_idio_vol_60": -1.0,
}
MSFE_DISCOUNT = 0.99
MSFE_HOLDOUT_FRAC = 0.25
MSFE_MIN_TRAIN_DATES = 8
MSFE_MIN_HOLDOUT_DATES = 4


def _subset_design(
    x: NDArray[np.float64],
    names: list[str] | tuple[str, ...] | None,
    wanted: tuple[str, ...],
) -> tuple[NDArray[np.float64], NDArray[np.intp], list[str] | None]:
    """Slice columns to the a priori daily-CS subset. Full design if none match."""
    x = np.asarray(x, dtype=float)
    n_char = int(x.shape[1]) if x.ndim == 2 else 0
    if names is None or n_char == 0 or len(list(names)) != n_char:
        idx = np.arange(n_char, dtype=np.intp)
        return x, idx, None
    wanted_set = set(wanted)
    idx = np.asarray([i for i, name in enumerate(names) if str(name) in wanted_set], dtype=np.intp)
    if idx.size == 0:
        idx = np.arange(n_char, dtype=np.intp)
        return x, idx, [str(n) for n in names]
    return x[:, idx], idx, [str(names[int(i)]) for i in idx]


class ColumnSubsetRanker(JoblibMixin):
    """Fit/predict an inner ranker on ``SHORT_HORIZON_FEATURES`` only."""

    def __init__(self, inner: Any, columns: tuple[str, ...], name: str) -> None:
        self.inner = inner
        self.columns = columns
        self._name = str(name)
        self._idx: NDArray[np.intp] | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> ColumnSubsetRanker:
        xs, idx, subnames = _subset_design(x, kwargs.get("features"), self.columns)
        self._idx = idx
        kw = dict(kwargs)
        if subnames is not None:
            kw["features"] = subnames
        self.inner.fit(xs, y, **kw)
        return self

    def predict(self, x: NDArray[np.float64], **kwargs: Any) -> NDArray[np.float64]:
        if self._idx is None:
            raise ValueError(f"{self._name} is not fitted")
        x = np.asarray(x, dtype=float)
        xs = x[:, self._idx]
        if kwargs:
            return np.asarray(self.inner.predict(xs, **kwargs), dtype=float)
        return np.asarray(self.inner.predict(xs), dtype=float)

    def metadata(self) -> ModelMeta:
        meta = self.inner.metadata()
        extra = dict(meta.extra or {})
        extra["subset"] = list(self.columns)
        extra["n_subset"] = int(self._idx.size) if self._idx is not None else 0
        return ModelMeta(
            family=meta.family,
            name=self._name,
            version=meta.version,
            extra=extra,
        )


class ReversalRanker(ClassicRanker):
    """Jegadeesh (1990) daily reversal only."""

    def __init__(self) -> None:
        super().__init__(
            signs={"cs_z_reversal_1": 1.0},
            name="reversal",
            paper="Jegadeesh, Journal of Finance 1990",
        )


class ClassicShortRanker(ClassicRanker):
    """A priori daily/weekly CS signs. Monthly zoo characteristics stay at 0."""

    def __init__(self) -> None:
        super().__init__(
            signs=CLASSIC_ST_SIGNS,
            name="classic_st",
            paper="Jegadeesh 1990; Lehmann 1990; Jegadeesh–Titman skip; Blitz residual mom; Bali MAX; Ang ivol",
        )


def _nested_date_cut(n_dates: int) -> tuple[int, int] | None:
    """Train/holdout date counts for nested MSFE. None → in-sample MSE fallback."""
    n = int(n_dates)
    if n < MSFE_MIN_TRAIN_DATES + MSFE_MIN_HOLDOUT_DATES:
        return None
    n_hold = min(max(MSFE_MIN_HOLDOUT_DATES, int(round(n * MSFE_HOLDOUT_FRAC))), n // 4)
    n_train = n - n_hold
    if n_train < MSFE_MIN_TRAIN_DATES or n_hold < MSFE_MIN_HOLDOUT_DATES:
        return None
    return n_train, n_hold


def _discounted_date_mse(
    pred: NDArray[np.float64],
    y: NDArray[np.float64],
    dates: NDArray[Any],
    theta: float,
) -> float:
    groups = date_groups(dates)
    n = len(groups)
    acc = 0.0
    wsum = 0.0
    for i, idx in enumerate(groups):
        err = np.asarray(pred, dtype=float)[idx] - np.asarray(y, dtype=float)[idx]
        finite = np.isfinite(err)
        if not finite.any():
            continue
        mse = float(np.mean(np.square(err[finite])))
        weight = float(theta) ** (n - 1 - i)
        acc += weight * mse
        wsum += weight
    if wsum <= 0.0 or not np.isfinite(acc):
        return float("inf")
    return acc / wsum


class MSFECombinationRanker(JoblibMixin):
    """Rapach discounted-MSFE combination of univariate CS OLS (train-nested)."""

    def __init__(self, theta: float = MSFE_DISCOUNT) -> None:
        if not np.isfinite(theta) or theta <= 0.0 or theta > 1.0:
            raise ValueError("combo_msfe discount theta must be in (0, 1]")
        self.theta = float(theta)
        self.intercepts: NDArray[np.float64] | None = None
        self.slopes: NDArray[np.float64] | None = None
        self.weights: NDArray[np.float64] | None = None

    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> MSFECombinationRanker:
        dates = kwargs.get("dates")
        if dates is None:
            raise ValueError("MSFE combination requires per-row dates")
        xx, yy, mask = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("MSFE combination fit has no finite rows")
        d = np.asarray(dates)[mask]
        groups = date_groups(d)
        n_dates = len(groups)
        n_char = int(xx.shape[1])
        cut = _nested_date_cut(n_dates)
        intercepts = np.zeros(n_char, dtype=float)
        slopes = np.zeros(n_char, dtype=float)
        msfe = np.full(n_char, np.inf, dtype=float)
        if cut is None:
            train_x, train_y, train_d = xx, yy, d
            hold_x = hold_y = hold_d = None
        else:
            n_train, _ = cut
            train_idx = np.concatenate(groups[:n_train])
            hold_idx = np.concatenate(groups[n_train:])
            train_x, train_y, train_d = xx[train_idx], yy[train_idx], d[train_idx]
            hold_x, hold_y, hold_d = xx[hold_idx], yy[hold_idx], d[hold_idx]
        for j in range(n_char):
            intercepts[j], beta = _ols_intercept(xx[:, [j]], yy)
            slopes[j] = float(beta[0]) if beta.size else 0.0
            inner_int, inner_beta = _ols_intercept(train_x[:, [j]], train_y)
            inner_slope = float(inner_beta[0]) if inner_beta.size else 0.0
            if hold_x is None:
                pred = inner_int + inner_slope * train_x[:, j]
                msfe[j] = _discounted_date_mse(pred, train_y, train_d, self.theta)
            else:
                pred = inner_int + inner_slope * hold_x[:, j]
                msfe[j] = _discounted_date_mse(pred, hold_y, hold_d, self.theta)
        finite_msfe = np.where(np.isfinite(msfe) & (msfe > 0.0), msfe, np.nan)
        if not np.isfinite(finite_msfe).any():
            w = np.ones(n_char, dtype=float)
        else:
            inv = 1.0 / np.where(np.isfinite(finite_msfe), finite_msfe, np.inf)
            if float(np.sum(inv)) <= 0.0:
                w = np.ones(n_char, dtype=float)
            else:
                w = inv
        self.intercepts = intercepts
        self.slopes = slopes
        self.weights = w / float(np.sum(w))
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.intercepts is None or self.slopes is None or self.weights is None:
            raise ValueError("MSFECombinationRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        stacked = self.intercepts + x * self.slopes
        return stacked @ self.weights

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="combo_msfe",
            version="v1",
            extra={
                "theta": self.theta,
                "paper": "Rapach, Strauss, Zhou, RFS 2010 discounted MSFE combination",
            },
        )


def make_ridge_st(alpha: float) -> ColumnSubsetRanker:
    return ColumnSubsetRanker(RidgeRanker(alpha), SHORT_HORIZON_FEATURES, "ridge_st")


def make_fm_st(alpha: float) -> ColumnSubsetRanker:
    return ColumnSubsetRanker(FamaMacBethRidgeRanker(alpha), SHORT_HORIZON_FEATURES, "fm_st")


def make_combo_ic_st() -> ColumnSubsetRanker:
    return ColumnSubsetRanker(ICWeightedCombinationRanker(), SHORT_HORIZON_FEATURES, "combo_ic_st")


class AdaptiveLassoRanker(JoblibMixin):
    """Zou (2006) adaptive LASSO: OLS weights, then L1 on reweighted columns.

    Columns are standardized first and the OLS weights are normalized so the
    largest weight is 1 (floor ``1e-3``). Zou's weights are only defined up
    to the scale absorbed by ``lambda``; without the normalization a
    near-zero OLS slope divides its column by ~1e-8 and coordinate descent
    stalls on the ill-conditioned design. The L1 step uses Gram-precomputed
    coordinate descent so expanding ``n\\gg p`` windows finish. ``alpha`` is
    quoted in units of the target's standard deviation on the standardized,
    unit-max-weight design.
    """

    def __init__(self, alpha: float = 0.01, power: float = 1.0) -> None:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("adaptive LASSO alpha must be finite and non-negative")
        if not np.isfinite(power) or power <= 0:
            raise ValueError("adaptive LASSO power must be finite and positive")
        self.alpha = float(alpha)
        self.power = float(power)
        self.coef: NDArray[np.float64] | None = None
        self.intercept: float = 0.0
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> AdaptiveLassoRanker:
        xx, yy, _ = _finite(x, y)
        if xx.shape[0] == 0:
            raise ValueError("adaptive LASSO fit has no finite rows")
        self.mean_ = np.mean(xx, axis=0)
        scale = np.std(xx, axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        xs = (xx - self.mean_) / self.scale_
        first = LinearRegression(fit_intercept=True).fit(xs, yy)
        raw_w = np.power(np.abs(np.asarray(first.coef_, dtype=float)), self.power)
        top = float(np.max(raw_w)) if raw_w.size else 0.0
        if not np.isfinite(top) or top <= 0:
            weights = np.ones_like(raw_w)
        else:
            weights = np.maximum(raw_w / top, 1e-3)
        scaled = xs / weights
        if self.alpha == 0.0:
            fit = LinearRegression(fit_intercept=True).fit(scaled, yy)
            self.intercept = float(fit.intercept_)
            raw = np.asarray(fit.coef_, dtype=float)
        else:
            alpha = _target_scaled_penalty(self.alpha, yy)
            self.intercept, raw = _l1_fit(scaled, yy, alpha)
        self.coef = raw / weights
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.coef is None or self.mean_ is None or self.scale_ is None:
            raise ValueError("AdaptiveLassoRanker is not fitted")
        x = np.where(np.isfinite(x), x, 0.0)
        return self.intercept + ((x - self.mean_) / self.scale_) @ self.coef

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="ranking",
            name="alasso",
            version="v1",
            extra={"alpha": self.alpha, "power": self.power, "paper": "Zou, JASA 2006 adaptive LASSO"},
        )
