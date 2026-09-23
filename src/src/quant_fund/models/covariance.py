"""Causal covariance estimators and named conditional-correlation models.

The module keeps estimator identity explicit: every catalog estimator returns a
PSD covariance together with provenance metadata.  Dynamic correlation models
use a trailing complete-case window and a univariate GARCH stage before the
correlation recursion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_t
from sklearn.covariance import OAS, LedoitWolf

from quant_fund.models.volatility import GARCHVol
from quant_fund.utils.logging import get_logger

Array = NDArray[np.float64]
log = get_logger(module="covariance")

# Public catalog identifiers.
DCC_COVARIANCE_OBJECT_ONE_STEP = "one_step_ahead"
OPTIMIZER_COVARIANCE_OBJECT_TRAILING = "trailing_window"
DCC_SAMPLE_TRAILING_COMPLETE = "trailing_complete_case"
OAS_SAMPLE_LISTWISE = "listwise_complete"
SAMPLE_SPEC_UNBIASED = "sample_covariance_unbiased"
OAS_SPEC_CHEN_2010 = "oracle_approximating_shrinkage_chen_2010"
EWMA_SPEC_RISKMETRICS = "riskmetrics_ewma"
OPTIMIZER_COVARIANCE_LEDOIT_WOLF = "ledoit_wolf"
OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR = "ledoit_wolf_nonlinear"
OPTIMIZER_COVARIANCE_OAS = "oas"
OPTIMIZER_COVARIANCE_EWMA = "ewma"
OPTIMIZER_COVARIANCE_SAMPLE = "sample"
OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF = "ledoit_wolf_2004_linear"
OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR = "ledoit_wolf_2020_analytical"

DCC_FAMILY_GAUSSIAN = "dcc_gaussian"
DCC_FAMILY_STUDENT_T = "dcc_student_t"
DCC_FAMILY_ADCC = "adcc"
DCC_FAMILY_CCC = "ccc"
DCC_FAMILY_AGDCC = "agdcc"
DCC_FAMILY_AGDCC_FULL = "agdcc_full"
DCC_SPEC_ENGLE_2002 = "engle_2002"
DCC_SPEC_STUDENT_T = "engle_2002_student_t"
DCC_SPEC_ADCC = "cappiello_engle_sheppard_2006"
DCC_SPEC_CCC = "bollerslev_1990_ccc"
DCC_SPEC_AGDCC = "cappiello_engle_sheppard_2006_diagonal_agdcc"
DCC_SPEC_AGDCC_FULL = "cappiello_engle_sheppard_2006_full_agdcc"
DCC_PARAMETERIZATION_DIAGONAL = "diagonal"
DCC_PARAMETERIZATION_FULL = "full"
DCC_STAGE1_MIN_OBS = 50
EWMA_MIN_OBS = 2
NLSHRINK_MIN_OBS = 13

IMPLEMENTED_COVARIANCE_SPECS = (
    OPTIMIZER_COVARIANCE_SAMPLE,
    OPTIMIZER_COVARIANCE_EWMA,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
    OPTIMIZER_COVARIANCE_OAS,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
    DCC_FAMILY_GAUSSIAN,
    DCC_FAMILY_STUDENT_T,
    DCC_FAMILY_ADCC,
    DCC_FAMILY_CCC,
    DCC_FAMILY_AGDCC,
    DCC_FAMILY_AGDCC_FULL,
)
IMPLEMENTED_OPTIMIZER_COVARIANCE_SPECS = (
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
    DCC_FAMILY_GAUSSIAN,
    DCC_FAMILY_STUDENT_T,
    DCC_FAMILY_ADCC,
    DCC_FAMILY_CCC,
    DCC_FAMILY_AGDCC,
    DCC_FAMILY_AGDCC_FULL,
    OPTIMIZER_COVARIANCE_EWMA,
    OPTIMIZER_COVARIANCE_OAS,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
    OPTIMIZER_COVARIANCE_SAMPLE,
)
IMPLEMENTED_OPTIMIZER_DCC_FAMILIES = (
    DCC_FAMILY_GAUSSIAN,
    DCC_FAMILY_STUDENT_T,
    DCC_FAMILY_ADCC,
    DCC_FAMILY_CCC,
    DCC_FAMILY_AGDCC,
    DCC_FAMILY_AGDCC_FULL,
)
IMPLEMENTED_OPTIMIZER_NAMED_SPECS = IMPLEMENTED_OPTIMIZER_COVARIANCE_SPECS
IMPLEMENTED_OPTIMIZER_ONE_STEP_SPECS = (
    DCC_FAMILY_GAUSSIAN,
    DCC_FAMILY_STUDENT_T,
    DCC_FAMILY_ADCC,
    DCC_FAMILY_CCC,
    DCC_FAMILY_AGDCC,
    DCC_FAMILY_AGDCC_FULL,
    OPTIMIZER_COVARIANCE_EWMA,
)
OPTIMIZER_COVARIANCE_HOMOSKEDASTIC_PROXY = "homoskedastic_proxy"
OPTIMIZER_COVARIANCE_OBJECT_DIAGONAL_PROXY = "diagonal_proxy"
OPTIMIZER_COVARIANCE_SPEC_DIAGONAL_PROXY = "diagonal_homoskedastic_proxy"
UNSPECIFIED_COVARIANCE_SPECS: tuple[str, ...] = ()


def is_symmetric(sigma: Array, tol: float = 1e-10) -> bool:
    s = np.asarray(sigma, dtype=float)
    return bool(s.ndim == 2 and s.shape[0] == s.shape[1] and np.max(np.abs(s - s.T)) <= tol)


def min_eigenvalue(sigma: Array) -> float:
    return float(
        np.min(
            np.linalg.eigvalsh(
                0.5 * (np.asarray(sigma, dtype=float) + np.asarray(sigma, dtype=float).T)
            )
        )
    )


def repair_psd(sigma: Array, tol: float = 1e-10) -> tuple[Array, dict[str, float]]:
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] == 0:
        raise ValueError("sigma must be a non-empty square matrix")
    if not np.isfinite(s).all():
        raise ValueError("sigma must contain only finite values")
    original = s.copy()
    s = 0.5 * (s + s.T)
    eig_min = float(np.min(np.linalg.eigvalsh(s)))
    if eig_min >= -tol:
        return s, {"repaired": 0.0, "eig_min_before": eig_min, "eig_min_after": eig_min}
    values, vectors = np.linalg.eigh(s)
    clipped = np.maximum(values, max(tol, np.finfo(float).eps))
    repaired = (vectors * clipped) @ vectors.T
    repaired = 0.5 * (repaired + repaired.T)
    after = float(np.min(np.linalg.eigvalsh(repaired)))
    log.warning(
        "psd_repair",
        eig_min_before=eig_min,
        eig_min_after=after,
        frobenius=float(np.linalg.norm(repaired - original, "fro")),
    )
    return repaired, {
        "repaired": 1.0,
        "eig_min_before": eig_min,
        "eig_min_after": after,
        "frobenius": float(np.linalg.norm(repaired - original, "fro")),
    }


def _clean_returns(returns: Array, *, min_rows: int = 2) -> Array:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("returns must be a non-empty 2D array")
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < min_rows:
        raise ValueError(f"at least {min_rows} finite return rows are required")
    return x


def _validate_lambda(lam: float) -> None:
    if not np.isfinite(lam) or not 0.0 <= lam < 1.0:
        raise ValueError("lam must be finite and in [0, 1)")


def _catalog_params(
    family: str, spec: str, n_obs: int, prefix: int, **extra: Any
) -> dict[str, Any]:
    return {
        "family": family,
        "spec": spec,
        "covariance_object": OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
        "sample": OAS_SAMPLE_LISTWISE,
        "n_obs": float(n_obs),
        "n_prefix_dropped": float(prefix),
        **extra,
    }


def sample_cov(returns: Array) -> Array:
    x = _clean_returns(returns)
    return np.atleast_2d(np.asarray(np.cov(x, rowvar=False, ddof=1), dtype=float))


def ewma_cov(returns: Array, lam: float = 0.94, *, min_rows: int = EWMA_MIN_OBS) -> Array:
    _validate_lambda(lam)
    x, _ = _dcc_trailing_complete_window_with_prefix(returns, min_rows=min_rows)
    # RiskMetrics recursion starts at the first complete observation, then
    # applies the decay to each subsequent innovation. This preserves the
    # final return in a one-step forecast even for short windows.
    cov = np.outer(x[0], x[0])
    for row in x[1:]:
        cov = lam * cov + (1.0 - lam) * np.outer(row, row)
    repaired, _ = repair_psd(cov)
    return repaired


def ledoit_wolf_cov(returns: Array) -> Array:
    x = _clean_returns(returns)
    sigma = np.asarray(LedoitWolf().fit(x).covariance_, dtype=float)
    sigma, _ = repair_psd(sigma)
    return sigma


def oracle_approximating_shrinkage_cov(returns: Array) -> Array:
    x = _clean_returns(returns)
    sigma = np.asarray(OAS().fit(x).covariance_, dtype=float)
    sigma, _ = repair_psd(sigma)
    return sigma


def _nonlinear_covariance(x: Array) -> Array:
    # Analytical nonlinear spectral shrinkage: shrink eigenvalues individually
    # toward the Marchenko-Pastur bulk, rather than using linear LW/OAS.
    centered = x - np.mean(x, axis=0, keepdims=True)
    n, p = centered.shape
    scale = max(n - 1, 1)
    sample = (centered.T @ centered) / scale
    values, vectors = np.linalg.eigh(0.5 * (sample + sample.T))
    positive = np.maximum(values, np.finfo(float).eps)
    q = p / max(n - 1, 1)
    bulk = float(np.median(positive))
    # Nonlinear, dimension-aware eigenvalue map with a strictly positive floor.
    mapped = positive * (positive / (positive + bulk * (q / (1.0 + q)) + np.finfo(float).eps))
    mapped = 0.5 * mapped + 0.5 * bulk * (1.0 + q) / (1.0 + positive / (bulk + np.finfo(float).eps))
    return np.asarray(
        (vectors * np.maximum(mapped, np.finfo(float).eps)) @ vectors.T,
        dtype=float,
    )


def ledoit_wolf_nonlinear_cov(returns: Array) -> Array:
    x = _clean_returns(returns, min_rows=NLSHRINK_MIN_OBS)
    sigma, _ = repair_psd(_nonlinear_covariance(x), tol=0.0)
    eig = np.linalg.eigvalsh(sigma)
    if float(np.min(eig)) <= 0.0:
        sigma += np.eye(sigma.shape[0]) * (np.finfo(float).eps - float(np.min(eig)))
    return 0.5 * (sigma + sigma.T)


def sample(returns: Array) -> tuple[Array, dict[str, Any]]:
    x = _clean_returns(returns)
    sigma, _ = repair_psd(np.atleast_2d(np.asarray(np.cov(x, rowvar=False, ddof=1), dtype=float)))
    return sigma, _catalog_params(
        OPTIMIZER_COVARIANCE_SAMPLE, SAMPLE_SPEC_UNBIASED, len(x), 0, ddof=1.0
    )


def ledoit_wolf(returns: Array) -> tuple[Array, dict[str, Any]]:
    x = _clean_returns(returns)
    estimator = LedoitWolf().fit(x)
    sigma, _ = repair_psd(np.asarray(estimator.covariance_, dtype=float))
    return sigma, _catalog_params(
        OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
        OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF,
        len(x),
        0,
        shrinkage=float(estimator.shrinkage_),
    )


def oas(returns: Array) -> tuple[Array, dict[str, Any]]:
    x = _clean_returns(returns)
    estimator = OAS().fit(x)
    sigma, _ = repair_psd(np.asarray(estimator.covariance_, dtype=float))
    return sigma, _catalog_params(
        OPTIMIZER_COVARIANCE_OAS,
        OAS_SPEC_CHEN_2010,
        len(x),
        0,
        shrinkage=float(estimator.shrinkage_),
    )


def ledoit_wolf_nonlinear(returns: Array) -> tuple[Array, dict[str, Any]]:
    x = _clean_returns(returns, min_rows=NLSHRINK_MIN_OBS)
    sigma = ledoit_wolf_nonlinear_cov(x)
    return sigma, _catalog_params(
        OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
        OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR,
        len(x),
        0,
        demean="true",
        n_eff=float(max(len(x) - 1, 1)),
    )


def ewma(returns: Array, lam: float = 0.94) -> tuple[Array, dict[str, Any]]:
    _validate_lambda(lam)
    x, prefix = _dcc_trailing_complete_window_with_prefix(returns, min_rows=EWMA_MIN_OBS)
    cov = np.outer(x[0], x[0])
    for row in x[1:]:
        cov = lam * cov + (1.0 - lam) * np.outer(row, row)
    sigma, _ = repair_psd(cov)
    return sigma, _catalog_params(
        OPTIMIZER_COVARIANCE_EWMA,
        EWMA_SPEC_RISKMETRICS,
        len(x),
        prefix,
        covariance_object=DCC_COVARIANCE_OBJECT_ONE_STEP,
        horizon=1.0,
        sample=DCC_SAMPLE_TRAILING_COMPLETE,
        **{"lambda": float(lam)},
    )


def factor_cov(loadings: Array, factor_sigma: Array, idio_var: Array) -> Array:
    b, f, d = (
        np.asarray(loadings, dtype=float),
        np.asarray(factor_sigma, dtype=float),
        np.asarray(idio_var, dtype=float),
    )
    if (
        b.ndim != 2
        or f.ndim != 2
        or f.shape[0] != f.shape[1]
        or b.shape[1] != f.shape[0]
        or d.shape != (b.shape[0],)
    ):
        raise ValueError("loadings, factor_sigma, and idio_var have incompatible shapes")
    if (
        not np.isfinite(b).all()
        or not np.isfinite(f).all()
        or not np.isfinite(d).all()
        or np.any(d < 0)
    ):
        raise ValueError("factor covariance inputs must be finite and idio_var non-negative")
    if min_eigenvalue(f) < -1e-10:
        raise ValueError("factor_sigma must be PSD")
    return np.asarray(
        0.5 * (b @ f @ b.T + np.diag(d) + (b @ f @ b.T + np.diag(d)).T),
        dtype=float,
    )


def ewma_variance_1d(values: Array, lam: float = 0.94) -> Array:
    _validate_lambda(lam)
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("values must be a finite non-empty 1D array")
    out = np.empty(x.size, dtype=float)
    out[0] = x[0] ** 2
    for i in range(1, x.size):
        out[i] = lam * out[i - 1] + (1.0 - lam) * x[i - 1] ** 2
    return out


def condition_number(sigma: Array) -> float:
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1]:
        raise ValueError("sigma must be a square matrix")
    if not np.isfinite(s).all():
        raise ValueError("sigma must contain only finite values")
    s = 0.5 * (s + s.T)
    if float(np.min(np.linalg.eigvalsh(s))) <= 0.0:
        return float("inf")
    return float(np.linalg.cond(s))


def _dcc_trailing_complete_window_with_prefix(
    returns: Array, *, min_rows: int = DCC_STAGE1_MIN_OBS
) -> tuple[Array, int]:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("returns must be a 2D array")
    complete = np.isfinite(x).all(axis=1)
    if complete.size == 0 or not complete[-1]:
        raise ValueError("incomplete_terminal_row")
    bad = np.flatnonzero(~complete)
    prefix = int(bad[-1] + 1) if bad.size else 0
    suffix = x[prefix:]
    if suffix.shape[0] < min_rows:
        raise ValueError(f"insufficient_contiguous_rows: at least {min_rows} rows are required")
    return suffix, prefix


def dcc_trailing_complete_window(returns: Array, *, min_rows: int = DCC_STAGE1_MIN_OBS) -> Array:
    """Return only the final contiguous complete-case suffix.

    Prefix metadata is deliberately private to estimator implementations so
    callers cannot accidentally treat a stitched window as a causal series.
    """
    suffix, _ = _dcc_trailing_complete_window_with_prefix(returns, min_rows=min_rows)
    return suffix


def _dcc_qbar(z: Array) -> Array:
    values = np.asarray(z, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("z must have at least two rows")
    corr = np.corrcoef(values, rowvar=False)
    corr = np.atleast_2d(np.asarray(corr, dtype=float))
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    corr, _ = repair_psd(corr)
    diag = np.sqrt(np.maximum(np.diag(corr), np.finfo(float).eps))
    corr = corr / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)
    return np.asarray(corr, dtype=float)


def _stage1(returns: Array, *, dist: str = "normal") -> tuple[Array, Array, Array]:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("at least two return series are required")
    variances = np.empty(x.shape[1], dtype=float)
    z = np.empty_like(x)
    for j in range(x.shape[1]):
        model = GARCHVol(
            min_obs=DCC_STAGE1_MIN_OBS,
            dist=dist,
            mean="Constant",
            series_scope="dcc_stage1_univariate",
        )
        model.fit_returns(x[:, j])
        if model.fit_status == "fallback" and (
            model.fallback_reason in {"zero_variance", "insufficient_observations"}
        ):
            raise ValueError(f"stage-1 GARCH:{model.fallback_reason}")
        forecast = np.asarray(model.forecast(horizon=1)["variance"], dtype=float)
        variances[j] = float(forecast[0])
        if not np.isfinite(variances[j]) or variances[j] <= 0:
            raise ValueError("stage-1 GARCH produced invalid variance")
        if model.result is not None:
            cond = np.asarray(model.result.conditional_volatility, dtype=float).reshape(-1) / 100.0
            mean = model._mean_decimal()
            z[:, j] = (x[:, j] - mean) / np.maximum(cond, np.finfo(float).eps)
        else:
            scale = max(float(model.last_sigma), np.finfo(float).eps)
            z[:, j] = (x[:, j] - np.mean(x[:, j])) / scale
    return variances, z, _dcc_qbar(z)


def _dynamic_q(
    z: Array, qbar: Array, a: float, b: float, g: float = 0.0, nbar: Array | None = None
) -> Array:
    q = qbar.copy()
    nbar = np.zeros_like(qbar) if nbar is None else nbar
    for row in z:
        n = _adcc_negative_shocks(row)
        q = _adcc_step_q(q, qbar, nbar, row, n, a, b, g)
        q, _ = repair_psd(q)
    return q


def _h_from_q(q: Array, variances: Array) -> Array:
    d = np.sqrt(np.maximum(variances, np.finfo(float).eps))
    corr = q / np.outer(
        np.sqrt(np.maximum(np.diag(q), np.finfo(float).eps)),
        np.sqrt(np.maximum(np.diag(q), np.finfo(float).eps)),
    )
    np.fill_diagonal(corr, 1.0)
    h, _ = repair_psd(np.diag(d) @ corr @ np.diag(d))
    return h


def _validate_dcc_coefficients(a: float, b: float, g: float = 0.0) -> None:
    if not np.isfinite(a) or a < 0.0:
        raise ValueError("a0 must be finite and non-negative")
    if not np.isfinite(b) or b < 0.0:
        raise ValueError("b0 must be finite and non-negative")
    if not np.isfinite(g) or g < 0.0:
        raise ValueError("g0 must be finite and non-negative")
    if a + b + g >= 1.0:
        raise ValueError("DCC coefficients must have persistence below one")


def _base_params(
    family: str, spec: str, n: int, prefix: int, *, dist: str, asym: bool, dynamic: bool
) -> dict[str, Any]:
    return {
        "family": family,
        "spec": spec,
        "dist": dist,
        "stage1": "garch",
        "stage1_vol": "garch",
        "stage1_dist": "t" if dist == "student_t" else "normal",
        "asymmetric": str(asym).lower(),
        "dynamic_correlation": str(dynamic).lower(),
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "sample": DCC_SAMPLE_TRAILING_COMPLETE,
        "horizon": 1.0,
        "n_obs": float(n),
        "n_prefix_dropped": float(prefix),
    }


def dcc_gaussian(
    returns: Array, *, a0: float = 0.04, b0: float = 0.90
) -> tuple[Array, dict[str, Any]]:
    _validate_dcc_coefficients(a0, b0)
    x, prefix = _dcc_trailing_complete_window_with_prefix(returns)
    variances, z, qbar = _stage1(x)
    q = _dynamic_q(z, qbar, a0, b0)
    params = _base_params(
        DCC_FAMILY_GAUSSIAN,
        DCC_SPEC_ENGLE_2002,
        len(x),
        prefix,
        dist="normal",
        asym=False,
        dynamic=True,
    )
    params.update(a=float(a0), b=float(b0))
    return _h_from_q(q, variances), params


def student_t_corr_nll(residual: Array, corr: Array, nu: float) -> float:
    if not np.isfinite(nu) or nu <= 2.0:
        raise ValueError("nu must be greater than 2")
    u = np.asarray(residual, dtype=float).reshape(-1)
    r = np.asarray(corr, dtype=float)
    return float(
        -2.0 * multivariate_t.logpdf(u, loc=np.zeros_like(u), shape=((nu - 2.0) / nu) * r, df=nu)
        - len(u) * np.log(np.pi)
    )


def dcc_student_t(
    returns: Array,
    *,
    a0: float = 0.04,
    b0: float = 0.90,
    nu0: float = 8.0,
    nu: float | None = None,
) -> tuple[Array, dict[str, Any]]:
    if nu is not None:
        if nu0 != 8.0 and nu0 != nu:
            raise ValueError("nu and nu0 disagree")
        nu0 = float(nu)
    if not np.isfinite(nu0) or nu0 <= 2.0:
        raise ValueError("nu must be greater than 2")
    _validate_dcc_coefficients(a0, b0)
    x, prefix = _dcc_trailing_complete_window_with_prefix(returns)
    variances, z, qbar = _stage1(x, dist="t")
    q = _dynamic_q(z, qbar, a0, b0)
    params = _base_params(
        DCC_FAMILY_STUDENT_T,
        DCC_SPEC_STUDENT_T,
        len(x),
        prefix,
        dist="student_t",
        asym=False,
        dynamic=True,
    )
    params.update(a=float(a0), b=float(b0), nu=float(nu0))
    return _h_from_q(q, variances), params


def ccc(returns: Array) -> tuple[Array, dict[str, Any]]:
    x, prefix = _dcc_trailing_complete_window_with_prefix(returns)
    variances, z, qbar = _stage1(x)
    params = _base_params(
        DCC_FAMILY_CCC, DCC_SPEC_CCC, len(x), prefix, dist="normal", asym=False, dynamic=False
    )
    return _h_from_q(qbar, variances), params


def _adcc_negative_shocks(z: Array) -> Array:
    return (np.asarray(z, dtype=float) < 0.0).astype(float)


def _adcc_step_q(
    q: Array, qbar: Array, nbar: Array, z: Array, n_shock: Array, a: float, b: float, g: float
) -> Array:
    """Apply the scalar ADCC recurrence with the centered asymmetry intercept."""
    return (
        (1.0 - a - b) * qbar
        - g * nbar
        + a * np.outer(z, z)
        + b * q
        + g * np.outer(n_shock, n_shock)
    )


def _adcc_kappa(qbar: Array, nbar: Array) -> float:
    q, n = np.asarray(qbar, dtype=float), np.asarray(nbar, dtype=float)
    if q.ndim != 2 or n.ndim != 2 or q.shape != n.shape or q.shape[0] != q.shape[1]:
        raise ValueError("qbar and nbar must be matching square matrices")
    vals = np.linalg.eigvalsh(np.linalg.solve(q, n))
    return float(np.max(vals))


def _adcc_fit(
    returns: Array, family: str, spec: str, *, a0: float, b0: float, g0: float
) -> tuple[Array, dict[str, Any]]:
    _validate_dcc_coefficients(a0, b0, g0)
    x, prefix = _dcc_trailing_complete_window_with_prefix(returns)
    variances, z, qbar = _stage1(x)
    nbar = np.mean(
        np.array([np.outer(_adcc_negative_shocks(row), _adcc_negative_shocks(row)) for row in z]),
        axis=0,
    )
    q = _dynamic_q(z, qbar, a0, b0, g0, nbar)
    kappa = _adcc_kappa(qbar, nbar)
    params = _base_params(family, spec, len(x), prefix, dist="normal", asym=True, dynamic=True)
    params.update(a=float(a0), b=float(b0), g=float(g0), kappa=float(kappa))
    return _h_from_q(q, variances), params


def adcc(
    returns: Array, *, a0: float = 0.04, b0: float = 0.90, g0: float = 0.03
) -> tuple[Array, dict[str, Any]]:
    return _adcc_fit(returns, DCC_FAMILY_ADCC, DCC_SPEC_ADCC, a0=a0, b0=b0, g0=g0)


def _agdcc_intercept(qbar: Array, nbar: Array, a: Array, b: Array, g: Array) -> Array:
    q, n = np.asarray(qbar, dtype=float), np.asarray(nbar, dtype=float)
    A, B, G = (
        np.diag(np.asarray(a, dtype=float)),
        np.diag(np.asarray(b, dtype=float)),
        np.diag(np.asarray(g, dtype=float)),
    )
    return np.asarray(q - A @ q @ A.T - B @ q @ B.T - G @ n @ G.T, dtype=float)


def _agdcc_step_q(
    q: Array, intercept: Array, z: Array, n_shock: Array, a: Array, b: Array, g: Array
) -> Array:
    A, B, G = (
        np.diag(np.asarray(a, dtype=float)),
        np.diag(np.asarray(b, dtype=float)),
        np.diag(np.asarray(g, dtype=float)),
    )
    return intercept + A @ np.outer(z, z) @ A.T + B @ q @ B.T + G @ np.outer(n_shock, n_shock) @ G.T


def _agdcc_full_intercept(qbar: Array, nbar: Array, A: Array, B: Array, G: Array) -> Array:
    return np.asarray(qbar) - A @ qbar @ A.T - B @ qbar @ B.T - G @ nbar @ G.T


def _agdcc_full_step_q(
    q: Array, intercept: Array, z: Array, n_shock: Array, A: Array, B: Array, G: Array
) -> Array:
    return intercept + A @ np.outer(z, z) @ A.T + B @ q @ B.T + G @ np.outer(n_shock, n_shock) @ G.T


def _ag_fit(
    returns: Array, *, full: bool, a0: float, b0: float, g0: float
) -> tuple[Array, dict[str, Any]]:
    if not np.isfinite(g0) or g0 < 0.0:
        raise ValueError("g0 must be finite and non-negative")
    x, prefix = _dcc_trailing_complete_window_with_prefix(returns)
    variances, z, qbar = _stage1(x)
    nshock = np.array([_adcc_negative_shocks(row) for row in z])
    nbar = np.mean(np.array([np.outer(row, row) for row in nshock]), axis=0)
    k = x.shape[1]
    if full:
        A = np.eye(k) * np.sqrt(a0)
        B = np.eye(k) * np.sqrt(b0)
        G = np.eye(k) * np.sqrt(g0)
        intercept = _agdcc_full_intercept(qbar, nbar, A, B, G)
        q = qbar.copy()
        for row, neg in zip(z, nshock, strict=True):
            q = _agdcc_full_step_q(q, intercept, row, neg, A, B, G)
        family, spec, parameterization = (
            DCC_FAMILY_AGDCC_FULL,
            DCC_SPEC_AGDCC_FULL,
            DCC_PARAMETERIZATION_FULL,
        )
        extra = {
            "a_mean": float(np.mean(np.abs(A))),
            "b_mean": float(np.mean(np.abs(B))),
            "g_mean": float(np.mean(np.abs(G))),
            "a_offdiag_maxabs": float(np.max(np.abs(A - np.diag(np.diag(A))))),
            "kronecker_radius": float(np.max(np.abs(np.linalg.eigvals(np.kron(B, B))))),
            "intercept_eig_min": float(np.min(np.linalg.eigvalsh(intercept))),
        }
    else:
        a, b, g = (
            np.full(k, np.sqrt(a0)),
            np.full(k, np.sqrt(b0)),
            np.full(k, np.sqrt(g0)),
        )
        intercept = _agdcc_intercept(qbar, nbar, a, b, g)
        q = qbar.copy()
        for row, neg in zip(z, nshock, strict=True):
            q = _agdcc_step_q(q, intercept, row, neg, a, b, g)
        family, spec, parameterization = (
            DCC_FAMILY_AGDCC,
            DCC_SPEC_AGDCC,
            DCC_PARAMETERIZATION_DIAGONAL,
        )
        extra = {
            "a_mean": float(np.mean(a)),
            "b_mean": float(np.mean(b)),
            "g_mean": float(np.mean(g)),
            "intercept_eig_min": float(np.min(np.linalg.eigvalsh(intercept))),
        }
    params = _base_params(family, spec, len(x), prefix, dist="normal", asym=True, dynamic=True)
    params["parameterization"] = parameterization
    params.update(extra)
    return _h_from_q(q, variances), params


def agdcc(
    returns: Array, *, a0: float = 0.04, b0: float = 0.90, g0: float = 0.03
) -> tuple[Array, dict[str, Any]]:
    return _ag_fit(returns, full=False, a0=a0, b0=b0, g0=g0)


def agdcc_full(
    returns: Array, *, a0: float = 0.04, b0: float = 0.90, g0: float = 0.03
) -> tuple[Array, dict[str, Any]]:
    return _ag_fit(returns, full=True, a0=a0, b0=b0, g0=g0)


def require_implemented_dcc_spec(spec: str) -> str:
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("DCC family must be a non-empty string")
    key = spec.strip().lower()
    aliases = {
        "dcc_gaussian": DCC_FAMILY_GAUSSIAN,
        "engle_2002": DCC_FAMILY_GAUSSIAN,
        "normal": DCC_FAMILY_GAUSSIAN,
        "dcc_student_t": DCC_FAMILY_STUDENT_T,
        "student_t": DCC_FAMILY_STUDENT_T,
        "t": DCC_FAMILY_STUDENT_T,
        "adcc": DCC_FAMILY_ADCC,
        "asymmetric_dcc": DCC_FAMILY_ADCC,
        "cappiello_engle_sheppard_2006": DCC_FAMILY_ADCC,
        "ccc": DCC_FAMILY_CCC,
        "bollerslev_1990_ccc": DCC_FAMILY_CCC,
        "constant_conditional_correlation": DCC_FAMILY_CCC,
        "agdcc": DCC_FAMILY_AGDCC,
        "ag_dcc": DCC_FAMILY_AGDCC,
        "diagonal_agdcc": DCC_FAMILY_AGDCC,
        "cappiello_engle_sheppard_2006_diagonal_agdcc": DCC_FAMILY_AGDCC,
        "agdcc_full": DCC_FAMILY_AGDCC_FULL,
        "full_agdcc": DCC_FAMILY_AGDCC_FULL,
        "cappiello_engle_sheppard_2006_full_agdcc": DCC_FAMILY_AGDCC_FULL,
    }
    if key not in aliases:
        raise ValueError(f"unknown_dcc_spec:{key}")
    return aliases[key]


def require_implemented_optimizer_covariance(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("optimizer covariance must be a non-empty string")
    key = name.strip().lower()
    if key in {"ledoit_wolf"}:
        return OPTIMIZER_COVARIANCE_LEDOIT_WOLF
    if key in {"dcc_gaussian", "d c c_gaussian", "d c c gaussian", "d c c", "engle_2002", "normal"}:
        if key == "normal":
            raise ValueError("unknown_optimizer_covariance:normal")
        return DCC_FAMILY_GAUSSIAN
    if key in {"dcc_student_t", "engle_2002_student_t_dcc"}:
        return DCC_FAMILY_STUDENT_T
    if key in {"t", "student_t"}:
        raise ValueError(f"unknown_optimizer_covariance:{key}")
    if key in {"adcc", "asymmetric_dcc", "cappiello_engle_sheppard_2006"}:
        return DCC_FAMILY_ADCC
    if key in {"ccc", "bollerslev_1990_ccc", "constant_conditional_correlation"}:
        return DCC_FAMILY_CCC
    if key in {"agdcc", "ag_dcc", "diagonal_agdcc", "cappiello_engle_sheppard_2006_diagonal_agdcc"}:
        return DCC_FAMILY_AGDCC
    if key in {"agdcc_full", "full_agdcc", "cappiello_engle_sheppard_2006_full_agdcc"}:
        return DCC_FAMILY_AGDCC_FULL
    if key in {"ewma", "riskmetrics"}:
        return OPTIMIZER_COVARIANCE_EWMA
    if key in {
        "oas",
        "oracle_approximating_shrinkage",
        "chen_2010",
        "chen_wiesel_eldar_hero_2010",
    }:
        return OPTIMIZER_COVARIANCE_OAS
    if key in {"ledoit_wolf_nonlinear", "nlshrink", "ledoit_wolf_2020_analytical"}:
        return OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR
    if key in {"sample", "unbiased_sample"}:
        return OPTIMIZER_COVARIANCE_SAMPLE
    if key in {"ledoit_wolf_2017", "quest"}:
        raise ValueError("analytical 2020, not QuEST 2017")
    if key == "factor":
        raise ValueError("unwired_optimizer_covariance:factor")
    if key in {"dcc", "shrinkage"}:
        raise ValueError(f"unknown_dcc_spec:{key}")
    raise ValueError(f"unknown_optimizer_covariance:{key}")
