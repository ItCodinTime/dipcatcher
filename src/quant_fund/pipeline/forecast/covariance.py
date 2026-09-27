"""Named optimizer covariance estimates.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.models.covariance import (
    DCC_COVARIANCE_OBJECT_ONE_STEP,
    DCC_FAMILY_ADCC,
    DCC_FAMILY_AGDCC,
    DCC_FAMILY_AGDCC_FULL,
    DCC_FAMILY_CCC,
    DCC_FAMILY_GAUSSIAN,
    DCC_FAMILY_STUDENT_T,
    DCC_SPEC_ADCC,
    DCC_SPEC_AGDCC,
    DCC_SPEC_AGDCC_FULL,
    DCC_SPEC_CCC,
    DCC_SPEC_ENGLE_2002,
    DCC_SPEC_STUDENT_T,
    DCC_STAGE1_MIN_OBS,
    EWMA_MIN_OBS,
    EWMA_SPEC_RISKMETRICS,
    IMPLEMENTED_OPTIMIZER_DCC_FAMILIES,
    NLSHRINK_MIN_OBS,
    OAS_SPEC_CHEN_2010,
    OPTIMIZER_COVARIANCE_EWMA,
    OPTIMIZER_COVARIANCE_HOMOSKEDASTIC_PROXY,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
    OPTIMIZER_COVARIANCE_OAS,
    OPTIMIZER_COVARIANCE_OBJECT_DIAGONAL_PROXY,
    OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
    OPTIMIZER_COVARIANCE_SAMPLE,
    OPTIMIZER_COVARIANCE_SPEC_DIAGONAL_PROXY,
    OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF,
    OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR,
    SAMPLE_SPEC_UNBIASED,
    adcc,
    agdcc,
    agdcc_full,
    ccc,
    dcc_gaussian,
    dcc_student_t,
    dcc_trailing_complete_window,
    ewma,
    ledoit_wolf,
    ledoit_wolf_nonlinear,
    oas,
    repair_psd,
    require_implemented_optimizer_covariance,
    sample,
)
from quant_fund.models.realized_garch import REALIZED_GARCH_FAMILY, REALIZED_GARCH_MEASURE
from quant_fund.schemas.forecast import MARKET_RISK_OVERLAY_REALIZED_GARCH

from .realized import (
    RealizedGarchMarketForecast,
    apply_market_variance_overlay_to_covariance,
    resolve_market_variance_overlay_asof,
)
from .state import Array, log

if TYPE_CHECKING:
    from .garch import GarchMarketForecast


@dataclass(frozen=True)
class OptimizerCovarianceEstimate:
    """Named trailing covariance used by ``optimize_asof`` and ``/risk/portfolio``."""

    sigma: Array
    security_ids: list[str]
    estimator: str
    covariance_object: str
    spec: str
    n_obs: int
    market_overlay: str | None
    overlay: GarchMarketForecast | RealizedGarchMarketForecast | None
    params: dict[str, float | str]
    unmeasured_reason: str | None = None
    fallback_reason: str | None = None


def _trailing_return_matrix(hist: pl.DataFrame, ids: list[str]) -> tuple[list[str], Array]:
    if "ret_1" not in hist.columns or hist.is_empty() or not ids:
        return [], np.empty((0, 0), dtype=float)
    wide = (
        hist.select(["event_time", "security_id", "ret_1"])
        .sort(["event_time", "security_id"])
        .pivot(on="security_id", index="event_time", values="ret_1")
        .sort("event_time")
    )
    cols = [sid for sid in ids if sid in wide.columns]
    if not cols:
        return [], np.empty((0, 0), dtype=float)
    return cols, np.asarray(wide.select(cols).to_numpy(), dtype=float)


def _named_dcc_optimizer_estimate(
    estimator: str,
    mat: Array,
    cols: list[str],
    finite_rows: int,
    psd_tol: float,
) -> OptimizerCovarianceEstimate:
    """Fit a named DCC optimizer path. Look up fitters at call time.

    Patches of ``dcc_gaussian`` / ``dcc_student_t`` / ``adcc`` / ``ccc`` /
    ``agdcc`` / ``agdcc_full`` on this module must still bind. A family or
    covariance-object mismatch fails closed so unrestricted AG-DCC cannot
    silently size as diagonal AG-DCC, scalar ADCC, Gaussian DCC,
    Student-t DCC, or CCC, and diagonal AG-DCC cannot silently size as
    those other specs either.
    """
    fitter: Callable[[Array], tuple[Array, dict[str, float | str]]]
    if estimator == DCC_FAMILY_GAUSSIAN:
        fitter = dcc_gaussian
        default_spec = DCC_SPEC_ENGLE_2002
    elif estimator == DCC_FAMILY_STUDENT_T:
        fitter = dcc_student_t
        default_spec = DCC_SPEC_STUDENT_T
    elif estimator == DCC_FAMILY_ADCC:
        fitter = adcc
        default_spec = DCC_SPEC_ADCC
    elif estimator == DCC_FAMILY_CCC:
        fitter = ccc
        default_spec = DCC_SPEC_CCC
    elif estimator == DCC_FAMILY_AGDCC:
        fitter = agdcc
        default_spec = DCC_SPEC_AGDCC
    elif estimator == DCC_FAMILY_AGDCC_FULL:
        fitter = agdcc_full
        default_spec = DCC_SPEC_AGDCC_FULL
    else:
        raise ValueError(f"unknown_optimizer_dcc_family:{estimator}")
    prefix = f"optimizer_covariance_failed:{estimator}"
    if len(cols) < 2:
        raise ValueError(f"{prefix}:fewer_than_two_securities")
    if finite_rows < DCC_STAGE1_MIN_OBS:
        raise ValueError(f"{prefix}:insufficient_finite_rows:{finite_rows}")
    try:
        dcc_trailing_complete_window(mat, min_rows=DCC_STAGE1_MIN_OBS)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    try:
        sigma, params = fitter(mat)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    sigma, _ = repair_psd(sigma, psd_tol)
    family = str(params.get("family", ""))
    if family != estimator:
        raise ValueError(f"{prefix}:unexpected_family:{family}")
    object_name = str(params.get("covariance_object", DCC_COVARIANCE_OBJECT_ONE_STEP))
    spec_name = str(params.get("spec", default_spec))
    if object_name != DCC_COVARIANCE_OBJECT_ONE_STEP:
        raise ValueError(f"{prefix}:unexpected_covariance_object:{object_name}")
    return OptimizerCovarianceEstimate(
        sigma=np.asarray(sigma, dtype=float),
        security_ids=cols,
        estimator=estimator,
        covariance_object=object_name,
        spec=spec_name,
        n_obs=int(float(params.get("n_obs", finite_rows))),
        market_overlay=None,
        overlay=None,
        params=dict(params),
    )


def _named_ewma_optimizer_estimate(
    mat: Array,
    cols: list[str],
    finite_rows: int,
    psd_tol: float,
    lam: float,
) -> OptimizerCovarianceEstimate:
    """Fit named RiskMetrics EWMA. Look up ``ewma`` at call time so patches bind.

    A family or covariance-object mismatch fails closed so EWMA cannot
    silently size as Ledoit–Wolf or DCC. The matrix is not GARCH-overlaid.
    """
    prefix = f"optimizer_covariance_failed:{OPTIMIZER_COVARIANCE_EWMA}"
    if len(cols) < 2:
        raise ValueError(f"{prefix}:fewer_than_two_securities")
    if finite_rows < EWMA_MIN_OBS:
        raise ValueError(f"{prefix}:insufficient_finite_rows:{finite_rows}")
    try:
        dcc_trailing_complete_window(mat, min_rows=EWMA_MIN_OBS)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    try:
        sigma, params = ewma(mat, lam=lam)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    sigma, _ = repair_psd(sigma, psd_tol)
    family = str(params.get("family", ""))
    if family != OPTIMIZER_COVARIANCE_EWMA:
        raise ValueError(f"{prefix}:unexpected_family:{family}")
    object_name = str(params.get("covariance_object", DCC_COVARIANCE_OBJECT_ONE_STEP))
    spec_name = str(params.get("spec", EWMA_SPEC_RISKMETRICS))
    if object_name != DCC_COVARIANCE_OBJECT_ONE_STEP:
        raise ValueError(f"{prefix}:unexpected_covariance_object:{object_name}")
    return OptimizerCovarianceEstimate(
        sigma=np.asarray(sigma, dtype=float),
        security_ids=cols,
        estimator=OPTIMIZER_COVARIANCE_EWMA,
        covariance_object=object_name,
        spec=spec_name,
        n_obs=int(float(params.get("n_obs", finite_rows))),
        market_overlay=None,
        overlay=None,
        params=dict(params),
    )


def _named_sample_optimizer_estimate(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
    mat: Array,
    cols: list[str],
    finite_rows: int,
    psd_tol: float,
) -> OptimizerCovarianceEstimate:
    """Fit named unbiased sample covariance. Look up ``sample`` at call time.

    A family or covariance-object mismatch fails closed so sample cannot
    silently size as Ledoit–Wolf, OAS, EWMA, or DCC. The matrix is
    GARCH/RGARCH overlay-scaled like trailing Ledoit–Wolf. When T>N the
    estimator stays sample rather than switching to Ledoit–Wolf.
    """
    prefix = f"optimizer_covariance_failed:{OPTIMIZER_COVARIANCE_SAMPLE}"
    if len(cols) < 2:
        raise ValueError(f"{prefix}:fewer_than_two_securities")
    if finite_rows < 2:
        raise ValueError(f"{prefix}:insufficient_finite_rows:{finite_rows}")
    try:
        sigma, params = sample(mat)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    sigma, _ = repair_psd(sigma, psd_tol)
    family = str(params.get("family", ""))
    if family != OPTIMIZER_COVARIANCE_SAMPLE:
        raise ValueError(f"{prefix}:unexpected_family:{family}")
    object_name = str(params.get("covariance_object", OPTIMIZER_COVARIANCE_OBJECT_TRAILING))
    spec_name = str(params.get("spec", SAMPLE_SPEC_UNBIASED))
    if object_name != OPTIMIZER_COVARIANCE_OBJECT_TRAILING:
        raise ValueError(f"{prefix}:unexpected_covariance_object:{object_name}")
    sigma, overlay, overlay_kind = apply_market_variance_overlay_to_covariance(
        config, frame, asof, sigma
    )
    return OptimizerCovarianceEstimate(
        sigma=np.asarray(sigma, dtype=float),
        security_ids=cols,
        estimator=OPTIMIZER_COVARIANCE_SAMPLE,
        covariance_object=object_name,
        spec=spec_name,
        n_obs=int(float(params.get("n_obs", finite_rows))),
        market_overlay=overlay_kind,
        overlay=overlay,
        params=dict(params),
    )


def _named_oas_optimizer_estimate(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
    mat: Array,
    cols: list[str],
    finite_rows: int,
    psd_tol: float,
) -> OptimizerCovarianceEstimate:
    """Fit named Chen OAS. Look up ``oas`` at call time so patches bind.

    A family or covariance-object mismatch fails closed so OAS cannot
    silently size as Ledoit–Wolf, sample, EWMA, or DCC. The matrix is
    GARCH/RGARCH overlay-scaled like trailing Ledoit–Wolf. When T<=N the
    estimator stays OAS rather than switching to sample.
    """
    prefix = f"optimizer_covariance_failed:{OPTIMIZER_COVARIANCE_OAS}"
    if len(cols) < 2:
        raise ValueError(f"{prefix}:fewer_than_two_securities")
    if finite_rows < 2:
        raise ValueError(f"{prefix}:insufficient_finite_rows:{finite_rows}")
    try:
        sigma, params = oas(mat)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    sigma, _ = repair_psd(sigma, psd_tol)
    family = str(params.get("family", ""))
    if family != OPTIMIZER_COVARIANCE_OAS:
        raise ValueError(f"{prefix}:unexpected_family:{family}")
    object_name = str(params.get("covariance_object", OPTIMIZER_COVARIANCE_OBJECT_TRAILING))
    spec_name = str(params.get("spec", OAS_SPEC_CHEN_2010))
    if object_name != OPTIMIZER_COVARIANCE_OBJECT_TRAILING:
        raise ValueError(f"{prefix}:unexpected_covariance_object:{object_name}")
    sigma, overlay, overlay_kind = apply_market_variance_overlay_to_covariance(
        config, frame, asof, sigma
    )
    return OptimizerCovarianceEstimate(
        sigma=np.asarray(sigma, dtype=float),
        security_ids=cols,
        estimator=OPTIMIZER_COVARIANCE_OAS,
        covariance_object=object_name,
        spec=spec_name,
        n_obs=int(float(params.get("n_obs", finite_rows))),
        market_overlay=overlay_kind,
        overlay=overlay,
        params=dict(params),
    )


def _named_ledoit_wolf_nonlinear_optimizer_estimate(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
    mat: Array,
    cols: list[str],
    finite_rows: int,
    psd_tol: float,
) -> OptimizerCovarianceEstimate:
    """Fit named analytical nonlinear Ledoit-Wolf. Look up at call time.

    A family or covariance-object mismatch fails closed so 2020 analytical
    nonlinear shrinkage cannot silently size as 2004 linear Ledoit-Wolf,
    OAS, sample, EWMA, or DCC. The matrix is GARCH/RGARCH overlay-scaled
    like trailing Ledoit-Wolf. When T<=N the estimator stays nonlinear
    rather than switching to sample or 2004 linear shrinkage.
    """
    prefix = f"optimizer_covariance_failed:{OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR}"
    if len(cols) < 2:
        raise ValueError(f"{prefix}:fewer_than_two_securities")
    if finite_rows < NLSHRINK_MIN_OBS:
        raise ValueError(f"{prefix}:insufficient_finite_rows:{finite_rows}")
    try:
        sigma, params = ledoit_wolf_nonlinear(mat)
    except ValueError as exc:
        raise ValueError(f"{prefix}:{exc}") from exc
    sigma, _ = repair_psd(sigma, psd_tol)
    family = str(params.get("family", ""))
    if family != OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR:
        raise ValueError(f"{prefix}:unexpected_family:{family}")
    object_name = str(params.get("covariance_object", OPTIMIZER_COVARIANCE_OBJECT_TRAILING))
    spec_name = str(params.get("spec", OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR))
    if object_name != OPTIMIZER_COVARIANCE_OBJECT_TRAILING:
        raise ValueError(f"{prefix}:unexpected_covariance_object:{object_name}")
    if spec_name == OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF:
        raise ValueError(f"{prefix}:unexpected_spec:{spec_name}")
    sigma, overlay, overlay_kind = apply_market_variance_overlay_to_covariance(
        config, frame, asof, sigma
    )
    return OptimizerCovarianceEstimate(
        sigma=np.asarray(sigma, dtype=float),
        security_ids=cols,
        estimator=OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
        covariance_object=object_name,
        spec=spec_name,
        n_obs=int(float(params.get("n_obs", finite_rows))),
        market_overlay=overlay_kind,
        overlay=overlay,
        params=dict(params),
    )


def _unmeasured_optimizer_covariance(reason: str, ids: list[str]) -> OptimizerCovarianceEstimate:
    n = len(ids)
    sigma = np.empty((0, 0), dtype=float) if n == 0 else np.diag(np.ones(n) * 0.02**2)
    return OptimizerCovarianceEstimate(
        sigma=sigma,
        security_ids=list(ids),
        estimator=OPTIMIZER_COVARIANCE_HOMOSKEDASTIC_PROXY,
        covariance_object=OPTIMIZER_COVARIANCE_OBJECT_DIAGONAL_PROXY,
        spec=OPTIMIZER_COVARIANCE_SPEC_DIAGONAL_PROXY,
        n_obs=0,
        market_overlay=None,
        overlay=None,
        params={},
        unmeasured_reason=reason,
        fallback_reason=reason,
    )


def estimate_optimizer_covariance_asof(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
    ids: list[str],
    hist: pl.DataFrame,
) -> OptimizerCovarianceEstimate:
    r"""Estimate the named optimizer covariance on PIT-filtered trailing returns.

    ``optimizer.covariance=ledoit_wolf`` (default) keeps trailing Ledoit-Wolf
    2004 plus the GARCH/RGARCH overlay and stays Ledoit-Wolf when T<=N
    rather than silently switching to sample. ``dcc_gaussian``,
    ``dcc_student_t``, ``adcc``, ``ccc``, ``agdcc``, ``agdcc_full``, and
    ``ewma`` return one-step-ahead H_{t+1} and do **not** apply that overlay.
    Named ``oas`` returns trailing Chen OAS plus the overlay and must not
    silently size as Ledoit-Wolf, sample, EWMA, or DCC; when T<=N it stays
    OAS rather than switching to sample. Named ``ledoit_wolf_nonlinear``
    returns trailing analytical 2020 nonlinear shrinkage plus the overlay
    and must not silently size as 2004 linear Ledoit-Wolf, OAS, sample,
    EWMA, or DCC; when T<=N it stays nonlinear rather than switching to
    sample or 2004. Named ``sample`` returns trailing unbiased sample
    covariance plus the overlay and must not silently size as Ledoit-Wolf,
    OAS, EWMA, or DCC; when T>N it stays sample rather than switching to
    Ledoit-Wolf. Sequential one-step samples use the trailing contiguous
    complete-case window; holes are not concatenated and an incomplete
    asof row fails closed rather than dropping \(r_t\) / \(z_t\).
    OAS, named sample, named nonlinear Ledoit-Wolf, and default
    Ledoit-Wolf listwise-delete. Those named paths are distinct and must
    not substitute for each other or for Ledoit-Wolf 2004. A failed or
    short named fit fails closed rather than silently substituting
    Ledoit-Wolf. Named unrestricted AG-DCC must not silently size as
    diagonal AG-DCC. Named diagonal AG-DCC must not silently size as
    scalar ADCC. Named CCC must not silently size as Gaussian DCC or
    Ledoit-Wolf. Factor stays unwired. This does not invent high-frequency
    RV or wire factor covariance.
    """
    estimator = require_implemented_optimizer_covariance(config.optimizer.covariance)
    cols, mat = _trailing_return_matrix(hist, ids)
    finite_rows = int(np.isfinite(mat).all(axis=1).sum()) if mat.size else 0
    if estimator == OPTIMIZER_COVARIANCE_SAMPLE:
        return _named_sample_optimizer_estimate(
            config,
            frame,
            asof,
            mat,
            cols,
            finite_rows,
            config.train.psd_eigen_tol,
        )
    if estimator == OPTIMIZER_COVARIANCE_OAS:
        return _named_oas_optimizer_estimate(
            config,
            frame,
            asof,
            mat,
            cols,
            finite_rows,
            config.train.psd_eigen_tol,
        )
    if estimator == OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR:
        return _named_ledoit_wolf_nonlinear_optimizer_estimate(
            config,
            frame,
            asof,
            mat,
            cols,
            finite_rows,
            config.train.psd_eigen_tol,
        )
    if estimator == OPTIMIZER_COVARIANCE_EWMA:
        return _named_ewma_optimizer_estimate(
            mat,
            cols,
            finite_rows,
            config.train.psd_eigen_tol,
            float(config.features.ewma_lambda),
        )
    if estimator in IMPLEMENTED_OPTIMIZER_DCC_FAMILIES:
        return _named_dcc_optimizer_estimate(
            estimator,
            mat,
            cols,
            finite_rows,
            config.train.psd_eigen_tol,
        )

    if "ret_1" not in hist.columns:
        return _unmeasured_optimizer_covariance("no_ret_1", ids)
    if len(cols) < 2:
        return _unmeasured_optimizer_covariance("fewer_than_two_securities", ids)
    if finite_rows < 2:
        return _unmeasured_optimizer_covariance("insufficient_finite_rows", ids)
    try:
        sigma, params = ledoit_wolf(mat)
    except ValueError:
        log.warning("covariance_fallback", reason="estimation_failed", n_ids=len(ids))
        return _unmeasured_optimizer_covariance("estimation_failed", ids)
    family = str(params.get("family", ""))
    prefix = f"optimizer_covariance_failed:{OPTIMIZER_COVARIANCE_LEDOIT_WOLF}"
    if family != OPTIMIZER_COVARIANCE_LEDOIT_WOLF:
        raise ValueError(f"{prefix}:unexpected_family:{family}")
    object_name = str(params.get("covariance_object", OPTIMIZER_COVARIANCE_OBJECT_TRAILING))
    spec_name = str(params.get("spec", OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF))
    if object_name != OPTIMIZER_COVARIANCE_OBJECT_TRAILING:
        raise ValueError(f"{prefix}:unexpected_covariance_object:{object_name}")
    sigma, _ = repair_psd(sigma, config.train.psd_eigen_tol)
    sigma, overlay, overlay_kind = apply_market_variance_overlay_to_covariance(
        config, frame, asof, sigma
    )
    return OptimizerCovarianceEstimate(
        sigma=np.asarray(sigma, dtype=float),
        security_ids=cols,
        estimator=OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
        covariance_object=object_name,
        spec=spec_name,
        n_obs=int(float(params.get("n_obs", finite_rows))),
        market_overlay=overlay_kind,
        overlay=overlay,
        params=dict(params),
    )


def market_risk_overlay_asof(
    config: AppConfig,
    frame: pl.DataFrame,
    asof: datetime,
) -> tuple[float | None, str | None]:
    """Causal date-level market sigma for paper/backtest ``check_order``."""
    overlay, source = resolve_market_variance_overlay_asof(config, frame, asof)
    if overlay is None:
        return None, None
    return float(overlay.sigma), source


def _market_overlay_note(overlay_kind: str) -> str:
    if overlay_kind == MARKET_RISK_OVERLAY_REALIZED_GARCH:
        return "realized_garch_market_cross_section"
    return "garch_market_cross_section"


def _market_overlay_diagnostics(
    overlay: GarchMarketForecast | RealizedGarchMarketForecast,
    overlay_kind: str,
) -> dict[str, float | str]:
    diagnostics: dict[str, float | str] = {
        "garch_market_sigma": float(overlay.sigma),
        "garch_market_variance": float(overlay.variance),
        "garch_cumulative_variance": float(overlay.cumulative_variance),
        "garch_horizon": float(overlay.horizon),
        "garch_n_obs": float(overlay.n_obs),
        "garch_series_scope": overlay.series_scope,
        "garch_fit_status": overlay.fit_status,
        "market_risk_overlay": overlay_kind,
    }
    if overlay_kind == MARKET_RISK_OVERLAY_REALIZED_GARCH:
        diagnostics["realized_measure"] = REALIZED_GARCH_MEASURE
        diagnostics["intraday_realized_variance"] = "false"
        diagnostics["garch_variance_family"] = REALIZED_GARCH_FAMILY
    return diagnostics


__all__ = [
    "OptimizerCovarianceEstimate",
    "estimate_optimizer_covariance_asof",
    "market_risk_overlay_asof",
]
