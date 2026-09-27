"""Wasserstein distributionally robust bounds on a scalar outcome law.

The mean bound is exact for every reference law. The mean-to-scale ratio
bound is the Gelbrich moment-disk minimum: tight for a univariate Gaussian
reference, and only a lower bound (sometimes vacuous) for any other law.
Neither bound is a confidence interval for an unknown true distribution.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.robustness.analytic import (
    gelbrich_worst_case_ratio,
    lipschitz_worst_case,
    plug_in_moments,
    wasserstein_worst_case_mean,
)

FloatArray = NDArray[np.float64]


def _ratio_status(kind: str, reference: str) -> str:
    if kind == "unbounded_below":
        if reference == "gaussian":
            return "proven_unbounded"
        return "vacuous_outer_bound"
    if reference == "gaussian":
        return "proven_tight"
    return "outer_bound"


def distributional_bounds(
    *,
    mean: float,
    scale: float | None,
    radius: float,
    reference: str,
    moment_source: str,
) -> dict[str, Any]:
    """Build the distributional block of a robustness scorecard.

    ``reference`` is ``gaussian`` or ``empirical``. ``moment_source`` records
    whether the moments were supplied as a population (``population``) or
    computed from a sample (``plug_in_sample``).
    """
    if reference not in {"gaussian", "empirical"}:
        raise ValueError("reference must be 'gaussian' or 'empirical'")
    if moment_source not in {"population", "plug_in_sample"}:
        raise ValueError("moment_source must be 'population' or 'plug_in_sample'")
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and non-negative")
    worst_mean = wasserstein_worst_case_mean(mean, radius)
    if scale is None or scale <= 0.0:
        ratio_value: float | None = None
        ratio_status = "undefined"
        ratio_case = "undefined_scale"
    else:
        ratio_value, ratio_case = gelbrich_worst_case_ratio(mean, scale, radius)
        ratio_status = _ratio_status(ratio_case, reference)
        if ratio_status == "vacuous_outer_bound":
            ratio_value = None
    return {
        "wasserstein_radius": float(radius),
        "reference": reference,
        "moment_source": moment_source,
        "reference_mean": float(mean),
        "reference_scale": None if scale is None else float(scale),
        "worst_case_mean": {
            "value": worst_mean,
            "wasserstein_order": 1,
            "holds_for_every_order_at_least": 1,
            "status": "proven",
            "formula": "translation_attains_mean_shift",
            "scope": "exact_for_any_reference_law",
        },
        "worst_case_ratio": {
            "value": ratio_value,
            "wasserstein_order": 2,
            "object": "mean_over_scale",
            "status": ratio_status,
            "case": ratio_case,
            "formula": "gelbrich1990_moment_disk",
            "scope": (
                "tight_for_univariate_gaussian"
                if reference == "gaussian"
                else "lower_bound_via_outer_moment_disk"
            ),
        },
        # Radius of a pure translation that makes the worst-case mean non-positive.
        # Exact, because the worst-case mean is mean - radius.
        "radius_to_nonpositive_mean": {
            "value": max(0.0, float(mean)),
            "status": "proven",
            "formula": "mean_minus_radius",
        },
    }


def bounds_from_outcomes(
    outcomes: FloatArray,
    radius: float,
    *,
    reference: str = "empirical",
) -> dict[str, Any]:
    """Plug-in moments of ``outcomes``, then :func:`distributional_bounds`."""
    mean, scale = plug_in_moments(np.asarray(outcomes, dtype=float))
    return distributional_bounds(
        mean=mean,
        scale=scale,
        radius=radius,
        reference=reference,
        moment_source="plug_in_sample",
    )


def lipschitz_shift(value: float, lipschitz: float, radius: float) -> dict[str, float | str]:
    """Lower and upper Lipschitz shifts, labeled as the KR / Esfahani–Kuhn bound."""
    return {
        "lower": lipschitz_worst_case(value, lipschitz, radius, lower=True),
        "upper": lipschitz_worst_case(value, lipschitz, radius, lower=False),
        "status": "proven",
        "formula": "kantorovich_rubinstein_lipschitz",
    }
