"""Weight constraints and fail-closed covariance validation.

Every engine in :mod:`quant_fund.research.allocation.engines` returns raw
weights that are passed through :func:`apply_constraints` before use. The
constraint layer is deterministic and *feasible by construction*: clipping
and gross rescaling can only shrink exposures, so the post-constraint book
always satisfies the bounds (verified again in evaluation as a sanity
check).

Covariance validation is fail closed: a non-square, asymmetric, non-finite,
non-PSD, or zero-variance matrix raises :class:`DegenerateCovarianceError`
rather than silently projecting or jittering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

#: Relative eigenvalue tolerance for the PSD check. Eigenvalues no more
#: negative than ``PSD_TOL * scale`` are accepted (floating-point noise on
#: nearly singular matrices); anything more negative fails closed.
PSD_TOL = 1e-10

#: Asymmetry tolerance for the covariance check (relative to max |Sigma|).
SYM_TOL = 1e-8

#: Zero-variance floors. ``np.cov`` on a constant column returns float
#: residue (~1e-37), not exact zero — so "zero variance" must be checked
#: numerically: an asset is degenerate when its variance is below an
#: absolute floor or below a tiny fraction of the largest variance (the
#: conditioning of such a book is unusable for allocation regardless).
VAR_ABS_FLOOR = 1e-30
VAR_REL_FLOOR = 1e-14


class DegenerateCovarianceError(ValueError):
    """Raised when a covariance matrix is unusable for allocation.

    Covers non-square, asymmetric, non-finite, non-positive-semidefinite
    matrices and zero-variance (zero/negative diagonal) assets. Engines
    never repair such input — they refuse, so a bad estimator upstream can
    never produce plausible-looking weights.
    """


class InfeasibleConstraintsError(ValueError):
    """Raised when the constraint set itself is malformed."""


@dataclass(frozen=True)
class AllocationConstraints:
    """Book constraints applied to every engine's raw output.

    - ``long_only``: clip negative weights to zero.
    - ``leverage_cap``: bound on gross exposure ``sum |w_i|``; scaled down
      (never up) when an engine's raw book exceeds it. A cap below 1 leaves
      the remainder as unallocated cash.
    - ``min_weight``: inclusion floor. Long-only positions strictly below
      this size are dropped to zero (never lifted to the floor — lifting
      could violate the leverage cap). Final positions are therefore either
      0 or ``>= min_weight``.
    - ``max_weight``: per-asset cap, applied before gross rescaling.
    """

    long_only: bool = True
    leverage_cap: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("leverage_cap", "max_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise InfeasibleConstraintsError(f"{name} must be positive and finite")
        if not np.isfinite(self.min_weight) or self.min_weight < 0.0:
            raise InfeasibleConstraintsError("min_weight must be nonnegative and finite")
        if self.min_weight > self.max_weight:
            raise InfeasibleConstraintsError("min_weight must not exceed max_weight")
        if self.min_weight > self.leverage_cap:
            raise InfeasibleConstraintsError(
                "min_weight must not exceed leverage_cap "
                "(a single minimum-size position must fit in the book)"
            )


def validate_covariance(cov: Array) -> Array:
    """Fail-closed covariance validation.

    Returns the matrix as a float64 array when it is finite, square,
    symmetric within ``SYM_TOL``, positive semidefinite within ``PSD_TOL``,
    and has a strictly positive diagonal (no zero-variance assets). Raises
    :class:`DegenerateCovarianceError` otherwise.
    """
    m = np.asarray(cov, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 1:
        raise DegenerateCovarianceError("covariance must be a nonempty square matrix")
    if not np.all(np.isfinite(m)):
        raise DegenerateCovarianceError("covariance must be finite")
    scale = max(float(np.abs(m).max()), 1e-30)
    if float(np.abs(m - m.T).max()) > SYM_TOL * scale:
        raise DegenerateCovarianceError("covariance must be symmetric")
    diag = np.diag(m)
    var_floor = max(VAR_ABS_FLOOR, VAR_REL_FLOOR * float(diag.max()))
    if np.any(diag <= var_floor):
        raise DegenerateCovarianceError("covariance has a zero-variance asset")
    if float(np.linalg.eigvalsh(m).min()) < -PSD_TOL * scale:
        raise DegenerateCovarianceError("covariance must be positive semidefinite")
    return m


def apply_constraints(weights: Array, constraints: AllocationConstraints) -> Array:
    """Project raw engine weights onto the constraint set.

    Order of operations — chosen so every step preserves feasibility:

    1. long-only clip (negatives -> 0),
    2. per-asset cap ``min(w_i, max_weight)`` (also clips shorts to
       ``-max_weight`` when ``long_only=False``),
    3. gross rescale down to ``leverage_cap`` when ``sum |w|`` exceeds it,
    4. inclusion floor: positions with ``|w_i| < min_weight`` drop to 0.

    Steps 3 and 4 only shrink positions, so the result is feasible by
    construction: ``|w_i| <= max_weight``, ``sum |w_i| <= leverage_cap``,
    and every nonzero position has ``|w_i| >= min_weight``.
    """
    w = np.asarray(weights, dtype=float).reshape(-1).copy()
    if w.size == 0 or not np.all(np.isfinite(w)):
        raise ValueError("weights must be a nonempty finite vector")
    if constraints.long_only:
        w = np.asarray(np.clip(w, 0.0, None), dtype=float)
    w = np.asarray(np.clip(w, -constraints.max_weight, constraints.max_weight), dtype=float)
    gross = float(np.abs(w).sum())
    if gross > constraints.leverage_cap:
        w *= constraints.leverage_cap / gross
    floor = constraints.min_weight
    if floor > 0.0:
        w[(w != 0.0) & (np.abs(w) < floor)] = 0.0
    return w


def weights_satisfy(
    weights: Array,
    constraints: AllocationConstraints,
    *,
    tol: float = 1e-12,
) -> list[str]:
    """Post-hoc feasibility audit. Returns a list of violations (empty = ok).

    Used by the evaluation loop as a fail-closed sanity check on the
    constraint layer itself.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    violations: list[str] = []
    if w.size == 0 or not np.all(np.isfinite(w)):
        return ["nonfinite_or_empty_weights"]
    if constraints.long_only and float(w.min()) < -tol:
        violations.append("long_only")
    gross = float(np.abs(w).sum())
    if gross > constraints.leverage_cap * (1.0 + tol):
        violations.append("leverage_cap")
    if float(np.abs(w).max()) > constraints.max_weight * (1.0 + tol):
        violations.append("max_weight")
    floor = constraints.min_weight
    if floor > 0.0:
        nonzero = np.abs(w[w != 0.0])
        if nonzero.size and float(nonzero.min()) < floor * (1.0 - 1e-9):
            violations.append("min_weight")
    return violations
