"""One CS-promotion rule for every race receipt.

A challenger clears only when that same name has positive mean IC, wins
the pairwise Diebold-Mariano test of -IC against the benchmark, and is
rejected by Romano-Wolf StepM, while White's reality check and Hansen's
SPA both reject the no-skill null. A model that is merely less negative
than the benchmark does not clear. A pooled pass that dies on the frozen
holdout does not clear. This module does not write a champion alias and
does not move blend_weight.
"""

from __future__ import annotations

import math

ALPHA = 0.05


def positive_mean_ic(mean_ic: float) -> bool:
    """Absolute skill. A less-negative IC is not enough."""
    try:
        value = float(mean_ic)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def clears_cs_promotion(
    *,
    name: str,
    mean_ic: float,
    dm_preferred: str,
    dm_p: float,
    stepm_rejected: list[str],
    rc_p: float,
    spa_p: float,
    alpha: float = ALPHA,
) -> bool:
    """True only when ``name`` itself clears every gate."""
    try:
        dm_value = float(dm_p)
        rc_value = float(rc_p)
        spa_value = float(spa_p)
        level = float(alpha)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in (dm_value, rc_value, spa_value, level)):
        return False
    return bool(
        positive_mean_ic(mean_ic)
        and dm_preferred == name
        and dm_value < level
        and name in stepm_rejected
        and rc_value < level
        and spa_value < level
    )


def names_clearing_both(selection: list[str], holdout: list[str]) -> list[str]:
    """Challengers that clear the CS rule on the selection window and the frozen holdout."""
    hold = set(holdout)
    return [name for name in selection if name in hold]


def clears_book_overlay(
    *,
    excess_means: list[float],
    rc_p: float,
    spa_p: float,
    alpha: float = ALPHA,
) -> bool:
    """Holdout book claim. Every excess mean must be positive, and RC and SPA both clear.

    Does not move blend_weight. A reality-check p-value alone is not enough.
    """
    try:
        rc_value = float(rc_p)
        spa_value = float(spa_p)
        level = float(alpha)
    except (TypeError, ValueError):
        return False
    if not excess_means or not all(math.isfinite(v) for v in (rc_value, spa_value, level)):
        return False
    return bool(
        all(positive_mean_ic(mean) for mean in excess_means)
        and rc_value < level
        and spa_value < level
    )
