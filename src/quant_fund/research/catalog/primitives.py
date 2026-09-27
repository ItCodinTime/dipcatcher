"""Shared scalar and IC-pack helpers for catalog honesty checks.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

import math
from typing import Any


def _finite_scalar(value: object) -> bool:
    """Return True iff *value* is a finite real number (bools rejected)."""
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number)


def _ic_pack_honesty_errors(
    blob: dict[str, Any],
    *,
    mean_ic_key: str,
    rank_ic_key: str | None,
    t_key: str,
    p_key: str,
    n_key: str | None,
) -> list[str]:
    """Shared IC-pack soft-verify: mean/rank/t finite; p∈[0,1]; n≥0."""
    errs: list[str] = []
    for key in (mean_ic_key, rank_ic_key, t_key):
        if key is None or key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    if p_key in blob:
        try:
            p = float(blob.get(p_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{p_key}_non_numeric")
        else:
            if p == p and abs(p) != float("inf") and not (0.0 <= p <= 1.0):
                errs.append(f"{p_key}_out_of_unit_interval")
            elif p == p and abs(p) == float("inf"):
                errs.append(f"{p_key}_non_finite_fail_closed")
    if n_key is not None and n_key in blob:
        try:
            n = float(blob.get(n_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{n_key}_non_numeric")
        else:
            if n == n and abs(n) != float("inf") and n < 0.0:
                errs.append(f"{n_key}_negative")
            elif n == n and abs(n) == float("inf"):
                errs.append(f"{n_key}_non_finite")
    return errs


def _finite_pair(blob: object, key_a: str, key_b: str) -> tuple[float, float] | None:
    """Return finite ``(a, b)`` for two blob keys, or None to skip soft-verify.

    Missing / non-numeric / NaN / ±inf on either side → None (never invent a
    comparison from absent evidence).
    """
    if not isinstance(blob, dict):
        return None
    a = blob.get(key_a)
    b = blob.get(key_b)
    if not _finite_scalar(a) or not _finite_scalar(b):
        return None
    return float(a), float(b)  # type: ignore[arg-type]
