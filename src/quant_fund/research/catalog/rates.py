"""Unit-interval and finite-rate catch-all honesty checks.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations


def northset_all_rate_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every top-level ``*_rate`` ∈ [0, 1] when finite.

    Covers candle-pattern rates (doji/hammer/…) and sweep_*_rate companions
    alongside identity rates. Skip kyle_*/METRICS_*. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_rate"):
            continue
        if key.startswith("kyle_") or "METRICS_" in key:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_all_t_ic_finite_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_t_ic`` is finite when present (signed OK).

    Catch-all companion to ``*_p_ic`` unit-interval. Skip best_feature_* /
    kyle_* / METRICS_*. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_t_ic"):
            continue
        if key.startswith(("best_feature", "kyle_")) or "METRICS_" in key:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
    return errs


def northset_all_mean_rank_ic_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_mean_rank_ic`` ∈ [-1, 1] when finite."""
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_mean_rank_ic"):
            continue
        if key.startswith(("best_feature", "kyle_")) or "METRICS_" in key:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
        elif not (-1.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_all_finite_rate_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_finite_rate`` ∈ [0, 1] when finite.

    Complements dedicated shape/structure helpers. Skip kyle_*/METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_finite_rate"):
            continue
        if key.startswith("kyle_") or "METRICS_" in key:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_all_floor_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_floor`` ∈ [0, 1] when finite and not None.

    Complements dedicated floors honesty. Skip kyle_*/METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_floor"):
            continue
        if key.startswith("kyle_") or "METRICS_" in key:
            continue
        if raw is None:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_all_p_ic_unit_interval_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_p_ic`` key ∈ [0, 1] when finite.

    Catch-all residual for IC packs not yet given a dedicated helper.
    Never invent missing keys. Research diagnostic only; never live Sharpe.
    Off kyle_ofi / METRICS_* / best_feature_* (Sergeant lane).
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_p_ic"):
            continue
        if key.startswith("best_feature"):
            continue  # Sergeant best_feature lane
        if raw is None:
            continue
        try:
            p = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if p != p:
            continue
        if abs(p) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
        elif not (0.0 <= p <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_all_n_dates_nonneg_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_n_dates`` companion ≥ 0 when finite.

    Pairs with ``*_p_ic`` catch-all. Skip best_feature_*. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_n_dates"):
            continue
        if key.startswith("best_feature"):
            continue
        try:
            n = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if n != n:
            continue
        if abs(n) == float("inf"):
            errs.append(f"{key}_non_finite")
        elif n < 0.0:
            errs.append(f"{key}_negative")
    return errs


def northset_all_share_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every top-level ``*_share`` ∈ [0, 1] when finite.

    Covers overnight_share, tob_*_share, sweep_*_reclaim/follow_share, etc.
    Skip kyle_*/METRICS_*. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_share"):
            continue
        if key.startswith("kyle_") or "METRICS_" in key:
            continue
        if raw is None:
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_all_fraction_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: every top-level ``*_fraction`` key ∈ [0, 1] when finite.

    Covers sweep fold-positive fractions and any future ``*_fraction`` rates
    without per-key helpers. Nested dicts are ignored (top-level only).
    """
    if not isinstance(blob, dict):
        return []
    errors: list[str] = []
    for key, raw in blob.items():
        if not str(key).endswith("_fraction"):
            continue
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{key}_non_numeric")
            continue
        if val != val:  # NaN ok
            continue
        if not (0.0 <= val <= 1.0):
            errors.append(f"{key}_out_of_unit_interval")
    return errors


__all__ = [
    "northset_all_finite_rate_unit_honesty_errors",
    "northset_all_floor_unit_honesty_errors",
    "northset_all_fraction_unit_honesty_errors",
    "northset_all_mean_rank_ic_unit_honesty_errors",
    "northset_all_n_dates_nonneg_honesty_errors",
    "northset_all_p_ic_unit_interval_honesty_errors",
    "northset_all_rate_unit_honesty_errors",
    "northset_all_share_unit_honesty_errors",
    "northset_all_t_ic_finite_honesty_errors",
]
