"""Northset and candle information-coefficient pack honesty checks.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from .primitives import _ic_pack_honesty_errors


def ofi_p_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ofi_p_ic / ofi_t_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. ≠ ofi_mean_ic alias path may coexist.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("ofi_p_ic", "ofi_t_ic"):
        if key not in blob:
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
    return errs


def microprice_p_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: microprice_p_ic / microprice_t_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. Companion aliases of microprice_minus_mid_bps IC.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("microprice_p_ic", "microprice_t_ic"):
        if key not in blob:
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
    return errs


def clv_p_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: clv_p_ic / clv_t_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. Companion aliases of close_location_value IC
    (≠ microprice_p_ic / microprice_t_ic). Research diagnostic only; never live Sharpe.
    Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("clv_p_ic", "clv_t_ic"):
        if key not in blob:
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
    return errs


def northset_queue_imbalance_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify queue_imbalance_* IC companions when present.

    - ``queue_imbalance_mean_ic`` / ``queue_imbalance_mean_rank_ic`` / ``_t_ic`` finite
    - ``queue_imbalance_p_ic`` ∈ [0, 1] when finite
    - ``queue_imbalance_n_dates`` ≥ 0 when finite
    Never equate to ``queue_imbalance_mean`` (mean ≠ IC) or ofi_* IC.
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "queue_imbalance_mean_ic",
        "queue_imbalance_mean_rank_ic",
        "queue_imbalance_t_ic",
    ):
        if key not in blob:
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
    if "queue_imbalance_p_ic" in blob:
        try:
            p = float(blob.get("queue_imbalance_p_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("queue_imbalance_p_ic_non_numeric")
        else:
            if p == p and abs(p) != float("inf") and not (0.0 <= p <= 1.0):
                errs.append("queue_imbalance_p_ic_out_of_unit_interval")
            elif p == p and abs(p) == float("inf"):
                errs.append("queue_imbalance_p_ic_non_finite_fail_closed")
    if "queue_imbalance_n_dates" in blob:
        try:
            n = float(blob.get("queue_imbalance_n_dates"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("queue_imbalance_n_dates_non_numeric")
        else:
            if n == n and abs(n) != float("inf") and n < 0.0:
                errs.append("queue_imbalance_n_dates_negative")
            elif n == n and abs(n) == float("inf"):
                errs.append("queue_imbalance_n_dates_non_finite")
    return errs


def northset_vpin_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify vpin_* IC pack companions beyond finite-only p/t.

    - ``vpin_mean_ic`` / ``vpin_mean_rank_ic`` / ``vpin_t_ic`` finite
    - ``vpin_p_ic`` ∈ [0, 1] when finite (stricter than finite-only)
    - ``vpin_n_dates`` ≥ 0 when finite
    Never equate to ``vpin_mean``. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("vpin_mean_ic", "vpin_mean_rank_ic", "vpin_t_ic"):
        if key not in blob:
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
    if "vpin_p_ic" in blob:
        try:
            p = float(blob.get("vpin_p_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("vpin_p_ic_non_numeric")
        else:
            if p == p and abs(p) != float("inf") and not (0.0 <= p <= 1.0):
                errs.append("vpin_p_ic_out_of_unit_interval")
            elif p == p and abs(p) == float("inf"):
                errs.append("vpin_p_ic_non_finite_fail_closed")
    if "vpin_n_dates" in blob:
        try:
            n = float(blob.get("vpin_n_dates"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("vpin_n_dates_non_numeric")
        else:
            if n == n and abs(n) != float("inf") and n < 0.0:
                errs.append("vpin_n_dates_negative")
            elif n == n and abs(n) == float("inf"):
                errs.append("vpin_n_dates_non_finite")
    return errs


def northset_ofi_lag_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ofi_lag_* IC companions when present.

    - ``ofi_lag_mean_ic`` / ``ofi_lag_mean_rank_ic`` / ``ofi_lag_t_ic`` finite
    - ``ofi_lag_p_ic`` ∈ [0, 1] when finite
    - ``ofi_lag_n_dates`` ≥ 0 when finite
    Never equate to ``ofi_mean_ic`` / ``ofi_p_ic`` or ``ofi_lag1_corr``.
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("ofi_lag_mean_ic", "ofi_lag_mean_rank_ic", "ofi_lag_t_ic"):
        if key not in blob:
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
    if "ofi_lag_p_ic" in blob:
        try:
            p = float(blob.get("ofi_lag_p_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("ofi_lag_p_ic_non_numeric")
        else:
            if p == p and abs(p) != float("inf") and not (0.0 <= p <= 1.0):
                errs.append("ofi_lag_p_ic_out_of_unit_interval")
            elif p == p and abs(p) == float("inf"):
                errs.append("ofi_lag_p_ic_non_finite_fail_closed")
    if "ofi_lag_n_dates" in blob:
        try:
            n = float(blob.get("ofi_lag_n_dates"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("ofi_lag_n_dates_non_numeric")
        else:
            if n == n and abs(n) != float("inf") and n < 0.0:
                errs.append("ofi_lag_n_dates_negative")
            elif n == n and abs(n) == float("inf"):
                errs.append("ofi_lag_n_dates_non_finite")
    return errs


def ofi_mean_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ofi_mean_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. ≠ imbalance_top_mean_ic.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "ofi_mean_ic" not in blob:
        return []
    try:
        x = float(blob.get("ofi_mean_ic"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["ofi_mean_ic_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["ofi_mean_ic_non_finite_fail_closed"]
    return []


def northset_all_mean_ic_finite_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``*_mean_ic`` is finite when present (signed OK)."""
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.endswith("_mean_ic"):
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


def northset_volume_over_range_ic_packs_honesty_errors(blob: object) -> list[str]:
    """Soft-verify volume_over_range(_abs)_* IC packs."""
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for stem in ("volume_over_range", "volume_over_range_abs"):
        errs.extend(
            _ic_pack_honesty_errors(
                blob,
                mean_ic_key=f"{stem}_mean_ic",
                rank_ic_key=f"{stem}_mean_rank_ic",
                t_key=f"{stem}_t_ic",
                p_key=f"{stem}_p_ic",
                n_key=f"{stem}_n_dates",
            )
        )
    return errs


def northset_amihud_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify amihud_* IC pack; never equate to amihud_mean."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="amihud_mean_ic",
        rank_ic_key="amihud_mean_rank_ic",
        t_key="amihud_t_ic",
        p_key="amihud_p_ic",
        n_key="amihud_n_dates",
    ) + _ic_pack_honesty_errors(
        blob,
        mean_ic_key="amihud_abs_mean_ic",
        rank_ic_key="amihud_abs_mean_rank_ic",
        t_key="amihud_abs_t_ic",
        p_key="amihud_abs_p_ic",
        n_key="amihud_abs_n_dates",
    )


def northset_imbalance_depth_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify imbalance_depth_* IC pack; never equate to mean_depth_imbalance."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="imbalance_depth_mean_ic",
        rank_ic_key="imbalance_depth_mean_rank_ic",
        t_key="imbalance_depth_t_ic",
        p_key="imbalance_depth_p_ic",
        n_key="imbalance_depth_n_dates",
    )


def northset_bid_log_size_slope_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify bid_log_size_slope_* IC pack."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="bid_log_size_slope_mean_ic",
        rank_ic_key="bid_log_size_slope_mean_rank_ic",
        t_key="bid_log_size_slope_t_ic",
        p_key="bid_log_size_slope_p_ic",
        n_key="bid_log_size_slope_n_dates",
    )


def northset_wick_skew_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify wick_skew_* IC pack on northset (finite mean/rank/t; p∈[0,1]; n≥0)."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="wick_skew_mean_ic",
        rank_ic_key="wick_skew_mean_rank_ic",
        t_key="wick_skew_t_ic",
        p_key="wick_skew_p_ic",
        n_key="wick_skew_n_dates",
    )


def northset_imbalance_top_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify imbalance_top_* IC pack (p∈[0,1], n_dates≥0, ICs finite).

    Never equate to ``mean_imbalance_top``. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="imbalance_top_mean_ic",
        rank_ic_key="imbalance_top_mean_rank_ic",
        t_key="imbalance_top_t_ic",
        p_key="imbalance_top_p_ic",
        n_key="imbalance_top_n_dates",
    )


def northset_clv_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify clv_* IC pack: p∈[0,1], t finite. Never equate to CLV mean."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="clv_mean_ic",
        rank_ic_key="clv_mean_rank_ic",
        t_key="clv_t_ic",
        p_key="clv_p_ic",
        n_key="clv_n_dates",
    )


def northset_microprice_bps_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify microprice_minus_mid_bps_* IC pack; never equate to microprice_p_ic alias alone."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="microprice_minus_mid_bps_mean_ic",
        rank_ic_key="microprice_minus_mid_bps_mean_rank_ic",
        t_key="microprice_minus_mid_bps_t_ic",
        p_key="microprice_minus_mid_bps_p_ic",
        n_key="microprice_minus_mid_bps_n_dates",
    )


def imbalance_top_mean_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: imbalance_top_mean_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. Companion to H22 discovery path.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "imbalance_top_mean_ic" not in blob:
        return []
    try:
        x = float(blob.get("imbalance_top_mean_ic"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["imbalance_top_mean_ic_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["imbalance_top_mean_ic_non_finite_fail_closed"]
    return []


__all__ = [
    "clv_p_ic_honesty_errors",
    "imbalance_top_mean_ic_honesty_errors",
    "microprice_p_ic_honesty_errors",
    "northset_all_mean_ic_finite_honesty_errors",
    "northset_amihud_ic_pack_honesty_errors",
    "northset_bid_log_size_slope_ic_pack_honesty_errors",
    "northset_clv_ic_pack_honesty_errors",
    "northset_imbalance_depth_ic_pack_honesty_errors",
    "northset_imbalance_top_ic_pack_honesty_errors",
    "northset_microprice_bps_ic_pack_honesty_errors",
    "northset_ofi_lag_ic_honesty_errors",
    "northset_queue_imbalance_ic_honesty_errors",
    "northset_volume_over_range_ic_packs_honesty_errors",
    "northset_vpin_ic_pack_honesty_errors",
    "northset_wick_skew_ic_pack_honesty_errors",
    "ofi_mean_ic_honesty_errors",
    "ofi_p_ic_honesty_errors",
]
