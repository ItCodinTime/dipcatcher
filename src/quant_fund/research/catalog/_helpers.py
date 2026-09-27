"""Shared private helpers and tunable constants for the research catalog."""

from __future__ import annotations

import math


def _iter_mapping_keys(obj: object) -> list[str]:
    """Collect nested mapping keys (dicts only; list elements walked)."""
    keys: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.append(str(k))
            keys.extend(_iter_mapping_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            keys.extend(_iter_mapping_keys(item))
    return keys


def _finite_scalar(value: object) -> bool:
    """Return True iff *value* is a finite real number (bools rejected)."""
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number)


_DATA_SNOOPING_P_KEYS = ("reality_check_p", "spa_p_lower", "spa_p_consistent", "spa_p_upper")


_JP_CV_SCOPE_FAMILIES = ("jackknife_plus", "cv_plus")


def _ic_pack_honesty_errors(
    blob: dict,
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


_NORTHSET_SWEEP_EVIDENCE_SCOPE_ALLOWED = frozenset(
    {"synthetic", "empirical_adjusted", "fixture_raw_unadjusted"}
)


_NORTHSET_IMPACT_ESTIMATOR_SCOPE_ALLOWED = frozenset({"per_security_equal_weight"})


_NORTHSET_YANG_ZHANG_QLIKE_SCOPE_ALLOWED = frozenset({"per_security_expanding_oos"})


_NORTHSET_VPIN_METHOD_ALLOWED = frozenset(
    {"count_window_bulk_ofi_proxy", "volume_bucket_bulk_ofi_proxy"}
)


_NORTHSET_CORWIN_SCHULTZ_PAIR_SCOPE_ALLOWED = frozenset({"prior_and_current_bar"})


_NORTHSET_OVERNIGHT_GAP_METHOD_ALLOWED = frozenset({"event_close_to_next_open"})


_NORTHSET_TWO_WAY_INFERENCE_INDEX_ALLOWED = frozenset({"event_rows_not_calendar_zeros"})


_NORTHSET_PRICE_BASIS_ALLOWED = frozenset({"split_adjusted", "raw_fixture_opt_out"})


_NORTHSET_RETURN_BASIS_ALLOWED = frozenset(
    {"total_return", "split_adjusted", "raw_fixture_opt_out"}
)


_KYLE_RESIDUAL_SPEARMAN_KEYS = (
    "residual_ofi_ex_depth_fwd_delta_mid_mean_spearman",
    "residual_depth_ex_ofi_fwd_delta_mid_mean_spearman",
    "residual_depth_ex_ofi_fwd_ret_1_mean_spearman",
    "residual_ofi_ex_depth_fwd_ret_1_mean_spearman",
    "residual_depth_ex_ofi_fwd_ret_2_mean_spearman",
    "residual_ofi_ex_depth_fwd_ret_2_mean_spearman",
    "residual_depth_ex_ofi_fwd_ret_3_mean_spearman",
    "residual_ofi_ex_depth_fwd_ret_3_mean_spearman",
)


_KYLE_FORBIDDEN_TOKENS = ("sharpe", "sortino", "calmar", "pnl", "nav", "live_pnl_claim")


def _kyle_ofi_blob(blob: object) -> dict | None:
    """Return nested kyle_ofi dict from a northset family blob, or the blob itself."""
    if not isinstance(blob, dict):
        return None
    if blob.get("family") == "kyle_ofi":
        return blob
    nest = blob.get("kyle_ofi")
    return nest if isinstance(nest, dict) else None


def _kyle_ofi_has_nest_diagnostic_marker(kyle: dict) -> bool:
    """True when any kyle nest diagnostic stamp is present/finite."""
    for key in _KYLE_RESIDUAL_SPEARMAN_KEYS:
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x == x:
            return True
    for key in (
        "kyle_lambda_depth_p50",
        "kyle_lambda_ofi_p50",
        "kyle_lambda_ofi_depth_spearman",
        "kyle_lambda_ofi_depth_pearson",
        "kyle_lambda_date_series_n_depth",
        "kyle_lambda_date_series_n_ofi",
        "join_coverage",
        "kyle_lambda_depth_mean",
        "kyle_lambda_ofi_mean",
    ):
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x == x:
            return True
    if "kyle_lambda_dispersion_window" in kyle:
        return True
    for key in ("book_dgp", "dgp", "data_source", "label", "book_source"):
        v = kyle.get(key)
        if isinstance(v, str) and v.strip():
            return True
    return False


_KYLE_OFI_IC_METHOD_ALLOWED = frozenset({"date_level_spearman_hac"})


_NORTHSET_SPREAD_REL_TOL = 1e-9


_NORTHSET_SPREAD_ABS_TOL = 1e-12


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


_CANDLE_FEATURE_IC_METHOD_ALLOWED = frozenset({"date_level_spearman_hac"})


def _candle_has_feature_ic_marker(blob: dict) -> bool:
    """True if any FEATURE_COLS-style ic_<col> spearman (not meta suffix) is present."""
    for key in blob:
        if not isinstance(key, str) or not key.startswith("ic_"):
            continue
        if key == "ic_method":
            continue
        if key.endswith(("_t", "_p", "_n_dates", "_pearson")):
            continue
        return True
    return False
