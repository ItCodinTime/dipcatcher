"""Northset book, spread, shape, and provenance receipt honesty checks.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

import math

from .primitives import _finite_pair, _finite_scalar


def mean_microprice_weight_balance_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_microprice_weight_balance ∈ [0, 1] when finite.

    Applies to northset and candle_order_book family receipts. NaN / absent → skip.
    Research diagnostic only; never live Sharpe / promotion.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_microprice_weight_balance")
    if val is None:
        return []
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        # Bools/strings are not probabilities; never coerce them into range.
        return ["mean_microprice_weight_balance_non_numeric"]
    x = float(val)
    if x != x:  # NaN
        return []
    if not (0.0 <= x <= 1.0):
        return ["mean_microprice_weight_balance_out_of_unit_interval"]
    return []


def join_coverage_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: join_coverage ∈ (0, 1] when finite, or [min_join, 1] when floor present.

    Applies to northset and candle_order_book fuse receipts. Uses
    ``book_join_coverage_floor`` when finite (northset fail-closed companion);
    otherwise requires open-unit interval (0, 1]. NaN / absent → skip.
    Research diagnostic only; never live Sharpe / promotion.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("join_coverage")
    if val is None:
        return []
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        # Bools/strings must not coerce into a passing coverage.
        return ["join_coverage_non_numeric"]
    x = float(val)
    if x != x:  # NaN
        return []
    if abs(x) == float("inf"):
        return ["join_coverage_non_finite_fail_closed"]

    floor: float | None = None
    raw_floor = blob.get("book_join_coverage_floor")
    if raw_floor is None:
        raw_floor = blob.get("min_join_coverage")
    try:
        f = float(raw_floor)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        f = float("nan")
    if f == f and abs(f) != float("inf") and 0.0 <= f <= 1.0:
        floor = f

    if floor is not None:
        if not (floor <= x <= 1.0):
            return ["join_coverage_outside_floor_to_one_fail_closed"]
        return []
    if not (0.0 < x <= 1.0):
        return ["join_coverage_outside_open_unit_interval_fail_closed"]
    return []


def mean_book_age_seconds_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_book_age_seconds ≥ 0 when finite.

    Fuse/join age diagnostic on northset + candle_order_book receipts. NaN
    (no book ages) skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_book_age_seconds")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if x < 0.0 or abs(x) == float("inf"):
        return ["mean_book_age_seconds_negative_or_non_finite"]
    return []


def book_age_seconds_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean/max book_age_seconds ≥ 0; max ≥ mean when both finite.

    Applies to northset and candle_order_book fuse receipts that stamp
    ``mean_book_age_seconds`` / ``max_book_age_seconds``. NaN / absent → skip.
    ±inf → fail-closed. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []

    def _finite(key: str) -> float | None:
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if x != x:
            return None
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
            return None
        return x

    mean = _finite("mean_book_age_seconds")
    mx = _finite("max_book_age_seconds")
    if mean is not None and mean < 0.0:
        errs.append("mean_book_age_seconds_negative")
    if mx is not None and mx < 0.0:
        errs.append("max_book_age_seconds_negative")
    if mean is not None and mx is not None and mx + 1e-12 < mean:
        errs.append("max_book_age_seconds_lt_mean")
    return errs


def structure_finite_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify structure finite-rate keys ∈ [0, 1] when finite.

    Checks:
    - aggregate ``structure_finite_rate`` (northset + candle receipts)
    - candle ``finite_rate_*`` companions (microprice_minus_mid / size concentration)

    Never equate northset aggregate with candle companions. NaN/absent skip;
    ±inf fail-closed. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    keys = (
        "structure_finite_rate",
        "finite_rate_microprice_minus_mid",
        "finite_rate_bid_size_concentration_top",
        "finite_rate_ask_size_concentration_top",
    )
    errs: list[str] = []
    for key in keys:
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_top_level_claim_honesty_errors(blob: object) -> list[str]:
    """Soft-verify northset top-level research_only + claim markers.

    When either key is present: research_only must be True and claim must equal
    ``research_diagnostic_only``. Coupling: ``research_only is True`` ⇒ claim
    key present and correct (fail-closed if claim missing). Absent both → skip.
    Research diagnostic only; never live Sharpe. Off kyle nest.
    """
    if not isinstance(blob, dict):
        return []
    if "research_only" not in blob and "claim" not in blob:
        return []
    errs: list[str] = []
    if "research_only" in blob and blob.get("research_only") is not True:
        errs.append("northset_research_only_missing_or_false")
    if blob.get("research_only") is True:
        if "claim" not in blob:
            errs.append("northset_claim_missing_while_research_only_true")
        elif blob.get("claim") != "research_diagnostic_only":
            errs.append("northset_claim_not_research_diagnostic_only")
    elif "claim" in blob:
        # claim ⇒ research_only: a present claim without the honesty flag is not
        # acceptable evidence (the coupling must hold in both directions).
        if "research_only" not in blob:
            errs.append("northset_research_only_missing_or_false")
        if blob.get("claim") != "research_diagnostic_only":
            errs.append("northset_claim_not_research_diagnostic_only")
    return errs


def size_concentration_top_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_bid/ask_size_concentration_top ∈ (0, 1] when finite.

    Candle_order_book structure receipts. NaN/absent skip; ±inf fail-closed.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("mean_bid_size_concentration_top", "mean_ask_size_concentration_top"):
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
            continue
        if not (0.0 < x <= 1.0):
            errs.append(f"{key}_out_of_open_unit_interval")
    return errs


def mean_microprice_minus_mid_honesty_errors(blob: object) -> list[str]:
    """Soft-verify microprice−mid means finite when present (signed OK).

    Keys: mean_microprice_minus_mid (price units) and mean_microprice_minus_mid_bps.
    Never equate the two. NaN/absent skip; ±inf fail-closed.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("mean_microprice_minus_mid", "mean_microprice_minus_mid_bps"):
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


def northset_log_size_slope_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_bid/ask_log_size_slope finite when present (not ±inf).

    Slopes may be negative (deeper levels thinner). NaN skipped.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("mean_bid_log_size_slope", "mean_ask_log_size_slope"):
        if key not in blob or blob.get(key) is None:
            continue
        val = blob.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errs.append(f"{key}_non_numeric")
            continue
        x = float(val)
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
    return errs


def northset_qlike_means_honesty_errors(blob: object) -> list[str]:
    """Soft-verify northset QLIKE keys ≥ 0 when finite.

    Parkinson / Garman-Klass / Rogers-Satchell / Yang-Zhang vs close-to-close
    QLIKE are losses — negative is dishonest. NaN skipped.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "parkinson_qlike_vs_cc",
        "garman_klass_qlike_vs_cc",
        "rogers_satchell_qlike_vs_cc",
        "yang_zhang_qlike_vs_cc",
        "overnight_plus_oc_qlike_vs_cc",
        "session_rv_qlike_vs_cc",
    ):
        if key not in blob or blob.get(key) is None:
            continue
        val = blob.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errs.append(f"{key}_non_numeric")
            continue
        x = float(val)
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            continue
        if x < 0.0:
            errs.append(f"{key}_negative")
    return errs


def northset_range_spread_honesty_errors(blob: object) -> list[str]:
    """Soft-verify range/spread estimator means when finite.

    - ``corwin_schultz_spread`` / ``abdi_ranaldo_spread`` ∈ [0, 1] (relative spreads)
    - ``roll_spread`` / ``mean_true_range`` / ``yang_zhang_variance`` ≥ 0
      (Roll can exceed 1 depending on scale)
    NaN skipped. Research diagnostic only; never live Sharpe. Off kyle_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    unit_keys = ("corwin_schultz_spread", "abdi_ranaldo_spread")
    nonneg_keys = ("roll_spread", "mean_true_range", "yang_zhang_variance")
    for key in unit_keys + nonneg_keys:
        if key not in blob:
            continue
        raw = blob.get(key)
        if raw is None:
            continue  # JSON null = unavailable (NaN), never non-numeric
        try:
            x = float(raw)
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            continue
        if key in unit_keys:
            if not (0.0 <= x <= 1.0):
                errs.append(f"{key}_out_of_unit_interval")
        elif x < 0.0:
            errs.append(f"{key}_negative")
    return errs


def amihud_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: amihud_mean ≥ 0 when finite.

    Amihud illiquidity is |ret| / dollar volume — negative is dishonest.
    NaN skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    try:
        x = float(blob.get("amihud_mean"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["amihud_mean_non_finite"]
    if x < 0.0:
        return ["amihud_mean_negative"]
    return []


def depth_shape_finite_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: depth_shape_finite_rate ∈ [0, 1] when finite.

    Candle_order_book / northset fuse rate of finite DEPTH_SHAPE fields.
    NaN (thin/missing shape cols) skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("depth_shape_finite_rate")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if not (0.0 <= x <= 1.0):
        return ["depth_shape_finite_rate_out_of_unit_interval"]
    return []


def mean_tob_notional_share_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_tob_notional_share ∈ (0, 1] when finite.

    NaN/absent skip; ±inf fail-closed. ≠ mean_tob_size_share.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_tob_notional_share" not in blob:
        return []
    try:
        x = float(blob.get("mean_tob_notional_share"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_tob_notional_share_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 < x <= 1.0):
        return ["mean_tob_notional_share_out_of_open_unit_interval"]
    return []


def mean_depth_imbalance_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_depth_imbalance ∈ [-1, 1] when finite.

    NaN/absent skip; ±inf fail-closed. ≠ imbalance_top / notional_imbalance.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_depth_imbalance" not in blob:
        return []
    try:
        x = float(blob.get("mean_depth_imbalance"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_depth_imbalance_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (-1.0 <= x <= 1.0):
        return ["mean_depth_imbalance_out_of_unit_interval"]
    return []


def mean_depth_imbalance_abs_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_depth_imbalance_abs ∈ [0, 1] when finite.

    Absolute depth imbalance — ≠ signed mean_depth_imbalance. NaN/absent skip.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_depth_imbalance_abs" not in blob:
        return []
    try:
        x = float(blob.get("mean_depth_imbalance_abs"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_depth_imbalance_abs_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
        return ["mean_depth_imbalance_abs_out_of_unit_interval"]
    return []


def mean_imbalance_top_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_imbalance_top ∈ [-1, 1] when finite.

    Top-of-book size imbalance — ≠ mean_depth_imbalance / mean_notional_imbalance.
    touch_size_imbalance is an alias; do not require a second receipt key.
    NaN/absent skip. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_imbalance_top" not in blob:
        return []
    try:
        x = float(blob.get("mean_imbalance_top"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_imbalance_top_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (-1.0 <= x <= 1.0):
        return ["mean_imbalance_top_out_of_unit_interval"]
    return []


def northset_shape_columns_ensured_book_panel_path_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``shape_columns_ensured`` ↔ ``book_panel_path`` stamp pair.

    Contract (``bench_northset`` / DATA_CONTRACTS): synth ensure runs when
    ``book_panel_path`` is absent — ``shape_columns_ensured is True`` ⇔ path
    None/empty; ``False`` ⇔ nonempty path string.

    When only one side present → skip. Research diagnostic only; never live
    Sharpe. Off nest invent / kyle_ofi overwrite (nest has its own path helper).
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "northset"):
        return []
    if "shape_columns_ensured" not in blob:
        return []
    flag = blob.get("shape_columns_ensured")
    if type(flag) is not bool:
        return []  # bool-flags helper owns type

    has_path_key = "book_panel_path" in blob
    path = blob.get("book_panel_path") if has_path_key else None
    path_nonempty = isinstance(path, str) and bool(path.strip())

    if flag is True:
        if has_path_key and path is not None and not isinstance(path, str):
            # Malformed path must not read as the benign "absent" case.
            return ["book_panel_path_non_string_while_shape_columns_ensured"]
        if has_path_key and path is not None and path_nonempty:
            return ["book_panel_path_set_while_shape_columns_ensured"]
        return []
    # flag False
    if has_path_key and not path_nonempty:
        return ["book_panel_path_missing_while_shape_columns_not_ensured"]
    if not has_path_key:
        return ["book_panel_path_missing_while_shape_columns_not_ensured"]
    return []


def northset_shape_columns_ensured_rates_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ``shape_columns_ensured is True`` ⇒ shape rates present + finite.

    When synth ensure ran, these three rates must be stamped and finite ∈ [0, 1]:
    ``depth_shape_finite_rate``, ``concentration_top_finite_rate``,
    ``queue_priority_finite_rate``. ``False`` / absent / non-bool → skip (bool
    helper owns type). Research diagnostic only; never live Sharpe.
    Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    flag = blob.get("shape_columns_ensured")
    if flag is not True:
        return []
    errs: list[str] = []
    for key in (
        "depth_shape_finite_rate",
        "concentration_top_finite_rate",
        "queue_priority_finite_rate",
    ):
        if key not in blob:
            errs.append(f"{key}_missing_while_shape_columns_ensured")
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric_while_shape_columns_ensured")
            continue
        if x != x or abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_while_shape_columns_ensured")
        elif not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval_while_shape_columns_ensured")
    return errs


def northset_metrics_required_finite_ok_rates_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ``metrics_required_finite_ok is True`` ⇒ companion rates ok.

    When the receipt claims REQUIRED LOB metrics were finite on a sample row,
    the stamped structure finite-rate companions must be present and finite
    ∈ [0, 1]. Does **not** invent per-key ``METRICS_REQUIRED_*`` receipt stamps
    (those stay inside book_metrics asserts). Off kyle_ofi. Research only.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("metrics_required_finite_ok") is not True:
        return []
    errs: list[str] = []
    for key in (
        "depth_shape_finite_rate",
        "concentration_top_finite_rate",
        "queue_priority_finite_rate",
        "side_notional_finite_rate",
        "tob_size_share_finite_rate",
        "structure_finite_rate",
    ):
        if key not in blob:
            errs.append(f"{key}_missing_while_metrics_required_finite_ok")
            continue
        val = blob.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errs.append(f"{key}_non_numeric_while_metrics_required_finite_ok")
            continue
        x = float(val)
        if x != x or abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_while_metrics_required_finite_ok")
        elif not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval_while_metrics_required_finite_ok")
    return errs


def northset_metrics_required_keys_finite_when_present_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: METRICS_REQUIRED keys on the receipt, if present, are finite.

    Does not invent keys. Any overlapping ``METRICS_REQUIRED_FINITE_KEYS`` stamp
    must parse finite (not NaN/±inf). Off kyle_ofi / METRICS_* invent. Research only.
    """
    if not isinstance(blob, dict):
        return []
    try:
        from quant_fund.microstructure.book_metrics import METRICS_REQUIRED_FINITE_KEYS
    except Exception:
        return []
    errs: list[str] = []
    for key in METRICS_REQUIRED_FINITE_KEYS:
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric_metrics_required")
            continue
        if x != x or abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_metrics_required")
    return errs


def northset_depth_notional_spread_over_mid_honesty_errors(blob: object) -> list[str]:
    """Soft-verify depth / side-notional / spread_over_mid receipt means.

    - mean_bid_depth / mean_ask_depth ≥ 0 when finite
    - mean_side_notional_proxy_bid / ask ≥ 0 when finite
    - mean_top_of_book_notional_proxy ≥ 0 when finite
    - mean_spread_over_mid ≥ 0 when finite (≠ mean_spread_bps scale)
    NaN/absent skip; ±inf fail. Research diagnostic only; never live Sharpe.
    Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "mean_bid_depth",
        "mean_ask_depth",
        "mean_side_notional_proxy_bid",
        "mean_side_notional_proxy_ask",
        "mean_top_of_book_notional_proxy",
        "mean_spread_over_mid",
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
            errs.append(f"{key}_non_finite")
            continue
        if x < 0.0:
            errs.append(f"{key}_negative")
    return errs


def northset_price_slope_tick_top_levels_honesty_errors(blob: object) -> list[str]:
    """Soft-verify price-slope / tick-spacing / top-size / n_levels means.

    - mean_bid/ask_log_price_slope finite when present (signed OK; ≠ size slopes)
    - mean_bid/ask_mean_log_tick_spacing finite when present (may be negative:
      log of sub-unit tick gaps)
    - mean_top_bid/ask_size ≥ 0 when finite
    - mean_n_bid/ask_levels ≥ 0 when finite
    NaN/absent skip. Research diagnostic only; never live Sharpe.
    Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("mean_bid_log_price_slope", "mean_ask_log_price_slope"):
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
            errs.append(f"{key}_non_finite")
    # Log tick spacing is finite-when-present only: gaps below 1.0 (e.g. 0.01
    # ticks) give legitimately negative logs — sign is not an honesty failure.
    for key in ("mean_bid_mean_log_tick_spacing", "mean_ask_mean_log_tick_spacing"):
        if key not in blob:
            continue
        raw = blob.get(key)
        if raw is None:
            continue
        try:
            x = float(raw)
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
    for key in (
        "mean_top_bid_size",
        "mean_top_ask_size",
        "mean_n_bid_levels",
        "mean_n_ask_levels",
    ):
        if key not in blob:
            continue
        raw = blob.get(key)
        if raw is None:
            continue
        try:
            x = float(raw)
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            continue
        if x < 0.0:
            errs.append(f"{key}_negative")
    return errs


def northset_overnight_rv_semi_honesty_errors(blob: object) -> list[str]:
    """Soft-verify stamped overnight / bipower / semi companions.

    - overnight_share ∈ [0, 1] when finite
    - session_mean_rv / session_mean_bv ≥ 0 when finite
    - semi_up / semi_down ≥ 0 when finite
    NaN/absent skip. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    if "overnight_share" in blob:
        try:
            x = float(blob.get("overnight_share"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("overnight_share_non_numeric")
        else:
            if x == x and abs(x) != float("inf") and not (0.0 <= x <= 1.0):
                errs.append("overnight_share_out_of_unit_interval")
            elif x == x and abs(x) == float("inf"):
                errs.append("overnight_share_non_finite")
    for key in ("session_mean_rv", "session_mean_bv", "semi_up", "semi_down"):
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
            errs.append(f"{key}_non_finite")
            continue
        if x < 0.0:
            errs.append(f"{key}_negative")
    return errs


def mean_notional_imbalance_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_notional_imbalance ∈ [-1, 1] when finite.

    NaN/absent skip; ±inf fail-closed. ≠ imbalance_top. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_notional_imbalance" not in blob:
        return []
    try:
        x = float(blob.get("mean_notional_imbalance"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_notional_imbalance_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (-1.0 <= x <= 1.0):
        return ["mean_notional_imbalance_out_of_unit_interval"]
    return []


def mean_queue_priority_honesty_errors(blob: object) -> list[str]:
    """Soft-verify queue priority means ∈ [0, 1] when finite.

    Keys: mean_queue_priority_proxy, mean_ask_queue_priority_proxy, and legacy
    mean_queue_priority alias. NaN/absent skip; ±inf fail-closed.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    keys = (
        "mean_queue_priority_proxy",
        "mean_ask_queue_priority_proxy",
        "mean_queue_priority",
    )
    for key in keys:
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def mean_tob_size_share_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_tob_size_share ∈ (0, 1] when finite.

    NaN/absent skip; ±inf fail-closed. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_tob_size_share" not in blob:
        return []
    val = blob.get("mean_tob_size_share")
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        # Bools/strings must not coerce into a passing share.
        return ["mean_tob_size_share_non_numeric"]
    x = float(val)
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 < x <= 1.0):
        return ["mean_tob_size_share_out_of_open_unit_interval"]
    return []


def mean_close_mid_abs_rel_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_close_mid_abs_rel ≥ 0 when finite.

    Candle close−mid absolute relative diagnostic (≠ mean_effective_spread /
    quoted book spread). NaN/absent skip; ±inf fail-closed.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_close_mid_abs_rel" not in blob:
        return []
    try:
        x = float(blob.get("mean_close_mid_abs_rel"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_close_mid_abs_rel_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["mean_close_mid_abs_rel_non_finite"]
    if x < 0.0:
        return ["mean_close_mid_abs_rel_negative"]
    return []


def northset_book_shape_finite_rates_honesty_errors(blob: object) -> list[str]:
    """Soft-verify northset book-shape finite_rate companions ∈ [0, 1].

    Keys: concentration_top_finite_rate, queue_priority_finite_rate,
    side_notional_finite_rate, tob_size_share_finite_rate.
    (depth_shape_finite_rate / gap_finite_rate have their own helpers.)
    NaN/absent skipped; ±inf / out-of-range fail. Research diagnostic only;
    never live Sharpe. Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "concentration_top_finite_rate",
        "queue_priority_finite_rate",
        "side_notional_finite_rate",
        "tob_size_share_finite_rate",
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
            errs.append(f"{key}_non_finite")
            continue
        if not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def book_uncrossed_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: book_uncrossed_rate ∈ [0, 1] when finite.

    Northset book integrity rate (H21 companion). NaN/absent skip.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("book_uncrossed_rate")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
        return ["book_uncrossed_rate_out_of_unit_interval"]
    return []


def ohlc_identity_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ohlc_identity_rate ∈ [0, 1] when finite.

    Daily bar OHLC identity rate (≠ session_ohlc_identity_rate). NaN skipped.
    Research diagnostic only; never live Sharpe. Does not touch kyle_* keys.
    """
    if not isinstance(blob, dict):
        return []
    try:
        x = float(blob.get("ohlc_identity_rate"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["ohlc_identity_rate_non_finite"]
    if not (0.0 <= x <= 1.0):
        return ["ohlc_identity_rate_out_of_unit_interval"]
    return []


def northset_n_bars_scored_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: n_bars / n_fused / n_scored when present — ints ≥0; chain order.

    Top-level northset receipt (≠ kyle nest n_fused/n_scored). When present:
    - each is a non-negative integer (NaN skip per key)
    - ``n_scored ≤ n_fused`` when both present
    - ``n_fused ≤ n_bars`` when both present
    - ``n_scored ≤ n_bars`` when both present (legacy pair)

    Never invent always-on ``min_names`` (unstamped). Research diagnostic only;
    never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []

    def _nonneg_int(key: str) -> int | None:
        if key not in blob:
            return None
        val = blob.get(key)
        try:
            x = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            return None
        if x != x:
            return None
        if abs(x) == float("inf") or x < 0 or abs(x - int(x)) > 1e-9:
            errs.append(f"{key}_not_nonneg_int")
            return None
        return int(x)

    nb = _nonneg_int("n_bars")
    nf = _nonneg_int("n_fused")
    ns = _nonneg_int("n_scored")
    if nb is not None and ns is not None and ns > nb:
        errs.append("n_scored_gt_n_bars")
    if nf is not None and ns is not None and ns > nf:
        errs.append("n_scored_gt_n_fused")
    if nb is not None and nf is not None and nf > nb:
        errs.append("n_fused_gt_n_bars")
    return errs


def robinhood_plus_claim_honesty_errors(blob: object) -> list[str]:
    """Soft-verify robinhood+ research_only / execution_claim / family stamps.

    When ``family == "robinhood_plus"``: research_only must be True,
    execution_claim must be research_only, and claim must be
    research_metric_only. Do not stamp a ``pnl`` key token. Never a live
    capital or promotion gate.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "robinhood_plus":
        return []
    errors: list[str] = []
    if blob.get("research_only") is not True:
        errors.append("robinhood_plus_research_only_invalid")
    if blob.get("execution_claim") != "research_only":
        errors.append("robinhood_plus_execution_claim_invalid")
    if blob.get("claim") != "research_metric_only":
        errors.append("robinhood_plus_claim_invalid")
    backend = blob.get("backend")
    if backend not in (None, "numpy", "torch"):
        errors.append("robinhood_plus_backend_invalid")
    sizes = blob.get("sizes_book")
    if sizes is True:
        try:
            ic_chal = float(blob.get("mean_ic_challenger"))  # type: ignore[arg-type]
            ic_champ = float(blob.get("mean_ic_champion"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append("robinhood_plus_sizes_book_without_ic_win")
        else:
            if not (ic_chal == ic_chal and ic_champ == ic_champ and ic_chal > ic_champ):
                errors.append("robinhood_plus_sizes_book_without_ic_win")
    return errors


def northset_dm_park_honesty_errors(blob: object) -> list[str]:
    """Soft-verify Diebold–Mariano vs Park companions on northset.

    For each of gk/rs/split vs park:
    - ``dm_*_vs_park_p`` ∈ [0, 1] when finite
    - ``dm_*_vs_park_stat`` finite when present (signed OK)
    - ``dm_*_vs_park_preferred`` ∈ allowed label set for that pair
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    allowed = {
        "gk": frozenset({"park", "gk"}),
        "rs": frozenset({"park", "rs"}),
        "split": frozenset({"park", "split"}),
    }
    for stem in ("gk", "rs", "split"):
        p_key = f"dm_{stem}_vs_park_p"
        s_key = f"dm_{stem}_vs_park_stat"
        pref_key = f"dm_{stem}_vs_park_preferred"
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
        if s_key in blob:
            try:
                s = float(blob.get(s_key))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append(f"{s_key}_non_numeric")
            else:
                if s == s and abs(s) == float("inf"):
                    errs.append(f"{s_key}_non_finite_fail_closed")
        if pref_key in blob:
            pref = blob.get(pref_key)
            # isinstance guard: unhashable JSON values (list/dict) would raise
            # TypeError on set membership instead of failing closed.
            if not isinstance(pref, str) or pref not in allowed[stem]:
                errs.append(f"{pref_key}_invalid")
    return errs


_NORTHSET_IMPACT_ESTIMATOR_SCOPE_ALLOWED = frozenset({"per_security_equal_weight"})
_NORTHSET_YANG_ZHANG_QLIKE_SCOPE_ALLOWED = frozenset({"per_security_expanding_oos"})
_NORTHSET_VPIN_METHOD_ALLOWED = frozenset(
    {"count_window_bulk_ofi_proxy", "volume_bucket_bulk_ofi_proxy"}
)
_NORTHSET_CORWIN_SCHULTZ_PAIR_SCOPE_ALLOWED = frozenset({"prior_and_current_bar"})
_NORTHSET_OVERNIGHT_GAP_METHOD_ALLOWED = frozenset({"event_close_to_next_open"})
_NORTHSET_TWO_WAY_INFERENCE_INDEX_ALLOWED = frozenset({"event_rows_not_calendar_zeros"})


def northset_receipt_dgp_data_source_honesty_errors(blob: object) -> list[str]:
    """Soft-verify always-on northset ``dgp`` / ``book_dgp`` ↔ ``data_source``.

    Stamped on ``bench_northset`` (≠ nest ``kyle_ofi`` twins — never equate).
    When present on the top-level receipt:

    - ``dgp`` and ``book_dgp`` both present → must match
    - ``data_source == "SYNTHETIC"`` ⇒ present dgp fields are ``synthetic_lob``
    - ``book_dgp``/``dgp`` ``synthetic_lob`` ⇒ ``data_source`` is ``SYNTHETIC`` when set
    - non-synthetic dgp + ``data_source`` set ⇒ ``data_source != "SYNTHETIC"``;
      if ``book_source`` also set, ``data_source == book_source``

    Skip when all three of dgp/book_dgp/data_source absent. Research diagnostic
    only; never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "northset"):
        return []

    malformed: list[str] = []

    def _str(key: str) -> str | None:
        if key not in blob or blob.get(key) is None:
            return None
        val = blob.get(key)
        if not isinstance(val, str):
            # Present-but-non-string provenance must not read as absent.
            malformed.append(f"{key}_non_str")
            return None
        s = val.strip()
        return s if s else None

    book_dgp = _str("book_dgp")
    dgp = _str("dgp")
    data_source = _str("data_source")
    book_source = _str("book_source")
    if book_dgp is None and dgp is None and data_source is None and not malformed:
        return []

    errs: list[str] = list(malformed)
    if book_dgp is not None and dgp is not None and book_dgp != dgp:
        errs.append("northset_dgp_book_dgp_mismatch")

    synth_dgp: bool | None = None
    if book_dgp is not None:
        synth_dgp = book_dgp == "synthetic_lob"
    elif dgp is not None:
        synth_dgp = dgp == "synthetic_lob"

    if synth_dgp is True and data_source is not None and data_source != "SYNTHETIC":
        errs.append("northset_synthetic_dgp_data_source_not_SYNTHETIC")
    if data_source == "SYNTHETIC":
        if book_dgp is not None and book_dgp != "synthetic_lob":
            errs.append("northset_SYNTHETIC_data_source_book_dgp_not_synthetic_lob")
        if dgp is not None and dgp != "synthetic_lob":
            errs.append("northset_SYNTHETIC_data_source_dgp_not_synthetic_lob")
        if book_dgp is None and dgp is None:
            errs.append("northset_SYNTHETIC_data_source_missing_dgp")
    if synth_dgp is False and data_source is not None:
        if data_source == "SYNTHETIC":
            errs.append("northset_nonsynthetic_dgp_data_source_SYNTHETIC")
        elif book_source is not None and data_source != book_source:
            errs.append("northset_nonsynthetic_data_source_ne_book_source")
    return errs


def northset_receipt_string_enum_honesty_errors(blob: object) -> list[str]:
    """Soft-verify northset string enum companions when present.

    - ``claim`` → research_diagnostic_only
    - ``research_only is True`` when present
    - ``session_l2_identity_gate`` → enforced|skipped
    - ``data_source`` / ``label``: SYNTHETIC source ⇔ SYN* label when both present
    Research diagnostic only; never live Sharpe. Off kyle nest.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "northset"):
        return []
    errs: list[str] = []
    if "claim" in blob and blob.get("claim") != "research_diagnostic_only":
        errs.append("northset_claim_not_research_diagnostic_only")
    if "research_only" in blob and blob.get("research_only") is not True:
        errs.append("northset_research_only_missing_or_false")
    if "session_l2_identity_gate" in blob:
        gate = blob.get("session_l2_identity_gate")
        if gate not in ("enforced", "skipped"):
            errs.append("session_l2_identity_gate_invalid")
    src = blob.get("data_source")
    lab = blob.get("label")
    if isinstance(src, str) and isinstance(lab, str):
        synth_src = src.upper() == "SYNTHETIC"
        synth_lab = lab.upper().startswith("SYN")
        if synth_src != synth_lab:
            errs.append("northset_data_source_label_synthetic_mismatch")
    return errs


def northset_product_stamp_honesty_errors(blob: object) -> list[str]:
    """Soft-verify stamped ``product`` on northset receipts.

    ``bench_northset`` stamps ``product: "Northset"``. When ``product`` is present
    and family is ``northset`` (or family absent): value must be the nonempty string
    ``Northset``. Skip other families / absent key. Research diagnostic only; never
    live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if "product" not in blob:
        return []
    fam = blob.get("family")
    if fam is not None and fam != "northset":
        return []
    val = blob.get("product")
    if not isinstance(val, str) or not val:
        return ["northset_product_not_nonempty_str"]
    if val != "Northset":
        return ["northset_product_unexpected_token"]
    return []


def northset_impact_proxy_warning_honesty_errors(blob: object) -> list[str]:
    """Soft-verify stamped ``impact_proxy_warning`` enum on northset receipts.

    DATA_CONTRACTS / benches: book-size / TOB proxies are not signed trade prints.
    When present on ``family == northset`` (or family absent with the key stamped):
    value must be the nonempty string
    ``depth_or_ofi_proxy_not_signed_trade_flow``. Skip other families / absent key.
    Research diagnostic only; never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if "impact_proxy_warning" not in blob:
        return []
    fam = blob.get("family")
    if fam is not None and fam != "northset":
        return []
    val = blob.get("impact_proxy_warning")
    expected = "depth_or_ofi_proxy_not_signed_trade_flow"
    if not isinstance(val, str) or not val:
        return ["impact_proxy_warning_not_nonempty_str"]
    if val != expected:
        return ["impact_proxy_warning_unexpected_token"]
    return []


def close_location_value_clv_alias_identity_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``close_location_value_*`` IC aliases match ``clv_*`` when both stamped.

    DATA_CONTRACTS: ``clv_p_ic`` / ``clv_t_ic`` are receipt aliases of
    ``close_location_value_p_ic`` / ``close_location_value_t_ic`` (H30 binds to
    ``clv_p_ic``). When both sides of a pair are finite they must match; keys stay
    distinct. Skip absent / non-finite sides. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    pairs = (
        ("close_location_value_p_ic", "clv_p_ic", "close_location_value_p_ic_clv_p_ic_mismatch"),
        ("close_location_value_t_ic", "clv_t_ic", "close_location_value_t_ic_clv_t_ic_mismatch"),
    )
    for long_k, short_k, err_token in pairs:
        if long_k == short_k:
            errs.append("close_location_value_clv_alias_keys_collapsed")
            continue
        if long_k not in blob or short_k not in blob:
            continue
        if not _finite_scalar(blob.get(long_k)) or not _finite_scalar(blob.get(short_k)):
            continue
        a = float(blob[long_k])
        b = float(blob[short_k])
        if not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12):
            errs.append(err_token)
    return errs


_NORTHSET_PRICE_BASIS_ALLOWED = frozenset({"split_adjusted", "raw_fixture_opt_out"})
_NORTHSET_RETURN_BASIS_ALLOWED = frozenset(
    {"total_return", "split_adjusted", "raw_fixture_opt_out"}
)


def northset_family_book_source_honesty_errors(blob: object) -> list[str]:
    """Soft-verify always-on ``family`` / ``book_source`` / ``label`` stamps when present.

    - When ``family`` is present: must be ``"northset"`` (skip other families so
      candle blobs are not dual-flagged — callers pass the northset family blob)
    - When ``book_source`` is present: nonempty string
    - When ``label`` is present: nonempty string (empty label can pass the
      SYNTHETIC↔SYN* match when ``data_source`` is also non-SYN)

    Skip when ``family`` is ``candle_order_book`` (wrong surface). Research
    diagnostic only; never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    fam = blob.get("family") if "family" in blob else None
    if fam == "candle_order_book":
        return []

    errs: list[str] = []
    if "family" in blob and blob.get("family") is not None and blob.get("family") != "northset":
        errs.append("northset_family_invalid")

    if "book_source" in blob and blob.get("book_source") is not None:
        bs = blob.get("book_source")
        if not isinstance(bs, str) or not bs.strip():
            errs.append("northset_book_source_empty_or_not_str")

    if "label" in blob and blob.get("label") is not None:
        lab = blob.get("label")
        if not isinstance(lab, str) or not lab.strip():
            errs.append("northset_label_empty_or_not_str")
    return errs


def northset_price_return_basis_honesty_errors(blob: object) -> list[str]:
    """Soft-verify always-on ``price_basis`` / ``return_basis`` stamps when present.

    Contract (DATA_CONTRACTS price_basis/return_basis matrix):
    - ``price_basis`` ∈ {split_adjusted, raw_fixture_opt_out}
    - ``return_basis`` ∈ {total_return, split_adjusted, raw_fixture_opt_out}
    - both ``raw_fixture_opt_out`` ⇒ must match each other
    - ``price_basis == split_adjusted`` ⇒ ``return_basis`` ∈ {total_return, split_adjusted}
      when return_basis present

    Absent keys skip. Research diagnostic only; never live Sharpe. Off nest /
    kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "northset"):
        return []

    errs: list[str] = []
    pb = blob.get("price_basis") if "price_basis" in blob else None
    rb = blob.get("return_basis") if "return_basis" in blob else None

    if pb is not None and (not isinstance(pb, str) or pb not in _NORTHSET_PRICE_BASIS_ALLOWED):
        errs.append("northset_price_basis_invalid")
    if rb is not None and (not isinstance(rb, str) or rb not in _NORTHSET_RETURN_BASIS_ALLOWED):
        errs.append("northset_return_basis_invalid")

    if isinstance(pb, str) and isinstance(rb, str):
        if pb == "raw_fixture_opt_out" and rb != "raw_fixture_opt_out":
            errs.append("northset_raw_fixture_price_return_basis_mismatch")
        if pb == "split_adjusted" and rb not in ("total_return", "split_adjusted"):
            errs.append("northset_split_adjusted_return_basis_invalid")
    return errs


def northset_depth_honesty_errors(blob: object) -> list[str]:
    """Soft-verify always-on northset ``depth`` stamp when present.

    LOB levels used by attach / shape floors — must be finite integer ≥ 1.
    Candle has a parallel check in ``candle_order_book_sizing_honesty_errors``;
    never equate the two surfaces. Skip absent/NaN. Research diagnostic only;
    never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "northset"):
        return []
    if "depth" not in blob or blob.get("depth") is None:
        return []
    try:
        d = float(blob.get("depth"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["northset_depth_non_numeric"]
    if d != d:
        return []
    if abs(d) == float("inf"):
        return ["northset_depth_non_finite"]
    if d < 1.0 or abs(d - int(d)) > 1e-9:
        return ["northset_depth_lt_one_or_not_int"]
    return []


def northset_component_sources_honesty_errors(blob: object) -> list[str]:
    """Soft-verify always-on ``component_sources`` receipt map when present.

    Contract (DATA_CONTRACTS evidence matrix / ``bench_northset`` stamp):
    - value is a ``dict``
    - required keys: ``bars``, ``book``, ``session_candles``, ``session_book``
      with nonempty string values
    - ``session_candles`` == ``synthetic_reconstruction``
    - when ``use_session_l2`` is bool: ``session_book`` is
      ``synthetic_reconstruction`` if True else ``disabled``
    - when ``book_source`` is a nonempty str: ``book`` must equal it

    Absent ``component_sources`` → skip. Nest has no component map — never equate.
    Research diagnostic only; never live Sharpe. Off kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if "component_sources" not in blob:
        return []
    cs = blob.get("component_sources")
    if not isinstance(cs, dict):
        return ["component_sources_not_dict"]

    errs: list[str] = []
    required = ("bars", "book", "session_candles", "session_book")
    for key in required:
        if key not in cs:
            errs.append(f"component_sources_missing_{key}")
            continue
        val = cs.get(key)
        if not isinstance(val, str) or not val.strip():
            errs.append(f"component_sources_{key}_empty_or_not_str")

    if errs:
        return errs

    if cs.get("session_candles") != "synthetic_reconstruction":
        errs.append("component_sources_session_candles_not_synthetic_reconstruction")

    use = blob.get("use_session_l2")
    if type(use) is bool:
        expect_book = "synthetic_reconstruction" if use else "disabled"
        if cs.get("session_book") != expect_book:
            errs.append("component_sources_session_book_mismatch_use_session_l2")

    book_source = blob.get("book_source")
    if isinstance(book_source, str) and book_source.strip() and cs.get("book") != book_source:
        errs.append("component_sources_book_ne_book_source")

    return errs


def northset_receipt_bool_flags_honesty_errors(blob: object) -> list[str]:
    """Soft-verify stamped northset receipt bool flags are real ``bool``.

    Covers keys stamped by ``bench_northset`` that are not already checked by
    ``book_hypothesis_eligible_honesty_errors``:

    - ``metrics_required_finite_ok``
    - ``shape_columns_ensured``
    - ``use_session_l2``
    - ``research_only``
    - ``include_kyle_ofi``
    - ``sweep_follow_control_sample_adequate``
    - ``sweep_reject_control_sample_adequate``

    When present, each must be ``type is bool`` (not int/str/None). True/False
    both valid. Absent skip. Top-level always-on only (≠ nest). Research
    diagnostic only; never live Sharpe. Off kyle_ofi overwrite / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "metrics_required_finite_ok",
        "shape_columns_ensured",
        "use_session_l2",
        "research_only",
        "include_kyle_ofi",
        "sweep_follow_control_sample_adequate",
        "sweep_reject_control_sample_adequate",
    ):
        if key not in blob:
            continue
        if type(blob.get(key)) is not bool:
            errs.append(f"{key}_not_bool")
    return errs


def book_hypothesis_eligible_honesty_errors(blob: object) -> list[str]:
    """Soft-verify book_hypothesis_eligible / session_book_hypothesis_eligible bools.

    When present, each key must be a real ``bool`` (not int/str/None). Content
    True/False is always valid — eligibility is a receipt flag, not a score.
    Absent keys skipped. Research diagnostic only; never live Sharpe.
    Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("book_hypothesis_eligible", "session_book_hypothesis_eligible"):
        if key not in blob:
            continue
        val = blob.get(key)
        if type(val) is not bool:
            errs.append(f"{key}_not_bool")
    return errs


def gap_finite_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: gap_finite_rate ∈ [0, 1] when finite.

    Northset overnight gap finite fraction (identities.gap_finite_rate).
    NaN (empty bars) skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("gap_finite_rate")
    if val is None:
        return []
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        # Bools/strings must not coerce into a passing rate.
        return ["gap_finite_rate_non_numeric"]
    x = float(val)
    if x != x:  # NaN
        return []
    if not (0.0 <= x <= 1.0):
        return ["gap_finite_rate_out_of_unit_interval"]
    return []


def vpin_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: vpin_mean ∈ [0, 1] when finite.

    Daily fused VPIN mean on northset receipt (not session_book_vpin_mean).
    NaN skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("vpin_mean")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if not (0.0 <= x <= 1.0):
        return ["vpin_mean_out_of_unit_interval"]
    return []


def northset_spread_means_honesty_errors(blob: object) -> list[str]:
    """Soft-verify northset mean_quoted/effective/spread_bps when finite.

    - each ≥ 0 (and not ±inf), including mean_half_spread[_bps]
    - half ≈ ½ quoted; half_bps ≈ ½ spread_bps
    - do not equate quoted ≈ effective on this fuse (distinct columns)
    NaN keys skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []

    def _finite(key: str) -> float | None:
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if x != x:
            return None
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite")
            return None
        return x

    quoted = _finite("mean_quoted_spread")
    effective = _finite("mean_effective_spread")
    spread_bps = _finite("mean_spread_bps")
    half = _finite("mean_half_spread")
    half_bps = _finite("mean_half_spread_bps")
    for key, val in (
        ("mean_quoted_spread", quoted),
        ("mean_effective_spread", effective),
        ("mean_spread_bps", spread_bps),
        ("mean_half_spread", half),
        ("mean_half_spread_bps", half_bps),
    ):
        if val is not None and val < 0.0:
            errs.append(f"{key}_negative")
    # Post-collision-fix contract: effective_spread is the ask−bid alias, so a
    # finite quoted/effective divergence is a dishonesty signal (book effective
    # must not track the candle close−mid diagnostic).
    if (
        quoted is not None
        and effective is not None
        and not math.isclose(quoted, effective, rel_tol=1e-6, abs_tol=1e-12)
    ):
        errs.append("mean_quoted_effective_spread_mismatch")
    if quoted is not None and half is not None:
        tol = 1e-6 + 1e-6 * abs(quoted)
        if abs(half - 0.5 * quoted) > tol:
            errs.append("mean_half_spread_not_half_quoted")
    if spread_bps is not None and half_bps is not None:
        tol = 1e-6 + 1e-6 * abs(spread_bps)
        if abs(half_bps - 0.5 * spread_bps) > tol:
            errs.append("mean_half_spread_bps_not_half_spread_bps")
    return errs


# --- Mac-only honesty helpers merged by Lt ---
# Spread-alias tolerance constants (restored: the merge dropped them while the
# helpers below call math.isclose with these names).
_NORTHSET_SPREAD_REL_TOL = 1e-9
_NORTHSET_SPREAD_ABS_TOL = 1e-12


def northset_half_spread_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ``mean_half_spread ≈ 0.5 * mean_quoted_spread`` when both finite.

    Missing / non-finite either side → skip. Research diagnostic only; never
    live Sharpe / promotion.
    """
    pair = _finite_pair(blob, "mean_half_spread", "mean_quoted_spread")
    if pair is None:
        return []
    half, quoted = pair
    if not math.isclose(
        half,
        0.5 * quoted,
        rel_tol=_NORTHSET_SPREAD_REL_TOL,
        abs_tol=_NORTHSET_SPREAD_ABS_TOL,
    ):
        return ["mean_half_spread_not_half_of_mean_quoted_spread"]
    return []


def northset_queue_imbalance_mean_alias_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ``queue_imbalance_mean`` ∈ [-1, 1] when finite.

    Alias companion to ``mean_queue_imbalance`` / session queue stamps — same
    signed-unit contract on the northset receipt key name.
    """
    if not isinstance(blob, dict):
        return []
    raw = blob.get("queue_imbalance_mean")
    if raw is None:
        return []
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return ["queue_imbalance_mean_non_numeric"]
    if val != val:
        return []
    if not (-1.0 <= val <= 1.0):
        return ["queue_imbalance_mean_out_of_signed_unit"]
    return []


def northset_microprice_weight_balance_honesty_errors(blob: object) -> list[str]:
    """Northset-named entry point for mean microprice weight-balance honesty.

    Identical contract to :func:`mean_microprice_weight_balance_honesty_errors`
    (∈ [0, 1] when finite; missing / NaN → skip). Research diagnostic only.
    """
    return mean_microprice_weight_balance_honesty_errors(blob)


def northset_spread_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ``mean_spread_bps ≈ 2 * mean_half_spread_bps`` when both finite.

    ``half_spread_bps = 1e4 * half_spread / mid`` and ``spread_bps = 2 * half_spread_bps``
    by construction; a receipt violating that is inconsistent. Missing /
    non-finite either side → skip. Research diagnostic only; never live Sharpe.
    """
    pair = _finite_pair(blob, "mean_spread_bps", "mean_half_spread_bps")
    if pair is None:
        return []
    spread_bps, half_bps = pair
    if not math.isclose(
        spread_bps,
        2.0 * half_bps,
        rel_tol=_NORTHSET_SPREAD_REL_TOL,
        abs_tol=_NORTHSET_SPREAD_ABS_TOL,
    ):
        return ["mean_spread_bps_not_double_mean_half_spread_bps"]
    return []


def northset_spread_receipt_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: ``mean_effective_spread ≈ mean_quoted_spread`` when both finite.

    Book ``effective_spread`` is an alias of the quoted spread
    (best_ask − best_bid); the candle diagnostic is ``mean_close_mid_abs_rel``
    only, never a second "effective" spread. Missing / non-finite either side
    → skip (thin external book panels stamp NaN). Research diagnostic only;
    never live Sharpe / promotion.
    """
    pair = _finite_pair(blob, "mean_effective_spread", "mean_quoted_spread")
    if pair is None:
        return []
    effective, quoted = pair
    if not math.isclose(
        effective,
        quoted,
        rel_tol=_NORTHSET_SPREAD_REL_TOL,
        abs_tol=_NORTHSET_SPREAD_ABS_TOL,
    ):
        return ["mean_effective_spread_diverges_from_mean_quoted_spread"]
    return []


def northset_queue_priority_le_size_concentration_honesty_errors(blob: object) -> list[str]:
    """Soft-verify queue priority means do not exceed same-side size concentration.

    Book identity: queue_priority ≤ size_concentration_top on each side when both
    finite (touch share of depth ≤ top-level concentration). Receipt means:

    - ``mean_queue_priority_proxy`` ≤ ``mean_bid_size_concentration_top``
    - ``mean_ask_queue_priority_proxy`` ≤ ``mean_ask_size_concentration_top``

    Skip pairs where either key absent/non-finite. Research diagnostic only;
    never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    pairs = (
        (
            "mean_queue_priority_proxy",
            "mean_bid_size_concentration_top",
            "mean_queue_priority_proxy_gt_mean_bid_size_concentration_top",
        ),
        (
            "mean_ask_queue_priority_proxy",
            "mean_ask_size_concentration_top",
            "mean_ask_queue_priority_proxy_gt_mean_ask_size_concentration_top",
        ),
    )
    for q_key, c_key, err in pairs:
        if q_key not in blob or c_key not in blob:
            continue
        try:
            q = float(blob.get(q_key))  # type: ignore[arg-type]
            c = float(blob.get(c_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if q != q or c != c or abs(q) == float("inf") or abs(c) == float("inf"):
            continue
        if q > c + 1e-9:
            errs.append(err)
    return errs


def northset_queue_priority_bid_ask_pair_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ask queue priority is a separate companion from bid/proxy.

    When ``mean_queue_priority_proxy`` is present and finite, require
    ``mean_ask_queue_priority_proxy`` also present (missing ask = dishonest
    collapse). Both ∈ [0, 1] when finite (delegates bounds to
    mean_queue_priority_honesty_errors). Never force equality between bid and
    ask means (Jensen / side asymmetry OK). Research diagnostic only.
    Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_queue_priority_proxy" not in blob:
        return []
    try:
        bid = float(blob.get("mean_queue_priority_proxy"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if bid != bid or abs(bid) == float("inf"):
        return []
    if "mean_ask_queue_priority_proxy" not in blob:
        return ["mean_ask_queue_priority_proxy_missing_while_bid_proxy_finite"]
    ask = blob.get("mean_ask_queue_priority_proxy")
    if not _finite_scalar(ask):
        # Present but non-finite is inconsistent, not a skip: the bid proxy is
        # finite, so the ask companion must be finite too.
        return ["mean_ask_queue_priority_proxy_nan_while_bid_proxy_finite"]
    return []


__all__ = [
    "amihud_mean_honesty_errors",
    "book_age_seconds_honesty_errors",
    "book_hypothesis_eligible_honesty_errors",
    "book_uncrossed_rate_honesty_errors",
    "close_location_value_clv_alias_identity_honesty_errors",
    "depth_shape_finite_rate_honesty_errors",
    "gap_finite_rate_honesty_errors",
    "join_coverage_honesty_errors",
    "mean_book_age_seconds_honesty_errors",
    "mean_close_mid_abs_rel_honesty_errors",
    "mean_depth_imbalance_abs_honesty_errors",
    "mean_depth_imbalance_honesty_errors",
    "mean_imbalance_top_honesty_errors",
    "mean_microprice_minus_mid_honesty_errors",
    "mean_microprice_weight_balance_honesty_errors",
    "mean_notional_imbalance_honesty_errors",
    "mean_queue_priority_honesty_errors",
    "mean_tob_notional_share_honesty_errors",
    "mean_tob_size_share_honesty_errors",
    "northset_book_shape_finite_rates_honesty_errors",
    "northset_component_sources_honesty_errors",
    "northset_depth_honesty_errors",
    "northset_depth_notional_spread_over_mid_honesty_errors",
    "northset_dm_park_honesty_errors",
    "northset_family_book_source_honesty_errors",
    "northset_half_spread_honesty_errors",
    "northset_impact_proxy_warning_honesty_errors",
    "northset_log_size_slope_honesty_errors",
    "northset_metrics_required_finite_ok_rates_honesty_errors",
    "northset_metrics_required_keys_finite_when_present_honesty_errors",
    "northset_microprice_weight_balance_honesty_errors",
    "northset_n_bars_scored_honesty_errors",
    "northset_overnight_rv_semi_honesty_errors",
    "northset_price_return_basis_honesty_errors",
    "northset_price_slope_tick_top_levels_honesty_errors",
    "northset_product_stamp_honesty_errors",
    "northset_qlike_means_honesty_errors",
    "northset_queue_imbalance_mean_alias_honesty_errors",
    "northset_queue_priority_bid_ask_pair_honesty_errors",
    "northset_queue_priority_le_size_concentration_honesty_errors",
    "northset_range_spread_honesty_errors",
    "northset_receipt_bool_flags_honesty_errors",
    "northset_receipt_dgp_data_source_honesty_errors",
    "northset_receipt_string_enum_honesty_errors",
    "northset_shape_columns_ensured_book_panel_path_honesty_errors",
    "northset_shape_columns_ensured_rates_honesty_errors",
    "northset_spread_bps_honesty_errors",
    "northset_spread_means_honesty_errors",
    "northset_spread_receipt_honesty_errors",
    "northset_top_level_claim_honesty_errors",
    "ohlc_identity_rate_honesty_errors",
    "robinhood_plus_claim_honesty_errors",
    "size_concentration_top_honesty_errors",
    "structure_finite_rate_honesty_errors",
    "vpin_mean_honesty_errors",
]
