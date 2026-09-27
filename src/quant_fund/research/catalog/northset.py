"""Northset receipt honesty checkers (``northset_*_honesty_errors``)."""

from __future__ import annotations

import math

from ._helpers import (
    _NORTHSET_CORWIN_SCHULTZ_PAIR_SCOPE_ALLOWED,
    _NORTHSET_IMPACT_ESTIMATOR_SCOPE_ALLOWED,
    _NORTHSET_OVERNIGHT_GAP_METHOD_ALLOWED,
    _NORTHSET_PRICE_BASIS_ALLOWED,
    _NORTHSET_RETURN_BASIS_ALLOWED,
    _NORTHSET_SPREAD_ABS_TOL,
    _NORTHSET_SPREAD_REL_TOL,
    _NORTHSET_SWEEP_EVIDENCE_SCOPE_ALLOWED,
    _NORTHSET_TWO_WAY_INFERENCE_INDEX_ALLOWED,
    _NORTHSET_VPIN_METHOD_ALLOWED,
    _NORTHSET_YANG_ZHANG_QLIKE_SCOPE_ALLOWED,
    _finite_pair,
    _finite_scalar,
    _ic_pack_honesty_errors,
)
from .session import (
    mean_microprice_weight_balance_honesty_errors,
)


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


def northset_shape_and_session_l2_floors_honesty_errors(blob: object) -> list[str]:
    """Soft-verify optional shape floors + session_l2_identity_floor ∈ [0, 1].

    When a floor key is present and not None/NaN, it must be finite and in [0, 1].
    Keys: depth_shape / concentration_top / queue_priority / side_notional /
    tob_size_share *_finite_floor, session_l2_identity_floor, and
    ``book_join_coverage_floor`` (stamped join companion).
    session_l2_identity_gate when present must be "enforced" or "skipped".
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    floor_keys = (
        "depth_shape_finite_floor",
        "concentration_top_finite_floor",
        "queue_priority_finite_floor",
        "side_notional_finite_floor",
        "tob_size_share_finite_floor",
        "session_l2_identity_floor",
        "book_join_coverage_floor",
    )
    for key in floor_keys:
        if key not in blob:
            continue
        val = blob.get(key)
        if val is None:
            continue
        try:
            x = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    if "session_l2_identity_gate" in blob:
        gate = blob.get("session_l2_identity_gate")
        if gate not in ("enforced", "skipped"):
            errs.append("session_l2_identity_gate_invalid")
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
            x = float(raw)  # type: ignore[arg-type]
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


def northset_queue_sweep_ofi_honesty_errors(blob: object) -> list[str]:
    """Soft-verify queue_imbalance_mean, sweep_any_rate, ofi_mean_ic when present.

    - queue_imbalance_mean ∈ [-1, 1] when finite
    - sweep_any_rate ∈ [0, 1] when finite
    - ofi_mean_ic finite when present (not ±inf); NaN skipped
    Research diagnostic only; never live Sharpe.
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

    qi = _finite("queue_imbalance_mean")
    if qi is not None and not (-1.0 <= qi <= 1.0):
        errs.append("queue_imbalance_mean_out_of_unit_interval")

    sar = _finite("sweep_any_rate")
    if sar is not None and not (0.0 <= sar <= 1.0):
        errs.append("sweep_any_rate_out_of_unit_interval")

    # ofi_mean_ic: finite-when-present only (IC can be negative)
    _ = _finite("ofi_mean_ic")
    return errs


def northset_include_kyle_ofi_nest_presence_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``include_kyle_ofi`` ↔ nested ``kyle_ofi`` presence.

    Stamp contract (``bench_northset``):
    - ``include_kyle_ofi is True`` ⇒ ``kyle_ofi`` is a dict with ``research_only is True``
    - ``include_kyle_ofi is False`` ⇒ ``kyle_ofi`` key absent

    Does not invent nest internals / does not overwrite kyle_ofi.py — presence only.
    Skip when flag absent. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "northset"):
        return []
    if "include_kyle_ofi" not in blob:
        return []
    flag = blob.get("include_kyle_ofi")
    if type(flag) is not bool:
        return []  # bool-flags helper owns type

    has_nest = "kyle_ofi" in blob
    nest = blob.get("kyle_ofi") if has_nest else None

    if flag is True:
        if not isinstance(nest, dict):
            return ["kyle_ofi_nest_missing_while_include_kyle_ofi_true"]
        if nest.get("research_only") is not True:
            return ["kyle_ofi_nest_research_only_not_true_while_include_kyle_ofi_true"]
        return []
    # flag False
    if has_nest:
        return ["kyle_ofi_nest_present_while_include_kyle_ofi_false"]
    return []


def northset_sweep_control_sample_adequate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``sweep_*_control_sample_adequate True`` ⇒ companions honest.

    For each of ``sweep_reject`` / ``sweep_follow``:
    - adequate True ⇒ ``*_control_n_dates`` ≥ 1 when present, ``*_control_diff_p``
      ∈ [0, 1] when finite, ``*_control_diff_t`` finite when present (not ±inf)
    - adequate False / absent → skip numeric companions (bool helper owns type)
    Does not invent kyle nest / METRICS_*. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for prefix in ("sweep_reject", "sweep_follow"):
        flag_key = f"{prefix}_control_sample_adequate"
        if flag_key not in blob:
            continue
        if blob.get(flag_key) is not True:
            continue
        n_key = f"{prefix}_control_n_dates"
        p_key = f"{prefix}_control_diff_p"
        t_key = f"{prefix}_control_diff_t"
        if n_key in blob:
            try:
                n = float(blob.get(n_key))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append(f"{n_key}_non_numeric_while_sample_adequate")
            else:
                if n != n or abs(n) == float("inf") or n < 1.0:
                    errs.append(f"{n_key}_lt_one_while_sample_adequate")
        if p_key in blob:
            try:
                p = float(blob.get(p_key))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append(f"{p_key}_non_numeric_while_sample_adequate")
            else:
                if p != p or abs(p) == float("inf") or not (0.0 <= p <= 1.0):
                    errs.append(f"{p_key}_invalid_while_sample_adequate")
        if t_key in blob:
            try:
                tv = float(blob.get(t_key))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append(f"{t_key}_non_numeric_while_sample_adequate")
            else:
                if tv != tv:
                    pass  # NaN skip — inadequate companions may leave NaN even if flag wrong
                elif abs(tv) == float("inf"):
                    errs.append(f"{t_key}_non_finite_while_sample_adequate")
    return errs


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
            x = float(raw)  # type: ignore[arg-type]
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
            x = float(raw)  # type: ignore[arg-type]
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


def northset_fwd_ret_after_sweep_honesty_errors(blob: object) -> list[str]:
    """Soft-verify mean_fwd_ret_after_* sweep companions finite when present.

    Signed conditional forward returns — may be negative; only ±inf / non-numeric
    fail. Keys: high/low reclaim and high/low follow. NaN/absent skip.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "mean_fwd_ret_after_high_reclaim",
        "mean_fwd_ret_after_low_reclaim",
        "mean_fwd_ret_after_high_follow",
        "mean_fwd_ret_after_low_follow",
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


def northset_vpin_sweep_fold_honesty_errors(blob: object) -> list[str]:
    """Soft-verify vpin_p/t_ic, n_sweep_low, sweep_*_fold_positive_fraction.

    - vpin_p_ic / vpin_t_ic finite when present (signed OK; ±inf fail-closed)
    - n_sweep_low ≥ 0 when present
    - sweep_reject_fold_positive_fraction / sweep_follow_fold_positive_fraction
      ∈ [0, 1] when finite
    NaN/absent skipped. Research diagnostic only; never live Sharpe.
    Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []

    for key in ("vpin_p_ic", "vpin_t_ic"):
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

    if "n_sweep_low" in blob:
        try:
            n = float(blob.get("n_sweep_low"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("n_sweep_low_non_numeric")
        else:
            if n != n or abs(n) == float("inf"):
                errs.append("n_sweep_low_non_finite")
            elif n < 0.0:
                errs.append("n_sweep_low_negative")

    for key in (
        "sweep_reject_fold_positive_fraction",
        "sweep_follow_fold_positive_fraction",
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


def northset_sweep_evidence_blob_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``sweep_evidence`` research-diagnostic contract.

    When ``sweep_evidence`` dict present:
    - ``research_only is True`` when key present
    - ``horizons`` non-empty list of positive ints when present
    - each event_studies row: ``sample_adequate`` bool; if False then ``reject_fdr`` is not True
    - ``n_events`` / ``n_dates`` ≥ 0 when finite
    Never promotes inadequate samples. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    ev = blob.get("sweep_evidence")
    if not isinstance(ev, dict):
        return []
    errs: list[str] = []
    if "research_only" in ev and ev.get("research_only") is not True:
        errs.append("sweep_evidence_research_only_missing_or_false")
    idx = ev.get("inference_index")
    if idx is not None and idx != "calendar_including_idle_zeros":
        errs.append("sweep_evidence_inference_index_invalid")
    if "horizons" in ev:
        hz = ev.get("horizons")
        if not isinstance(hz, list) or not hz:
            errs.append("sweep_evidence_horizons_invalid")
        else:
            for h in hz:
                try:
                    hf = float(h)
                except (TypeError, ValueError):
                    errs.append("sweep_evidence_horizons_invalid")
                    break
                # Reject non-finite / non-integral values before int() (inf raises).
                if hf != hf or abs(hf) == float("inf") or hf <= 0.0 or hf != int(hf):
                    errs.append("sweep_evidence_horizons_invalid")
                    break
    studies = ev.get("event_studies")
    if isinstance(studies, list):
        for i, row in enumerate(studies):
            if not isinstance(row, dict):
                errs.append(f"sweep_evidence_event_studies_{i}_not_dict")
                continue
            if "sample_adequate" in row and type(row.get("sample_adequate")) is not bool:
                errs.append(f"sweep_evidence_event_studies_{i}_sample_adequate_not_bool")
            if "reject_fdr" in row and type(row.get("reject_fdr")) is not bool:
                # Truthy non-bools (1, "true") must not bypass the inadequate-
                # sample promotion guard below.
                errs.append(f"sweep_evidence_event_studies_{i}_reject_fdr_not_bool")
            if row.get("sample_adequate") is False and row.get("reject_fdr") is True:
                errs.append(f"sweep_evidence_event_studies_{i}_reject_fdr_with_inadequate_sample")
            for nk in ("n_events", "n_dates"):
                if nk not in row:
                    continue
                try:
                    n = float(row.get(nk))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    errs.append(f"sweep_evidence_event_studies_{i}_{nk}_non_numeric")
                    continue
                if n == n and abs(n) != float("inf") and n < 0.0:
                    errs.append(f"sweep_evidence_event_studies_{i}_{nk}_negative")
    return errs


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


def northset_sweep_evidence_scope_honesty_errors(blob: object) -> list[str]:
    """Soft-verify sweep_evidence_scope / impact_estimator_scope when present.

    Stamp contract (``bench_northset`` / DATA_CONTRACTS):
    - ``sweep_evidence_scope`` ∈ {synthetic, empirical_adjusted, fixture_raw_unadjusted}
    - when ``data_source == "SYNTHETIC"`` and scope present → scope must be ``synthetic``
    - when ``price_basis == "split_adjusted"`` and ``data_source`` is not SYNTHETIC
      and scope present → scope must be ``empirical_adjusted``
    - when ``price_basis == "raw_fixture_opt_out"`` and ``data_source`` is not SYNTHETIC
      and scope present → scope must be ``fixture_raw_unadjusted``
    - ``impact_estimator_scope`` ∈ {per_security_equal_weight} when present (stamp)
    - ``include_kyle_ofi`` bool when present (legacy companion check)

    Research diagnostic only; never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    scope = blob.get("sweep_evidence_scope") if "sweep_evidence_scope" in blob else None
    if scope is not None:
        # isinstance guard: unhashable JSON values would raise TypeError on
        # set membership instead of failing closed.
        if not isinstance(scope, str) or scope not in _NORTHSET_SWEEP_EVIDENCE_SCOPE_ALLOWED:
            errs.append("sweep_evidence_scope_invalid")
        else:
            data_source = blob.get("data_source")
            price_basis = blob.get("price_basis")
            if isinstance(data_source, str) and data_source == "SYNTHETIC":
                if scope != "synthetic":
                    errs.append("sweep_evidence_scope_not_synthetic_despite_SYNTHETIC_data_source")
            elif isinstance(price_basis, str):
                if price_basis == "split_adjusted" and scope != "empirical_adjusted":
                    errs.append(
                        "sweep_evidence_scope_not_empirical_adjusted_despite_split_adjusted"
                    )
                elif price_basis == "raw_fixture_opt_out" and scope != "fixture_raw_unadjusted":
                    errs.append("sweep_evidence_scope_not_fixture_raw_despite_raw_fixture_opt_out")
    if "impact_estimator_scope" in blob:
        ies = blob.get("impact_estimator_scope")
        # Stamp contract: bench_northset hardcodes per_security_equal_weight.
        # isinstance guard: unhashable JSON values would raise TypeError.
        if not isinstance(ies, str) or ies not in _NORTHSET_IMPACT_ESTIMATOR_SCOPE_ALLOWED:
            errs.append("impact_estimator_scope_invalid")
    if "include_kyle_ofi" in blob and type(blob.get("include_kyle_ofi")) is not bool:
        errs.append("include_kyle_ofi_not_bool")
    if blob.get("family") == "northset":
        yz_scope = blob.get("yang_zhang_qlike_scope")
        yz_qlike = blob.get("yang_zhang_qlike_vs_cc")
        yz_finite = False
        try:
            yz_f = float(yz_qlike)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            yz_f = float("nan")
        yz_finite = yz_f == yz_f and abs(yz_f) != float("inf")
        if (yz_finite or "yang_zhang_qlike_scope" in blob) and (
            not isinstance(yz_scope, str)
            or yz_scope not in _NORTHSET_YANG_ZHANG_QLIKE_SCOPE_ALLOWED
        ):
            errs.append("yang_zhang_qlike_scope_invalid")
        vpin_method = blob.get("vpin_method")
        vpin_val = blob.get("vpin_mean")
        vpin_finite = False
        try:
            vp = float(vpin_val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            vp = float("nan")
        vpin_finite = vp == vp and abs(vp) != float("inf")
        if (vpin_finite or "vpin_method" in blob) and (
            not isinstance(vpin_method, str) or vpin_method not in _NORTHSET_VPIN_METHOD_ALLOWED
        ):
            errs.append("vpin_method_invalid")
        cs_scope = blob.get("corwin_schultz_pair_scope")
        cs_spread = blob.get("corwin_schultz_spread")
        cs_finite = False
        try:
            cs_f = float(cs_spread)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            cs_f = float("nan")
        cs_finite = cs_f == cs_f and abs(cs_f) != float("inf")
        if (cs_finite or "corwin_schultz_pair_scope" in blob) and (
            not isinstance(cs_scope, str)
            or cs_scope not in _NORTHSET_CORWIN_SCHULTZ_PAIR_SCOPE_ALLOWED
        ):
            errs.append("corwin_schultz_pair_scope_invalid")
        gap_method = blob.get("sweep_overnight_gap_method")
        gap_p = blob.get("sweep_follow_overnight_gap_p")
        gap_finite = False
        try:
            gap_f = float(gap_p)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            gap_f = float("nan")
        gap_finite = gap_f == gap_f and abs(gap_f) != float("inf")
        if (gap_finite or "sweep_overnight_gap_method" in blob) and (
            not isinstance(gap_method, str)
            or gap_method not in _NORTHSET_OVERNIGHT_GAP_METHOD_ALLOWED
        ):
            errs.append("sweep_overnight_gap_method_invalid")
        two_way_idx = blob.get("sweep_two_way_inference_index")
        two_way_p = blob.get("sweep_follow_two_way_cluster_p")
        two_way_finite = False
        try:
            tw_f = float(two_way_p)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            tw_f = float("nan")
        two_way_finite = tw_f == tw_f and abs(tw_f) != float("inf")
        if (two_way_finite or "sweep_two_way_inference_index" in blob) and (
            not isinstance(two_way_idx, str)
            or two_way_idx not in _NORTHSET_TWO_WAY_INFERENCE_INDEX_ALLOWED
        ):
            errs.append("sweep_two_way_inference_index_invalid")
    return errs


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


def northset_session_close_ic_packs_honesty_errors(blob: object) -> list[str]:
    """Soft-verify session_close_* IC packs (mid/spread/imbalance/depths/micro).

    Distinct from session snaps / n_session counts. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    stems = (
        "session_close_mid",
        "session_close_spread_bps",
        "session_close_imbalance",
        "session_close_bid_depth",
        "session_close_ask_depth",
        "session_close_micro_bps",
        "session_spread_bps_mean",
        "session_imbalance_mean",
    )
    for stem in stems:
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


def northset_sweep_signed_ic_packs_honesty_errors(blob: object) -> list[str]:
    """Soft-verify sweep_*_signed IC packs; never equate reject/follow/depth."""
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for stem in (
        "sweep_reject_signed",
        "sweep_follow_signed",
        "sweep_depth_signed",
    ):
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


def northset_candle_body_ret_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_body_ret_* IC pack on northset."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="candle_body_ret_mean_ic",
        rank_ic_key="candle_body_ret_mean_rank_ic",
        t_key="candle_body_ret_t_ic",
        p_key="candle_body_ret_p_ic",
        n_key="candle_body_ret_n_dates",
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


def northset_session_book_vpin_ic_pack_honesty_errors(blob: object) -> list[str]:
    """Soft-verify session_book_vpin_* IC pack; ≠ daily vpin_* IC."""
    if not isinstance(blob, dict):
        return []
    return _ic_pack_honesty_errors(
        blob,
        mean_ic_key="session_book_vpin_mean_ic",
        rank_ic_key="session_book_vpin_mean_rank_ic",
        t_key="session_book_vpin_t_ic",
        p_key="session_book_vpin_p_ic",
        n_key="session_book_vpin_n_dates",
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


def northset_lag_corr_and_sweep_count_honesty_errors(blob: object) -> list[str]:
    """Soft-verify mid_lag1_corr, ofi_lag1_corr, n_sweep_high when present.

    - mid_lag1_corr ∈ [-1, 1] when finite (panel mean of per-name lag-1 corr)
    - ofi_lag1_corr ∈ [-1, 1] when finite
    - n_sweep_high ≥ 0 when present (integer count; non-numeric / ±inf dishonest)
    - mid_lag1_n_securities / ofi_lag1_n_securities ≥ 0 when finite
    NaN skipped for corr keys. Research diagnostic only; never live Sharpe.
    Does not touch kyle_* / book_metrics METRICS_* keys.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []

    for key in ("mid_lag1_corr", "ofi_lag1_corr"):
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
        if not (-1.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")

    if "n_sweep_high" in blob:
        try:
            n = float(blob.get("n_sweep_high"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("n_sweep_high_non_numeric")
        else:
            if n != n or abs(n) == float("inf"):
                errs.append("n_sweep_high_non_finite")
            elif n < 0.0:
                errs.append("n_sweep_high_negative")

    for key in ("mid_lag1_n_securities", "ofi_lag1_n_securities"):
        if key not in blob:
            continue
        try:
            n = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            continue
        if n != n:
            continue
        if abs(n) == float("inf"):
            errs.append(f"{key}_non_finite")
        elif n < 0.0:
            errs.append(f"{key}_negative")
        # When matching corr is finite, n_securities should be present (already is)
    return errs


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


def northset_session_l2_enforced_identity_rates_present_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: when session L2 identity gate is enforced, rates are present.

    If ``session_l2_identity_gate == "enforced"``, require the four session
    identity rate keys. Distinct from daily ``ohlc_identity_rate``.
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("session_l2_identity_gate") != "enforced":
        return []
    errs: list[str] = []
    for key in (
        "session_ohlc_identity_rate",
        "session_reconstructs_daily_rate",
        "session_volume_conservation_rate",
        "session_chain_rate",
    ):
        if key not in blob:
            errs.append(f"{key}_missing_while_session_l2_identity_gate_enforced")
    return errs


def northset_session_identity_rates_honesty_errors(blob: object) -> list[str]:
    """Soft-verify session L2 identity rates ∈ [0, 1] when finite.

    Keys: session_ohlc_identity_rate, session_reconstructs_daily_rate,
    session_volume_conservation_rate, session_chain_rate. NaN skipped.
    Research diagnostic only;
    never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "session_ohlc_identity_rate",
        "session_reconstructs_daily_rate",
        "session_volume_conservation_rate",
        "session_chain_rate",
    ):
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x != x:
            continue
        if not (0.0 <= x <= 1.0):
            errs.append(f"{key}_out_of_unit_interval")
    return errs


def northset_session_imbalance_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_imbalance_mean ∈ [-1, 1] when finite.

    Session-L2 path mean of ``session_imbalance_mean`` — not daily
    ``imbalance_top`` / ``queue_imbalance``. Skip if missing or non-finite.
    Research diagnostic only; never live Sharpe / promotion.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_imbalance_mean")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if not (-1.0 <= x <= 1.0):
        return ["mean_session_imbalance_mean_out_of_unit_interval"]
    return []


def northset_session_close_micro_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_close_micro_bps finite when present (not ±inf).

    Last-snap session micro — not daily ``microprice_minus_mid_bps`` mean.
    NaN (absent / session L2 off) is fine. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_close_micro_bps")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if abs(x) == float("inf"):
        return ["mean_session_close_micro_bps_non_finite"]
    return []


def northset_session_close_depth_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_close_bid/ask_depth ≥ 0 when finite.

    Last-snap depths — not daily ``bid_depth`` / ``ask_depth`` means.
    NaN (absent / session L2 off) is fine. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    errors: list[str] = []
    for key in ("mean_session_close_bid_depth", "mean_session_close_ask_depth"):
        val = blob.get(key)
        try:
            x = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x != x:  # NaN
            continue
        if x < 0.0 or abs(x) == float("inf"):
            errors.append(f"{key}_negative_or_non_finite")
    return errors


def northset_session_imbalance_std_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_imbalance_std ≥ 0 when finite.

    Path dispersion — not path mean / last-snap close imbalance.
    NaN (absent / session L2 off) is fine. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_imbalance_std")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if x < 0.0 or abs(x) == float("inf"):
        return ["mean_session_imbalance_std_negative_or_non_finite"]
    return []


def northset_session_close_imbalance_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_close_imbalance ∈ [-1, 1] when finite.

    Last-snap close imbalance — not path mean / daily imbalance_top.
    NaN (absent / session L2 off) is fine. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_close_imbalance")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if not (-1.0 <= x <= 1.0):
        return ["mean_session_close_imbalance_out_of_unit_interval"]
    return []


def northset_session_close_mid_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_close_mid > 0 when finite.

    Last-snap mid — not daily mid/close. NaN skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_close_mid")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if not (x > 0.0) or abs(x) == float("inf"):
        return ["mean_session_close_mid_non_positive_or_non_finite"]
    return []


def northset_session_close_spread_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_close_spread_bps ≥ 0 when finite.

    Last-snap session close spread — not path ``mean_session_spread_bps_mean`` /
    daily ``mean_spread_bps``. Uses ≥0 (not >0): vendor_book_map can emit
    ``spread_bps=0`` when mid≤0 / locked book (``otherwise(0.0)``); synthetic
    clips to [1, 80] but honesty must not reject a real zero. NaN skipped.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_close_spread_bps")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if x < 0.0 or abs(x) == float("inf"):
        return ["mean_session_close_spread_bps_negative_or_non_finite"]
    return []


def northset_session_spread_bps_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_spread_bps_mean ≥ 0 when finite.

    Session-L2 path mean of ``session_spread_bps_mean`` — not last-snap close /
    daily ``mean_spread_bps``. ≥0 matches book_metrics / vendor zero-spread.
    NaN skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_spread_bps_mean")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if x < 0.0 or abs(x) == float("inf"):
        return ["mean_session_spread_bps_mean_negative_or_non_finite"]
    return []


def northset_session_book_snaps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_book_snaps > 0 when finite.

    Session-L2 snap count mean — skip NaN (session L2 off). Research diagnostic
    only; never live Sharpe / promotion.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_book_snaps")
    if val is None:
        return []
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        # Present-but-malformed must not read as absent/NaN.
        return ["mean_session_book_snaps_non_numeric"]
    x = float(val)
    if x != x:  # NaN
        return []
    if not (x > 0.0) or abs(x) == float("inf"):
        return ["mean_session_book_snaps_non_positive_or_non_finite"]
    return []


def northset_session_ofi_sum_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify session_ofi_sum_* IC companions when present.

    - ``session_ofi_sum_mean_ic`` / ``session_ofi_sum_mean_rank_ic`` / ``_t_ic`` finite
    - ``session_ofi_sum_p_ic`` ∈ [0, 1] when finite
    - ``session_ofi_sum_n_dates`` ≥ 0 when finite
    Never equate to daily ``ofi_mean_ic`` / ``ofi_p_ic`` (different feature).
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in (
        "session_ofi_sum_mean_ic",
        "session_ofi_sum_mean_rank_ic",
        "session_ofi_sum_t_ic",
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
    if "session_ofi_sum_p_ic" in blob:
        try:
            p = float(blob.get("session_ofi_sum_p_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("session_ofi_sum_p_ic_non_numeric")
        else:
            if p == p and abs(p) != float("inf") and not (0.0 <= p <= 1.0):
                errs.append("session_ofi_sum_p_ic_out_of_unit_interval")
            elif p == p and abs(p) == float("inf"):
                errs.append("session_ofi_sum_p_ic_non_finite_fail_closed")
    if "session_ofi_sum_n_dates" in blob:
        try:
            n = float(blob.get("session_ofi_sum_n_dates"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("session_ofi_sum_n_dates_non_numeric")
        else:
            if n == n and abs(n) != float("inf") and n < 0.0:
                errs.append("session_ofi_sum_n_dates_negative")
            elif n == n and abs(n) == float("inf"):
                errs.append("session_ofi_sum_n_dates_non_finite")
    return errs


def northset_session_ofi_sum_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_ofi_sum_mean finite when present (not ±inf).

    Session-path mean of ``session_ofi_sum`` — signed OFI sum is unbounded, so
    no unit-interval claim; only reject ±inf. NaN (session L2 off) skipped.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("session_ofi_sum_mean")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if abs(x) == float("inf"):
        return ["session_ofi_sum_mean_non_finite"]
    return []


def northset_session_book_vpin_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_book_vpin_mean ∈ [0, 1] when finite.

    Session book VPIN proxy is ``|ofi_sum| / ofi_abs_sum`` (synthetic_lob) →
    unit interval by construction. NaN skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("session_book_vpin_mean")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:  # NaN
        return []
    if not (0.0 <= x <= 1.0):
        return ["session_book_vpin_mean_out_of_unit_interval"]
    return []


def northset_session_ofi_abs_sum_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: mean_session_ofi_abs_sum ≥ 0 when finite.

    Path mean of ``session_ofi_abs_sum`` (VPIN denominator companion) — not
    ``|session_ofi_sum_mean|``. NaN skipped. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("mean_session_ofi_abs_sum")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if x < 0.0 or abs(x) == float("inf"):
        return ["mean_session_ofi_abs_sum_negative_or_non_finite"]
    return []


def northset_session_ofi_abs_dominates_sum_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: |session_ofi_sum_mean| ≤ mean_session_ofi_abs_sum when both finite.

    Path identity for VPIN denom companion. Never equate ofi_abs mean to
    |ofi_sum_mean| (Jensen) as a claimed equality. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    try:
        s = float(blob.get("session_ofi_sum_mean"))  # type: ignore[arg-type]
        a = float(blob.get("mean_session_ofi_abs_sum"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if s != s or a != a:
        return []
    if abs(s) == float("inf") or abs(a) == float("inf"):
        return []
    if abs(s) > a + 1e-9:
        return ["session_ofi_abs_sum_less_than_abs_ofi_sum_mean"]
    return []


def northset_session_close_mid_micro_pair_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: when close_micro is finite, close_mid must also be finite.

    Companion keys from the same last-snap session aggregate. Both NaN (session
    L2 off) is fine; micro finite + mid NaN/±inf is dishonest pairing.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []

    def _parse(key: str) -> float | None:
        val = blob.get(key)
        try:
            x = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return x

    micro = _parse("mean_session_close_micro_bps")
    mid = _parse("mean_session_close_mid")
    if micro is None or micro != micro:  # missing/NaN micro → skip
        return []
    if abs(micro) == float("inf"):
        return []  # micro non-finite handled by micro helper
    # micro finite → mid must be finite (NaN/missing/±inf fail)
    if mid is None or mid != mid or abs(mid) == float("inf"):
        return ["mean_session_close_mid_nan_while_micro_finite"]
    return []


def northset_kyle_r2_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify always-on ``kyle_r2`` / ``kyle_ofi_r2`` ∈ [0, 1] when finite.

    Name-mean OLS R² from ``kyle_lambda`` helpers (signed_volume vs ofi paths).
    NaN/absent skip; ±inf fail-closed. Research diagnostic only; never live Sharpe.
    Always-on surface — not nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    for key in ("kyle_r2", "kyle_ofi_r2"):
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


def northset_sweep_fold_positive_rates_honesty_errors(blob: object) -> list[str]:
    """Soft-verify sweep fold-positive rates ∈ [0, 1] when finite.

    Explicit residual (also covered by ``*_fraction`` catch-all):
    ``sweep_min_fold_positive_fraction``, ``sweep_reject_fold_positive_fraction``,
    ``sweep_follow_fold_positive_fraction``. Min threshold must be in (0, 1]
    when finite (0 is not a meaningful fold floor). Off kyle invent.
    Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    keys = (
        "sweep_min_fold_positive_fraction",
        "sweep_reject_fold_positive_fraction",
        "sweep_follow_fold_positive_fraction",
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
            continue
        if key == "sweep_min_fold_positive_fraction" and not (0.0 < x <= 1.0):
            errs.append("sweep_min_fold_positive_fraction_not_positive_unit")
    return errs


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


def candle_spread_alias_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle spread alias identities + nonneg when stamped.

    - each of quoted / effective / half / spread_bps / half_bps ≥ 0 when finite
    - ``mean_quoted_spread ≈ mean_effective_spread`` when both finite
    - ``mean_half_spread ≈ 0.5 * mean_quoted_spread``
    - ``mean_spread_bps ≈ 2 * mean_half_spread_bps``
    - ``mean_spread_bps ≈ 1e4 * mean_spread_over_mid`` when both finite
      (book_metrics: spread_bps = 1e4 * spread/mid; spread_over_mid = spread/mid)
    Never invent missing keys. Research diagnostic only; never live Sharpe.
    Off kyle invent / IC↔gap mesh.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    for key in (
        "mean_quoted_spread",
        "mean_effective_spread",
        "mean_half_spread",
        "mean_half_spread_bps",
        "mean_spread_bps",
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
        if abs(x) == float("inf") or x < 0.0:
            errs.append(f"{key}_negative_or_non_finite")
    qe = _finite_pair(blob, "mean_quoted_spread", "mean_effective_spread")
    if qe is not None:
        quoted, effective = qe
        if not math.isclose(
            quoted,
            effective,
            rel_tol=_NORTHSET_SPREAD_REL_TOL,
            abs_tol=_NORTHSET_SPREAD_ABS_TOL,
        ):
            errs.append("mean_quoted_spread_not_equal_mean_effective_spread")
    errs.extend(northset_half_spread_honesty_errors(blob))
    errs.extend(northset_spread_bps_honesty_errors(blob))
    # spread_bps = 1e4 * spread_over_mid by construction (same mid)
    pair_bps_mid = _finite_pair(blob, "mean_spread_bps", "mean_spread_over_mid")
    if pair_bps_mid is not None:
        spread_bps, over_mid = pair_bps_mid
        if not math.isclose(
            spread_bps,
            1e4 * over_mid,
            rel_tol=_NORTHSET_SPREAD_REL_TOL,
            abs_tol=max(_NORTHSET_SPREAD_ABS_TOL, 1e-6),
        ):
            errs.append("mean_spread_bps_not_1e4_times_mean_spread_over_mid")
    return errs


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


def northset_structure_finite_rate_distinct_from_candle_honesty_errors(blob: object) -> list[str]:
    """Soft-verify northset ``structure_finite_rate`` is distinct from candle companions.

    Northset aggregates concentration_top / queue_priority / side_notional /
    tob_size_share finite rates — never candle ``finite_rate_*`` keys
    (microprice_minus_mid / size concentration). When any northset companion
    rate is present with ``structure_finite_rate``:

    - candle ``finite_rate_*`` keys must be absent (family mix = dishonest)
    - if all four companions + aggregate are finite, aggregate ≈ nanmean(companions)

    NaN/absent skip. Research diagnostic only; never live Sharpe. Off kyle / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    northset_companions = (
        "concentration_top_finite_rate",
        "queue_priority_finite_rate",
        "side_notional_finite_rate",
        "tob_size_share_finite_rate",
    )
    candle_companions = (
        "finite_rate_microprice_minus_mid",
        "finite_rate_bid_size_concentration_top",
        "finite_rate_ask_size_concentration_top",
    )
    has_northset = any(k in blob for k in northset_companions)
    if not has_northset or "structure_finite_rate" not in blob:
        return []
    errs: list[str] = []
    for key in candle_companions:
        if key in blob:
            errs.append("northset_structure_finite_rate_mixed_with_candle_finite_rate_companions")
            break
    vals: list[float] = []
    for key in northset_companions:
        if key not in blob:
            continue
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x != x or abs(x) == float("inf"):
            continue
        vals.append(x)
    try:
        agg = float(blob.get("structure_finite_rate"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return errs
    if agg != agg or abs(agg) == float("inf"):
        return errs
    if len(vals) == 4:
        import math

        expected = sum(vals) / 4.0
        if not math.isclose(agg, expected, rel_tol=1e-9, abs_tol=1e-12):
            errs.append("northset_structure_finite_rate_not_nanmean_of_companion_rates")
    return errs


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
