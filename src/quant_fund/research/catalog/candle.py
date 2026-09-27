"""candle_order_book receipt honesty checks.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

import math

from .primitives import _finite_pair, _ic_pack_honesty_errors
from .receipt import (
    _NORTHSET_SPREAD_ABS_TOL,
    _NORTHSET_SPREAD_REL_TOL,
    mean_microprice_minus_mid_honesty_errors,
    northset_half_spread_honesty_errors,
    northset_spread_bps_honesty_errors,
)


def candle_microprice_minus_mid_finite_pack_honesty_errors(blob: object) -> list[str]:
    """Pack soft-verify: candle/northset microprice−mid means finite when stamped.

    Delegates to :func:`mean_microprice_minus_mid_honesty_errors` (price + bps).
    Research diagnostic only; never live Sharpe. Off kyle invent.
    """
    return mean_microprice_minus_mid_honesty_errors(blob)


def candle_spread_bps_nonneg_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``mean_spread_bps`` ≥ 0 when finite (candle or northset stamp).

    Spread in bps cannot be negative. Complements half-spread identity helper
    (which only checks 2× half). NaN/absent skip. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_spread_bps" not in blob:
        return []
    try:
        x = float(blob.get("mean_spread_bps"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_spread_bps_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or x < 0.0:
        return ["mean_spread_bps_negative_or_non_finite"]
    return []


def candle_log_slopes_finite_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle bid/ask log size+price slopes finite when stamped.

    When scored/stamped: means must not be ±inf (NaN skip). Signed OK.
    Research diagnostic only; off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    errs: list[str] = []
    for key in (
        "mean_bid_log_size_slope",
        "mean_ask_log_size_slope",
        "mean_bid_log_price_slope",
        "mean_ask_log_price_slope",
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


def candle_log_tick_spacing_finite_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle ``mean_*_mean_log_tick_spacing`` finite when stamped.

    Log tick spacing may be negative (log of a small relative tick). Do **not**
    require ≥0 (northset price-slope helper's ≥0 contract is northset-scale).
    NaN skip; ±inf fail-closed. Research diagnostic only; off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    errs: list[str] = []
    for key in (
        "mean_bid_mean_log_tick_spacing",
        "mean_ask_mean_log_tick_spacing",
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


def mean_candle_dir_x_imbalance_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``mean_candle_dir_x_imbalance`` ∈ [-1, 1] when finite.

    FEATURE_COLS interaction of ternary direction and imbalance — mean must lie
    in signed unit. NaN/absent skip; ±inf fail-closed. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    if "mean_candle_dir_x_imbalance" not in blob:
        return []
    try:
        x = float(blob.get("mean_candle_dir_x_imbalance"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_candle_dir_x_imbalance_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (-1.0 <= x <= 1.0):
        return ["mean_candle_dir_x_imbalance_out_of_signed_unit"]
    return []


def candle_structure_ic_implies_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify structure LOB IC⇒mean pairs on candle_order_book.

    Covers imbalance_top, queue_imbalance, tob_size_share, bid/ask size concentration.
    Never equate IC to mean. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    specs = (
        ("ic_imbalance_top", "mean_imbalance_top", -1.0, 1.0),
        ("ic_queue_imbalance", "mean_queue_imbalance", -1.0, 1.0),
        ("ic_tob_size_share", "mean_tob_size_share", 0.0, 1.0),
        ("ic_bid_size_concentration_top", "mean_bid_size_concentration_top", 0.0, 1.0),
        ("ic_ask_size_concentration_top", "mean_ask_size_concentration_top", 0.0, 1.0),
    )
    for ic_key, mean_key, lo, hi in specs:
        if ic_key not in blob:
            continue
        if mean_key not in blob:
            errs.append(f"{mean_key}_missing_while_{ic_key}_scored")
            continue
        try:
            m = float(blob.get(mean_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{mean_key}_non_numeric")
            continue
        if m != m:
            errs.append(f"{mean_key}_nan_while_{ic_key}_scored")
        elif abs(m) == float("inf") or not (lo <= m <= hi):
            errs.append(f"{mean_key}_out_of_unit_interval")
    return errs


def candle_depth_imbalance_ic_implies_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify depth imbalance IC⇒mean pairs on candle_order_book.

    - ``ic_imbalance_depth`` ⇒ ``mean_depth_imbalance`` ∈ [-1, 1]
    - ``ic_depth_imbalance_abs`` ⇒ ``mean_depth_imbalance_abs`` ∈ [0, 1]
    Never equate IC to mean; never equate abs to signed. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    if "ic_imbalance_depth" in blob:
        if "mean_depth_imbalance" not in blob:
            errs.append("mean_depth_imbalance_missing_while_imbalance_depth_ic_scored")
        else:
            try:
                m = float(blob.get("mean_depth_imbalance"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append("mean_depth_imbalance_non_numeric")
            else:
                if m != m:
                    errs.append("mean_depth_imbalance_nan_while_imbalance_depth_ic_scored")
                elif abs(m) == float("inf") or not (-1.0 <= m <= 1.0):
                    errs.append("mean_depth_imbalance_out_of_unit_interval")
    if "ic_depth_imbalance_abs" in blob:
        if "mean_depth_imbalance_abs" not in blob:
            errs.append("mean_depth_imbalance_abs_missing_while_depth_imbalance_abs_ic_scored")
        else:
            try:
                m = float(blob.get("mean_depth_imbalance_abs"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append("mean_depth_imbalance_abs_non_numeric")
            else:
                if m != m:
                    errs.append("mean_depth_imbalance_abs_nan_while_depth_imbalance_abs_ic_scored")
                elif abs(m) == float("inf") or not (0.0 <= m <= 1.0):
                    errs.append("mean_depth_imbalance_abs_out_of_unit_interval")
    return errs


def candle_order_book_claim_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_order_book research_only + claim markers.

    When ``family == "candle_order_book"`` and either key present:
    ``research_only is True`` and ``claim == "research_diagnostic_only"``.
    Coupling: ``research_only is True`` ⇒ claim present and correct.
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    if "research_only" not in blob and "claim" not in blob:
        return []
    errs: list[str] = []
    if "research_only" in blob and blob.get("research_only") is not True:
        errs.append("candle_research_only_missing_or_false")
    if blob.get("research_only") is True:
        if "claim" not in blob:
            errs.append("candle_claim_missing_while_research_only_true")
        elif blob.get("claim") != "research_diagnostic_only":
            errs.append("candle_claim_not_research_diagnostic_only")
    elif "claim" in blob:
        # claim ⇒ research_only: a present claim without the honesty flag is not
        # acceptable evidence (the coupling must hold in both directions).
        if "research_only" not in blob:
            errs.append("candle_research_only_missing_or_false")
        if blob.get("claim") != "research_diagnostic_only":
            errs.append("candle_claim_not_research_diagnostic_only")
    return errs


def candle_all_ic_pearson_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every ``ic_*_pearson`` ∈ [-1, 1] when finite on candle receipts."""
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.startswith("ic_") or not key.endswith("_pearson"):
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


def candle_all_ic_p_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every candle ``ic_*_p`` ∈ [0, 1] when finite.

    Companion catch-all to :func:`candle_all_ic_pearson_unit_honesty_errors`.
    Skips non-p suffixes (``_pearson``, ``_n_dates``, bare ``ic_*``).
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.startswith("ic_") or not key.endswith("_p"):
            continue
        if key.endswith("_pearson"):
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


def candle_all_ic_t_finite_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every candle ``ic_*_t`` is finite when present (not ±inf)."""
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.startswith("ic_") or not key.endswith("_t"):
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


def candle_all_ic_n_dates_nonneg_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every candle ``ic_*_n_dates`` ≥ 0 when finite."""
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.startswith("ic_") or not key.endswith("_n_dates"):
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
        if abs(x) == float("inf") or x < 0.0:
            errs.append(f"{key}_negative_or_non_finite")
    return errs


def candle_join_coverage_and_chain_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle join_coverage ∈ (0, 1], chain ints, and ratio identity.

    - join_coverage ∈ (0, 1] when finite
    - n_scored ≤ n_fused ≤ n_bars when pairs present
    - when join_coverage + n_fused + n_bars (n_bars>0) all finite:
      join_coverage ≈ n_fused / n_bars (fuse stamps lit(fused.height/n_candles))

    Complements sizing honesty. Research diagnostic only; never live Sharpe.
    Off kyle invent / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    if "join_coverage" in blob:
        try:
            j = float(blob.get("join_coverage"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("join_coverage_non_numeric")
        else:
            if j == j and abs(j) != float("inf") and not (0.0 < j <= 1.0):
                errs.append("join_coverage_out_of_open_unit_interval")
            elif j == j and abs(j) == float("inf"):
                errs.append("join_coverage_non_finite")

    # chain ints when present
    def _as_int(key: str) -> int | None:
        if key not in blob:
            return None
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            return None
        if x != x or abs(x) == float("inf") or x < 0 or x != int(x):
            errs.append(f"{key}_not_nonneg_int")
            return None
        return int(x)

    n_bars = _as_int("n_bars")
    n_fused = _as_int("n_fused")
    n_scored = _as_int("n_scored")
    if n_fused is not None and n_bars is not None and n_fused > n_bars:
        errs.append("n_fused_gt_n_bars")
    if n_scored is not None and n_fused is not None and n_scored > n_fused:
        errs.append("n_scored_gt_n_fused")

    # Ratio identity: join_coverage is stamped as fused.height / n_candle_rows
    if "join_coverage" in blob and n_bars is not None and n_fused is not None and n_bars > 0:
        try:
            j = float(blob.get("join_coverage"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass  # non-numeric already flagged above
        else:
            if j == j and abs(j) != float("inf"):
                expected = float(n_fused) / float(n_bars)
                if not math.isclose(j, expected, rel_tol=1e-9, abs_tol=1e-12):
                    errs.append("join_coverage_not_n_fused_over_n_bars")
    return errs


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


def candle_feature_cols_ic_implies_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: every scored FEATURE_COLS IC has a mean_* companion when family is candle.

    Uses ``mean_<col>`` (and ``mean_depth_imbalance`` for imbalance_depth).
    Bounds: concentration/tob/queue_priority/depth_abs/mwb/spread_over_mid ∈[0,1];
    imbalance_* ∈[-1,1]; others finite-only. Never equate IC to mean.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    try:
        from quant_fund.microstructure.bench import FEATURE_COLS
    except Exception:
        return []
    unit01 = {
        "depth_imbalance_abs",
        "spread_over_mid",
        "bid_size_concentration_top",
        "ask_size_concentration_top",
        "queue_priority_proxy",
        "ask_queue_priority_proxy",
        "tob_size_share",
        "microprice_weight_balance",
        "candle_body_frac",
        "candle_range_frac",
    }
    unit_signed = {
        "imbalance_top",
        "imbalance_depth",
        "queue_imbalance",
        "notional_imbalance",
        "candle_direction",
        "candle_dir_x_imbalance",
    }
    errs: list[str] = []
    for col in FEATURE_COLS:
        ic_key = f"ic_{col}"
        if ic_key not in blob:
            continue
        mean_key = "mean_depth_imbalance" if col == "imbalance_depth" else f"mean_{col}"
        if mean_key not in blob:
            errs.append(f"{mean_key}_missing_while_{ic_key}_scored")
            continue
        try:
            m = float(blob.get(mean_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{mean_key}_non_numeric")
            continue
        if m != m:
            errs.append(f"{mean_key}_nan_while_{ic_key}_scored")
            continue
        if abs(m) == float("inf"):
            errs.append(f"{mean_key}_non_finite")
            continue
        if col in unit01 and not (0.0 <= m <= 1.0) or col in unit_signed and not (-1.0 <= m <= 1.0):
            errs.append(f"{mean_key}_out_of_unit_interval")
    return errs


def candle_all_finite_rate_prefix_honesty_errors(blob: object) -> list[str]:
    """Soft-verify every candle ``finite_rate_*`` key ∈ [0, 1] when finite.

    Candle stamps use the ``finite_rate_<feature>`` prefix (≠ northset
    ``*_finite_rate`` suffix catch-all). NaN skip; ±inf / OOB fail-closed.
    Research diagnostic only; off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    errs: list[str] = []
    for key, raw in blob.items():
        if not isinstance(key, str) or not key.startswith("finite_rate_"):
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


def candle_ofi_qp_slope_ic_implies_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ofi / queue_priority / size-slope IC⇒mean on candle_order_book."""
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    errs: list[str] = []
    specs = (
        ("ic_ofi", "mean_ofi", None, None),  # signed unbounded
        ("ic_queue_priority_proxy", "mean_queue_priority_proxy", 0.0, 1.0),
        ("ic_ask_queue_priority_proxy", "mean_ask_queue_priority_proxy", 0.0, 1.0),
        ("ic_bid_log_size_slope", "mean_bid_log_size_slope", None, None),
        ("ic_ask_log_size_slope", "mean_ask_log_size_slope", None, None),
        ("ic_microprice_minus_mid", "mean_microprice_minus_mid", None, None),
        ("ic_microprice_minus_mid_bps", "mean_microprice_minus_mid_bps", None, None),
        ("ic_bid_mean_log_tick_spacing", "mean_bid_mean_log_tick_spacing", None, None),
        ("ic_ask_mean_log_tick_spacing", "mean_ask_mean_log_tick_spacing", None, None),
        ("ic_bid_log_price_slope", "mean_bid_log_price_slope", None, None),
        ("ic_ask_log_price_slope", "mean_ask_log_price_slope", None, None),
    )
    for ic_key, mean_key, lo, hi in specs:
        if ic_key not in blob:
            continue
        if mean_key not in blob:
            errs.append(f"{mean_key}_missing_while_{ic_key}_scored")
            continue
        try:
            m = float(blob.get(mean_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{mean_key}_non_numeric")
            continue
        if m != m:
            errs.append(f"{mean_key}_nan_while_{ic_key}_scored")
        elif abs(m) == float("inf"):
            errs.append(f"{mean_key}_non_finite")
        elif lo is not None and hi is not None and not (lo <= m <= hi):
            errs.append(f"{mean_key}_out_of_unit_interval")
    return errs


def candle_ofi_and_queue_imbalance_means_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle fuse means when stamped (not only when IC scored).

    - ``mean_ofi`` finite when present (signed unbounded)
    - ``mean_queue_imbalance`` ∈ [-1, 1] when finite
    NaN skip. Complements IC⇒mean helpers. Research diagnostic only; off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    errs: list[str] = []
    if "mean_ofi" in blob:
        try:
            x = float(blob.get("mean_ofi"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("mean_ofi_non_numeric")
        else:
            if x == x and abs(x) == float("inf"):
                errs.append("mean_ofi_non_finite")
    if "mean_queue_imbalance" in blob:
        try:
            q = float(blob.get("mean_queue_imbalance"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("mean_queue_imbalance_non_numeric")
        else:
            if q == q and (abs(q) == float("inf") or not (-1.0 <= q <= 1.0)):
                errs.append("mean_queue_imbalance_out_of_signed_unit")
    return errs


def candle_feature_ofi_finite_honesty_errors(blob: object) -> list[str]:
    """Soft-verify FEATURE_COLS ``ofi`` mean finite when stamped or IC-scored.

    Delegates stamped-mean check to :func:`candle_ofi_and_queue_imbalance_means_honesty_errors`
    and IC⇒mean finite to :func:`candle_feature_cols_ic_implies_mean_honesty_errors`
    (ofi is unbounded). Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    errs.extend(candle_ofi_and_queue_imbalance_means_honesty_errors(blob))
    # only keep ofi-related
    return [e for e in errs if "ofi" in e]


def candle_notional_imbalance_ic_implies_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: when notional IC is scored, mean_notional_imbalance ∈ [-1, 1].

    ``ic_notional_imbalance`` on candle_order_book ⇒ mean present and ∈ [-1, 1].
    Never equate IC to the mean. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    if "ic_notional_imbalance" not in blob:
        return []
    if "mean_notional_imbalance" not in blob:
        return ["mean_notional_imbalance_missing_while_notional_ic_scored"]
    try:
        m = float(blob.get("mean_notional_imbalance"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_notional_imbalance_non_numeric"]
    if m != m:
        return ["mean_notional_imbalance_nan_while_notional_ic_scored"]
    if abs(m) == float("inf") or not (-1.0 <= m <= 1.0):
        return ["mean_notional_imbalance_out_of_unit_interval"]
    return []


def candle_spread_over_mid_ic_implies_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: when spread_over_mid IC is scored, mean ≥ 0.

    ``ic_spread_over_mid`` on candle_order_book ⇒ ``mean_spread_over_mid`` present
    and ≥ 0 when finite. Never equate IC to the mean. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    if "ic_spread_over_mid" not in blob:
        return []
    if "mean_spread_over_mid" not in blob:
        return ["mean_spread_over_mid_missing_while_spread_over_mid_ic_scored"]
    try:
        m = float(blob.get("mean_spread_over_mid"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_spread_over_mid_non_numeric"]
    if m != m:
        return ["mean_spread_over_mid_nan_while_spread_over_mid_ic_scored"]
    if abs(m) == float("inf") or m < 0.0:
        return ["mean_spread_over_mid_negative_or_non_finite"]
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


def candle_spread_bps_ic_matches_spread_over_mid_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle ``ic_spread_bps`` ≈ ``ic_spread_over_mid`` when both finite.

    book_metrics: ``spread_bps = 1e4 * spread_over_mid`` — a positive scale, so
    date-level Spearman (and Pearson) ICs must match. Complements the mean-side
    ``mean_spread_bps ≈ 1e4 * mean_spread_over_mid`` check in
    :func:`candle_spread_alias_honesty_errors`.

    When both IC keys are finite:
    - bare Spearman ICs must be isclose
    - ``ic_*_pearson`` companions must be isclose when both finite

    Skip absent / non-finite either side. Candle family only.
    Research diagnostic only; never live Sharpe. Off sibling invent / half_spread
    Sergeant lane / kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    errs: list[str] = []

    def _finite(key: str) -> float | None:
        if key not in blob:
            return None
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"{key}_non_numeric")
            return None
        if x != x or abs(x) == float("inf"):
            return None
        return x

    spearman_bps = _finite("ic_spread_bps")
    spearman_mid = _finite("ic_spread_over_mid")
    if (
        spearman_bps is not None
        and spearman_mid is not None
        and not math.isclose(spearman_bps, spearman_mid, rel_tol=1e-9, abs_tol=1e-12)
    ):
        errs.append("ic_spread_bps_diverges_from_ic_spread_over_mid")

    pearson_bps = _finite("ic_spread_bps_pearson")
    pearson_mid = _finite("ic_spread_over_mid_pearson")
    if (
        pearson_bps is not None
        and pearson_mid is not None
        and not math.isclose(pearson_bps, pearson_mid, rel_tol=1e-9, abs_tol=1e-12)
    ):
        errs.append("ic_spread_bps_pearson_diverges_from_ic_spread_over_mid_pearson")
    return errs


def candle_frac_and_spread_x_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle structure fracs + spread×range companion.

    When stamped on ``candle_order_book`` (or family absent):
    - ``mean_candle_range_frac`` / ``mean_candle_body_frac`` ∈ [0, 1] when finite
    - ``mean_imbalance_x_body_frac`` ∈ [-1, 1] when finite
    - ``mean_spread_x_range`` ≥ 0 when finite
    NaN skipped. Research diagnostic only; never live Sharpe. Off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    fam = blob.get("family")
    if fam not in (None, "candle_order_book"):
        return []
    errs: list[str] = []
    for key in ("mean_candle_range_frac", "mean_candle_body_frac"):
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
    if "mean_imbalance_x_body_frac" in blob:
        try:
            x = float(blob.get("mean_imbalance_x_body_frac"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("mean_imbalance_x_body_frac_non_numeric")
        else:
            if x == x and abs(x) != float("inf") and not (-1.0 <= x <= 1.0):
                errs.append("mean_imbalance_x_body_frac_out_of_signed_unit")
            elif x == x and abs(x) == float("inf"):
                errs.append("mean_imbalance_x_body_frac_non_finite")
    if "mean_spread_x_range" in blob:
        try:
            x = float(blob.get("mean_spread_x_range"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("mean_spread_x_range_non_numeric")
        else:
            if x == x and abs(x) == float("inf"):
                errs.append("mean_spread_x_range_non_finite")
            elif x == x and x < 0.0:
                errs.append("mean_spread_x_range_negative")
    return errs


def candle_direction_mean_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``mean_candle_direction`` ∈ [-1, 1] when finite.

    FEATURE_COLS ``candle_direction`` is ternary ∈ {-1, 0, 1}; the receipt mean
    must therefore lie in [-1, 1]. NaN/absent skip. Research diagnostic only.
    Off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    fam = blob.get("family")
    if fam not in (None, "candle_order_book"):
        return []
    if "mean_candle_direction" not in blob:
        return []
    try:
        x = float(blob.get("mean_candle_direction"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_candle_direction_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf") or not (-1.0 <= x <= 1.0):
        return ["mean_candle_direction_out_of_signed_unit"]
    return []


def candle_wick_skew_and_body_ret_means_honesty_errors(blob: object) -> list[str]:
    """Soft-verify unbounded candle FEATURE_COLS means finite when stamped.

    Keys: ``mean_wick_skew``, ``mean_candle_body_ret``, ``mean_signed_vol_x_imbalance``.
    Present numeric values must not be ±inf (NaN skip). Research diagnostic only;
    never live Sharpe. Off kyle invent.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    errs: list[str] = []
    for key in (
        "mean_wick_skew",
        "mean_candle_body_ret",
        "mean_signed_vol_x_imbalance",
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


def candle_signed_vol_x_imbalance_mean_honesty_errors(blob: object) -> list[str]:
    """Alias: soft-verify ``mean_signed_vol_x_imbalance`` finite via the finite-means pack."""
    if not isinstance(blob, dict):
        return []
    slim = (
        {
            "family": blob.get("family"),
            "mean_signed_vol_x_imbalance": blob["mean_signed_vol_x_imbalance"],
        }
        if "mean_signed_vol_x_imbalance" in blob
        else {"family": blob.get("family")}
    )
    return candle_wick_skew_and_body_ret_means_honesty_errors(slim)


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


def candle_structure_finite_rate_covers_companions_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle ``structure_finite_rate`` ≈ nanmean of finite_rate companions.

    Bench stamps ``structure_finite_rate`` via nanmean of:
    ``finite_rate_microprice_minus_mid``,
    ``finite_rate_bid_size_concentration_top``,
    ``finite_rate_ask_size_concentration_top``.

    When the aggregate and all three companions are finite, require equality.
    Partial / NaN companions → skip (thin books). Candle family only.
    Research diagnostic only; never live Sharpe. Off kyle invent / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    fam = blob.get("family")
    if fam not in (None, "candle_order_book"):
        return []
    companions = (
        "finite_rate_microprice_minus_mid",
        "finite_rate_bid_size_concentration_top",
        "finite_rate_ask_size_concentration_top",
    )
    if "structure_finite_rate" not in blob:
        return []
    if not all(k in blob for k in companions):
        return []
    try:
        agg = float(blob.get("structure_finite_rate"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["structure_finite_rate_non_numeric"]
    if agg != agg or abs(agg) == float("inf"):
        return []
    vals: list[float] = []
    for key in companions:
        try:
            x = float(blob.get(key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return []
        if x != x or abs(x) == float("inf"):
            return []
        vals.append(x)
    import math

    expected = sum(vals) / float(len(vals))
    if not math.isclose(agg, expected, rel_tol=1e-9, abs_tol=1e-12):
        return ["candle_structure_finite_rate_not_nanmean_of_finite_rate_companions"]
    return []


def candle_mwb_scored_implies_mean_unit_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: when MWB IC is scored on candle, mean MWB ∈ [0, 1].

    Fuse path contract: ``ic_microprice_weight_balance`` present ⇒
    ``mean_microprice_weight_balance`` present and ∈ [0, 1] when finite.
    Never equate IC to the mean. NaN mean with finite IC is dishonest.
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    if "ic_microprice_weight_balance" not in blob:
        return []
    if "mean_microprice_weight_balance" not in blob:
        return ["mean_microprice_weight_balance_missing_while_mwb_ic_scored"]
    try:
        m = float(blob.get("mean_microprice_weight_balance"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["mean_microprice_weight_balance_non_numeric"]
    if m != m:
        return ["mean_microprice_weight_balance_nan_while_mwb_ic_scored"]
    if abs(m) == float("inf") or not (0.0 <= m <= 1.0):
        return ["mean_microprice_weight_balance_out_of_unit_interval"]
    return []


def candle_microprice_weight_balance_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ic_microprice_weight_balance companions when scored.

    When ``ic_microprice_weight_balance`` is present: finite-when-present for IC/t;
    ``ic_microprice_weight_balance_p`` ∈ [0, 1]; ``_n_dates`` ≥ 0. Never equate
    to mean_microprice_weight_balance (mean ≠ IC). Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    if "ic_microprice_weight_balance" not in blob:
        return []
    # Reuse pattern via the general IC helper for this key slice
    slice_blob = {
        k: v
        for k, v in blob.items()
        if isinstance(k, str) and k.startswith("ic_microprice_weight_balance")
    }
    return candle_feature_cols_ic_honesty_errors(slice_blob)


def candle_feature_cols_ic_completeness_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle FEATURE_COLS IC keys are present when the family scored.

    When ``family == "candle_order_book"`` and (``n_scored`` > 0 or any spearman
    ``ic_*`` key is present), every column in ``FEATURE_COLS`` must expose the
    Spearman/Pearson companion pack: ``ic_<col>``, ``ic_<col>_pearson``,
    ``ic_<col>_t``, ``ic_<col>_p``, and ``ic_<col>_n_dates`` (values may be NaN).
    Skip other families. Research diagnostic only; never live Sharpe.
    Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") != "candle_order_book":
        return []
    try:
        from quant_fund.microstructure.bench import FEATURE_COLS
    except Exception:
        return []
    n_scored = blob.get("n_scored")
    try:
        scored = float(n_scored) if n_scored is not None else float("nan")
    except (TypeError, ValueError):
        scored = float("nan")
    has_ic = any(
        isinstance(k, str)
        and k.startswith("ic_")
        and not k.endswith(("_t", "_p", "_n_dates", "_pearson"))
        and k != "ic_method"
        for k in blob
    )
    if not (has_ic or (scored == scored and scored > 0.0)):
        return []
    errs: list[str] = []
    for col in FEATURE_COLS:
        for suf in ("", "_pearson", "_t", "_p", "_n_dates"):
            key = f"ic_{col}{suf}"
            if key not in blob:
                errs.append(f"{key}_missing_from_candle_feature_cols_receipt")
    return errs


def candle_feature_cols_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_order_book FEATURE_COLS IC receipt companions.

    When present on a candle family blob:
    - ``ic_*_p`` ∈ [0, 1] when finite (Spearman HAC p-values)
    - ``ic_*_n_dates`` ≥ 0 when finite
    - bare ``ic_<feature>`` / ``ic_*_t`` / ``ic_*_pearson`` finite-when-present (signed OK; ±inf fail)
    - ``mean_abs_ic`` ∈ [0, 1] when finite
    - bare ``ic_*`` / ``ic_*_pearson`` / ``best_feature_ic`` ∈ [-1, 1] when finite
    - ``best_feature_ic_key`` / ``best_feature_ic`` pair identity: nonempty key requires
      finite matching value (max |spearman|); finite value requires nonempty key
    - ``mean_abs_ic`` ≈ mean(|ic_*| spearman) when both present
    - ``|best_feature_ic|`` ≥ ``mean_abs_ic`` when both finite (best covers mean;
      still holds when feature ``ic_*`` keys are absent)

    Never equate with northset ofi/microprice/clv IC keys. NaN/absent skip.
    Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    errs: list[str] = []
    spearman: dict[str, float] = {}

    for key, raw in blob.items():
        if not isinstance(key, str) or not key.startswith("ic_"):
            continue
        try:
            x = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            # Skip non-numeric meta (e.g. ic_method=date_level_spearman_hac).
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errs.append(f"{key}_non_finite_fail_closed")
            continue
        if key.endswith("_p"):
            if not (0.0 <= x <= 1.0):
                errs.append(f"{key}_out_of_unit_interval")
        elif key.endswith("_n_dates"):
            if x < 0.0:
                errs.append(f"{key}_negative")
        elif key.endswith(("_t", "_pearson")):
            # pearson correlation ∈ [-1, 1]; t may be unbounded
            if key.endswith("_pearson") and not (-1.0 <= x <= 1.0):
                errs.append(f"{key}_out_of_unit_interval")
        else:
            # bare ic_<feature> spearman mean ∈ [-1, 1]
            if not (-1.0 <= x <= 1.0):
                errs.append(f"{key}_out_of_unit_interval")
            else:
                spearman[key] = x

    if "mean_abs_ic" in blob:
        try:
            mai = float(blob.get("mean_abs_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("mean_abs_ic_non_numeric")
        else:
            if mai == mai and abs(mai) != float("inf"):
                if mai < 0.0:
                    errs.append("mean_abs_ic_negative")
                elif mai > 1.0:
                    errs.append("mean_abs_ic_out_of_unit_interval")
            elif mai == mai and abs(mai) == float("inf"):
                errs.append("mean_abs_ic_non_finite_fail_closed")

    import math

    best_key = blob.get("best_feature_ic_key")
    has_best_key = best_key is not None and best_key != ""
    has_best_ic = "best_feature_ic" in blob and blob.get("best_feature_ic") is not None

    if has_best_key:
        if (
            not isinstance(best_key, str)
            or not best_key.startswith("ic_")
            or best_key.endswith(("_t", "_p", "_n_dates", "_pearson"))
        ):
            errs.append("best_feature_ic_key_not_ic_spearman_key")
        elif best_key not in blob:
            errs.append("best_feature_ic_key_missing_from_receipt")
        if not has_best_ic:
            errs.append("best_feature_ic_missing_despite_best_feature_ic_key")
        else:
            try:
                best_val = float(blob.get("best_feature_ic"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                errs.append("best_feature_ic_non_numeric")
            else:
                if best_val != best_val:
                    errs.append("best_feature_ic_nan_despite_best_feature_ic_key")
                elif abs(best_val) == float("inf"):
                    errs.append("best_feature_ic_non_finite_fail_closed")
                elif not (-1.0 <= best_val <= 1.0) or not (-1.0 <= best_val <= 1.0):
                    errs.append("best_feature_ic_out_of_unit_interval")
                elif spearman:
                    max_abs = max(abs(v) for v in spearman.values())
                    if not math.isclose(abs(best_val), max_abs, rel_tol=1e-9, abs_tol=1e-12):
                        errs.append("best_feature_ic_not_max_abs_spearman_ic")
                    keyed = spearman.get(best_key) if isinstance(best_key, str) else None
                    if keyed is None or not math.isclose(
                        best_val, keyed, rel_tol=1e-9, abs_tol=1e-12
                    ):
                        errs.append("best_feature_ic_mismatch_best_feature_ic_key")
    elif has_best_ic:
        try:
            best_val = float(blob.get("best_feature_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("best_feature_ic_non_numeric")
        else:
            if best_val == best_val and abs(best_val) != float("inf"):
                errs.append("best_feature_ic_key_missing_despite_best_feature_ic")
            elif best_val == best_val and abs(best_val) == float("inf"):
                errs.append("best_feature_ic_non_finite_fail_closed")

    # mean_abs_ic identity: ≈ mean(|spearman ic_*|) when both sides present
    if "mean_abs_ic" in blob and spearman:
        try:
            mai = float(blob.get("mean_abs_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass  # already flagged non-numeric above
        else:
            if mai == mai and abs(mai) != float("inf"):
                expected = sum(abs(v) for v in spearman.values()) / len(spearman)
                if not math.isclose(mai, expected, rel_tol=1e-9, abs_tol=1e-12):
                    errs.append("mean_abs_ic_not_mean_abs_spearman_ic")

    # |best_feature_ic| covers mean_abs_ic even when feature ic_* keys absent
    if "best_feature_ic" in blob and "mean_abs_ic" in blob:
        try:
            best_val = float(blob.get("best_feature_ic"))  # type: ignore[arg-type]
            mai = float(blob.get("mean_abs_ic"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass  # non-numeric already flagged above when applicable
        else:
            if (
                best_val == best_val
                and mai == mai
                and abs(best_val) != float("inf")
                and abs(mai) != float("inf")
                and abs(best_val) + 1e-12 < mai
            ):
                errs.append("best_feature_ic_abs_lt_mean_abs_ic")

    return errs


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


def candle_order_book_dgp_data_source_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_order_book ``dgp`` / ``book_dgp`` ↔ ``data_source``.

    Parallel to always-on northset helper — never equate the two surfaces.
    When present on candle family blob:

    - ``dgp`` and ``book_dgp`` both present → must match
    - ``data_source == "SYNTHETIC"`` ⇒ present dgp fields are ``synthetic_lob``
    - ``book_dgp``/``dgp`` ``synthetic_lob`` ⇒ ``data_source`` is ``SYNTHETIC`` when set
    - non-synthetic dgp + ``data_source`` set ⇒ not ``SYNTHETIC``; if ``book_source``
      also set, ``data_source == book_source``

    Skip non-candle families / all three absent. Research diagnostic only; never
    live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if blob.get("family") not in (None, "candle_order_book"):
        return []
    # require candle marker or explicit family
    if (
        blob.get("family") is None
        and "best_feature_ic" not in blob
        and "mean_abs_ic" not in blob
        and not any(k in blob for k in ("finite_rate_microprice_minus_mid", "ic_method"))
    ):
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
        errs.append("candle_dgp_book_dgp_mismatch")

    synth_dgp: bool | None = None
    if book_dgp is not None:
        synth_dgp = book_dgp == "synthetic_lob"
    elif dgp is not None:
        synth_dgp = dgp == "synthetic_lob"

    if synth_dgp is True and data_source is not None and data_source != "SYNTHETIC":
        errs.append("candle_synthetic_dgp_data_source_not_SYNTHETIC")
    if data_source == "SYNTHETIC":
        if book_dgp is not None and book_dgp != "synthetic_lob":
            errs.append("candle_SYNTHETIC_data_source_book_dgp_not_synthetic_lob")
        if dgp is not None and dgp != "synthetic_lob":
            errs.append("candle_SYNTHETIC_data_source_dgp_not_synthetic_lob")
        if book_dgp is None and dgp is None:
            errs.append("candle_SYNTHETIC_data_source_missing_dgp")
    if synth_dgp is False and data_source is not None:
        if data_source == "SYNTHETIC":
            errs.append("candle_nonsynthetic_dgp_data_source_SYNTHETIC")
        elif book_source is not None and data_source != book_source:
            errs.append("candle_nonsynthetic_data_source_ne_book_source")
    return errs


def candle_order_book_family_provenance_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_order_book ``family`` / ``book_source`` / ``label`` stamps.

    When present on a candle family blob:
    - ``family`` == ``candle_order_book``
    - ``book_source`` nonempty string
    - ``label`` nonempty string

    Skip non-candle families. Never equate with northset / kyle nest twins.
    Research diagnostic only; never live Sharpe. Off kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    fam = blob.get("family") if "family" in blob else None
    if fam is not None and fam != "candle_order_book":
        return []
    # If family absent, only run when candle markers present
    if (
        fam is None
        and not any(
            k in blob
            for k in ("best_feature_ic", "mean_abs_ic", "finite_rate_microprice_minus_mid")
        )
        and "book_source" not in blob
        and "label" not in blob
    ):
        return []

    errs: list[str] = []
    if (
        "family" in blob
        and blob.get("family") is not None
        and (blob.get("family") != "candle_order_book")
    ):
        errs.append("candle_family_invalid")
    if "book_source" in blob and blob.get("book_source") is not None:
        bs = blob.get("book_source")
        if not isinstance(bs, str) or not bs.strip():
            errs.append("candle_book_source_empty_or_not_str")
    if "label" in blob and blob.get("label") is not None:
        lab = blob.get("label")
        if not isinstance(lab, str) or not lab.strip():
            errs.append("candle_label_empty_or_not_str")
    return errs


def candle_order_book_sizing_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_order_book sizing stamps: min_names / depth / n_bars / n_fused / n_scored.

    Candle bench stamps these (CLI echoes n_fused + min_names). When present:
    - ``min_names`` finite integer ≥ 1
    - ``depth`` finite integer ≥ 1 (LOB levels used by attach)
    - ``n_bars`` / ``n_fused`` / ``n_scored`` non-negative integers (NaN skip per key)
    - ``n_scored ≤ n_fused ≤ n_bars`` when pairs present

    Candle-family only (skip other families). Never equate with northset / kyle nest
    twins. Research diagnostic only; never live Sharpe. Off kyle_ofi / METRICS_*.
    """
    if not isinstance(blob, dict):
        return []
    fam = blob.get("family")
    if fam is not None and fam != "candle_order_book":
        return []

    errs: list[str] = []

    if "min_names" in blob and blob.get("min_names") is not None:
        try:
            mn = float(blob.get("min_names"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("candle_min_names_non_numeric")
        else:
            if mn != mn or abs(mn) == float("inf"):
                errs.append("candle_min_names_non_finite")
            elif mn < 1.0 or abs(mn - int(mn)) > 1e-9:
                errs.append("candle_min_names_lt_one_or_not_int")

    if "depth" in blob and blob.get("depth") is not None:
        try:
            d = float(blob.get("depth"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append("candle_depth_non_numeric")
        else:
            if d != d or abs(d) == float("inf"):
                errs.append("candle_depth_non_finite")
            elif d < 1.0 or abs(d - int(d)) > 1e-9:
                errs.append("candle_depth_lt_one_or_not_int")

    def _nonneg_int(key: str) -> int | None:
        if key not in blob:
            return None
        val = blob.get(key)
        try:
            x = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errs.append(f"candle_{key}_non_numeric")
            return None
        if x != x:
            return None
        if abs(x) == float("inf") or x < 0 or abs(x - int(x)) > 1e-9:
            errs.append(f"candle_{key}_not_nonneg_int")
            return None
        return int(x)

    nb = _nonneg_int("n_bars")
    nf = _nonneg_int("n_fused")
    ns = _nonneg_int("n_scored")
    if nb is not None and ns is not None and ns > nb:
        errs.append("candle_n_scored_gt_n_bars")
    if nf is not None and ns is not None and ns > nf:
        errs.append("candle_n_scored_gt_n_fused")
    if nb is not None and nf is not None and nf > nb:
        errs.append("candle_n_fused_gt_n_bars")
    return errs


def candle_order_book_ic_method_honesty_errors(blob: object) -> list[str]:
    """Soft-verify candle_order_book ``ic_method`` for FEATURE_COLS date-IC / HAC.

    When any ``ic_<feature>`` spearman key is present:
    - ``ic_method`` must be present and ``date_level_spearman_hac``
    Finite ``ic_<feature>`` with companion ``ic_<feature>_n_dates`` present → n_dates ≥ 1.
    Missing nest / no IC markers → skip. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    # Optional family gate: if family stamped, must be candle_order_book
    fam = blob.get("family")
    if fam is not None and fam != "candle_order_book":
        return []
    if not _candle_has_feature_ic_marker(blob):
        return []
    errors: list[str] = []
    method = blob.get("ic_method")
    if method is None or (isinstance(method, str) and not method.strip()):
        errors.append("candle_order_book_ic_method_missing")
    elif not isinstance(method, str) or method not in _CANDLE_FEATURE_IC_METHOD_ALLOWED:
        errors.append("candle_order_book_ic_method_invalid")
    for key, val in blob.items():
        if not isinstance(key, str) or not key.startswith("ic_"):
            continue
        if key.endswith(("_t", "_p", "_n_dates", "_pearson")):
            continue
        try:
            x = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if x != x:
            continue
        nd_key = f"{key}_n_dates"
        if nd_key not in blob:
            continue
        try:
            nd = float(blob.get(nd_key))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            errors.append(f"{nd_key}_non_numeric")
            continue
        if nd != nd or nd < 1.0:
            errors.append(f"{nd_key}_lt_1")
    return errors


__all__ = [
    "candle_all_finite_rate_prefix_honesty_errors",
    "candle_all_ic_n_dates_nonneg_honesty_errors",
    "candle_all_ic_p_unit_honesty_errors",
    "candle_all_ic_pearson_unit_honesty_errors",
    "candle_all_ic_t_finite_honesty_errors",
    "candle_depth_imbalance_ic_implies_mean_honesty_errors",
    "candle_direction_mean_honesty_errors",
    "candle_feature_cols_ic_completeness_honesty_errors",
    "candle_feature_cols_ic_honesty_errors",
    "candle_feature_cols_ic_implies_mean_honesty_errors",
    "candle_feature_ofi_finite_honesty_errors",
    "candle_frac_and_spread_x_honesty_errors",
    "candle_join_coverage_and_chain_honesty_errors",
    "candle_log_slopes_finite_honesty_errors",
    "candle_log_tick_spacing_finite_honesty_errors",
    "candle_microprice_minus_mid_finite_pack_honesty_errors",
    "candle_microprice_weight_balance_ic_honesty_errors",
    "candle_mwb_scored_implies_mean_unit_honesty_errors",
    "candle_notional_imbalance_ic_implies_mean_honesty_errors",
    "candle_ofi_and_queue_imbalance_means_honesty_errors",
    "candle_ofi_qp_slope_ic_implies_mean_honesty_errors",
    "candle_order_book_claim_honesty_errors",
    "candle_order_book_dgp_data_source_honesty_errors",
    "candle_order_book_family_provenance_honesty_errors",
    "candle_order_book_ic_method_honesty_errors",
    "candle_order_book_sizing_honesty_errors",
    "candle_signed_vol_x_imbalance_mean_honesty_errors",
    "candle_spread_alias_honesty_errors",
    "candle_spread_bps_ic_matches_spread_over_mid_ic_honesty_errors",
    "candle_spread_bps_nonneg_honesty_errors",
    "candle_spread_over_mid_ic_implies_mean_honesty_errors",
    "candle_structure_finite_rate_covers_companions_honesty_errors",
    "candle_structure_ic_implies_mean_honesty_errors",
    "candle_wick_skew_and_body_ret_means_honesty_errors",
    "mean_candle_dir_x_imbalance_honesty_errors",
    "northset_candle_body_ret_ic_pack_honesty_errors",
    "northset_structure_finite_rate_distinct_from_candle_honesty_errors",
]
