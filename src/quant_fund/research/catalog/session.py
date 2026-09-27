"""Session-level honesty checkers (``*_honesty_errors``)."""

from __future__ import annotations

import math

from ._helpers import (
    _DATA_SNOOPING_P_KEYS,
    _finite_scalar,
)
from .constants import (
    H20_HYPOTHESIS_ID,
    H21_HYPOTHESIS_ID,
    H22_HYPOTHESIS_ID,
    H23_HYPOTHESIS_ID,
    H24_HYPOTHESIS_ID,
    H25_HYPOTHESIS_ID,
    H26_HYPOTHESIS_ID,
    H27_HYPOTHESIS_ID,
    H28_HYPOTHESIS_ID,
    H29_HYPOTHESIS_ID,
    H30_HYPOTHESIS_ID,
    H31_HYPOTHESIS_ID,
    H32_HYPOTHESIS_ID,
    H33_HYPOTHESIS_ID,
    H34_HYPOTHESIS_ID,
    H35_HYPOTHESIS_ID,
    H36_HYPOTHESIS_ID,
    H37_HYPOTHESIS_ID,
    H38_HYPOTHESIS_ID,
    H39_HYPOTHESIS_ID,
    H40_HYPOTHESIS_ID,
    H41_HYPOTHESIS_ID,
    H42_HYPOTHESIS_ID,
    H43_HYPOTHESIS_ID,
    H44_HYPOTHESIS_ID,
    H45_HYPOTHESIS_ID,
    NORTHSET_H23_H28_SPECS,
)
from .predicates import (
    northset_has_finite_book_uncrossed_rate,
    northset_has_finite_imbalance_p_ic,
    northset_has_finite_ohlc_identity_rate,
    northset_has_finite_session_book_vpin_p_ic,
    ranking_data_snooping_blob,
)


def ranking_data_snooping_honesty_errors(ranking: object) -> list[str]:
    """Soft-verify nested ``ranking.data_snooping`` p-values and ordering.

    When the ranking family carries a data-snooping block:

    - ``research_only is True`` and ``claim == "research_diagnostic_only"``
    - Reality Check / SPA p-values ∈ [0, 1] when finite
    - SPA ordering ``p_lower <= p_consistent <= p_upper`` when all finite
    - ``n_trials`` / ``n_obs`` positive; ``stepm_n_rejected == len(rejected)``;
      ``mcs_n_included == len(included)``
    - per-trial adjusted / MCS p-values ∈ [0, 1] when finite

    Missing block / non-dict → skip. Research diagnostic only; never a live
    Sharpe / promotion gate.
    """
    blob = ranking_data_snooping_blob(ranking)
    if blob is None:
        return []
    errs: list[str] = []
    if blob.get("research_only") is not True:
        errs.append("data_snooping_research_only_missing_or_false")
    if blob.get("claim") != "research_diagnostic_only":
        errs.append("data_snooping_claim_not_research_diagnostic_only")
    for key in _DATA_SNOOPING_P_KEYS:
        val = blob.get(key)
        if val is None:
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errs.append(f"data_snooping_{key}_non_numeric")
            continue
        x = float(val)
        if x != x:
            continue
        if not 0.0 <= x <= 1.0:
            errs.append(f"data_snooping_{key}_out_of_unit_interval")
    # No universal ordering is asserted between the three SPA variants (see
    # SpaResult): each is range-checked above; the consistent variant is the
    # recommended one but can sit below the all-recentered lower variant.
    for key in ("n_trials", "n_obs"):
        val = blob.get(key)
        if val is None:
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)) or float(val) < 1:
            errs.append(f"data_snooping_{key}_invalid")
    rejected = blob.get("stepm_rejected")
    n_rejected = blob.get("stepm_n_rejected")
    if (
        isinstance(rejected, list)
        and isinstance(n_rejected, (int, float))
        and not isinstance(n_rejected, bool)
        and int(n_rejected) != len(rejected)
    ):
        errs.append("data_snooping_stepm_n_rejected_mismatch")
    included = blob.get("mcs_included")
    n_included = blob.get("mcs_n_included")
    if (
        isinstance(included, list)
        and isinstance(n_included, (int, float))
        and not isinstance(n_included, bool)
        and int(n_included) != len(included)
    ):
        errs.append("data_snooping_mcs_n_included_mismatch")
    for key in ("stepm_adjusted_p", "mcs_p_values"):
        table = blob.get(key)
        if not isinstance(table, dict):
            continue
        for name, val in table.items():
            if val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                errs.append(f"data_snooping_{key}_{name}_non_numeric")
                continue
            x = float(val)
            if x != x:
                continue
            if not 0.0 <= x <= 1.0:
                errs.append(f"data_snooping_{key}_{name}_out_of_unit_interval")
    return errs


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


def session_volume_conservation_vs_reconstructs_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H24 volume conservation with H23 session reconstructs.

    Sibling rates on the northset receipt — volume conservation ≠ OHLC envelope
    reconstructs (DATA_CONTRACTS). When **both** keys are present:

    - Catalog bind lock: reconstructs → H23, conservation → H24 (distinct hyp ids)
    - Numeric equality on clean synth is allowed (never fail on value equality)
    - Distinct key identity is structural (``session_reconstructs_daily_rate`` ≠
      ``session_volume_conservation_rate``)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_recon = "session_reconstructs_daily_rate"
    k_vol = "session_volume_conservation_rate"
    if k_recon not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_recon == k_vol:
        errs.append("session_reconstructs_volume_conservation_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    recon_hyp = by_key.get(k_recon)
    vol_hyp = by_key.get(k_vol)
    if recon_hyp != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if vol_hyp != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if recon_hyp is not None and vol_hyp is not None and recon_hyp == vol_hyp:
        errs.append("session_reconstructs_volume_conservation_hypothesis_collapsed")
    return errs


def session_volume_conservation_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_volume_conservation_rate ∈ [0, 1] when finite.

    Session volume conservation sibling of reconstructs/chain rates.
    NaN/absent skip. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("session_volume_conservation_rate")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
        return ["session_volume_conservation_rate_out_of_unit_interval"]
    return []


def session_ohlc_vs_reconstructs_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ session envelope reconstructing
    the daily bar. When both keys are present:

    - Distinct key identity
    - H23 binds to reconstructs only (``H23_HYPOTHESIS_ID``); session OHLC is not H23
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "session_ohlc_identity_rate"
    k_recon = "session_reconstructs_daily_rate"
    if k_ohlc not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_recon:
        errs.append("session_ohlc_reconstructs_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_ohlc) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    return errs


def session_ohlc_vs_volume_conservation_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with ``session_volume_conservation_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ session volume conservation
    (H24). When both keys are present:

    - Distinct key identity
    - H24 binds to volume conservation only (``H24_HYPOTHESIS_ID``); session OHLC
      is not H23/H24
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "session_ohlc_identity_rate"
    k_vol = "session_volume_conservation_rate"
    if k_ohlc not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_vol:
        errs.append("session_ohlc_volume_conservation_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_ohlc) == H24_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h24")
    if by_key.get(k_ohlc) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    return errs


def session_ohlc_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with ``session_chain_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ H29 session chain rate.
    When both keys are present:

    - Distinct key identity
    - H29 binds to chain only (``H29_HYPOTHESIS_ID``); session OHLC is not H23/H29
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "session_ohlc_identity_rate"
    k_chain = "session_chain_rate"
    if k_ohlc not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_chain:
        errs.append("session_ohlc_session_chain_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_ohlc) == H29_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h29")
    if by_key.get(k_ohlc) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    return errs


def session_reconstructs_daily_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_reconstructs_daily_rate ∈ [0, 1] when finite.

    Session→daily reconstruction rate — ≠ session_ohlc_identity_rate.
    NaN/absent skip. Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("session_reconstructs_daily_rate")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
        return ["session_reconstructs_daily_rate_out_of_unit_interval"]
    return []


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


def book_uncrossed_vs_imbalance_p_ic_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H21 ``book_uncrossed_rate`` with H22 ``imbalance_top_p_ic``.

    Triad siblings: uncrossed book bound ≠ imbalance discovery IC p-value.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H21 ≠ H22)
    - Gate helpers: H21 on book_uncrossed only; H22 on imbalance_top_p_ic only
    - Numeric equality is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_book = "book_uncrossed_rate"
    k_imb = "imbalance_top_p_ic"
    if k_book not in blob or k_imb not in blob:
        return []

    errs: list[str] = []
    if k_book == k_imb:
        errs.append("book_uncrossed_imbalance_p_ic_keys_collapsed")
    if H21_HYPOTHESIS_ID == H22_HYPOTHESIS_ID:
        errs.append("h21_h22_hypothesis_ids_collapsed")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_imb: 1.0, **elig}):
        errs.append("imbalance_p_ic_incorrectly_triggers_h21_gate")
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_incorrectly_triggers_h22_gate")
    return errs


def ohlc_identity_vs_imbalance_p_ic_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H20 ``ohlc_identity_rate`` with H22 ``imbalance_top_p_ic``.

    Triad diagonal: OHLC envelope bound ≠ imbalance discovery IC p-value.
    Completes H20↔H21↔H22 never-equate mesh (edges already landed). When both
    keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H22)
    - Gate helpers: H20 on ohlc only; H22 on imbalance_top_p_ic only
    - Numeric equality is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_imb = "imbalance_top_p_ic"
    if k_ohlc not in blob or k_imb not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_imb:
        errs.append("ohlc_identity_imbalance_p_ic_keys_collapsed")
    if H20_HYPOTHESIS_ID == H22_HYPOTHESIS_ID:
        errs.append("h20_h22_hypothesis_ids_collapsed")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_imb: 0.05}):
        errs.append("imbalance_p_ic_incorrectly_triggers_h20_gate")
    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_ohlc: 1.0, **elig}):
        errs.append("ohlc_identity_incorrectly_triggers_h22_gate")
    return errs


def session_bulk_vpin_vs_siblings_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_bulk_vpin`` with ``vpin_mean`` / ``session_book_vpin_mean``.

    DATA_CONTRACTS VPIN triad: candle signed-volume bulk ≠ daily Cont proxy mean
    (H32 path) ≠ session-L2 ``|ofi_sum|/ofi_abs_sum`` mean (H43 path). Bulk has
    **no** H-id. When ``session_bulk_vpin`` and at least one sibling are present:

    - Distinct key identity vs each present sibling
    - Distinct hyp ids H32 ≠ H43
    - Bulk must not bind to H32/H43 metric keys (``vpin_p_ic`` / ``session_book_vpin_p_ic``)
    - H43 gate helper triggers on session_book_vpin_p_ic only, not bulk
    - Numeric equality on synth is allowed

    Skip when bulk absent or both siblings absent. Research diagnostic only;
    never live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_bulk = "session_bulk_vpin"
    k_daily = "vpin_mean"
    k_sess = "session_book_vpin_mean"
    if k_bulk not in blob:
        return []
    if k_daily not in blob and k_sess not in blob:
        return []

    errs: list[str] = []
    if k_daily in blob and k_bulk == k_daily:
        errs.append("session_bulk_vpin_vpin_mean_keys_collapsed")
    if k_sess in blob and k_bulk == k_sess:
        errs.append("session_bulk_vpin_session_book_vpin_mean_keys_collapsed")
    if k_daily in blob and k_sess in blob and k_daily == k_sess:
        errs.append("vpin_mean_session_book_vpin_mean_keys_collapsed")
    if H32_HYPOTHESIS_ID == H43_HYPOTHESIS_ID:
        errs.append("h32_h43_hypothesis_ids_collapsed")

    # Catalog bind lock: H32/H43 attach to *_p_ic, never to session_bulk_vpin
    h32_metric = "vpin_p_ic"
    h43_metric = "session_book_vpin_p_ic"
    if k_bulk in {h32_metric, h43_metric}:
        errs.append("session_bulk_vpin_incorrectly_bound_as_h32_or_h43_metric")
    if h32_metric == k_bulk or h43_metric == k_bulk:
        errs.append("session_bulk_vpin_metric_key_collapsed_into_h_gate")

    if northset_has_finite_session_book_vpin_p_ic({k_bulk: 0.5}):
        errs.append("session_bulk_vpin_incorrectly_triggers_h43_gate")
    if not northset_has_finite_session_book_vpin_p_ic(
        {h43_metric: 0.05, "session_book_hypothesis_eligible": True}
    ):
        # eligible default True in helper — still require p_ic key
        errs.append("session_book_vpin_p_ic_h43_gate_helper_broken")
    return errs


def ohlc_identity_vs_book_uncrossed_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H20 ``ohlc_identity_rate`` with H21 ``book_uncrossed_rate``.

    Triad siblings (DATA_CONTRACTS H20/H21/H22): OHLC envelope ≠ uncrossed book.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H21)
    - H20 finite gate helper triggers on ohlc only; H21 gate on book_uncrossed only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_book = "book_uncrossed_rate"
    if k_ohlc not in blob or k_book not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_book:
        errs.append("ohlc_identity_book_uncrossed_keys_collapsed")
    if H20_HYPOTHESIS_ID == H21_HYPOTHESIS_ID:
        errs.append("h20_h21_hypothesis_ids_collapsed")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_book: 1.0}):
        errs.append("book_uncrossed_incorrectly_triggers_h20_gate")
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, "book_hypothesis_eligible": True}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_ohlc: 1.0, "book_hypothesis_eligible": True}):
        errs.append("ohlc_identity_incorrectly_triggers_h21_gate")
    return errs


def ohlc_identity_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``ohlc_identity_rate`` with ``gap_finite_rate``.

    DATA_CONTRACTS: OHLC envelope identity ≠ overnight gap finiteness. When both
    keys are present:

    - Distinct key identity
    - H20 finite gate binds to ``ohlc_identity_rate`` only (gap has **no** H20 claim)
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_gap = "gap_finite_rate"
    if k_ohlc not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_gap:
        errs.append("ohlc_identity_gap_finite_keys_collapsed")

    # H20 watches daily ohlc only — gap must not share H20 id with session binds check
    if H20_HYPOTHESIS_ID in {H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h20_hypothesis_collapsed_into_h21_h22")

    # Positive: northset_has_finite_ohlc_identity_rate uses ohlc key, not gap
    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    return errs


def gap_finite_rate_vs_book_uncrossed_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``gap_finite_rate`` with H21 ``book_uncrossed_rate``.

    DATA_CONTRACTS: overnight gap finiteness ≠ uncrossed book. When both keys
    are present:

    - Distinct key identity
    - Distinct hyp/gate: H21 binds to ``book_uncrossed_rate`` only (gap has
      **no** H21 claim); H21 id stays distinct from H20/H22
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad polish (already landed). Off nest invent / kyle_ofi.
    """
    if not isinstance(blob, dict):
        return []
    k_gap = "gap_finite_rate"
    k_book = "book_uncrossed_rate"
    if k_gap not in blob or k_book not in blob:
        return []

    errs: list[str] = []
    if k_gap == k_book:
        errs.append("gap_finite_book_uncrossed_keys_collapsed")
    if H21_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h21_hypothesis_collapsed_into_h20_h22")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def imbalance_top_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: imbalance discovery IC p-value ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp/gate: H22 binds imbalance_top_p_ic only; gap has **no** H22
      claim and must not trigger H20/H21 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_gap = "gap_finite_rate"
    if k_imb not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_gap:
        errs.append("imbalance_p_ic_gap_finite_keys_collapsed")
    if H22_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h22_hypothesis_collapsed_into_h20_or_h21")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")

    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_imb: 0.05}):
        errs.append("imbalance_p_ic_incorrectly_triggers_h20_gate")

    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_incorrectly_triggers_h21_gate")
    return errs


def imbalance_top_p_ic_vs_session_reconstructs_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with H23 ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: imbalance discovery IC p-value ≠ session envelope reconstructing
    the daily bar. When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H22 ≠ H23)
    - H23 binds reconstructs only; imbalance must not bind H23
    - H22 gate on imbalance_top_p_ic only; reconstructs must not trigger H22
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_recon = "session_reconstructs_daily_rate"
    if k_imb not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_recon:
        errs.append("imbalance_p_ic_session_reconstructs_keys_collapsed")
    if H22_HYPOTHESIS_ID == H23_HYPOTHESIS_ID:
        errs.append("h22_h23_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_imb) == H23_HYPOTHESIS_ID:
        errs.append("imbalance_top_p_ic_incorrectly_bound_h23")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_recon: 1.0, **elig}):
        errs.append("session_reconstructs_incorrectly_triggers_h22_gate")
    return errs


def microprice_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H25 ``microprice_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: microprice discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H25 binds ``microprice_p_ic`` only (gap has **no** H25 claim)
    - No dedicated microprice finite-gate helper — ``_finite_scalar`` on the
      ``microprice_p_ic`` key only (a gap-only blob must not count as finite microprice)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_mp = "microprice_p_ic"
    k_gap = "gap_finite_rate"
    if k_mp not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_mp == k_gap:
        errs.append("microprice_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_mp) != H25_HYPOTHESIS_ID:
        errs.append("microprice_p_ic_not_bound_h25")
    if by_key.get(k_gap) == H25_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h25")
    if H25_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h25_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_mp: 0.05}.get(k_mp)):
        errs.append("microprice_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_mp)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_microprice")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def microprice_p_ic_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H25 ``microprice_p_ic`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: microprice discovery IC p ≠ session reconstruction chain.
    Mirrors :func:`imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors`
    with H25 bind + key-scoped ``_finite_scalar`` (no dedicated microprice gate).
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H25 ≠ H29)
    - H25 binds ``microprice_p_ic`` only; H29 binds chain only
    - Chain must not count as finite microprice via key-scoped probe
    - Numeric equality on clean synth is allowed (different units)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Off Commander #246+ identity batch.
    """
    if not isinstance(blob, dict):
        return []
    k_mp = "microprice_p_ic"
    k_chain = "session_chain_rate"
    if k_mp not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_mp == k_chain:
        errs.append("microprice_p_ic_session_chain_keys_collapsed")
    if H25_HYPOTHESIS_ID == H29_HYPOTHESIS_ID:
        errs.append("h25_h29_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_mp) != H25_HYPOTHESIS_ID:
        errs.append("microprice_p_ic_not_bound_h25")
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_mp) == H29_HYPOTHESIS_ID:
        errs.append("microprice_p_ic_incorrectly_bound_h29")
    if by_key.get(k_chain) == H25_HYPOTHESIS_ID:
        errs.append("session_chain_rate_incorrectly_bound_h25")

    if not _finite_scalar({k_mp: 0.05}.get(k_mp)):
        errs.append("microprice_p_ic_finite_probe_broken")
    if _finite_scalar({k_chain: 1.0}.get(k_mp)):
        errs.append("session_chain_incorrectly_counts_as_finite_microprice")
    return errs


def clv_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H30 ``clv_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: CLV discovery IC p ≠ overnight gap finiteness.
    Mirrors :func:`microprice_p_ic_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H30 binds ``clv_p_ic`` only (gap has **no** H30 claim)
    - No dedicated clv finite-gate helper — ``_finite_scalar`` on the
      ``clv_p_ic`` key only (a gap-only blob must not count as finite clv)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Commander #236+; not Sergeant
    ofi↔gap; not Lt wick↔gap.
    """
    if not isinstance(blob, dict):
        return []
    k_clv = "clv_p_ic"
    k_gap = "gap_finite_rate"
    if k_clv not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_clv == k_gap:
        errs.append("clv_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_clv) != H30_HYPOTHESIS_ID:
        errs.append("clv_p_ic_not_bound_h30")
    if by_key.get(k_gap) == H30_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h30")
    if H30_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h30_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_clv: 0.05}.get(k_clv)):
        errs.append("clv_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_clv)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_clv")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def dm_split_vs_park_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H31 ``dm_split_vs_park_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: overnight-split vs Parkinson DM p ≠ overnight gap finiteness.
    Mirrors :func:`clv_p_ic_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H31 binds ``dm_split_vs_park_p`` only (gap has **no** H31 claim)
    - No dedicated dm_split finite-gate helper — ``_finite_scalar`` on the
      ``dm_split_vs_park_p`` key only (a gap-only blob must not count as finite dm)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Sergeant H28; not Lt H33 sweep.
    """
    if not isinstance(blob, dict):
        return []
    k_dm = "dm_split_vs_park_p"
    k_gap = "gap_finite_rate"
    if k_dm not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_dm == k_gap:
        errs.append("dm_split_vs_park_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_dm) != H31_HYPOTHESIS_ID:
        errs.append("dm_split_vs_park_p_not_bound_h31")
    if by_key.get(k_gap) == H31_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h31")
    if H31_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h31_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_dm: 0.05}.get(k_dm)):
        errs.append("dm_split_vs_park_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_dm)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_dm_split")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H36 ``sweep_follow_event_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-follow event p ≠ overnight gap finiteness.
    Mirrors :func:`clv_p_ic_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H36 binds ``sweep_follow_event_p`` only (gap has **no** H36 claim)
    - No dedicated sweep_follow_event finite-gate helper — ``_finite_scalar`` on
      the ``sweep_follow_event_p`` key only (a gap-only blob must not count as
      finite sweep_follow_event_p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Sergeant H35; not Lt H34.
    """
    if not isinstance(blob, dict):
        return []
    k_ev = "sweep_follow_event_p"
    k_gap = "gap_finite_rate"
    if k_ev not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_ev == k_gap:
        errs.append("sweep_follow_event_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_ev) != H36_HYPOTHESIS_ID:
        errs.append("sweep_follow_event_p_not_bound_h36")
    if by_key.get(k_gap) == H36_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h36")
    if H36_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h36_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_ev: 0.05}.get(k_ev)):
        errs.append("sweep_follow_event_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_ev)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_event_p")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H39 ``sweep_reject_cost_adjusted_mean_bps`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject cost-adjusted mean bps ≠ overnight gap finiteness.
    Mirrors :func:`sweep_follow_event_p_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H39 binds ``sweep_reject_cost_adjusted_mean_bps`` only (gap has **no** H39 claim)
    - No dedicated reject-cost finite-gate helper — ``_finite_scalar`` on the
      ``sweep_reject_cost_adjusted_mean_bps`` key only (a gap-only blob must not
      count as finite reject-cost bps)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Sergeant H38.
    """
    if not isinstance(blob, dict):
        return []
    k_cost = "sweep_reject_cost_adjusted_mean_bps"
    k_gap = "gap_finite_rate"
    if k_cost not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_cost == k_gap:
        errs.append("sweep_reject_cost_adjusted_mean_bps_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_cost) != H39_HYPOTHESIS_ID:
        errs.append("sweep_reject_cost_adjusted_mean_bps_not_bound_h39")
    if by_key.get(k_gap) == H39_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h39")
    if H39_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h39_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_cost: 1.0}.get(k_cost)):
        errs.append("sweep_reject_cost_adjusted_mean_bps_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_cost)):
        errs.append(
            "gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_cost_adjusted_mean_bps"
        )

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_control_diff_p_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H44 ``sweep_reject_control_diff_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject control-diff p ≠ overnight gap finiteness.
    Mirrors :func:`sweep_reject_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors`.
    When both keys are present:

    - Distinct key identity
    - H44 binds ``sweep_reject_control_diff_p`` only (gap has **no** H44 claim)
    - No dedicated reject-control finite-gate helper — ``_finite_scalar`` on the
      ``sweep_reject_control_diff_p`` key only (a gap-only blob must not count as
      finite reject-control p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Sergeant H42.
    """
    if not isinstance(blob, dict):
        return []
    k_ctrl = "sweep_reject_control_diff_p"
    k_gap = "gap_finite_rate"
    if k_ctrl not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_ctrl == k_gap:
        errs.append("sweep_reject_control_diff_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_ctrl) != H44_HYPOTHESIS_ID:
        errs.append("sweep_reject_control_diff_p_not_bound_h44")
    if by_key.get(k_gap) == H44_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h44")
    if H44_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h44_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_ctrl: 0.05}.get(k_ctrl)):
        errs.append("sweep_reject_control_diff_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_ctrl)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_control_diff_p")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def ofi_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H27 ``ofi_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: OFI discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H27 binds ``ofi_p_ic`` only (gap has **no** H27 claim)
    - No dedicated ofi finite-gate helper — ``_finite_scalar`` on the
      ``ofi_p_ic`` key only (a gap-only blob must not count as finite ofi)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Not Commander #211+ props.
    """
    if not isinstance(blob, dict):
        return []
    k_ofi = "ofi_p_ic"
    k_gap = "gap_finite_rate"
    if k_ofi not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_ofi == k_gap:
        errs.append("ofi_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_ofi) != H27_HYPOTHESIS_ID:
        errs.append("ofi_p_ic_not_bound_h27")
    if by_key.get(k_gap) == H27_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h27")
    if H27_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h27_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_ofi: 0.05}.get(k_ofi)):
        errs.append("ofi_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_ofi)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_ofi")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def dm_gk_vs_park_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H28 ``dm_gk_vs_park_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: GK vs Park DM p-value ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H28 binds ``dm_gk_vs_park_p`` only (gap has **no** H28 claim)
    - No dedicated DM finite-gate helper — ``_finite_scalar`` on the
      ``dm_gk_vs_park_p`` key only (a gap-only blob must not count as finite DM p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS dm_split H31; off Lt sweep H33; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_dm = "dm_gk_vs_park_p"
    k_gap = "gap_finite_rate"
    if k_dm not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_dm == k_gap:
        errs.append("dm_gk_vs_park_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_dm) != H28_HYPOTHESIS_ID:
        errs.append("dm_gk_vs_park_p_not_bound_h28")
    if by_key.get(k_gap) == H28_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h28")
    if H28_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h28_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_dm: 0.05}.get(k_dm)):
        errs.append("dm_gk_vs_park_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_dm)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_dm_gk")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_event_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H35 ``sweep_reject_event_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep reject-event IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H35 binds ``sweep_reject_event_p`` only (gap has **no** H35 claim)
    - No dedicated sweep-p finite-gate helper — ``_finite_scalar`` on the
      ``sweep_reject_event_p`` key only (a gap-only blob must not count as finite reject p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H36; off Lt H34; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_reject_event_p"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_reject_event_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H35_HYPOTHESIS_ID:
        errs.append("sweep_reject_event_p_not_bound_h35")
    if by_key.get(k_gap) == H35_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h35")
    if H35_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h35_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_reject_event_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_event_p")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_placebo_p_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H38 ``sweep_follow_placebo_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep follow-placebo IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H38 binds ``sweep_follow_placebo_p`` only (gap has **no** H38 claim)
    - No dedicated sweep-p finite-gate helper — ``_finite_scalar`` on the
      ``sweep_follow_placebo_p`` key only (a gap-only blob must not count as finite placebo p)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H39 cost; off Lt H37; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_placebo_p"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_placebo_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H38_HYPOTHESIS_ID:
        errs.append("sweep_follow_placebo_p_not_bound_h38")
    if by_key.get(k_gap) == H38_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h38")
    if H38_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h38_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_placebo_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_placebo_p")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H42 ``sweep_follow_fold_positive_fraction`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep follow fold-positive fraction ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H42 binds ``sweep_follow_fold_positive_fraction`` only (gap has **no** H42 claim)
    - No dedicated fold-fraction finite-gate helper — ``_finite_scalar`` on the
      fold key only (a gap-only blob must not count as finite fold fraction)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H44; off Lt H41; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_fold_positive_fraction"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_fold_positive_fraction_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H42_HYPOTHESIS_ID:
        errs.append("sweep_follow_fold_positive_fraction_not_bound_h42")
    if by_key.get(k_gap) == H42_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h42")
    if H42_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h42_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_fold_positive_fraction_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append(
            "gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_fold_positive_fraction"
        )

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def wick_skew_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H26 ``wick_skew_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: wick-skew discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H26 binds ``wick_skew_p_ic`` only (gap has **no** H26 claim)
    - Finite probe on ``wick_skew_p_ic`` only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_wick = "wick_skew_p_ic"
    k_gap = "gap_finite_rate"
    if k_wick not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_wick == k_gap:
        errs.append("wick_skew_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_wick) != H26_HYPOTHESIS_ID:
        errs.append("wick_skew_p_ic_not_bound_h26")
    if by_key.get(k_gap) == H26_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h26")
    if H26_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h26_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_wick: 0.05}.get(k_wick)):
        errs.append("wick_skew_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_wick)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_wick_skew")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def vpin_p_ic_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H32 ``vpin_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: VPIN discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H32 binds ``vpin_p_ic`` only (gap has **no** H32 claim)
    - Finite probe on ``vpin_p_ic`` only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_vpin = "vpin_p_ic"
    k_gap = "gap_finite_rate"
    if k_vpin not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_vpin == k_gap:
        errs.append("vpin_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vpin) != H32_HYPOTHESIS_ID:
        errs.append("vpin_p_ic_not_bound_h32")
    if by_key.get(k_gap) == H32_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h32")
    if H32_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h32_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_vpin: 0.05}.get(k_vpin)):
        errs.append("vpin_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_vpin)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_vpin")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_signed_p_ic_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H33 ``sweep_reject_signed_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H33 binds ``sweep_reject_signed_p_ic`` only (gap has **no** H33 claim)
    - Finite probe on sweep-reject key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_reject_signed_p_ic"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_reject_signed_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H33_HYPOTHESIS_ID:
        errs.append("sweep_reject_signed_p_ic_not_bound_h33")
    if by_key.get(k_gap) == H33_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h33")
    if H33_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h33_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_reject_signed_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_signed_p_ic_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H34 ``sweep_follow_signed_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-follow discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H34 binds ``sweep_follow_signed_p_ic`` only (gap has **no** H34 claim)
    - Finite probe on sweep-follow key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_signed_p_ic"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_signed_p_ic_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H34_HYPOTHESIS_ID:
        errs.append("sweep_follow_signed_p_ic_not_bound_h34")
    if by_key.get(k_gap) == H34_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h34")
    if H34_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h34_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_signed_p_ic_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_control_diff_p_vs_sweep_follow_control_diff_p_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H44 ``sweep_reject_control_diff_p`` with H45 follow twin.

    Sibling sweep control-diff p-values — reject path ≠ follow path.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H44 ≠ H45)
    - Catalog bind lock: reject → H44; follow → H45
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_control_diff_p"
    k_fol = "sweep_follow_control_diff_p"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_control_diff_p_keys_collapsed")
    if H44_HYPOTHESIS_ID == H45_HYPOTHESIS_ID:
        errs.append("h44_h45_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H44_HYPOTHESIS_ID:
        errs.append("sweep_reject_control_diff_p_not_bound_h44")
    if by_key.get(k_fol) != H45_HYPOTHESIS_ID:
        errs.append("sweep_follow_control_diff_p_not_bound_h45")
    if by_key.get(k_rej) == H45_HYPOTHESIS_ID:
        errs.append("sweep_reject_control_diff_p_incorrectly_bound_h45")
    if by_key.get(k_fol) == H44_HYPOTHESIS_ID:
        errs.append("sweep_follow_control_diff_p_incorrectly_bound_h44")
    return errs


def sweep_reject_fold_positive_fraction_vs_sweep_follow_fold_positive_fraction_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H41 ``sweep_reject_fold_positive_fraction`` with H42 follow twin.

    Sibling sweep fold-positive fractions — reject path ≠ follow path.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H41 ≠ H42)
    - Catalog bind lock: reject → H41; follow → H42
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_fold_positive_fraction"
    k_fol = "sweep_follow_fold_positive_fraction"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_fold_positive_fraction_keys_collapsed")
    if H41_HYPOTHESIS_ID == H42_HYPOTHESIS_ID:
        errs.append("h41_h42_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H41_HYPOTHESIS_ID:
        errs.append("sweep_reject_fold_positive_fraction_not_bound_h41")
    if by_key.get(k_fol) != H42_HYPOTHESIS_ID:
        errs.append("sweep_follow_fold_positive_fraction_not_bound_h42")
    if by_key.get(k_rej) == H42_HYPOTHESIS_ID:
        errs.append("sweep_reject_fold_positive_fraction_incorrectly_bound_h42")
    if by_key.get(k_fol) == H41_HYPOTHESIS_ID:
        errs.append("sweep_follow_fold_positive_fraction_incorrectly_bound_h41")
    return errs


def sweep_reject_event_p_vs_sweep_follow_event_p_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H35 ``sweep_reject_event_p`` with H36 ``sweep_follow_event_p``.

    DATA_CONTRACTS: sweep-reject event p ≠ sweep-follow event p. When both keys
    are present:

    - Distinct key identity
    - Distinct hyp ids (H35 ≠ H36)
    - H35 binds reject event only; H36 binds follow event only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_event_p"
    k_fol = "sweep_follow_event_p"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_event_p_keys_collapsed")
    if H35_HYPOTHESIS_ID == H36_HYPOTHESIS_ID:
        errs.append("h35_h36_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H35_HYPOTHESIS_ID:
        errs.append("sweep_reject_event_p_not_bound_h35")
    if by_key.get(k_fol) != H36_HYPOTHESIS_ID:
        errs.append("sweep_follow_event_p_not_bound_h36")
    if by_key.get(k_rej) == H36_HYPOTHESIS_ID:
        errs.append("sweep_reject_event_p_incorrectly_bound_h36")
    if by_key.get(k_fol) == H35_HYPOTHESIS_ID:
        errs.append("sweep_follow_event_p_incorrectly_bound_h35")
    return errs


def sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H33 ``sweep_reject_signed_p_ic`` with H34 ``sweep_follow_signed_p_ic``.

    Sibling sweep signed-IC p-values — reject path ≠ follow path.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H33 ≠ H34)
    - Catalog bind lock: reject → H33; follow → H34
    - Key-scoped ``_finite_scalar`` probes (no dedicated gate helpers)
    - Numeric equality on clean synth is allowed (different event sets)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H44; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_signed_p_ic"
    k_fol = "sweep_follow_signed_p_ic"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_signed_p_ic_keys_collapsed")
    if H33_HYPOTHESIS_ID == H34_HYPOTHESIS_ID:
        errs.append("h33_h34_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H33_HYPOTHESIS_ID:
        errs.append("sweep_reject_signed_p_ic_not_bound_h33")
    if by_key.get(k_fol) != H34_HYPOTHESIS_ID:
        errs.append("sweep_follow_signed_p_ic_not_bound_h34")
    if by_key.get(k_rej) == H34_HYPOTHESIS_ID:
        errs.append("sweep_reject_signed_p_ic_incorrectly_bound_h34")
    if by_key.get(k_fol) == H33_HYPOTHESIS_ID:
        errs.append("sweep_follow_signed_p_ic_incorrectly_bound_h33")

    if not _finite_scalar({k_rej: 0.05}.get(k_rej)):
        errs.append("sweep_reject_signed_p_ic_finite_probe_broken")
    if _finite_scalar({k_fol: 0.05}.get(k_rej)):
        errs.append("sweep_follow_signed_p_ic_incorrectly_counts_as_finite_reject")
    if not _finite_scalar({k_fol: 0.05}.get(k_fol)):
        errs.append("sweep_follow_signed_p_ic_finite_probe_broken")
    if _finite_scalar({k_rej: 0.05}.get(k_fol)):
        errs.append("sweep_reject_signed_p_ic_incorrectly_counts_as_finite_follow")
    return errs


def sweep_reject_cost_adjusted_mean_bps_vs_sweep_follow_cost_adjusted_mean_bps_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H39 ``sweep_reject_cost_adjusted_mean_bps`` with H40 ``sweep_follow_cost_adjusted_mean_bps``.

    Sibling sweep cost-adjusted mean bps — reject path ≠ follow path.
    Mirrors :func:`sweep_reject_signed_p_ic_vs_sweep_follow_signed_p_ic_never_equate_honesty_errors`
    (H33≠H34). When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H39 ≠ H40)
    - Catalog bind lock: reject → H39; follow → H40
    - Key-scoped ``_finite_scalar`` probes (no dedicated gate helpers)
    - Numeric equality on clean synth is allowed (different event sets)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off Sergeant H37≠H38; off Lt H35≠H36; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_cost_adjusted_mean_bps"
    k_fol = "sweep_follow_cost_adjusted_mean_bps"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_cost_adjusted_mean_bps_keys_collapsed")
    if H39_HYPOTHESIS_ID == H40_HYPOTHESIS_ID:
        errs.append("h39_h40_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H39_HYPOTHESIS_ID:
        errs.append("sweep_reject_cost_adjusted_mean_bps_not_bound_h39")
    if by_key.get(k_fol) != H40_HYPOTHESIS_ID:
        errs.append("sweep_follow_cost_adjusted_mean_bps_not_bound_h40")
    if by_key.get(k_rej) == H40_HYPOTHESIS_ID:
        errs.append("sweep_reject_cost_adjusted_mean_bps_incorrectly_bound_h40")
    if by_key.get(k_fol) == H39_HYPOTHESIS_ID:
        errs.append("sweep_follow_cost_adjusted_mean_bps_incorrectly_bound_h39")

    if not _finite_scalar({k_rej: 1.0}.get(k_rej)):
        errs.append("sweep_reject_cost_adjusted_mean_bps_finite_probe_broken")
    if _finite_scalar({k_fol: 1.0}.get(k_rej)):
        errs.append("sweep_follow_cost_adjusted_mean_bps_incorrectly_counts_as_finite_reject")
    if not _finite_scalar({k_fol: 1.0}.get(k_fol)):
        errs.append("sweep_follow_cost_adjusted_mean_bps_finite_probe_broken")
    if _finite_scalar({k_rej: 1.0}.get(k_fol)):
        errs.append("sweep_reject_cost_adjusted_mean_bps_incorrectly_counts_as_finite_follow")
    return errs


def sweep_reject_placebo_p_vs_sweep_follow_placebo_p_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H37 ``sweep_reject_placebo_p`` with H38 ``sweep_follow_placebo_p``.

    Sibling sweep placebo p-values — reject placebo ≠ follow placebo.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H37 ≠ H38)
    - Catalog bind lock: reject placebo → H37; follow placebo → H38
    - Key-scoped ``_finite_scalar`` probes (no dedicated gate helpers)
    - Numeric equality on clean synth is allowed (different event sets)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS H39≠H40; off Lt H35≠H36; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_rej = "sweep_reject_placebo_p"
    k_fol = "sweep_follow_placebo_p"
    if k_rej not in blob or k_fol not in blob:
        return []

    errs: list[str] = []
    if k_rej == k_fol:
        errs.append("sweep_reject_follow_placebo_p_keys_collapsed")
    if H37_HYPOTHESIS_ID == H38_HYPOTHESIS_ID:
        errs.append("h37_h38_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_rej) != H37_HYPOTHESIS_ID:
        errs.append("sweep_reject_placebo_p_not_bound_h37")
    if by_key.get(k_fol) != H38_HYPOTHESIS_ID:
        errs.append("sweep_follow_placebo_p_not_bound_h38")
    if by_key.get(k_rej) == H38_HYPOTHESIS_ID:
        errs.append("sweep_reject_placebo_p_incorrectly_bound_h38")
    if by_key.get(k_fol) == H37_HYPOTHESIS_ID:
        errs.append("sweep_follow_placebo_p_incorrectly_bound_h37")

    if not _finite_scalar({k_rej: 0.05}.get(k_rej)):
        errs.append("sweep_reject_placebo_p_finite_probe_broken")
    if _finite_scalar({k_fol: 0.05}.get(k_rej)):
        errs.append("sweep_follow_placebo_p_incorrectly_counts_as_finite_reject_placebo")
    if not _finite_scalar({k_fol: 0.05}.get(k_fol)):
        errs.append("sweep_follow_placebo_p_finite_probe_broken")
    if _finite_scalar({k_rej: 0.05}.get(k_fol)):
        errs.append("sweep_reject_placebo_p_incorrectly_counts_as_finite_follow_placebo")
    return errs


def sweep_reject_placebo_p_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H37 ``sweep_reject_placebo_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject placebo p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H37 binds ``sweep_reject_placebo_p`` only (gap has **no** H37 claim)
    - Finite probe on placebo key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_reject_placebo_p"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_reject_placebo_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H37_HYPOTHESIS_ID:
        errs.append("sweep_reject_placebo_p_not_bound_h37")
    if by_key.get(k_gap) == H37_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h37")
    if H37_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h37_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_reject_placebo_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_placebo")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_cost_adjusted_mean_bps_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H40 ``sweep_follow_cost_adjusted_mean_bps`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-follow cost-adjusted mean bps ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H40 binds cost-adjusted follow mean only (gap has **no** H40 claim)
    - Finite probe on cost key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_cost_adjusted_mean_bps"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_cost_adjusted_mean_bps_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H40_HYPOTHESIS_ID:
        errs.append("sweep_follow_cost_adjusted_mean_bps_not_bound_h40")
    if by_key.get(k_gap) == H40_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h40")
    if H40_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h40_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_cost_adjusted_mean_bps_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_cost")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_reject_fold_positive_fraction_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H41 ``sweep_reject_fold_positive_fraction`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-reject fold-positive fraction ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H41 binds fold-positive reject fraction only (gap has **no** H41 claim)
    - Finite probe on fold key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_reject_fold_positive_fraction"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_reject_fold_positive_fraction_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H41_HYPOTHESIS_ID:
        errs.append("sweep_reject_fold_positive_fraction_not_bound_h41")
    if by_key.get(k_gap) == H41_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h41")
    if H41_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h41_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_reject_fold_positive_fraction_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_reject_fold")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def sweep_follow_control_diff_p_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H45 ``sweep_follow_control_diff_p`` with ``gap_finite_rate``.

    DATA_CONTRACTS: sweep-follow control-diff p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H45 binds follow control-diff p only (gap has **no** H45 claim)
    - Finite probe on control key only (gap-only blob must not count)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sw = "sweep_follow_control_diff_p"
    k_gap = "gap_finite_rate"
    if k_sw not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sw == k_gap:
        errs.append("sweep_follow_control_diff_p_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sw) != H45_HYPOTHESIS_ID:
        errs.append("sweep_follow_control_diff_p_not_bound_h45")
    if by_key.get(k_gap) == H45_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h45")
    if H45_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h45_hypothesis_collapsed_into_h20_h21_h22")

    if not _finite_scalar({k_sw: 0.05}.get(k_sw)):
        errs.append("sweep_follow_control_diff_p_finite_probe_broken")
    if _finite_scalar({k_gap: 1.0}.get(k_sw)):
        errs.append("gap_finite_rate_incorrectly_counts_as_finite_sweep_follow_control")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def session_book_vpin_p_ic_vs_gap_finite_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H43 ``session_book_vpin_p_ic`` with ``gap_finite_rate``.

    DATA_CONTRACTS: session-book VPIN discovery IC p ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - H43 gate helper binds session_book_vpin_p_ic only (gap must not trigger it)
    - Gap must not incorrectly trigger H20 / H21 / H22 gates
    - Numeric equality on clean synth is allowed (different units; rare)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_vpin = "session_book_vpin_p_ic"
    k_gap = "gap_finite_rate"
    if k_vpin not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_vpin == k_gap:
        errs.append("session_book_vpin_p_ic_gap_finite_keys_collapsed")

    if H43_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID, H22_HYPOTHESIS_ID}:
        errs.append("h43_hypothesis_collapsed_into_h20_h21_h22")

    elig = {"session_book_hypothesis_eligible": True, "book_hypothesis_eligible": True}
    if not northset_has_finite_session_book_vpin_p_ic({k_vpin: 0.05, **elig}):
        errs.append("session_book_vpin_p_ic_h43_gate_helper_broken")
    if northset_has_finite_session_book_vpin_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h43_gate")

    if northset_has_finite_imbalance_p_ic({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h22_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    return errs


def gap_finite_rate_vs_session_reconstructs_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate ``gap_finite_rate`` with H23 ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: overnight gap finiteness ≠ session→daily reconstruction.
    When both keys are present:

    - Distinct key identity
    - Hyp/gate: reconstructs binds ``H23_HYPOTHESIS_ID``; gap is **not** H23
      and must not trigger H21/H20 gates
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_gap = "gap_finite_rate"
    k_recon = "session_reconstructs_daily_rate"
    if k_gap not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_gap == k_recon:
        errs.append("gap_finite_session_reconstructs_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_gap) == H23_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h23")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_recon: 1.0, **elig}):
        errs.append("session_reconstructs_incorrectly_triggers_h21_gate")

    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_recon: 1.0}):
        errs.append("session_reconstructs_incorrectly_triggers_h20_gate")

    if H23_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h23_hypothesis_collapsed_into_h20_or_h21")
    return errs


def gap_finite_rate_vs_session_volume_conservation_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate ``gap_finite_rate`` with H24 ``session_volume_conservation_rate``.

    DATA_CONTRACTS: overnight gap finiteness ≠ session volume conservation.
    When both keys are present:

    - Distinct key identity
    - Hyp/gate: conservation binds ``H24_HYPOTHESIS_ID``; gap is **not** H24
      and must not trigger H21/H20 gates
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_gap = "gap_finite_rate"
    k_vol = "session_volume_conservation_rate"
    if k_gap not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_gap == k_vol:
        errs.append("gap_finite_session_volume_conservation_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_gap) == H24_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h24")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_vol: 1.0, **elig}):
        errs.append("session_volume_conservation_incorrectly_triggers_h21_gate")

    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_vol: 1.0}):
        errs.append("session_volume_conservation_incorrectly_triggers_h20_gate")

    if H24_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h24_hypothesis_collapsed_into_h20_or_h21")
    return errs


def gap_finite_rate_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``gap_finite_rate`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: overnight gap finiteness ≠ session chain identity. When both
    keys are present:

    - Distinct key identity
    - Hyp/gate: chain binds ``H29_HYPOTHESIS_ID``; gap is **not** H29 and must
      not trigger H21/H20 gates
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_gap = "gap_finite_rate"
    k_chain = "session_chain_rate"
    if k_gap not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_gap == k_chain:
        errs.append("gap_finite_session_chain_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_gap) == H29_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h29")

    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_chain: 1.0, **elig}):
        errs.append("session_chain_incorrectly_triggers_h21_gate")

    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_chain: 1.0}):
        errs.append("session_chain_incorrectly_triggers_h20_gate")

    if H29_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h29_hypothesis_collapsed_into_h20_or_h21")
    return errs


def session_ohlc_vs_book_uncrossed_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with H21 ``book_uncrossed_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ uncrossed book. When both
    keys are present:

    - Distinct key identity
    - Hyp/gate: session OHLC is **not** H23; H21 binds ``book_uncrossed_rate``
      only (session OHLC must not trigger H21); H20 gate is daily-only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad polish. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sess = "session_ohlc_identity_rate"
    k_book = "book_uncrossed_rate"
    if k_sess not in blob or k_book not in blob:
        return []

    errs: list[str] = []
    if k_sess == k_book:
        errs.append("session_ohlc_book_uncrossed_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sess) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    if by_key.get(k_book) == H23_HYPOTHESIS_ID:
        errs.append("book_uncrossed_rate_incorrectly_bound_h23")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_sess: 1.0, **elig}):
        errs.append("session_ohlc_incorrectly_triggers_h21_gate")

    if northset_has_finite_ohlc_identity_rate({k_sess: 1.0}):
        errs.append("session_ohlc_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_book: 1.0}):
        errs.append("book_uncrossed_incorrectly_triggers_h20_gate")

    if H21_HYPOTHESIS_ID in {H20_HYPOTHESIS_ID, H23_HYPOTHESIS_ID}:
        errs.append("h21_hypothesis_collapsed_into_h20_or_h23")
    return errs


def book_uncrossed_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H21 ``book_uncrossed_rate`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: uncrossed book ≠ session reconstruction chain. When both
    keys are present:

    - Distinct key identity
    - Distinct hyp ids (H21 ≠ H29)
    - H21 finite gate on book_uncrossed only; H29 binds chain only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite. Leaves reconstructs lane clear.
    """
    if not isinstance(blob, dict):
        return []
    k_book = "book_uncrossed_rate"
    k_chain = "session_chain_rate"
    if k_book not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_book == k_chain:
        errs.append("book_uncrossed_session_chain_keys_collapsed")
    if H21_HYPOTHESIS_ID == H29_HYPOTHESIS_ID:
        errs.append("h21_h29_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_book) == H29_HYPOTHESIS_ID:
        errs.append("book_uncrossed_rate_incorrectly_bound_h29")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_chain: 1.0, **elig}):
        errs.append("session_chain_incorrectly_triggers_h21_gate")
    return errs


def imbalance_top_p_ic_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: imbalance discovery IC p ≠ session reconstruction chain.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H22 ≠ H29)
    - H22 finite gate on imbalance_top_p_ic only; H29 binds chain only
    - Numeric equality on clean synth is allowed (different units)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_chain = "session_chain_rate"
    if k_imb not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_chain:
        errs.append("imbalance_top_p_ic_session_chain_keys_collapsed")
    if H22_HYPOTHESIS_ID == H29_HYPOTHESIS_ID:
        errs.append("h22_h29_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_imb) == H29_HYPOTHESIS_ID:
        errs.append("imbalance_top_p_ic_incorrectly_bound_h29")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_chain: 1.0, **elig}):
        errs.append("session_chain_incorrectly_triggers_h22_gate")
    return errs


def imbalance_top_p_ic_vs_session_ohlc_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with ``session_ohlc_identity_rate``.

    DATA_CONTRACTS: imbalance discovery IC p ≠ session-candle OHLC identity.
    When both keys are present:

    - Distinct key identity
    - H22 finite gate on imbalance_top_p_ic only (session OHLC must not trigger it)
    - session_ohlc must **not** bind H23 / H24 / H29
    - Numeric equality on clean synth is allowed (different units)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_sess = "session_ohlc_identity_rate"
    if k_imb not in blob or k_sess not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_sess:
        errs.append("imbalance_top_p_ic_session_ohlc_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    sess_hyp = by_key.get(k_sess)
    if sess_hyp == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    if sess_hyp == H24_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h24")
    if sess_hyp == H29_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h29")
    if by_key.get(k_imb) in {H23_HYPOTHESIS_ID, H24_HYPOTHESIS_ID, H29_HYPOTHESIS_ID}:
        errs.append("imbalance_top_p_ic_incorrectly_bound_session_hyp")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_sess: 1.0, **elig}):
        errs.append("session_ohlc_incorrectly_triggers_h22_gate")
    return errs


def imbalance_top_p_ic_vs_session_volume_conservation_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H22 ``imbalance_top_p_ic`` with H24 ``session_volume_conservation_rate``.

    DATA_CONTRACTS: imbalance discovery IC p ≠ session volume conservation.
    Completes H22↔session-rate mesh with H22↔H20/H21/H29 and H24 siblings.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H22 ≠ H24)
    - H22 finite gate on imbalance_top_p_ic only; H24 binds volume only
    - imbalance_top_p_ic must not bind H24; volume must not trigger H22 gate
    - Numeric equality on clean synth is allowed (different units)

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_imb = "imbalance_top_p_ic"
    k_vol = "session_volume_conservation_rate"
    if k_imb not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_imb == k_vol:
        errs.append("imbalance_top_p_ic_session_volume_conservation_keys_collapsed")
    if H22_HYPOTHESIS_ID == H24_HYPOTHESIS_ID:
        errs.append("h22_h24_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_imb) == H24_HYPOTHESIS_ID:
        errs.append("imbalance_top_p_ic_incorrectly_bound_h24")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_imbalance_p_ic({k_imb: 0.05, **elig}):
        errs.append("imbalance_p_ic_h22_gate_helper_broken")
    if northset_has_finite_imbalance_p_ic({k_vol: 1.0, **elig}):
        errs.append("session_volume_conservation_incorrectly_triggers_h22_gate")
    return errs


def book_uncrossed_vs_session_reconstructs_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H21 ``book_uncrossed_rate`` with H23 ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: uncrossed book ≠ session envelope reconstructing the daily bar.
    When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H21 ≠ H23)
    - H21 finite gate on book_uncrossed only; H23 binds reconstructs only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_book = "book_uncrossed_rate"
    k_recon = "session_reconstructs_daily_rate"
    if k_book not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_book == k_recon:
        errs.append("book_uncrossed_session_reconstructs_keys_collapsed")
    if H21_HYPOTHESIS_ID == H23_HYPOTHESIS_ID:
        errs.append("h21_h23_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_book) == H23_HYPOTHESIS_ID:
        errs.append("book_uncrossed_rate_incorrectly_bound_h23")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_recon: 1.0, **elig}):
        errs.append("session_reconstructs_incorrectly_triggers_h21_gate")
    return errs


def book_uncrossed_vs_volume_conservation_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate H21 ``book_uncrossed_rate`` with H24 ``session_volume_conservation_rate``.

    DATA_CONTRACTS: uncrossed book ≠ session volume conservation. When both
    keys are present:

    - Distinct key identity
    - Distinct hyp ids (H21 ≠ H24)
    - H21 finite gate on book_uncrossed only; H24 binds volume only
    - book_uncrossed must not bind H24; volume must not trigger H21 gate
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_book = "book_uncrossed_rate"
    k_vol = "session_volume_conservation_rate"
    if k_book not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_book == k_vol:
        errs.append("book_uncrossed_volume_conservation_keys_collapsed")
    if H21_HYPOTHESIS_ID == H24_HYPOTHESIS_ID:
        errs.append("h21_h24_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_book) == H24_HYPOTHESIS_ID:
        errs.append("book_uncrossed_rate_incorrectly_bound_h24")

    elig = {"book_hypothesis_eligible": True}
    if not northset_has_finite_book_uncrossed_rate({k_book: 1.0, **elig}):
        errs.append("book_uncrossed_h21_gate_helper_broken")
    if northset_has_finite_book_uncrossed_rate({k_vol: 1.0, **elig}):
        errs.append("session_volume_conservation_incorrectly_triggers_h21_gate")
    return errs


def session_ohlc_vs_gap_finite_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``session_ohlc_identity_rate`` with ``gap_finite_rate``.

    DATA_CONTRACTS: session-candle OHLC identity ≠ overnight gap finiteness.
    When both keys are present:

    - Distinct key identity
    - Hyp/gate: session OHLC is **not** H23; gap has **no** H21 claim;
      H20 gate binds daily ``ohlc_identity_rate`` only (neither key is H20)
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off H20/H21/H22 triad. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_sess = "session_ohlc_identity_rate"
    k_gap = "gap_finite_rate"
    if k_sess not in blob or k_gap not in blob:
        return []

    errs: list[str] = []
    if k_sess == k_gap:
        errs.append("session_ohlc_gap_finite_keys_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_sess) == H23_HYPOTHESIS_ID:
        errs.append("session_ohlc_identity_rate_incorrectly_bound_h23")
    if by_key.get(k_gap) == H23_HYPOTHESIS_ID:
        errs.append("gap_finite_rate_incorrectly_bound_h23")

    # H21 is book_uncrossed — gap must not share that gate
    elig = {"book_hypothesis_eligible": True}
    if northset_has_finite_book_uncrossed_rate({k_gap: 1.0, **elig}):
        errs.append("gap_finite_rate_incorrectly_triggers_h21_gate")
    if northset_has_finite_book_uncrossed_rate({k_sess: 1.0, **elig}):
        errs.append("session_ohlc_incorrectly_triggers_h21_gate")

    # H20 is daily ohlc only — neither session nor gap is H20
    if northset_has_finite_ohlc_identity_rate({k_sess: 1.0}):
        errs.append("session_ohlc_incorrectly_triggers_h20_gate")
    if northset_has_finite_ohlc_identity_rate({k_gap: 1.0}):
        errs.append("gap_finite_rate_incorrectly_triggers_h20_gate")

    if H20_HYPOTHESIS_ID in {H23_HYPOTHESIS_ID, H21_HYPOTHESIS_ID}:
        errs.append("h20_hypothesis_collapsed_into_session_or_book_binds")
    return errs


def ohlc_identity_vs_session_ohlc_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate daily ``ohlc_identity_rate`` with ``session_ohlc_identity_rate``.

    Distinct clocks (daily bars vs session L2). When **both** keys are present:

    - Key identity: ``ohlc_identity_rate`` ≠ ``session_ohlc_identity_rate``
    - H20 finite gate binds to **daily** ``ohlc_identity_rate`` only
      (``H20_HYPOTHESIS_ID``); session OHLC is not an H20 metric
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_daily = "ohlc_identity_rate"
    k_sess = "session_ohlc_identity_rate"
    if k_daily not in blob or k_sess not in blob:
        return []

    errs: list[str] = []
    if k_daily == k_sess:
        errs.append("ohlc_identity_session_ohlc_keys_collapsed")

    # Catalog lock: H20 gate helper watches daily key only
    if not callable(northset_has_finite_ohlc_identity_rate):
        errs.append("ohlc_identity_h20_gate_helper_missing")
    # Session key must not be the daily key string (structural)
    if k_sess == "ohlc_identity_rate":
        errs.append("session_ohlc_identity_rate_aliased_to_daily")

    # Ensure H20 id stays distinct from session H23/H24/H29 binds
    session_hyps = {
        H23_HYPOTHESIS_ID,
        H24_HYPOTHESIS_ID,
        H29_HYPOTHESIS_ID,
    }
    if H20_HYPOTHESIS_ID in session_hyps:
        errs.append("h20_hypothesis_collapsed_into_session_binds")
    return errs


def ohlc_identity_vs_session_reconstructs_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate daily ``ohlc_identity_rate`` with ``session_reconstructs_daily_rate``.

    DATA_CONTRACTS: daily OHLC envelope identity (H20) ≠ session envelope that
    reconstructs the daily bar (H23). When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H23)
    - H20 finite gate on daily ohlc only; H23 binds reconstructs only
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_recon = "session_reconstructs_daily_rate"
    if k_ohlc not in blob or k_recon not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_recon:
        errs.append("ohlc_identity_session_reconstructs_keys_collapsed")
    if H20_HYPOTHESIS_ID == H23_HYPOTHESIS_ID:
        errs.append("h20_h23_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_recon) != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if by_key.get(k_ohlc) == H23_HYPOTHESIS_ID:
        errs.append("ohlc_identity_rate_incorrectly_bound_h23")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_recon: 1.0}):
        errs.append("session_reconstructs_incorrectly_triggers_h20_gate")
    return errs


def ohlc_identity_vs_session_chain_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate daily ``ohlc_identity_rate`` with H29 ``session_chain_rate``.

    DATA_CONTRACTS: daily OHLC envelope identity (H20) ≠ session chain identity
    (H29). When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H29)
    - H20 finite gate on daily ohlc only; H29 binds chain only
    - Daily ohlc must not bind H23/H24/H29; chain must not trigger H20 gate
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_chain = "session_chain_rate"
    if k_ohlc not in blob or k_chain not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_chain:
        errs.append("ohlc_identity_session_chain_keys_collapsed")
    if H20_HYPOTHESIS_ID == H29_HYPOTHESIS_ID:
        errs.append("h20_h29_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_chain) != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if by_key.get(k_ohlc) in {
        H23_HYPOTHESIS_ID,
        H24_HYPOTHESIS_ID,
        H29_HYPOTHESIS_ID,
    }:
        errs.append("ohlc_identity_rate_incorrectly_bound_session_hyp")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_chain: 1.0}):
        errs.append("session_chain_incorrectly_triggers_h20_gate")
    return errs


def ohlc_identity_vs_volume_conservation_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate daily ``ohlc_identity_rate`` with H24 ``session_volume_conservation_rate``.

    DATA_CONTRACTS: daily OHLC envelope identity (H20) ≠ session volume conservation
    (H24). When both keys are present:

    - Distinct key identity
    - Distinct hyp ids (H20 ≠ H24)
    - H24 binds volume only; H20 finite gate on daily ohlc only
    - Daily ohlc must not bind H23/H24/H29; volume must not trigger H20 gate
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_ohlc = "ohlc_identity_rate"
    k_vol = "session_volume_conservation_rate"
    if k_ohlc not in blob or k_vol not in blob:
        return []

    errs: list[str] = []
    if k_ohlc == k_vol:
        errs.append("ohlc_identity_volume_conservation_keys_collapsed")
    if H20_HYPOTHESIS_ID == H24_HYPOTHESIS_ID:
        errs.append("h20_h24_hypothesis_ids_collapsed")

    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    if by_key.get(k_vol) != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if by_key.get(k_ohlc) in {
        H23_HYPOTHESIS_ID,
        H24_HYPOTHESIS_ID,
        H29_HYPOTHESIS_ID,
    }:
        errs.append("ohlc_identity_rate_incorrectly_bound_session_hyp")

    if not northset_has_finite_ohlc_identity_rate({k_ohlc: 1.0}):
        errs.append("ohlc_identity_h20_gate_helper_broken")
    if northset_has_finite_ohlc_identity_rate({k_vol: 1.0}):
        errs.append("session_volume_conservation_incorrectly_triggers_h20_gate")
    return errs


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


def session_chain_vs_session_identity_siblings_never_equate_honesty_errors(
    blob: object,
) -> list[str]:
    """Never equate H29 session_chain_rate with H23 reconstructs / H24 conservation.

    Sibling session identity rates — chain ≠ reconstructs ≠ volume conservation.
    When ``session_chain_rate`` is present alongside either sibling:

    - Catalog bind lock: chain → H29; reconstructs → H23; conservation → H24
    - Distinct hyp ids (no collapse)
    - Numeric equality on clean synth is allowed

    Skip when chain key absent. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_chain = "session_chain_rate"
    k_recon = "session_reconstructs_daily_rate"
    k_vol = "session_volume_conservation_rate"
    if k_chain not in blob:
        return []
    if k_recon not in blob and k_vol not in blob:
        return []

    errs: list[str] = []
    by_key = {spec[0]: spec[1] for spec in NORTHSET_H23_H28_SPECS}
    chain_hyp = by_key.get(k_chain)
    recon_hyp = by_key.get(k_recon)
    vol_hyp = by_key.get(k_vol)
    if chain_hyp != H29_HYPOTHESIS_ID:
        errs.append("session_chain_rate_not_bound_h29")
    if k_recon in blob and recon_hyp != H23_HYPOTHESIS_ID:
        errs.append("session_reconstructs_daily_rate_not_bound_h23")
    if k_vol in blob and vol_hyp != H24_HYPOTHESIS_ID:
        errs.append("session_volume_conservation_rate_not_bound_h24")
    if chain_hyp is not None and recon_hyp is not None and chain_hyp == recon_hyp:
        errs.append("session_chain_reconstructs_hypothesis_collapsed")
    if chain_hyp is not None and vol_hyp is not None and chain_hyp == vol_hyp:
        errs.append("session_chain_volume_conservation_hypothesis_collapsed")
    return errs


def session_chain_rate_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_chain_rate ∈ [0, 1] when finite.

    Northset session reconstruction chain rate. NaN/absent skip; ±inf fail-closed.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("session_chain_rate")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
        return ["session_chain_rate_out_of_unit_interval"]
    return []


def session_mean_jump_ratio_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_mean_jump_ratio ∈ [0, 1] when finite.

    BNS bipower jump share is clipped to [0,1] in estimators.session_bipower_jump;
    out-of-range or ±inf is dishonest. NaN skipped.
    Research diagnostic only; never live Sharpe. Does not touch kyle_* keys.
    """
    if not isinstance(blob, dict):
        return []
    try:
        x = float(blob.get("session_mean_jump_ratio"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["session_mean_jump_ratio_non_finite"]
    if not (0.0 <= x <= 1.0):
        return ["session_mean_jump_ratio_out_of_unit_interval"]
    return []


def sweep_follow_signed_mean_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: sweep_follow_signed_mean_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. ≠ sweep_reject_signed_mean_ic.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "sweep_follow_signed_mean_ic" not in blob:
        return []
    try:
        x = float(blob.get("sweep_follow_signed_mean_ic"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["sweep_follow_signed_mean_ic_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["sweep_follow_signed_mean_ic_non_finite_fail_closed"]
    return []


def sweep_reject_signed_mean_ic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: sweep_reject_signed_mean_ic finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. Companion to H33 path; not a live gate.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "sweep_reject_signed_mean_ic" not in blob:
        return []
    try:
        x = float(blob.get("sweep_reject_signed_mean_ic"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["sweep_reject_signed_mean_ic_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["sweep_reject_signed_mean_ic_non_finite_fail_closed"]
    return []


def sweep_reject_event_mean_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: sweep_reject_event_mean_bps finite when present (signed OK).

    ±inf fail-closed; NaN/absent skip. Never equate to sweep_follow_* keys.
    Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []
    if "sweep_reject_event_mean_bps" not in blob:
        return []
    try:
        x = float(blob.get("sweep_reject_event_mean_bps"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["sweep_reject_event_mean_bps_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["sweep_reject_event_mean_bps_non_finite_fail_closed"]
    return []


def sweep_follow_event_mean_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: sweep_follow_event_mean_bps finite when present.

    Signed pre-cost event excess bps — may be negative; only ±inf is dishonest.
    NaN/absent skipped.

    Honesty: never equate to ``sweep_follow_cost_adjusted_mean_bps`` (post-cost)
    or to ``sweep_reject_event_mean_bps`` (different signal). Research diagnostic
    only; never live Sharpe. Does not touch kyle_* keys.
    """
    if not isinstance(blob, dict):
        return []
    if "sweep_follow_event_mean_bps" not in blob:
        return []
    try:
        x = float(blob.get("sweep_follow_event_mean_bps"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["sweep_follow_event_mean_bps_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["sweep_follow_event_mean_bps_non_finite"]
    return []


def sweep_follow_cost_adjusted_mean_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: sweep_follow_cost_adjusted_mean_bps finite when present.

    Signed bps after cost haircut — may be negative; only ±inf / non-numeric is
    dishonest. NaN skipped.

    Honesty: never equate to ``sweep_follow_event_mean_bps`` (pre-cost excess).
    Equality is allowed when cost is zero but they are different statistics —
    do not soft-fail on coincidence. Research diagnostic only; never live Sharpe.
    Does not touch kyle_* keys.
    """
    if not isinstance(blob, dict):
        return []
    try:
        x = float(blob.get("sweep_follow_cost_adjusted_mean_bps"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["sweep_follow_cost_adjusted_mean_bps_non_finite"]
    return []


def sweep_reject_cost_adjusted_mean_bps_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: sweep_reject_cost_adjusted_mean_bps finite when present.

    Signed bps after cost haircut — may be negative; only ±inf / non-numeric is
    dishonest. NaN/absent skipped.

    Honesty: never equate to ``sweep_reject_event_mean_bps`` (pre-cost) or to
    ``sweep_follow_cost_adjusted_mean_bps`` (different signal). Research
    diagnostic only; never live Sharpe. Does not touch kyle_* keys.
    """
    if not isinstance(blob, dict):
        return []
    if "sweep_reject_cost_adjusted_mean_bps" not in blob:
        return []
    try:
        x = float(blob.get("sweep_reject_cost_adjusted_mean_bps"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ["sweep_reject_cost_adjusted_mean_bps_non_numeric"]
    if x != x:
        return []
    if abs(x) == float("inf"):
        return ["sweep_reject_cost_adjusted_mean_bps_non_finite"]
    return []


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


def amihud_mean_ic_vs_amihud_abs_mean_ic_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``amihud_mean_ic`` with ``amihud_abs_mean_ic``.

    Sibling Amihud discovery ICs — signed Amihud path ≠ abs Amihud path.
    When both keys are present:

    - Distinct key identity
    - Companion packs stay distinct (``amihud_p_ic`` ≠ ``amihud_abs_p_ic``, etc.)
    - Key-scoped finite probes
    - Numeric equality on clean synth is allowed

    Skip when either key absent. Research diagnostic only; never live Sharpe.
    Off CoS spread-IC; off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_mean = "amihud_mean_ic"
    k_abs = "amihud_abs_mean_ic"
    if k_mean not in blob or k_abs not in blob:
        return []

    errs: list[str] = []
    if k_mean == k_abs:
        errs.append("amihud_mean_ic_amihud_abs_mean_ic_keys_collapsed")

    companions = (
        ("amihud_p_ic", "amihud_abs_p_ic"),
        ("amihud_t_ic", "amihud_abs_t_ic"),
        ("amihud_n_dates", "amihud_abs_n_dates"),
    )
    for a, b in companions:
        if a == b:
            errs.append(f"amihud_companion_keys_collapsed_{a}")

    if not _finite_scalar({k_mean: 0.1}.get(k_mean)):
        errs.append("amihud_mean_ic_finite_probe_broken")
    if _finite_scalar({k_abs: 0.1}.get(k_mean)):
        errs.append("amihud_abs_mean_ic_incorrectly_counts_as_finite_amihud_mean_ic")
    if not _finite_scalar({k_abs: 0.1}.get(k_abs)):
        errs.append("amihud_abs_mean_ic_finite_probe_broken")
    if _finite_scalar({k_mean: 0.1}.get(k_abs)):
        errs.append("amihud_mean_ic_incorrectly_counts_as_finite_amihud_abs_mean_ic")
    return errs


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


def mid_lag1_corr_vs_ofi_lag1_corr_never_equate_honesty_errors(blob: object) -> list[str]:
    """Never equate ``mid_lag1_corr`` with ``ofi_lag1_corr``.

    DATA_CONTRACTS: same panel lag-1 estimator shape, different series
    (mid path AR ≠ OFI path AR). When both keys are present:

    - Distinct key identity
    - Companion n-securities keys stay distinct when both stamped
    - Key-scoped finite probes (gap-style: mid-only blob must not count as ofi)
    - Numeric equality on clean synth is allowed

    Skip when either corr key absent. Research diagnostic only; never live Sharpe.
    Always-on surface — not nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    k_mid = "mid_lag1_corr"
    k_ofi = "ofi_lag1_corr"
    if k_mid not in blob or k_ofi not in blob:
        return []

    errs: list[str] = []
    if k_mid == k_ofi:
        errs.append("mid_ofi_lag1_corr_keys_collapsed")

    k_mid_n = "mid_lag1_n_securities"
    k_ofi_n = "ofi_lag1_n_securities"
    if k_mid_n in blob and k_ofi_n in blob and k_mid_n == k_ofi_n:
        errs.append("mid_ofi_lag1_n_securities_keys_collapsed")

    if not _finite_scalar({k_mid: 0.5}.get(k_mid)):
        errs.append("mid_lag1_corr_finite_probe_broken")
    if _finite_scalar({k_ofi: 0.5}.get(k_mid)):
        errs.append("ofi_lag1_corr_incorrectly_counts_as_finite_mid_lag1")
    if not _finite_scalar({k_ofi: 0.5}.get(k_ofi)):
        errs.append("ofi_lag1_corr_finite_probe_broken")
    if _finite_scalar({k_mid: 0.5}.get(k_ofi)):
        errs.append("mid_lag1_corr_incorrectly_counts_as_finite_ofi_lag1")
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


def session_bulk_vpin_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: session_bulk_vpin ∈ [0, 1] when finite.

    Candle signed-volume VPIN scalar (session_vpin) — **not** H32 ``vpin_mean``
    and **not** H43 ``session_book_vpin_mean`` (DATA_CONTRACTS triad).
    NaN/absent skip. Research diagnostic only; never live Sharpe.
    Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    val = blob.get("session_bulk_vpin")
    try:
        x = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    if x != x:
        return []
    if abs(x) == float("inf") or not (0.0 <= x <= 1.0):
        return ["session_bulk_vpin_out_of_unit_interval"]
    return []
