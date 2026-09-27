"""Northset session-path mean and identity honesty checks.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from .primitives import _ic_pack_honesty_errors


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


def northset_use_session_l2_gate_consistency_errors(blob: object) -> list[str]:
    """Soft-verify ``use_session_l2`` ↔ ``session_l2_identity_gate`` when both present.

    Stamp contract from ``bench_northset``:
    - ``use_session_l2 is True`` ⇔ gate ``"enforced"``
    - ``use_session_l2 is False`` ⇔ gate ``"skipped"``

    One present without the other → skip (pair incomplete). Invalid gate string
    still caught by floor/string-enum helpers. Research diagnostic only; never
    live Sharpe. Off nest invent / kyle_ofi overwrite.
    """
    if not isinstance(blob, dict):
        return []
    if "use_session_l2" not in blob or "session_l2_identity_gate" not in blob:
        return []
    use = blob.get("use_session_l2")
    gate = blob.get("session_l2_identity_gate")
    if type(use) is not bool:
        return []  # bool-flags helper owns type
    if gate not in ("enforced", "skipped"):
        return []  # enum helper owns invalid gate
    if use and gate != "enforced":
        return ["use_session_l2_true_gate_not_enforced"]
    if (not use) and gate != "skipped":
        return ["use_session_l2_false_gate_not_skipped"]
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


def northset_session_book_snaps_n_session_consistency_errors(blob: object) -> list[str]:
    """Soft-verify mean_session_book_snaps vs n_session_book_rows / n_session_candles.

    Fail-closed when both sides of the session-L2 coverage pair are present and
    disagree:

    - Finite ``mean_session_book_snaps`` (path on) → ``n_session_book_rows`` must
      be present and ``> 0``; if ``n_session_candles`` is also present, rows must
      equal candles (1:1 session book vs session candle panel totals).
    - ``n_session_book_rows > 0`` with ``mean_session_book_snaps`` present → mean
      must be finite and ``> 0``.

    Skip when mean is NaN/absent and rows are 0/absent (session L2 off).
    Never equate ``mean_session_book_snaps`` to ``n_session_candles`` (per-parent
    mean ≠ panel row count). Research diagnostic only; never live Sharpe.
    """
    if not isinstance(blob, dict):
        return []

    has_mean = "mean_session_book_snaps" in blob
    has_rows = "n_session_book_rows" in blob and blob.get("n_session_book_rows") is not None
    has_candles = "n_session_candles" in blob and blob.get("n_session_candles") is not None

    mean_x: float | None = None
    if has_mean:
        raw_mean = blob.get("mean_session_book_snaps")
        if raw_mean is None:
            mean_x = None
        elif isinstance(raw_mean, bool) or not isinstance(raw_mean, (int, float)):
            # Present-but-malformed must not read as absent/NaN.
            return ["mean_session_book_snaps_non_numeric"]
        else:
            mean_x = float(raw_mean)

    # Parse through float first: int(float("inf")) raises OverflowError, which
    # a JSON receipt with Infinity counts would otherwise crash on.
    rows_n: int | None = None
    if has_rows:
        try:
            rows_f = float(blob.get("n_session_book_rows"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return ["n_session_book_rows_non_integer"]
        if rows_f != rows_f or abs(rows_f) == float("inf") or rows_f != int(rows_f):
            return ["n_session_book_rows_non_integer"]
        rows_n = int(rows_f)

    candles_n: int | None = None
    if has_candles:
        try:
            candles_f = float(blob.get("n_session_candles"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return ["n_session_candles_non_integer"]
        if candles_f != candles_f or abs(candles_f) == float("inf") or candles_f != int(candles_f):
            return ["n_session_candles_non_integer"]
        candles_n = int(candles_f)

    mean_finite = mean_x is not None and mean_x == mean_x and abs(mean_x) != float("inf")
    mean_active = mean_finite and mean_x > 0.0  # type: ignore[operator]
    rows_pos = rows_n is not None and rows_n > 0

    # Session L2 off: NaN/absent mean + zero/absent rows → skip
    if not mean_active and not rows_pos:
        return []

    errs: list[str] = []

    if mean_active:
        if not has_rows:
            errs.append("n_session_book_rows_missing_despite_mean_session_book_snaps")
        elif rows_n is not None and rows_n <= 0:
            errs.append("n_session_book_rows_non_positive_despite_mean_session_book_snaps")
        elif rows_n is not None and candles_n is not None and rows_n != candles_n:
            errs.append("n_session_book_rows_ne_n_session_candles")

    if rows_pos and has_mean and (not mean_finite or not (mean_x > 0.0)):  # type: ignore[operator]
        errs.append("mean_session_book_snaps_non_positive_despite_n_session_book_rows")

    return errs


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


NORTHSET_SESSION_MEANS_HONESTY_HELPERS = (
    northset_session_imbalance_mean_honesty_errors,
    northset_session_close_micro_bps_honesty_errors,
    northset_session_close_depth_honesty_errors,
    northset_session_imbalance_std_honesty_errors,
    northset_session_close_imbalance_honesty_errors,
    northset_session_close_mid_honesty_errors,
    northset_session_close_mid_micro_pair_honesty_errors,
    northset_session_close_spread_bps_honesty_errors,
    northset_session_spread_bps_mean_honesty_errors,
    northset_session_book_snaps_honesty_errors,
    northset_session_book_snaps_n_session_consistency_errors,
    northset_session_ofi_sum_mean_honesty_errors,
    northset_session_ofi_sum_ic_honesty_errors,
    northset_session_ofi_abs_sum_honesty_errors,
    northset_session_ofi_abs_dominates_sum_honesty_errors,
    northset_session_book_vpin_mean_honesty_errors,
)


def northset_session_means_honesty_errors(blob: object) -> list[str]:
    """Dispatcher: fan into all session-mean soft-verify helpers.

    Keeps individual helpers for unit tests; verify.py can call this once.
    Research diagnostic only; never live Sharpe / promotion.
    """
    errors: list[str] = []
    for fn in NORTHSET_SESSION_MEANS_HONESTY_HELPERS:
        errors.extend(fn(blob))
    return errors


__all__ = [
    "NORTHSET_SESSION_MEANS_HONESTY_HELPERS",
    "northset_session_book_snaps_honesty_errors",
    "northset_session_book_snaps_n_session_consistency_errors",
    "northset_session_book_vpin_ic_pack_honesty_errors",
    "northset_session_book_vpin_mean_honesty_errors",
    "northset_session_close_depth_honesty_errors",
    "northset_session_close_ic_packs_honesty_errors",
    "northset_session_close_imbalance_honesty_errors",
    "northset_session_close_micro_bps_honesty_errors",
    "northset_session_close_mid_honesty_errors",
    "northset_session_close_mid_micro_pair_honesty_errors",
    "northset_session_close_spread_bps_honesty_errors",
    "northset_session_identity_rates_honesty_errors",
    "northset_session_imbalance_mean_honesty_errors",
    "northset_session_imbalance_std_honesty_errors",
    "northset_session_l2_enforced_identity_rates_present_honesty_errors",
    "northset_session_means_honesty_errors",
    "northset_session_ofi_abs_dominates_sum_honesty_errors",
    "northset_session_ofi_abs_sum_honesty_errors",
    "northset_session_ofi_sum_ic_honesty_errors",
    "northset_session_ofi_sum_mean_honesty_errors",
    "northset_session_spread_bps_mean_honesty_errors",
    "northset_shape_and_session_l2_floors_honesty_errors",
    "northset_use_session_l2_gate_consistency_errors",
    "session_bulk_vpin_honesty_errors",
    "session_chain_rate_honesty_errors",
    "session_mean_jump_ratio_honesty_errors",
    "session_reconstructs_daily_rate_honesty_errors",
    "session_volume_conservation_rate_honesty_errors",
]
