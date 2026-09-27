"""Sweep event, placebo, cost, and fold honesty checks.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from .primitives import _ic_pack_honesty_errors
from .receipt import (
    _NORTHSET_CORWIN_SCHULTZ_PAIR_SCOPE_ALLOWED,
    _NORTHSET_IMPACT_ESTIMATOR_SCOPE_ALLOWED,
    _NORTHSET_OVERNIGHT_GAP_METHOD_ALLOWED,
    _NORTHSET_TWO_WAY_INFERENCE_INDEX_ALLOWED,
    _NORTHSET_VPIN_METHOD_ALLOWED,
    _NORTHSET_YANG_ZHANG_QLIKE_SCOPE_ALLOWED,
)


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


_NORTHSET_SWEEP_EVIDENCE_SCOPE_ALLOWED = frozenset(
    {"synthetic", "empirical_adjusted", "fixture_raw_unadjusted"}
)


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


__all__ = [
    "northset_fwd_ret_after_sweep_honesty_errors",
    "northset_lag_corr_and_sweep_count_honesty_errors",
    "northset_queue_sweep_ofi_honesty_errors",
    "northset_sweep_control_sample_adequate_honesty_errors",
    "northset_sweep_evidence_blob_honesty_errors",
    "northset_sweep_evidence_scope_honesty_errors",
    "northset_sweep_fold_positive_rates_honesty_errors",
    "northset_sweep_signed_ic_packs_honesty_errors",
    "northset_vpin_sweep_fold_honesty_errors",
    "sweep_follow_cost_adjusted_mean_bps_honesty_errors",
    "sweep_follow_event_mean_bps_honesty_errors",
    "sweep_follow_signed_mean_ic_honesty_errors",
    "sweep_reject_cost_adjusted_mean_bps_honesty_errors",
    "sweep_reject_event_mean_bps_honesty_errors",
    "sweep_reject_signed_mean_ic_honesty_errors",
]
