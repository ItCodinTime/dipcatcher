"""Nested kyle_ofi residual-flow and dispersion honesty checks.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from typing import Any

from .receipt import join_coverage_honesty_errors


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


# ---------------------------------------------------------------------------
# Soft-verify: nested northset["kyle_ofi"] residual_flow / dispersion honesty
# (Sergeant / research diagnostic only — never live Sharpe / promotion.)
# ---------------------------------------------------------------------------

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


def _kyle_ofi_blob(blob: object) -> dict[str, Any] | None:
    """Return nested kyle_ofi dict from a northset family blob, or the blob itself."""
    if not isinstance(blob, dict):
        return None
    if blob.get("family") == "kyle_ofi":
        return blob
    nest = blob.get("kyle_ofi")
    return nest if isinstance(nest, dict) else None


def kyle_residual_flow_honesty_errors(blob: object) -> list[str]:
    """Soft-verify residual_flow receipt keys on nested ``kyle_ofi`` (or bare receipt).

    When any residual_* spearman key is present and finite:
    - ``research_only`` must be True
    - ``claim`` must be ``research_diagnostic_only``
    - companion ``*_t`` / ``*_n_dates`` keys must exist for that spearman
    - no Sharpe/pnl-token keys
    Missing nest / non-finite residual keys → skip (empty).
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    active = []
    for key in _KYLE_RESIDUAL_SPEARMAN_KEYS:
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            continue
        if x == x:  # finite
            active.append(key)
    if not active:
        return []

    errors: list[str] = []
    if kyle.get("research_only") is not True:
        errors.append("kyle_ofi_residual_research_only_missing")
    if kyle.get("claim") != "research_diagnostic_only":
        errors.append("kyle_ofi_residual_claim_invalid")
    for key in active:
        stem = key[: -len("_mean_spearman")]
        if f"{stem}_t" not in kyle:
            errors.append(f"kyle_ofi_residual_missing_t:{stem}")
        if f"{stem}_n_dates" not in kyle:
            errors.append(f"kyle_ofi_residual_missing_n_dates:{stem}")
    leaked = [k for k in kyle if any(tok in str(k).lower() for tok in _KYLE_FORBIDDEN_TOKENS)]
    if leaked:
        errors.append("kyle_ofi_residual_forbidden_metric_keys")
    return errors


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


def kyle_lambda_dispersion_honesty_errors(blob: object) -> list[str]:
    """Soft-verify Kyle λ dispersion stamps when present on nested ``kyle_ofi``.

    If ``kyle_lambda_depth_p50`` (or ofi twin) is finite: require research_only,
    research_diagnostic_only claim, and no Sharpe/pnl keys. Skip if absent/NaN.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    markers = ("kyle_lambda_depth_p50", "kyle_lambda_ofi_p50", "kyle_lambda_dispersion_window")
    present = False
    for key in markers:
        if key not in kyle:
            continue
        if key == "kyle_lambda_dispersion_window":
            present = True
            break
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            continue
        if x == x:
            present = True
            break
    if not present:
        return []
    errors: list[str] = []
    if kyle.get("research_only") is not True:
        errors.append("kyle_ofi_dispersion_research_only_missing")
    if kyle.get("claim") != "research_diagnostic_only":
        errors.append("kyle_ofi_dispersion_claim_invalid")
    leaked = [k for k in kyle if any(tok in str(k).lower() for tok in _KYLE_FORBIDDEN_TOKENS)]
    if leaked:
        errors.append("kyle_ofi_dispersion_forbidden_metric_keys")
    return errors


def kyle_lambda_ofi_depth_corr_honesty_errors(blob: object) -> list[str]:
    """Soft-verify OFI↔depth Kyle-λ corr stamps on nested ``kyle_ofi``.

    When ``kyle_lambda_ofi_depth_spearman`` or ``kyle_lambda_ofi_depth_pearson``
    is finite: require ``research_only``, ``claim=research_diagnostic_only``,
    companions ``kyle_lambda_ofi_depth_n_dates`` / ``*_prod_hac_t`` / ``*_prod_hac_p``,
    and no Sharpe/pnl-token keys. Missing / non-finite markers → skip.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    markers = ("kyle_lambda_ofi_depth_spearman", "kyle_lambda_ofi_depth_pearson")
    present = False
    for key in markers:
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            continue
        if x == x:
            present = True
            break
    if not present:
        return []
    errors: list[str] = []
    if kyle.get("research_only") is not True:
        errors.append("kyle_ofi_depth_corr_research_only_missing")
    if kyle.get("claim") != "research_diagnostic_only":
        errors.append("kyle_ofi_depth_corr_claim_invalid")
    for req in (
        "kyle_lambda_ofi_depth_n_dates",
        "kyle_lambda_ofi_depth_prod_hac_t",
        "kyle_lambda_ofi_depth_prod_hac_p",
    ):
        if req not in kyle:
            errors.append(f"kyle_ofi_depth_corr_missing:{req}")
    leaked = [k for k in kyle if any(tok in str(k).lower() for tok in _KYLE_FORBIDDEN_TOKENS)]
    if leaked:
        errors.append("kyle_ofi_depth_corr_forbidden_metric_keys")
    return errors


def kyle_ofi_join_coverage_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` join_coverage + book_source honesty.

    When ``join_coverage`` is finite: reuse :func:`join_coverage_honesty_errors`
    (∈(0,1], or [floor,1] if ``min_join_coverage`` / ``book_join_coverage_floor``
    present) and require nonempty ``book_source``. NaN / absent → skip.
    Research diagnostic only; never live Sharpe / promotion.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    val = kyle.get("join_coverage")
    if val is None:
        return []
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        # Present-but-malformed must not read as absent/NaN.
        return ["kyle_ofi_join_coverage_non_numeric"]
    x = float(val)
    if x != x:  # NaN
        return []
    errors: list[str] = list(join_coverage_honesty_errors(kyle))
    src = kyle.get("book_source")
    if not (isinstance(src, str) and str(src).strip()):
        errors.append("kyle_ofi_join_coverage_book_source_missing")
    return errors


def kyle_lambda_date_series_honesty_errors(blob: object) -> list[str]:
    """Soft-verify dump/date-series companion stamps on nested ``kyle_ofi``.

    When ``kyle_lambda_date_series_n_depth`` or ``kyle_lambda_date_series_n_ofi``
    is present and finite: require ``research_only``, ``claim=research_diagnostic_only``,
    and no Sharpe/pnl-token keys. Aligns with CLI ``--dump-lambda-series``
    research_only panel contract. NaN / absent → skip.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    markers = ("kyle_lambda_date_series_n_depth", "kyle_lambda_date_series_n_ofi")
    present = False
    for key in markers:
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            continue
        if x == x:
            present = True
            break
    if not present:
        return []
    errors: list[str] = []
    if kyle.get("research_only") is not True:
        errors.append("kyle_ofi_date_series_research_only_missing")
    if kyle.get("claim") != "research_diagnostic_only":
        errors.append("kyle_ofi_date_series_claim_invalid")
    leaked = [k for k in kyle if any(tok in str(k).lower() for tok in _KYLE_FORBIDDEN_TOKENS)]
    if leaked:
        errors.append("kyle_ofi_date_series_forbidden_metric_keys")
    return errors


def kyle_ofi_synthetic_source_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` book_dgp / dgp / data_source SYNTHETIC consistency.

    Contract mirrors ``bench_kyle_ofi_fused`` stamping:
    - ``book_dgp`` / ``dgp`` ``synthetic_lob`` ⇔ ``data_source == "SYNTHETIC"``
    - when both ``dgp`` and ``book_dgp`` present, they must match
    - non-synthetic ``book_dgp`` ⇒ ``data_source`` equals ``book_source`` (when both set)
    Absent / empty markers → skip. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _str(key: str) -> str | None:
        v = kyle.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    book_dgp = _str("book_dgp")
    dgp = _str("dgp")
    data_source = _str("data_source")
    book_source = _str("book_source")
    if book_dgp is None and dgp is None and data_source is None:
        return []

    errors: list[str] = []
    if book_dgp is not None and dgp is not None and book_dgp != dgp:
        errors.append("kyle_ofi_dgp_book_dgp_mismatch")

    synth_dgp = None
    if book_dgp is not None:
        synth_dgp = book_dgp == "synthetic_lob"
    elif dgp is not None:
        synth_dgp = dgp == "synthetic_lob"

    if synth_dgp is True and data_source is not None and data_source != "SYNTHETIC":
        errors.append("kyle_ofi_synthetic_dgp_data_source_not_SYNTHETIC")
    if data_source == "SYNTHETIC":
        # require synthetic_lob on whichever dgp field is present
        if book_dgp is not None and book_dgp != "synthetic_lob":
            errors.append("kyle_ofi_SYNTHETIC_data_source_book_dgp_not_synthetic_lob")
        if dgp is not None and dgp != "synthetic_lob":
            errors.append("kyle_ofi_SYNTHETIC_data_source_dgp_not_synthetic_lob")
        if book_dgp is None and dgp is None:
            errors.append("kyle_ofi_SYNTHETIC_data_source_missing_dgp")
    if synth_dgp is False and data_source is not None:
        if data_source == "SYNTHETIC":
            errors.append("kyle_ofi_nonsynthetic_dgp_data_source_SYNTHETIC")
        elif book_source is not None and data_source != book_source:
            errors.append("kyle_ofi_nonsynthetic_data_source_ne_book_source")
    return errors


def kyle_ofi_label_synthetic_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nest ``label`` SYN* vs ``data_source`` / dgp stamping.

    When ``label`` uppercases to a SYN* prefix (bench default path):
    - ``book_dgp``/``dgp`` ``synthetic_lob`` ⇒ ``data_source`` must be ``SYNTHETIC``
    - non-synthetic dgp ⇒ ``data_source`` must equal ``book_source`` (when set)
    Absent / non-SYN label → skip. Thin fail-closed; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    label = kyle.get("label")
    if label is None or not str(label).strip():
        return []
    if not str(label).upper().startswith("SYN"):
        return []

    def _str(key: str) -> str | None:
        v = kyle.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    book_dgp = _str("book_dgp")
    dgp = _str("dgp")
    data_source = _str("data_source")
    book_source = _str("book_source")
    if data_source is None:
        return []

    synth_dgp = None
    if book_dgp is not None:
        synth_dgp = book_dgp == "synthetic_lob"
    elif dgp is not None:
        synth_dgp = dgp == "synthetic_lob"

    errors: list[str] = []
    if synth_dgp is True and data_source != "SYNTHETIC":
        errors.append("kyle_ofi_SYN_label_data_source_not_SYNTHETIC")
    if synth_dgp is False:
        if data_source == "SYNTHETIC":
            errors.append("kyle_ofi_SYN_label_nonsynthetic_dgp_data_source_SYNTHETIC")
        elif book_source is not None and data_source != book_source:
            errors.append("kyle_ofi_SYN_label_data_source_ne_book_source")
    if synth_dgp is None and data_source != "SYNTHETIC":
        # SYN* label with no dgp stamped still expects SYNTHETIC data_source
        errors.append("kyle_ofi_SYN_label_data_source_not_SYNTHETIC")
    return errors


def _kyle_ofi_has_nest_diagnostic_marker(kyle: dict[str, Any]) -> bool:
    """True when any kyle nest diagnostic stamp is present/finite."""
    for key in _KYLE_RESIDUAL_SPEARMAN_KEYS:
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])
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
            x = float(kyle[key])
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


def kyle_ofi_nest_claim_honesty_errors(blob: object) -> list[str]:
    """Soft-verify: any nest diagnostic marker ⇒ research_only + claim.

    Umbrella fail-closed across residual/dispersion/ofi_depth/join/date_series/
    source stamps so a partial nest cannot omit honesty labels. Skip when no
    markers. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    if not _kyle_ofi_has_nest_diagnostic_marker(kyle):
        return []
    errors: list[str] = []
    if kyle.get("research_only") is not True:
        errors.append("kyle_ofi_nest_research_only_missing")
    if kyle.get("claim") != "research_diagnostic_only":
        errors.append("kyle_ofi_nest_claim_invalid")
    leaked = [k for k in kyle if any(tok in str(k).lower() for tok in _KYLE_FORBIDDEN_TOKENS)]
    if leaked:
        errors.append("kyle_ofi_nest_forbidden_metric_keys")
    return errors


_KYLE_OFI_IC_METHOD_ALLOWED = frozenset({"date_level_spearman_hac"})


def kyle_ofi_ic_method_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` ``ic_method`` stamp honesty.

    Contract: bench/residual paths stamp ``ic_method=date_level_spearman_hac``.
    - If ``ic_method`` present: must be in the allowed set (fail-closed otherwise).
    - If any IC diagnostic marker is finite (residual spearman, λ means, ofi_depth
      corr, dispersion window/p50) and ``ic_method`` absent → missing error.
    Absent markers and absent method → skip. Research diagnostic only.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _finite(key: str) -> bool:
        if key not in kyle:
            return False
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            return False
        return x == x

    ic_markers = False
    for key in _KYLE_RESIDUAL_SPEARMAN_KEYS:
        if _finite(key):
            ic_markers = True
            break
    if not ic_markers:
        for key in (
            "kyle_lambda_depth_mean",
            "kyle_lambda_ofi_mean",
            "kyle_lambda_depth_p50",
            "kyle_lambda_ofi_p50",
            "kyle_lambda_ofi_depth_spearman",
            "kyle_lambda_ofi_depth_pearson",
        ):
            if _finite(key):
                ic_markers = True
                break
    if not ic_markers and "kyle_lambda_dispersion_window" in kyle:
        ic_markers = True

    method = kyle.get("ic_method")
    errors: list[str] = []
    if method is None or (isinstance(method, str) and not method.strip()):
        if ic_markers:
            errors.append("kyle_ofi_ic_method_missing")
        return errors
    method_s = str(method).strip()
    if method_s not in _KYLE_OFI_IC_METHOD_ALLOWED:
        errors.append("kyle_ofi_ic_method_invalid")
    return errors


def kyle_ofi_family_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` ``family`` stamp when diagnostics present.

    When any nest diagnostic marker is present/finite: ``family`` must be exactly
    ``kyle_ofi``. Also fail-closed when a kyle-shaped receipt (λ/residual/ofi_depth
    keys) carries a wrong ``family`` — ``_kyle_ofi_blob`` would otherwise skip.
    Bare empty / northset-only scalars → skip. Research diagnostic only.
    """
    if not isinstance(blob, dict):
        return []
    nest = blob.get("kyle_ofi")
    if isinstance(nest, dict):
        kyle = nest
    elif blob.get("family") == "kyle_ofi" or any(
        k in blob
        for k in (
            "kyle_lambda_depth_mean",
            "kyle_lambda_ofi_mean",
            "kyle_lambda_ofi_depth_spearman",
            "kyle_lambda_dispersion_window",
            "residual_ofi_ex_depth_fwd_delta_mid_mean_spearman",
        )
    ):
        kyle = blob
    else:
        return []
    if not _kyle_ofi_has_nest_diagnostic_marker(kyle):
        return []
    if kyle.get("family") != "kyle_ofi":
        return ["kyle_ofi_family_invalid"]
    return []


def kyle_ofi_hac_lags_honesty_errors(blob: object) -> list[str]:
    """Soft-verify optional ``hac_lags`` stamp on nested ``kyle_ofi``.

    When ``hac_lags`` is present: must be finite and ≥ 0 (integer-valued).
    Absent → skip (receipt may omit the key without editing ``kyle_ofi.py``).
    Also: when both depth rolling HAC lo/hi finite, require lo ≤ hi (and ofi twin).
    Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    errors: list[str] = []
    if "hac_lags" in kyle and kyle.get("hac_lags") is not None:
        try:
            x = float(kyle["hac_lags"])
        except (TypeError, ValueError):
            return ["kyle_ofi_hac_lags_non_numeric"]
        if x != x or abs(x) == float("inf"):
            errors.append("kyle_ofi_hac_lags_non_finite")
        elif x < 0:
            errors.append("kyle_ofi_hac_lags_negative")
        elif abs(x - round(x)) > 1e-12:
            errors.append("kyle_ofi_hac_lags_not_integer")

    def _pair_lo_hi(lo_key: str, hi_key: str, err: str) -> None:
        if lo_key not in kyle or hi_key not in kyle:
            return
        try:
            lo = float(kyle[lo_key])
            hi = float(kyle[hi_key])
        except (TypeError, ValueError):
            return
        if lo != lo or hi != hi:
            return
        if lo > hi:
            errors.append(err)

    _pair_lo_hi(
        "kyle_lambda_depth_rolling_hac_lo",
        "kyle_lambda_depth_rolling_hac_hi",
        "kyle_ofi_depth_rolling_hac_lo_gt_hi",
    )
    _pair_lo_hi(
        "kyle_lambda_ofi_rolling_hac_lo",
        "kyle_lambda_ofi_rolling_hac_hi",
        "kyle_ofi_ofi_rolling_hac_lo_gt_hi",
    )
    return errors


def kyle_ofi_min_names_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` ``min_names`` stamp.

    When ``min_names`` present: finite integer-valued and ≥ 1.
    When bench-level markers present (``kyle_lambda_depth_mean`` / ``n_fused``)
    and ``min_names`` absent → missing. Else skip. Research diagnostic only.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    benchish = False
    for key in ("kyle_lambda_depth_mean", "kyle_lambda_ofi_mean", "n_fused", "n_scored"):
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            continue
        if x == x:
            benchish = True
            break
    if "min_names" not in kyle or kyle.get("min_names") is None:
        return ["kyle_ofi_min_names_missing"] if benchish else []
    try:
        x = float(kyle["min_names"])
    except (TypeError, ValueError):
        return ["kyle_ofi_min_names_non_numeric"]
    errors: list[str] = []
    if x != x or abs(x) == float("inf"):
        errors.append("kyle_ofi_min_names_non_finite")
    elif x < 1:
        errors.append("kyle_ofi_min_names_lt_one")
    elif abs(x - round(x)) > 1e-12:
        errors.append("kyle_ofi_min_names_not_integer")
    return errors


def kyle_ofi_n_fused_scored_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` ``n_fused`` / ``n_scored`` consistency.

    When either count is present: both must be finite integer-valued ≥ 0 and
    ``n_scored ≤ n_fused``. One without the other → missing companion.
    Absent both → skip. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    has_f = "n_fused" in kyle and kyle.get("n_fused") is not None
    has_s = "n_scored" in kyle and kyle.get("n_scored") is not None
    if not has_f and not has_s:
        return []
    errors: list[str] = []
    if has_f ^ has_s:
        errors.append("kyle_ofi_n_fused_scored_pair_incomplete")
        return errors

    def _parse(key: str) -> float | None:
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_{key}_non_numeric")
            return None
        if x != x or abs(x) == float("inf"):
            errors.append(f"kyle_ofi_{key}_non_finite")
            return None
        if x < 0:
            errors.append(f"kyle_ofi_{key}_negative")
        if abs(x - round(x)) > 1e-12:
            errors.append(f"kyle_ofi_{key}_not_integer")
        return x

    fused = _parse("n_fused")
    scored = _parse("n_scored")
    if fused is not None and scored is not None and scored > fused:
        errors.append("kyle_ofi_n_scored_gt_n_fused")
    return errors


def kyle_ofi_book_panel_path_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` ``book_panel_path`` stamp.

    Absent key or explicit ``None`` → skip (synthetic benches omit a path).
    When present and not ``None``: must be a nonempty string. Empty / whitespace
    fail-closed. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    if "book_panel_path" not in kyle:
        return []
    v = kyle.get("book_panel_path")
    if v is None:
        return []
    if not isinstance(v, str) or not str(v).strip():
        return ["kyle_ofi_book_panel_path_empty"]
    return []


def kyle_ofi_min_join_coverage_pair_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``min_join_coverage`` / ``book_join_coverage_floor`` vs join_coverage.

    When a floor stamp is present: finite and ∈ [0, 1]. When ``join_coverage`` is
    also finite: require ``join_coverage >= floor`` (and ≤ 1). Absent floor → skip
    (``kyle_ofi_join_coverage_honesty_errors`` still covers open-unit join alone).
    Research diagnostic only.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    raw_floor = kyle.get("min_join_coverage")
    if raw_floor is None:
        raw_floor = kyle.get("book_join_coverage_floor")
    if raw_floor is None:
        return []
    try:
        floor = float(raw_floor)
    except (TypeError, ValueError):
        return ["kyle_ofi_min_join_coverage_non_numeric"]
    errors: list[str] = []
    if floor != floor or abs(floor) == float("inf"):
        errors.append("kyle_ofi_min_join_coverage_non_finite")
        return errors
    if not (0.0 <= floor <= 1.0):
        errors.append("kyle_ofi_min_join_coverage_outside_unit_interval")
    if "join_coverage" not in kyle or kyle.get("join_coverage") is None:
        return errors
    try:
        cov = float(kyle["join_coverage"])
    except (TypeError, ValueError):
        return errors
    if cov != cov or abs(cov) == float("inf"):
        return errors
    if cov + 1e-12 < floor:
        errors.append("kyle_ofi_join_coverage_below_min_join_coverage")
    if cov > 1.0 + 1e-12:
        errors.append("kyle_ofi_join_coverage_above_one_with_floor")
    return errors


def kyle_ofi_dispersion_window_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``kyle_lambda_dispersion_window`` int stamp on nested ``kyle_ofi``.

    When dispersion markers present (``kyle_lambda_depth_p50`` / ``kyle_lambda_ofi_p50``
    finite, or window key expected): window must be present, finite, integer-valued,
    and ≥ 1. Absent markers and absent window → skip. Research diagnostic only.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _finite(key: str) -> bool:
        if key not in kyle:
            return False
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            return False
        return x == x

    disp_markers = _finite("kyle_lambda_depth_p50") or _finite("kyle_lambda_ofi_p50")
    if (
        "kyle_lambda_dispersion_window" not in kyle
        or kyle.get("kyle_lambda_dispersion_window") is None
    ):
        return ["kyle_ofi_dispersion_window_missing"] if disp_markers else []
    try:
        x = float(kyle["kyle_lambda_dispersion_window"])
    except (TypeError, ValueError):
        return ["kyle_ofi_dispersion_window_non_numeric"]
    errors: list[str] = []
    if x != x or abs(x) == float("inf"):
        errors.append("kyle_ofi_dispersion_window_non_finite")
    elif x < 1:
        errors.append("kyle_ofi_dispersion_window_lt_one")
    elif abs(x - round(x)) > 1e-12:
        errors.append("kyle_ofi_dispersion_window_not_integer")
    return errors


def kyle_ofi_diagnostic_string_honesty_errors(blob: object) -> list[str]:
    """Soft-verify optional nest ``diagnostic`` string stamp.

    Full ``bench_kyle_ofi_fused`` omits ``diagnostic`` (skip). Sub-receipts
    (residual / λ / ofi_depth_corr) stamp it: when the key is present it must be
    a nonempty string without Sharpe/pnl tokens. Research diagnostic only.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    if "diagnostic" not in kyle:
        return []
    v = kyle.get("diagnostic")
    if v is None or not isinstance(v, str) or not str(v).strip():
        return ["kyle_ofi_diagnostic_empty"]
    s = str(v).strip()
    if any(tok in s.lower() for tok in _KYLE_FORBIDDEN_TOKENS):
        return ["kyle_ofi_diagnostic_forbidden_token"]
    return []


def kyle_ofi_lambda_decile_order_honesty_errors(blob: object) -> list[str]:
    """Soft-verify Kyle λ dispersion decile order on nested ``kyle_ofi``.

    For depth and ofi sides independently: when p10/p50/p90 are all finite,
    require ``p10 ≤ p50 ≤ p90``. Partial / absent / NaN triples → skip that side.
    Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _finite(key: str) -> float | None:
        if key not in kyle:
            return None
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            return None
        if x != x or abs(x) == float("inf"):
            return None
        return x

    errors: list[str] = []
    for _side, keys, err in (
        (
            "depth",
            (
                "kyle_lambda_depth_p10",
                "kyle_lambda_depth_p50",
                "kyle_lambda_depth_p90",
            ),
            "kyle_ofi_depth_decile_order_invalid",
        ),
        (
            "ofi",
            (
                "kyle_lambda_ofi_p10",
                "kyle_lambda_ofi_p50",
                "kyle_lambda_ofi_p90",
            ),
            "kyle_ofi_ofi_decile_order_invalid",
        ),
    ):
        present = [v for v in (_finite(k) for k in keys) if v is not None]
        if len(present) != len(keys):
            continue
        p10, p50, p90 = present
        if not (p10 <= p50 <= p90):
            errors.append(err)
    return errors


def kyle_ofi_std_iqr_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` λ ``std`` / ``iqr`` stamps are ≥ 0 when finite.

    Covers depth/ofi ``kyle_lambda_*_std`` and ``kyle_lambda_*_iqr``. Absent / NaN
    → skip that key. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    errors: list[str] = []
    for key in (
        "kyle_lambda_depth_std",
        "kyle_lambda_ofi_std",
        "kyle_lambda_depth_iqr",
        "kyle_lambda_ofi_iqr",
    ):
        if key not in kyle:
            continue
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_{key}_non_numeric")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errors.append(f"kyle_ofi_{key}_non_finite")
        elif x < 0:
            errors.append(f"kyle_ofi_{key}_negative")
    return errors


def kyle_ofi_rolling_mean_hac_band_honesty_errors(blob: object) -> list[str]:
    """Soft-verify rolling mean present/finite when HAC band lo/hi are finite.

    For depth and ofi sides: if both ``*_rolling_hac_lo`` and ``*_rolling_hac_hi``
    are finite, require ``*_rolling_mean`` present and finite. Also lo ≤ hi
    (parity with hac_lags helper). Absent band → skip. Research diagnostic only.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _finite(key: str) -> float | None:
        if key not in kyle:
            return None
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            return None
        if x != x or abs(x) == float("inf"):
            return None
        return x

    errors: list[str] = []
    for side in ("depth", "ofi"):
        lo = _finite(f"kyle_lambda_{side}_rolling_hac_lo")
        hi = _finite(f"kyle_lambda_{side}_rolling_hac_hi")
        if lo is None or hi is None:
            continue
        if lo > hi:
            errors.append(f"kyle_ofi_{side}_rolling_hac_lo_gt_hi")
        mean = _finite(f"kyle_lambda_{side}_rolling_mean")
        if mean is None:
            errors.append(f"kyle_ofi_{side}_rolling_mean_missing_with_hac_band")
    return errors


def kyle_ofi_n_dates_companion_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``*_n_dates`` ≥ 1 when companion IC / t stamps are finite.

    Companions:
    - finite ``*_mean_spearman`` ⇒ ``*_n_dates`` present, finite, integer-valued, ≥ 1
    - finite IC-style ``*_t`` (excl. ``*hac*`` / ``*rolling*``) ⇒ matching ``*_n_dates``
      for residual / flow IC stems, or side-level ``kyle_lambda_{depth,ofi}_n_dates``
      for bare ``kyle_lambda_{depth,ofi}_t`` (not per-target fwd λ t-stats)
    Absent companions → skip. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _finite(key: str) -> float | None:
        if key not in kyle:
            return None
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            return None
        if x != x or abs(x) == float("inf"):
            return None
        return x

    def _n_dates_stem_for_t(stem: str) -> str | None:
        if stem in {"kyle_lambda_depth", "kyle_lambda_ofi"}:
            return stem
        ic_prefixes = (
            "residual_",
            "ofi_delta_mid_",
            "signed_depth_",
            "depth_flow_",
            "ofi_flow_",
            "ofi_fwd_",
            "signed_depth_fwd_",
        )
        if stem.startswith(ic_prefixes):
            return stem
        return None

    required: set[str] = set()
    for key in kyle:
        if not isinstance(key, str):
            continue
        if key.endswith("_mean_spearman") and _finite(key) is not None:
            required.add(key[: -len("_mean_spearman")])
        elif (
            key.endswith("_t")
            and "hac" not in key
            and "rolling" not in key
            and _finite(key) is not None
        ):
            mapped = _n_dates_stem_for_t(key[: -len("_t")])
            if mapped is not None:
                required.add(mapped)

    errors: list[str] = []
    for stem in sorted(required):
        nkey = f"{stem}_n_dates"
        if nkey not in kyle or kyle.get(nkey) is None:
            errors.append(f"kyle_ofi_n_dates_missing:{stem}")
            continue
        try:
            n = float(kyle[nkey])
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_n_dates_non_numeric:{stem}")
            continue
        if n != n or abs(n) == float("inf"):
            errors.append(f"kyle_ofi_n_dates_non_finite:{stem}")
        elif n < 1:
            errors.append(f"kyle_ofi_n_dates_lt_one:{stem}")
        elif abs(n - round(n)) > 1e-12:
            errors.append(f"kyle_ofi_n_dates_not_integer:{stem}")
    return errors


def kyle_ofi_pvalue_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` p-value stamps ∈ [0, 1] when finite.

    Any key ending in ``_p`` (IC / HAC p-values) that parses as finite must lie
    in the closed unit interval. Absent / NaN → skip that key. Research
    diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    errors: list[str] = []
    for key, raw in kyle.items():
        if not isinstance(key, str) or not key.endswith("_p"):
            continue
        # deciles are *_p10/*_p50/*_p90 — they do not end with lone "_p"
        try:
            x = float(raw)
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_pvalue_non_numeric:{key}")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errors.append(f"kyle_ofi_pvalue_non_finite:{key}")
        elif not (0.0 <= x <= 1.0):
            errors.append(f"kyle_ofi_pvalue_outside_unit_interval:{key}")
    return errors


def kyle_ofi_spearman_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` Spearman (and Pearson) corr stamps ∈ [-1, 1].

    Any key ending in ``_mean_spearman``, ``_spearman``, or ``_pearson`` that is
    finite must lie in the closed interval. Absent / NaN → skip. Research
    diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    errors: list[str] = []
    for key, raw in kyle.items():
        if not isinstance(key, str):
            continue
        if not (
            key.endswith("_mean_spearman") or key.endswith("_spearman") or key.endswith("_pearson")
        ):
            continue
        try:
            x = float(raw)
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_corr_non_numeric:{key}")
            continue
        if x != x:
            continue
        if abs(x) == float("inf"):
            errors.append(f"kyle_ofi_corr_non_finite:{key}")
        elif not (-1.0 <= x <= 1.0):
            errors.append(f"kyle_ofi_corr_outside_unit_interval:{key}")
    return errors


def kyle_ofi_tstat_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nested ``kyle_ofi`` t-stat stamps are finite when present.

    - Any key ending in ``_t`` that is present/non-null must parse finite (not NaN/±inf).
    - Finite ``*_mean_spearman`` ⇒ companion ``*_t`` present and finite (IC companion
      to p / n_dates soft-verify).
    Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []

    def _finite(key: str) -> float | None:
        if key not in kyle or kyle.get(key) is None:
            return None
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            return None
        if x != x or abs(x) == float("inf"):
            return None
        return x

    errors: list[str] = []
    for key, raw in kyle.items():
        if not isinstance(key, str) or not key.endswith("_t"):
            continue
        if raw is None:
            continue
        try:
            x = float(raw)
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_tstat_non_numeric:{key}")
            continue
        if x != x or abs(x) == float("inf"):
            errors.append(f"kyle_ofi_tstat_non_finite:{key}")

    for key in list(kyle.keys()):
        if not isinstance(key, str) or not key.endswith("_mean_spearman"):
            continue
        if _finite(key) is None:
            continue
        stem = key[: -len("_mean_spearman")]
        tkey = f"{stem}_t"
        if tkey not in kyle or kyle.get(tkey) is None:
            errors.append(f"kyle_ofi_tstat_missing:{stem}")
            continue
        # non-finite already flagged above if key present; ensure companion miss covered
        if _finite(tkey) is None and (
            f"kyle_ofi_tstat_non_finite:{tkey}" not in errors
            and f"kyle_ofi_tstat_non_numeric:{tkey}" not in errors
        ):
            errors.append(f"kyle_ofi_tstat_non_finite:{tkey}")
    return errors


def kyle_ofi_label_nonempty_honesty_errors(blob: object) -> list[str]:
    """Soft-verify nest ``label`` nonempty when diagnostic markers are present.

    Scan residual: SYN* / claim helpers skip blank labels. When any nest
    diagnostic marker is present, ``label`` must be a nonempty string.
    Absent markers → skip. Research diagnostic only; never live Sharpe.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    if not _kyle_ofi_has_nest_diagnostic_marker(kyle):
        return []
    label = kyle.get("label")
    if label is None or not isinstance(label, str) or not str(label).strip():
        return ["kyle_ofi_label_empty_with_markers"]
    return []


def kyle_ofi_date_series_counts_honesty_errors(blob: object) -> list[str]:
    """Soft-verify ``kyle_lambda_date_series_n_{depth,ofi}`` ≥ 0 ints when present.

    Complements claim soft-verify on those stamps. Absent → skip.
    """
    kyle = _kyle_ofi_blob(blob)
    if kyle is None:
        return []
    errors: list[str] = []
    for key in ("kyle_lambda_date_series_n_depth", "kyle_lambda_date_series_n_ofi"):
        if key not in kyle or kyle.get(key) is None:
            continue
        try:
            x = float(kyle[key])
        except (TypeError, ValueError):
            errors.append(f"kyle_ofi_{key}_non_numeric")
            continue
        if x != x or abs(x) == float("inf"):
            errors.append(f"kyle_ofi_{key}_non_finite")
        elif x < 0:
            errors.append(f"kyle_ofi_{key}_negative")
        elif abs(x - round(x)) > 1e-12:
            errors.append(f"kyle_ofi_{key}_not_integer")
    return errors


def kyle_ofi_nest_honesty_errors(blob: object) -> list[str]:
    """Dispatcher: full kyle nest soft-verify suite (incl. label nonempty + date_series n)."""
    errors: list[str] = []
    errors.extend(kyle_residual_flow_honesty_errors(blob))
    errors.extend(kyle_lambda_dispersion_honesty_errors(blob))
    errors.extend(kyle_lambda_ofi_depth_corr_honesty_errors(blob))
    errors.extend(kyle_ofi_join_coverage_honesty_errors(blob))
    errors.extend(kyle_lambda_date_series_honesty_errors(blob))
    errors.extend(kyle_ofi_synthetic_source_honesty_errors(blob))
    errors.extend(kyle_ofi_label_synthetic_honesty_errors(blob))
    errors.extend(kyle_ofi_nest_claim_honesty_errors(blob))
    errors.extend(kyle_ofi_ic_method_honesty_errors(blob))
    errors.extend(kyle_ofi_family_honesty_errors(blob))
    errors.extend(kyle_ofi_hac_lags_honesty_errors(blob))
    errors.extend(kyle_ofi_min_names_honesty_errors(blob))
    errors.extend(kyle_ofi_n_fused_scored_honesty_errors(blob))
    errors.extend(kyle_ofi_book_panel_path_honesty_errors(blob))
    errors.extend(kyle_ofi_min_join_coverage_pair_honesty_errors(blob))
    errors.extend(kyle_ofi_dispersion_window_honesty_errors(blob))
    errors.extend(kyle_ofi_diagnostic_string_honesty_errors(blob))
    errors.extend(kyle_ofi_lambda_decile_order_honesty_errors(blob))
    errors.extend(kyle_ofi_std_iqr_honesty_errors(blob))
    errors.extend(kyle_ofi_rolling_mean_hac_band_honesty_errors(blob))
    errors.extend(kyle_ofi_n_dates_companion_honesty_errors(blob))
    errors.extend(kyle_ofi_pvalue_honesty_errors(blob))
    errors.extend(kyle_ofi_spearman_honesty_errors(blob))
    errors.extend(kyle_ofi_tstat_honesty_errors(blob))
    errors.extend(kyle_ofi_label_nonempty_honesty_errors(blob))
    errors.extend(kyle_ofi_date_series_counts_honesty_errors(blob))
    return errors


__all__ = [
    "kyle_lambda_date_series_honesty_errors",
    "kyle_lambda_dispersion_honesty_errors",
    "kyle_lambda_ofi_depth_corr_honesty_errors",
    "kyle_ofi_book_panel_path_honesty_errors",
    "kyle_ofi_date_series_counts_honesty_errors",
    "kyle_ofi_diagnostic_string_honesty_errors",
    "kyle_ofi_dispersion_window_honesty_errors",
    "kyle_ofi_family_honesty_errors",
    "kyle_ofi_hac_lags_honesty_errors",
    "kyle_ofi_ic_method_honesty_errors",
    "kyle_ofi_join_coverage_honesty_errors",
    "kyle_ofi_label_nonempty_honesty_errors",
    "kyle_ofi_label_synthetic_honesty_errors",
    "kyle_ofi_lambda_decile_order_honesty_errors",
    "kyle_ofi_min_join_coverage_pair_honesty_errors",
    "kyle_ofi_min_names_honesty_errors",
    "kyle_ofi_n_dates_companion_honesty_errors",
    "kyle_ofi_n_fused_scored_honesty_errors",
    "kyle_ofi_nest_claim_honesty_errors",
    "kyle_ofi_nest_honesty_errors",
    "kyle_ofi_pvalue_honesty_errors",
    "kyle_ofi_rolling_mean_hac_band_honesty_errors",
    "kyle_ofi_spearman_honesty_errors",
    "kyle_ofi_std_iqr_honesty_errors",
    "kyle_ofi_synthetic_source_honesty_errors",
    "kyle_ofi_tstat_honesty_errors",
    "kyle_residual_flow_honesty_errors",
    "northset_include_kyle_ofi_nest_presence_honesty_errors",
    "northset_kyle_r2_unit_honesty_errors",
]
