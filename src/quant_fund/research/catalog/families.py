"""Benchmark family vocabulary, forbidden-key policy, and hypothesis registry."""

from __future__ import annotations

import math

from ._helpers import (
    _iter_mapping_keys,
)
from .constants import (
    COVERAGE_GUARANTEE_SCOPE_MARGINAL,
    DIST_CRPS_EPROCESS_REQUIRED_WHEN_DM,
    DIST_CRPS_SCALED_EPROCESS_REQUIRED_WHEN_DM,
    DM_CRPS_MARKER_KEY,
    DM_CRPS_SCALED_MARKER_KEY,
    ES_MARKER_KEYS,
    FORBIDDEN_RESEARCH_METRIC_KEYS,
    H1_EXPECTED_FAMILY,
    H1_HYPOTHESIS_ID,
    H2_EXPECTED_FAMILY,
    H2_HYPOTHESIS_ID,
    H3_EXPECTED_FAMILY,
    H3_HYPOTHESIS_ID,
    H4_EXPECTED_FAMILY,
    H4_HYPOTHESIS_ID,
    H4B_EXPECTED_FAMILY,
    H4B_HYPOTHESIS_ID,
    H7_EXPECTED_FAMILY,
    H7_HYPOTHESIS_ID,
    H8_EXPECTED_FAMILY,
    H8_HYPOTHESIS_ID,
    H9_EXPECTED_FAMILY,
    H9_HYPOTHESIS_ID,
    H10_EXPECTED_FAMILY,
    H10_HYPOTHESIS_ID,
    H11_EXPECTED_FAMILY,
    H11_HYPOTHESIS_ID,
    H12_EXPECTED_FAMILY,
    H12_HYPOTHESIS_ID,
    H15_EXPECTED_FAMILY,
    H15_HYPOTHESIS_ID,
    H16_H18_EXPECTED_FAMILY,
    H16_HYPOTHESIS_ID,
    H17_HYPOTHESIS_ID,
    H18_HYPOTHESIS_ID,
    H19_EXPECTED_FAMILY,
    H19_HYPOTHESIS_ID,
    H20_EXPECTED_FAMILY,
    H20_HYPOTHESIS_ID,
    H21_EXPECTED_FAMILY,
    H21_HYPOTHESIS_ID,
    H22_EXPECTED_FAMILY,
    H22_HYPOTHESIS_ID,
    H43_EXPECTED_FAMILY,
    H43_HYPOTHESIS_ID,
    H99_EXPECTED_FAMILY,
    H99_HYPOTHESIS_ID,
    KUPIEC_MARKER_KEYS,
    TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS,
    TAIL_VAR_BATTERY_REQUIRED_WHEN_KUPIEC,
)
from .predicates import (
    conformal_aci_blob,
    conformal_mondrian_aci_blob,
)


def family_blob_forbidden_metrics_absent(payload: object) -> bool:
    """Return True iff *payload* has no forbidden research-headline metric keys.

    Fail closed: any mapping key whose underscore tokens include sharpe / sortino /
    calmar / pnl / nav marks the blob unclean. Values are not scanned (keys only).

    Scope: research family / scorecard blobs only. Paper ``analytics_export`` may
    contain equity ``nav_*`` / stress ``*_pnl`` diagnostics; validate those with
    ``validate_analytics_export`` (live_pnl_claim fail-closed), not this helper.
    """
    for key in _iter_mapping_keys(payload):
        parts = str(key).lower().replace("-", "_").split("_")
        if any(tok in FORBIDDEN_RESEARCH_METRIC_KEYS for tok in parts if tok):
            return False
    return True


def family_blob_executed(payload: object) -> bool:
    """True iff family payload is truthy (scorecard ``executed`` twin)."""
    return bool(payload)


def family_blob_nonempty(payload: object) -> bool:
    """True iff family payload is truthy (scorecard ``nonempty`` twin)."""
    return bool(payload)


def family_blob_has_finite_observation(payload: object) -> bool:
    """True iff *payload* recursively contains at least one finite observation.

    Lifted from research.agent ``_benchmark_scorecard`` nested ``has_observation``
    (Day Wave 44). Semantics:
    - dict / list / tuple → any child is an observation
    - bool → True
    - int (incl. numpy integer scalars via ``.item()``) → True
    - float (incl. numpy floating) → ``math.isfinite``
    - str → nonempty strip and not ``nan`` / ``none`` (case-insensitive)
    - else → False

    Empty ``{}`` / all-NaN blobs → False. Used by soft scorecard forge verify.
    """
    if isinstance(payload, dict):
        return any(family_blob_has_finite_observation(item) for item in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(family_blob_has_finite_observation(item) for item in payload)
    if isinstance(payload, bool):
        return True
    if isinstance(payload, int):
        return True
    if isinstance(payload, float):
        return math.isfinite(payload)
    # Numpy scalars are not always subclasses of int/float — recurse via .item().
    if (
        hasattr(payload, "dtype")
        and hasattr(payload, "item")
        and not isinstance(payload, (bytes, bytearray, memoryview))
    ):
        try:
            return family_blob_has_finite_observation(payload.item())
        except (ValueError, TypeError, AttributeError):
            return False
    return (
        isinstance(payload, str)
        and bool(payload.strip())
        and payload.lower() not in {"nan", "none"}
    )


def tail_var_battery_missing_keys(payload: object) -> list[str]:
    """Return missing Christoffersen keys when a nonempty blob has Kupiec markers.

    Contract (Day Wave 18 soft honesty — research diagnostic only):
    - Empty ``{}`` or non-dict → no required keys (tiny panels may omit battery).
    - Nonempty dict without ``kupiec_p`` / ``kupiec_lr`` → skip (no Kupiec ⇒ no
      battery contract).
    - Nonempty dict with either Kupiec marker (even NaN) → require CC + ind key
      *presence*; values may be NaN. Missing keys returned in catalog order for
      ``tail_var_battery_incomplete:<key>`` errors in ``verify_research_artifact``.
    """
    if not isinstance(payload, dict) or not payload:
        return []
    keys = {str(k) for k in payload}
    if not (keys & KUPIEC_MARKER_KEYS):
        return []
    return [key for key in TAIL_VAR_BATTERY_REQUIRED_WHEN_KUPIEC if key not in keys]


def tail_var_battery_keys_present(payload: object) -> bool:
    """Return True iff soft VaR-battery key presence holds (see missing_keys)."""
    return not tail_var_battery_missing_keys(payload)


def dist_crps_eprocess_missing_keys(payload: object) -> list[str]:
    """Return missing e_dm_crps_* keys when a nonempty blob has DM CRPS markers.

    Contract (Day Wave 20 soft honesty — research diagnostic only):
    - Empty ``{}`` or non-dict → no required keys.
    - Nonempty dict without ``dm_crps_p`` / ``dm_crps_scaled_p`` → skip.
    - ``dm_crps_p`` present (even NaN) → require unscaled ``e_dm_crps_*`` presence.
    - ``dm_crps_scaled_p`` present (even NaN) → require scaled ``e_dm_crps_scaled_*``.
    - Values may be NaN / False / 0; missing keys returned in catalog order for
      ``dist_crps_eprocess_incomplete:<key>`` errors in ``verify_research_artifact``.
    """
    if not isinstance(payload, dict) or not payload:
        return []
    keys = {str(k) for k in payload}
    missing: list[str] = []
    if DM_CRPS_MARKER_KEY in keys:
        missing.extend(key for key in DIST_CRPS_EPROCESS_REQUIRED_WHEN_DM if key not in keys)
    if DM_CRPS_SCALED_MARKER_KEY in keys:
        missing.extend(key for key in DIST_CRPS_SCALED_EPROCESS_REQUIRED_WHEN_DM if key not in keys)
    return missing


def dist_crps_eprocess_keys_present(payload: object) -> bool:
    """Return True iff soft distribution CRPS e-process key presence holds."""
    return not dist_crps_eprocess_missing_keys(payload)


def tail_es_battery_missing_keys(payload: object) -> list[str]:
    """Return missing Acerbi/FZ keys when a nonempty blob has ES/VaR markers.

    Contract (Day Wave 21 soft honesty — research diagnostic only):
    - Empty ``{}`` or non-dict → no required keys (tiny panels may omit battery).
    - Nonempty dict without ``es_95`` / ``realized_es`` / ``var_95`` → skip
      (no ES/VaR forecast markers ⇒ no ES-battery contract).
    - Nonempty dict with any ES marker (even NaN) → require Acerbi Z1/Z2 +
      ``fissler_ziegel_mean`` + ``es_hit_count`` key *presence*; values may be
      NaN. Missing keys returned in catalog order for
      ``tail_es_battery_incomplete:<key>`` errors in ``verify_research_artifact``.
    - Does **not** double-require VaR-battery / Christoffersen keys — keep
      orthogonal (Kupiec ⇒ Christoffersen; ES markers ⇒ Acerbi/FZ). When both
      Kupiec and ES markers are present, both batteries apply independently.
    """
    if not isinstance(payload, dict) or not payload:
        return []
    keys = {str(k) for k in payload}
    if not (keys & ES_MARKER_KEYS):
        return []
    return [key for key in TAIL_ES_BATTERY_REQUIRED_WHEN_ES_MARKERS if key not in keys]


def tail_es_battery_keys_present(payload: object) -> bool:
    """Return True iff soft ES-battery key presence holds (see missing_keys)."""
    return not tail_es_battery_missing_keys(payload)


def hypotheses_include_h4b(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff *hypotheses* contains ``H4b_var_christoffersen_cc``.

    When *require_calibration* is True (default), the matching row must also
    have ``family == "calibration"`` (Day Wave 17 mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H4B_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H4B_EXPECTED_FAMILY)
    return False


def hypotheses_include_h4(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff *hypotheses* contains ``H4_var_kupiec``.

    When *require_calibration* is True (default), the matching row must also
    have ``family == "calibration"`` (agent mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H4_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H4_EXPECTED_FAMILY)
    return False


def hypotheses_include_h3(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff *hypotheses* contains ``H3_vol_dm``.

    When *require_discovery* is True (default), the matching row must also
    have ``family == "discovery"`` (agent mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H3_HYPOTHESIS_ID:
            continue
        return not (require_discovery and hyp.get("family") != H3_EXPECTED_FAMILY)
    return False


def rankers_oracle_raw(rankers: object) -> dict | None:
    """Return the ``oracle_raw`` ranker dict from *rankers*, or None.

    Skips non-dicts and names starting with ``_`` (same filter as agent
    ``model_rankers``). First matching ``name == "oracle_raw"`` wins.
    """
    if not isinstance(rankers, list):
        return None
    for row in rankers:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not isinstance(name, str) or name.startswith("_"):
            continue
        if name == "oracle_raw":
            return row
    return None


def hypotheses_include_h1(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff *hypotheses* contains ``H1_ranking_oracle``.

    When *require_discovery* is True (default), the matching row must also
    have ``family == "discovery"`` (agent mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H1_HYPOTHESIS_ID:
            continue
        return not (require_discovery and hyp.get("family") != H1_EXPECTED_FAMILY)
    return False


def hypotheses_include_h2(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff *hypotheses* contains ``H2_decile_mono``.

    When *require_discovery* is True (default), the matching row must also
    have ``family == "discovery"`` (agent mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H2_HYPOTHESIS_ID:
            continue
        return not (require_discovery and hyp.get("family") != H2_EXPECTED_FAMILY)
    return False


def hypotheses_include_h99(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff *hypotheses* contains ``H46_ranking_data_snooping`` (discovery)."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H99_HYPOTHESIS_ID:
            continue
        return not (require_discovery and hyp.get("family") != H99_EXPECTED_FAMILY)
    return False


conformal_aci_payload = conformal_aci_blob


def hypotheses_include_h7(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff *hypotheses* contains ``H7_aci_coverage``.

    When *require_calibration* is True (default), the matching row must also
    have ``family == "calibration"`` (agent mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H7_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H7_EXPECTED_FAMILY)
    return False


conformal_mondrian_aci_payload = conformal_mondrian_aci_blob


def hypotheses_include_h8(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff *hypotheses* contains ``H8_mondrian_high_vol``.

    When *require_calibration* is True (default), the matching row must also
    have ``family == "calibration"`` (agent mint contract).
    """
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict):
            continue
        if hyp.get("id") != H8_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H8_EXPECTED_FAMILY)
    return False


def hypotheses_include_h11(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain the CRC calibration hypothesis H11."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H11_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H11_EXPECTED_FAMILY)
    return False


def hypotheses_include_h12(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain the weighted CQR calibration hypothesis H12."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H12_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H12_EXPECTED_FAMILY)
    return False


def hypotheses_include_h9(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain the e-process ACI calibration hypothesis H9."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H9_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H9_EXPECTED_FAMILY)
    return False


def hypotheses_include_h10(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the Jackknife+ bound hypothesis H10."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H10_HYPOTHESIS_ID:
            continue
        return not (require_bound and hyp.get("family") != H10_EXPECTED_FAMILY)
    return False


def hypotheses_include_h15(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the CV+ bound hypothesis H15."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H15_HYPOTHESIS_ID:
            continue
        return not (require_bound and hyp.get("family") != H15_EXPECTED_FAMILY)
    return False


def hypotheses_include_id(
    hypotheses: object,
    hyp_id: str,
    *,
    require_family: str | None = H16_H18_EXPECTED_FAMILY,
) -> bool:
    """True iff hypotheses contain ``hyp_id`` (optionally requiring family)."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != hyp_id:
            continue
        return require_family is None or hyp.get("family") == require_family
    return False


def hypotheses_include_h16(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain panel localized CQR calibration hypothesis H16."""
    fam = H16_H18_EXPECTED_FAMILY if require_calibration else None
    return hypotheses_include_id(hypotheses, H16_HYPOTHESIS_ID, require_family=fam)


def hypotheses_include_h17(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain panel online CRC calibration hypothesis H17."""
    fam = H16_H18_EXPECTED_FAMILY if require_calibration else None
    return hypotheses_include_id(hypotheses, H17_HYPOTHESIS_ID, require_family=fam)


def hypotheses_include_h18(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain panel portfolio conformal calibration hypothesis H18."""
    fam = H16_H18_EXPECTED_FAMILY if require_calibration else None
    return hypotheses_include_id(hypotheses, H18_HYPOTHESIS_ID, require_family=fam)


def hypotheses_include_h19(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the conformal-rank bound hypothesis H19."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H19_HYPOTHESIS_ID:
            continue
        return not (require_bound and hyp.get("family") != H19_EXPECTED_FAMILY)
    return False


def hypotheses_include_h20(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the Northset OHLC bound hypothesis H20."""
    fam = H20_EXPECTED_FAMILY if require_bound else None
    return hypotheses_include_id(hypotheses, H20_HYPOTHESIS_ID, require_family=fam)


def hypotheses_include_h21(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the Northset book bound hypothesis H21."""
    fam = H21_EXPECTED_FAMILY if require_bound else None
    return hypotheses_include_id(hypotheses, H21_HYPOTHESIS_ID, require_family=fam)


def hypotheses_include_h22(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff hypotheses contain the Northset imbalance discovery hypothesis H22."""
    fam = H22_EXPECTED_FAMILY if require_discovery else None
    return hypotheses_include_id(hypotheses, H22_HYPOTHESIS_ID, require_family=fam)


def hypotheses_include_h43(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff hypotheses contain H43 session-book VPIN discovery row."""
    fam = H43_EXPECTED_FAMILY if require_discovery else None
    return hypotheses_include_id(hypotheses, H43_HYPOTHESIS_ID, require_family=fam)


def jp_cv_blob_requires_marginal_coverage_scope(blob: object) -> bool:
    """True iff nonempty family blob exposes coverage and/or coverage_floor keys."""
    if not isinstance(blob, dict) or not blob:
        return False
    return "coverage" in blob or "coverage_floor" in blob


def coverage_guarantee_scope_is_marginal(blob: object) -> bool:
    """True iff blob declares coverage_guarantee_scope=marginal_exchangeable."""
    return (
        isinstance(blob, dict)
        and blob.get("coverage_guarantee_scope") == COVERAGE_GUARANTEE_SCOPE_MARGINAL
    )
