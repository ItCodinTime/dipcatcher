"""Hypothesis-table consistency gates.

Split out of the original module. Import the parent path; it re-exports these names.
"""

from __future__ import annotations

from typing import Any

from .primitives import _finite_scalar

# Soft H4b H-table consistency (Day Wave 26 research diagnostic).
# When ``families["tail"]`` exposes a *finite* ``christoffersen_cc_p``, the
# notebook must mint ``H4b_var_christoffersen_cc`` (calibration) — same gate as
# Day Wave 17 ``_finite_number`` mint. Non-finite / missing cc_p → skip.
# Research-receipt fail-closed; not a live capital / promotion gate.
H4B_HYPOTHESIS_ID = "H4b_var_christoffersen_cc"
H4B_EXPECTED_FAMILY = "calibration"


def tail_has_finite_christoffersen_cc_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``christoffersen_cc_p``."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("christoffersen_cc_p"))


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


def h4b_hypothesis_consistency_errors(tail: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H4b notebook consistency (Day Wave 26).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``christoffersen_cc_p`` → no errors (skip).
    - Finite ``christoffersen_cc_p`` without ``H4b_var_christoffersen_cc``
      (calibration) → ``hypothesis_h4b_missing_despite_finite_christoffersen_cc_p``.
    """
    if not tail_has_finite_christoffersen_cc_p(tail):
        return []
    if hypotheses_include_h4b(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h4b_missing_despite_finite_christoffersen_cc_p"]


# Soft H4 Kupiec H-table consistency (Day Wave 29 research diagnostic).
# When ``families["tail"]`` exposes a *finite* ``kupiec_p``, the notebook must
# mint ``H4_var_kupiec`` (calibration) — same gate as agent ``_finite_number``
# mint. Non-finite / missing kupiec_p → skip.
# Research-receipt fail-closed; not a live capital / promotion gate.
H4_HYPOTHESIS_ID = "H4_var_kupiec"
H4_EXPECTED_FAMILY = "calibration"


def tail_has_finite_kupiec_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``kupiec_p``."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("kupiec_p"))


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


def h4_hypothesis_consistency_errors(tail: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H4 notebook consistency (Day Wave 29).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``kupiec_p`` → no errors (skip).
    - Finite ``kupiec_p`` without ``H4_var_kupiec``
      (calibration) → ``hypothesis_h4_missing_despite_finite_kupiec_p``.
    """
    if not tail_has_finite_kupiec_p(tail):
        return []
    if hypotheses_include_h4(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h4_missing_despite_finite_kupiec_p"]


# Soft H3 vol-DM H-table consistency (Day Wave 30 research diagnostic).
# When ``families["volatility"]`` exposes a *finite* ``dm_p``, the notebook must
# mint ``H3_vol_dm`` (discovery) — same gate as agent ``_finite_number`` mint.
# Non-finite / missing dm_p → skip.
# Research-receipt fail-closed; not a live capital / promotion gate.
H3_HYPOTHESIS_ID = "H3_vol_dm"
H3_EXPECTED_FAMILY = "discovery"


def volatility_has_finite_dm_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``dm_p``."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("dm_p"))


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


def h3_hypothesis_consistency_errors(volatility: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H3 notebook consistency (Day Wave 30).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``dm_p`` → no errors (skip).
    - Finite ``dm_p`` without ``H3_vol_dm``
      (discovery) → ``hypothesis_h3_missing_despite_finite_dm_p``.
    """
    if not volatility_has_finite_dm_p(volatility):
        return []
    if hypotheses_include_h3(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h3_missing_despite_finite_dm_p"]


# Soft H1/H2 oracle ranking H-table consistency (Day Wave 31 research diagnostic).
# Oracle lives in ``notebook["rankers"]`` (name ``oracle_raw``), NOT families.
# Skip ranker names starting with ``_`` (agent model_rankers filter).
# Finite ``p_ic`` → require ``H1_ranking_oracle`` (discovery); finite ``ls_p`` →
# require ``H2_decile_mono`` (discovery). Gates are independent (mirror agent mint).
# Non-finite / missing / no oracle_raw → skip each gate independently.
# Research-receipt fail-closed; not a live capital / promotion gate.
H1_HYPOTHESIS_ID = "H1_ranking_oracle"
H1_EXPECTED_FAMILY = "discovery"
H2_HYPOTHESIS_ID = "H2_decile_mono"
H2_EXPECTED_FAMILY = "discovery"


def rankers_oracle_raw(rankers: object) -> dict[str, Any] | None:
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


def oracle_has_finite_p_ic(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``p_ic`` (oracle_raw row)."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("p_ic"))


def oracle_has_finite_ls_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``ls_p`` (oracle_raw row)."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("ls_p"))


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


def h1_hypothesis_consistency_errors(rankers: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H1 notebook consistency (Day Wave 31).

    Contract (research diagnostic only):
    - No ``oracle_raw`` / non-dict / missing / non-finite ``p_ic`` → no errors (skip).
    - Finite ``p_ic`` without ``H1_ranking_oracle``
      (discovery) → ``hypothesis_h1_missing_despite_finite_p_ic``.
    """
    oracle = rankers_oracle_raw(rankers)
    if oracle is None or not oracle_has_finite_p_ic(oracle):
        return []
    if hypotheses_include_h1(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h1_missing_despite_finite_p_ic"]


def h2_hypothesis_consistency_errors(rankers: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H2 notebook consistency (Day Wave 31).

    Contract (research diagnostic only):
    - No ``oracle_raw`` / non-dict / missing / non-finite ``ls_p`` → no errors (skip).
    - Finite ``ls_p`` without ``H2_decile_mono``
      (discovery) → ``hypothesis_h2_missing_despite_finite_ls_p``.
    """
    oracle = rankers_oracle_raw(rankers)
    if oracle is None or not oracle_has_finite_ls_p(oracle):
        return []
    if hypotheses_include_h2(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h2_missing_despite_finite_ls_p"]


# Data-snooping battery over the ranker universe (Reality Check / SPA / StepM /
# MCS). When ``families["ranking"]["data_snooping"]`` exposes a *finite*
# consistent SPA p-value, the notebook must mint ``H46_ranking_data_snooping``
# (discovery) — same gate as the agent ``_finite_number`` mint. Non-finite /
# missing / no block → skip. Research-receipt fail-closed; not a live capital
# gate and never a live Sharpe claim.
H99_HYPOTHESIS_ID = "H99_ranking_data_snooping"
H99_EXPECTED_FAMILY = "discovery"
_DATA_SNOOPING_P_KEYS = ("reality_check_p", "spa_p_lower", "spa_p_consistent", "spa_p_upper")


def ranking_data_snooping_blob(ranking: object) -> dict[str, Any] | None:
    """Return the nested ``data_snooping`` dict from a ranking family blob."""
    if not isinstance(ranking, dict):
        return None
    blob = ranking.get("data_snooping")
    return blob if isinstance(blob, dict) and blob else None


def data_snooping_has_finite_spa_p(blob: object) -> bool:
    """True iff the data-snooping blob exposes a finite consistent SPA p-value."""
    if not isinstance(blob, dict):
        return False
    return _finite_scalar(blob.get("spa_p_consistent"))


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


def h99_hypothesis_consistency_errors(ranking: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H46 data-snooping notebook consistency.

    Contract (research diagnostic only):
    - No ``data_snooping`` block / non-dict / non-finite ``spa_p_consistent``
      → no errors (skip).
    - Finite consistent SPA p without ``H46_ranking_data_snooping`` (discovery)
      → ``hypothesis_h99_missing_despite_finite_spa_p``.
    """
    blob = ranking_data_snooping_blob(ranking)
    if blob is None or not data_snooping_has_finite_spa_p(blob):
        return []
    if hypotheses_include_h99(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h99_missing_despite_finite_spa_p"]


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


# Soft H7 ACI coverage H-table consistency (Day Wave 32 research diagnostic).
# When ``families["conformal"]["aci"]`` exposes a *finite* ``kupiec_p``, the
# notebook must mint ``H7_aci_coverage`` (calibration) — same gate as agent
# ``_finite_number`` mint on ACI miss-Kupiec. Non-finite / missing / no aci → skip.
# Research-receipt fail-closed; not a live capital / promotion gate.
H7_HYPOTHESIS_ID = "H7_aci_coverage"
H7_EXPECTED_FAMILY = "calibration"


def conformal_aci_blob(conformal: object) -> dict[str, Any] | None:
    """Return ``families['conformal']['aci']`` dict, or None.

    *conformal* is ``families.get("conformal")`` (the conformal family blob),
    not the full families dict.
    """
    if not isinstance(conformal, dict):
        return None
    aci = conformal.get("aci")
    return aci if isinstance(aci, dict) else None


# Back-compat alias (Day Wave 32 rename payload→blob).
conformal_aci_payload = conformal_aci_blob


def aci_has_finite_kupiec_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``kupiec_p`` (ACI row)."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("kupiec_p"))


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


def h7_hypothesis_consistency_errors(conformal: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H7 notebook consistency (Day Wave 32).

    Contract (research diagnostic only):
    - No ``aci`` / non-dict / missing / non-finite ``kupiec_p`` → no errors (skip).
    - Finite ACI ``kupiec_p`` without ``H7_aci_coverage``
      (calibration) → ``hypothesis_h7_missing_despite_finite_aci_kupiec_p``.
    """
    aci = conformal_aci_blob(conformal)
    if aci is None or not aci_has_finite_kupiec_p(aci):
        return []
    if hypotheses_include_h7(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h7_missing_despite_finite_aci_kupiec_p"]


# Soft H8 Mondrian high-vol H-table consistency (Day Wave 33 research diagnostic).
# When ``families["conformal"]["mondrian_aci"]`` exposes a *finite* ``high_x_kupiec_p``,
# the notebook must mint ``H8_mondrian_high_vol`` (calibration) — same gate as agent
# ``_finite_number`` mint on Mondrian high-X miss-Kupiec. Non-finite / missing /
# no mondrian_aci → skip. Research-receipt fail-closed; not a live capital / promotion gate.
H8_HYPOTHESIS_ID = "H8_mondrian_high_vol"
H8_EXPECTED_FAMILY = "calibration"


def conformal_mondrian_aci_blob(conformal: object) -> dict[str, Any] | None:
    """Return ``families['conformal']['mondrian_aci']`` dict, or None.

    *conformal* is ``families.get("conformal")`` (the conformal family blob),
    not the full families dict.
    """
    if not isinstance(conformal, dict):
        return None
    mond = conformal.get("mondrian_aci")
    return mond if isinstance(mond, dict) else None


def mondrian_aci_has_finite_high_x_kupiec_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``high_x_kupiec_p`` (Mondrian row)."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("high_x_kupiec_p"))


# Brief / back-compat aliases (Day Wave 33; mirror H7 payload alias).
conformal_mondrian_aci_payload = conformal_mondrian_aci_blob
mondrian_has_finite_high_x_kupiec_p = mondrian_aci_has_finite_high_x_kupiec_p


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


def h8_hypothesis_consistency_errors(conformal: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H8 notebook consistency (Day Wave 33).

    Contract (research diagnostic only):
    - No ``mondrian_aci`` / non-dict / missing / non-finite ``high_x_kupiec_p`` → no errors (skip).
    - Finite Mondrian ``high_x_kupiec_p`` without ``H8_mondrian_high_vol``
      (calibration) → ``hypothesis_h8_missing_despite_finite_high_x_kupiec_p``.
    """
    mond = conformal_mondrian_aci_blob(conformal)
    if mond is None or not mondrian_aci_has_finite_high_x_kupiec_p(mond):
        return []
    if hypotheses_include_h8(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h8_missing_despite_finite_high_x_kupiec_p"]


# Soft H11 CRC coverage H-table consistency (Day Wave 34 research diagnostic).
# When ``families["crc"]`` exposes a *finite* ``kupiec_p``, the notebook must mint
# ``H11_crc_var`` (calibration) — same gate as agent ``_finite_number`` mint on CRC
# Kupiec. Non-finite / missing → skip. Research-receipt fail-closed; not a live
# capital / promotion gate.
H11_HYPOTHESIS_ID = "H11_crc_var"
H11_EXPECTED_FAMILY = "calibration"


def crc_has_finite_kupiec_p(crc: object) -> bool:
    """True iff the CRC family exposes a finite Kupiec p-value."""
    return isinstance(crc, dict) and _finite_scalar(crc.get("kupiec_p"))


def hypotheses_include_h11(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain the CRC calibration hypothesis H11."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H11_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H11_EXPECTED_FAMILY)
    return False


def h11_hypothesis_consistency_errors(crc: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H11 notebook consistency (Day Wave 34).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``kupiec_p`` → no errors (skip).
    - Finite CRC ``kupiec_p`` without ``H11_crc_var`` (calibration) →
      ``hypothesis_h11_missing_despite_finite_crc_kupiec_p``.
    """
    if not crc_has_finite_kupiec_p(crc):
        return []
    if hypotheses_include_h11(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h11_missing_despite_finite_crc_kupiec_p"]


# Soft H12 weighted CQR coverage H-table consistency (Day Wave 35 research diagnostic).
# When ``families["weighted_conformal"]`` exposes a *finite* ``kupiec_p``, the notebook
# must mint ``H12_weighted_cqr`` (calibration) — same gate as agent ``_finite_number``
# mint on weighted CQR Kupiec. Non-finite / missing → skip. Research-receipt fail-closed;
# not a live capital / promotion gate.
H12_HYPOTHESIS_ID = "H12_weighted_cqr"
H12_EXPECTED_FAMILY = "calibration"


def weighted_conformal_has_finite_kupiec_p(wcqr: object) -> bool:
    """True iff the weighted_conformal family exposes a finite Kupiec p-value."""
    return isinstance(wcqr, dict) and _finite_scalar(wcqr.get("kupiec_p"))


def hypotheses_include_h12(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain the weighted CQR calibration hypothesis H12."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H12_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H12_EXPECTED_FAMILY)
    return False


def h12_hypothesis_consistency_errors(wcqr: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H12 notebook consistency (Day Wave 35).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``kupiec_p`` → no errors (skip).
    - Finite weighted_conformal ``kupiec_p`` without ``H12_weighted_cqr``
      (calibration) → ``hypothesis_h12_missing_despite_finite_wcqr_kupiec_p``.
    """
    if not weighted_conformal_has_finite_kupiec_p(wcqr):
        return []
    if hypotheses_include_h12(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h12_missing_despite_finite_wcqr_kupiec_p"]


# Soft H9 e-process ACI H-table consistency (Day Wave 36 research diagnostic).
# When ``families["evalues"]`` exposes a *finite* ``e_sup``, the notebook must mint
# ``H9_eprocess_aci`` (calibration) — same gate as agent ``_finite_number`` mint on
# e-process e_sup. Non-finite / missing → skip. Research-receipt fail-closed; not a
# live capital / promotion gate.
H9_HYPOTHESIS_ID = "H9_eprocess_aci"
H9_EXPECTED_FAMILY = "calibration"


def evalues_has_finite_e_sup(evalues: object) -> bool:
    """True iff the evalues family exposes a finite e_sup."""
    return isinstance(evalues, dict) and _finite_scalar(evalues.get("e_sup"))


def hypotheses_include_h9(hypotheses: object, *, require_calibration: bool = True) -> bool:
    """True iff hypotheses contain the e-process ACI calibration hypothesis H9."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H9_HYPOTHESIS_ID:
            continue
        return not (require_calibration and hyp.get("family") != H9_EXPECTED_FAMILY)
    return False


def h9_hypothesis_consistency_errors(evalues: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H9 notebook consistency (Day Wave 36).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``e_sup`` → no errors (skip).
    - Finite evalues ``e_sup`` without ``H9_eprocess_aci`` (calibration) →
      ``hypothesis_h9_missing_despite_finite_e_sup``.
    """
    if not evalues_has_finite_e_sup(evalues):
        return []
    if hypotheses_include_h9(hypotheses, require_calibration=True):
        return []
    return ["hypothesis_h9_missing_despite_finite_e_sup"]


# Soft H10 Jackknife+ coverage H-table consistency (Day Wave 37 research diagnostic).
# When ``families["jackknife_plus"]`` exposes a *finite* ``coverage``, the notebook
# must mint ``H10_jackknife_coverage`` (bound) — same gate as agent ``_finite_number``
# mint on Jackknife+ coverage. Non-finite / missing → skip. Research-receipt fail-closed;
# not a live capital / promotion gate. Note: expected family is ``bound`` (not calibration).
H10_HYPOTHESIS_ID = "H10_jackknife_coverage"
H10_EXPECTED_FAMILY = "bound"


def jackknife_plus_has_finite_coverage(jp: object) -> bool:
    """True iff the jackknife_plus family exposes a finite coverage."""
    return isinstance(jp, dict) and _finite_scalar(jp.get("coverage"))


def hypotheses_include_h10(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the Jackknife+ bound hypothesis H10."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H10_HYPOTHESIS_ID:
            continue
        return not (require_bound and hyp.get("family") != H10_EXPECTED_FAMILY)
    return False


def h10_hypothesis_consistency_errors(jp: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H10 notebook consistency (Day Wave 37).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``coverage`` → no errors (skip).
    - Finite jackknife_plus ``coverage`` without ``H10_jackknife_coverage``
      (bound) → ``hypothesis_h10_missing_despite_finite_coverage``.
    """
    if not jackknife_plus_has_finite_coverage(jp):
        return []
    if hypotheses_include_h10(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h10_missing_despite_finite_coverage"]


# Soft H15 CV+ floor H-table consistency (Day Wave 38 research diagnostic).
# When ``families["cv_plus"]`` exposes *finite* ``coverage`` *and* finite
# ``coverage_floor``, the notebook must mint ``H15_cv_plus_floor`` (bound) —
# same gate as agent ``_finite_number`` mint on both fields. Non-finite /
# missing either field → skip. Research-receipt fail-closed; not a live capital
# / promotion gate. Note: expected family is ``bound`` (not calibration).
H15_HYPOTHESIS_ID = "H15_cv_plus_floor"
H15_EXPECTED_FAMILY = "bound"


def cv_plus_has_finite_coverage_and_floor(cvp: object) -> bool:
    """True iff the cv_plus family exposes finite coverage and coverage_floor."""
    return (
        isinstance(cvp, dict)
        and _finite_scalar(cvp.get("coverage"))
        and _finite_scalar(cvp.get("coverage_floor"))
    )


def hypotheses_include_h15(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the CV+ bound hypothesis H15."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H15_HYPOTHESIS_ID:
            continue
        return not (require_bound and hyp.get("family") != H15_EXPECTED_FAMILY)
    return False


def h15_hypothesis_consistency_errors(cvp: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H15 notebook consistency (Day Wave 38).

    Contract (research diagnostic only):
    - Non-dict / missing / non-finite ``coverage`` or ``coverage_floor`` → no errors (skip).
    - Finite cv_plus ``coverage`` + ``coverage_floor`` without ``H15_cv_plus_floor``
      (bound) → ``hypothesis_h15_missing_despite_finite_coverage_and_floor``.
    """
    if not cv_plus_has_finite_coverage_and_floor(cvp):
        return []
    if hypotheses_include_h15(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h15_missing_despite_finite_coverage_and_floor"]


# Soft H16–H18 panel Kupiec H-table consistency (Day Wave 39 research diagnostic).
# When a *panel* family blob (localized_conformal / online_crc / portfolio_conformal)
# exposes a *finite* ``kupiec_p`` and ``dgp != "fixture"``, the notebook must mint the
# matching calibration hypothesis (agent mint). Fixture DGP never shares this H-table —
# skip. Non-finite / missing → skip. Research-receipt fail-closed; not a live capital /
# promotion gate. One parameterized helper keeps the three rows thin.
H16_H18_EXPECTED_FAMILY = "calibration"
H16_H18_PANEL_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "localized_conformal",
        "H16_localized_cqr",
        "hypothesis_h16_missing_despite_finite_kupiec_p",
    ),
    (
        "online_crc",
        "H17_online_crc",
        "hypothesis_h17_missing_despite_finite_kupiec_p",
    ),
    (
        "portfolio_conformal",
        "H18_portfolio_conformal",
        "hypothesis_h18_missing_despite_finite_kupiec_p",
    ),
)
H16_HYPOTHESIS_ID = "H16_localized_cqr"
H17_HYPOTHESIS_ID = "H17_online_crc"
H18_HYPOTHESIS_ID = "H18_portfolio_conformal"


def panel_family_has_finite_kupiec_p(blob: object) -> bool:
    """True iff panel family blob has a finite panel Kupiec p and is not fixture DGP.

    Panel families (H16–H18) may expose either ``kupiec_p`` or the clustered
    variant ``date_clustered_p``; either finite value satisfies the gate.
    """
    if not isinstance(blob, dict):
        return False
    if str(blob.get("dgp") or "") == "fixture":
        return False
    return _finite_scalar(blob.get("kupiec_p")) or _finite_scalar(blob.get("date_clustered_p"))


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


def h16_h18_panel_kupiec_consistency_errors(families: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H16–H18 panel Kupiec notebook consistency (Day Wave 39).

    Contract (research diagnostic only):
    - Non-dict families / missing family / fixture ``dgp`` / non-finite ``kupiec_p`` → skip.
    - Finite panel ``kupiec_p`` without matching H16/H17/H18 (calibration) → error token.
    """
    if not isinstance(families, dict):
        return []
    errors: list[str] = []
    for family_key, hyp_id, err_token in H16_H18_PANEL_SPECS:
        blob = families.get(family_key)
        if not panel_family_has_finite_kupiec_p(blob):
            continue
        if hypotheses_include_id(hypotheses, hyp_id, require_family=H16_H18_EXPECTED_FAMILY):
            continue
        errors.append(err_token)
    return errors


# Soft H19 conformal-rank FDR bound H-table consistency (Day Wave 40 research diagnostic).
# When ``families["conformal_rank"]`` exposes a *finite* ``fdr`` and ``dgp != "fixture"``,
# the notebook must mint ``H19_conformal_rank`` (bound) — same gate as agent mint
# (``_finite_number`` on fdr; fixture DGP never shares this H-table). Non-finite /
# missing / fixture → skip. Alpha defaults to 0.20 at mint time when missing; soft
# verify only gates finite-fdr + non-fixture → bound H19 presence. Research-receipt
# fail-closed; not a live capital / promotion gate. Note: expected family is ``bound``.
H19_HYPOTHESIS_ID = "H19_conformal_rank"
H19_EXPECTED_FAMILY = "bound"


def conformal_rank_has_finite_fdr(topk: object) -> bool:
    """True iff conformal_rank family has finite fdr and is not fixture DGP."""
    if not isinstance(topk, dict):
        return False
    if str(topk.get("dgp") or "") == "fixture":
        return False
    return _finite_scalar(topk.get("fdr"))


def hypotheses_include_h19(hypotheses: object, *, require_bound: bool = True) -> bool:
    """True iff hypotheses contain the conformal-rank bound hypothesis H19."""
    if not isinstance(hypotheses, list):
        return False
    for hyp in hypotheses:
        if not isinstance(hyp, dict) or hyp.get("id") != H19_HYPOTHESIS_ID:
            continue
        return not (require_bound and hyp.get("family") != H19_EXPECTED_FAMILY)
    return False


def h19_hypothesis_consistency_errors(topk: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H19 notebook consistency (Day Wave 40).

    Contract (research diagnostic only):
    - Non-dict / missing / fixture ``dgp`` / non-finite ``fdr`` → no errors (skip).
    - Finite conformal_rank ``fdr`` (non-fixture) without ``H19_conformal_rank``
      (bound) → ``hypothesis_h19_missing_despite_finite_fdr``.
    """
    if not conformal_rank_has_finite_fdr(topk):
        return []
    if hypotheses_include_h19(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h19_missing_despite_finite_fdr"]


# Soft H20–H22 Northset notebook consistency (catalog v2 research diagnostic).
# Finite ``ohlc_identity_rate`` → bound H20; finite ``book_uncrossed_rate`` → bound
# H21; finite ``imbalance_top_p_ic`` → discovery H22. Non-finite / missing → skip.
# Research-receipt fail-closed; not a live capital / promotion gate.
H20_HYPOTHESIS_ID = "H20_northset_ohlc"
H20_EXPECTED_FAMILY = "bound"
H21_HYPOTHESIS_ID = "H21_northset_book"
H21_EXPECTED_FAMILY = "bound"
H22_HYPOTHESIS_ID = "H22_northset_imbalance"
H22_EXPECTED_FAMILY = "discovery"


def northset_has_finite_ohlc_identity_rate(blob: object) -> bool:
    """True iff the northset family exposes a finite ohlc_identity_rate."""
    return isinstance(blob, dict) and _finite_scalar(blob.get("ohlc_identity_rate"))


def northset_has_finite_book_uncrossed_rate(blob: object) -> bool:
    """True iff the northset family exposes a finite book_uncrossed_rate."""
    return (
        isinstance(blob, dict)
        and blob.get("book_hypothesis_eligible", True) is not False
        and _finite_scalar(blob.get("book_uncrossed_rate"))
    )


def northset_has_finite_imbalance_p_ic(blob: object) -> bool:
    """True iff the northset family exposes a finite imbalance_top_p_ic."""
    return (
        isinstance(blob, dict)
        and blob.get("book_hypothesis_eligible", True) is not False
        and _finite_scalar(blob.get("imbalance_top_p_ic"))
    )


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


def h20_hypothesis_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H20 Northset OHLC identity consistency."""
    if not northset_has_finite_ohlc_identity_rate(blob):
        return []
    if hypotheses_include_h20(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h20_missing_despite_finite_ohlc_identity_rate"]


def h21_hypothesis_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H21 Northset uncrossed-book consistency."""
    if not northset_has_finite_book_uncrossed_rate(blob):
        return []
    if hypotheses_include_h21(hypotheses, require_bound=True):
        return []
    return ["hypothesis_h21_missing_despite_finite_book_uncrossed_rate"]


def h22_hypothesis_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for H22 Northset imbalance IC consistency."""
    if not northset_has_finite_imbalance_p_ic(blob):
        return []
    if hypotheses_include_h22(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h22_missing_despite_finite_imbalance_p_ic"]


# Soft H23–H51 Northset expansion (catalog v2). Finite metric → matching H-row.
# Bound: session reconstruct / volume conservation / session chain / OOT sign.
# Discovery: microprice IC, wick-skew IC, OFI IC, GK-vs-Parkinson DM,
# CLV IC, overnight-split vs Parkinson DM, VPIN IC, sweep-reject IC,
# sweep-follow IC, event studies, placebos, matched/liq controls, name cluster,
# two-way cluster, untradeable overnight gap.
# Non-finite / missing → skip.
H23_HYPOTHESIS_ID = "H23_northset_session"
H24_HYPOTHESIS_ID = "H24_northset_volume"
H25_HYPOTHESIS_ID = "H25_northset_microprice"
H26_HYPOTHESIS_ID = "H26_northset_wick"
H27_HYPOTHESIS_ID = "H27_northset_ofi"
H28_HYPOTHESIS_ID = "H28_northset_gk"
H29_HYPOTHESIS_ID = "H29_northset_chain"
H30_HYPOTHESIS_ID = "H30_northset_clv"
H31_HYPOTHESIS_ID = "H31_northset_overnight"
H32_HYPOTHESIS_ID = "H32_northset_vpin"
H33_HYPOTHESIS_ID = "H33_northset_sweep_reject"
H34_HYPOTHESIS_ID = "H34_northset_sweep_follow"
H35_HYPOTHESIS_ID = "H35_northset_reject_event"
H36_HYPOTHESIS_ID = "H36_northset_follow_event"
H37_HYPOTHESIS_ID = "H37_northset_reject_placebo"
H38_HYPOTHESIS_ID = "H38_northset_follow_placebo"
H39_HYPOTHESIS_ID = "H39_northset_reject_cost"
H40_HYPOTHESIS_ID = "H40_northset_follow_cost"
H41_HYPOTHESIS_ID = "H41_northset_reject_stability"
H42_HYPOTHESIS_ID = "H42_northset_follow_stability"
H44_HYPOTHESIS_ID = "H44_northset_reject_control"
H45_HYPOTHESIS_ID = "H45_northset_follow_control"
H46_HYPOTHESIS_ID = "H46_northset_reject_liq_control"
H47_HYPOTHESIS_ID = "H47_northset_follow_liq_control"
H48_HYPOTHESIS_ID = "H48_northset_follow_oot"
H49_HYPOTHESIS_ID = "H49_northset_follow_name_cluster"
H50_HYPOTHESIS_ID = "H50_northset_follow_two_way_cluster"
H51_HYPOTHESIS_ID = "H51_northset_follow_overnight_gap"
NORTHSET_H23_H28_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (
        "session_reconstructs_daily_rate",
        H23_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h23_missing_despite_finite_session_reconstructs_daily_rate",
    ),
    (
        "session_volume_conservation_rate",
        H24_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h24_missing_despite_finite_session_volume_conservation_rate",
    ),
    (
        "microprice_p_ic",
        H25_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h25_missing_despite_finite_microprice_p_ic",
    ),
    (
        "wick_skew_p_ic",
        H26_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h26_missing_despite_finite_wick_skew_p_ic",
    ),
    (
        "ofi_p_ic",
        H27_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h27_missing_despite_finite_ofi_p_ic",
    ),
    (
        "dm_gk_vs_park_p",
        H28_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h28_missing_despite_finite_dm_gk_vs_park_p",
    ),
    (
        "session_chain_rate",
        H29_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h29_missing_despite_finite_session_chain_rate",
    ),
    (
        "clv_p_ic",
        H30_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h30_missing_despite_finite_clv_p_ic",
    ),
    (
        "dm_split_vs_park_p",
        H31_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h31_missing_despite_finite_dm_split_vs_park_p",
    ),
    (
        "vpin_p_ic",
        H32_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h32_missing_despite_finite_vpin_p_ic",
    ),
    (
        "sweep_reject_signed_p_ic",
        H33_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h33_missing_despite_finite_sweep_reject_signed_p_ic",
    ),
    (
        "sweep_follow_signed_p_ic",
        H34_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h34_missing_despite_finite_sweep_follow_signed_p_ic",
    ),
    (
        "sweep_reject_event_p",
        H35_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h35_missing_despite_finite_sweep_reject_event_p",
    ),
    (
        "sweep_follow_event_p",
        H36_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h36_missing_despite_finite_sweep_follow_event_p",
    ),
    (
        "sweep_reject_placebo_p",
        H37_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h37_missing_despite_finite_sweep_reject_placebo_p",
    ),
    (
        "sweep_follow_placebo_p",
        H38_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h38_missing_despite_finite_sweep_follow_placebo_p",
    ),
    (
        "sweep_reject_cost_adjusted_mean_bps",
        H39_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h39_missing_despite_finite_sweep_reject_cost_adjusted_mean_bps",
    ),
    (
        "sweep_follow_cost_adjusted_mean_bps",
        H40_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h40_missing_despite_finite_sweep_follow_cost_adjusted_mean_bps",
    ),
    (
        "sweep_reject_fold_positive_fraction",
        H41_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h41_missing_despite_finite_sweep_reject_fold_positive_fraction",
    ),
    (
        "sweep_follow_fold_positive_fraction",
        H42_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h42_missing_despite_finite_sweep_follow_fold_positive_fraction",
    ),
    (
        "sweep_reject_control_diff_p",
        H44_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h44_missing_despite_finite_sweep_reject_control_diff_p",
    ),
    (
        "sweep_follow_control_diff_p",
        H45_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h45_missing_despite_finite_sweep_follow_control_diff_p",
    ),
    (
        "sweep_reject_liq_control_diff_p",
        H46_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h46_missing_despite_finite_sweep_reject_liq_control_diff_p",
    ),
    (
        "sweep_follow_liq_control_diff_p",
        H47_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h47_missing_despite_finite_sweep_follow_liq_control_diff_p",
    ),
    (
        "sweep_follow_oot_holdout_mean_bps",
        H48_HYPOTHESIS_ID,
        "bound",
        "hypothesis_h48_missing_despite_finite_sweep_follow_oot_holdout_mean_bps",
    ),
    (
        "sweep_follow_name_cluster_p",
        H49_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h49_missing_despite_finite_sweep_follow_name_cluster_p",
    ),
    (
        "sweep_follow_two_way_cluster_p",
        H50_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h50_missing_despite_finite_sweep_follow_two_way_cluster_p",
    ),
    (
        "sweep_follow_overnight_gap_p",
        H51_HYPOTHESIS_ID,
        "discovery",
        "hypothesis_h51_missing_despite_finite_sweep_follow_overnight_gap_p",
    ),
)
H43_HYPOTHESIS_ID = "H43_northset_session_book_vpin"
H43_EXPECTED_FAMILY = "discovery"


def northset_has_finite_session_book_vpin_p_ic(blob: object) -> bool:
    """True iff northset exposes finite session_book_vpin_p_ic (multi-snap path)."""
    return (
        isinstance(blob, dict)
        and blob.get("session_book_hypothesis_eligible", True) is not False
        and _finite_scalar(blob.get("session_book_vpin_p_ic"))
    )


def hypotheses_include_h43(hypotheses: object, *, require_discovery: bool = True) -> bool:
    """True iff hypotheses contain H43 session-book VPIN discovery row."""
    fam = H43_EXPECTED_FAMILY if require_discovery else None
    return hypotheses_include_id(hypotheses, H43_HYPOTHESIS_ID, require_family=fam)


def h43_hypothesis_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Soft-verify: finite session_book_vpin_p_ic → H43 discovery row present."""
    if not northset_has_finite_session_book_vpin_p_ic(blob):
        return []
    if hypotheses_include_h43(hypotheses, require_discovery=True):
        return []
    return ["hypothesis_h43_missing_despite_finite_session_book_vpin_p_ic"]


def northset_h23_h28_consistency_errors(blob: object, hypotheses: object) -> list[str]:
    """Return soft-verify errors for Northset H23–H51 notebook consistency."""
    if not isinstance(blob, dict):
        return []
    errors: list[str] = []
    book_metrics = {"microprice_p_ic", "ofi_p_ic", "vpin_p_ic"}
    for key, hyp_id, family, token in NORTHSET_H23_H28_SPECS:
        if key in book_metrics and blob.get("book_hypothesis_eligible", True) is False:
            continue
        if not _finite_scalar(blob.get(key)):
            continue
        if hypotheses_include_id(hypotheses, hyp_id, require_family=family):
            continue
        errors.append(token)
    return errors


# Soft Wave41 honesty-key receipt verify (Day Wave 42 research diagnostic).
# When nonempty ``jackknife_plus`` / ``cv_plus`` exposes ``coverage`` and/or
# ``coverage_floor`` (values may be NaN — key presence only), the blob must
# also expose ``coverage_guarantee_scope == "marginal_exchangeable"`` (Wave41
# honesty key). Empty ``{}`` / missing coverage keys → skip. Missing key →
# ``coverage_guarantee_scope_missing:<fam>``; wrong value →
# ``coverage_guarantee_scope_invalid:<fam>``. Research-receipt fail-closed;
# not a live capital / promotion gate; does **not** expand soft-verify H-table
# sprawl (H5/H6/H13/H14 remain paused). No forged soft scorecard flag.
COVERAGE_GUARANTEE_SCOPE_MARGINAL = "marginal_exchangeable"
_JP_CV_SCOPE_FAMILIES = ("jackknife_plus", "cv_plus")


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


def coverage_guarantee_scope_consistency_errors(families: object) -> list[str]:
    """Return soft-verify errors for Jackknife+/CV+ marginal scope honesty (Day Wave 42).

    Contract (research diagnostic only):
    - Non-dict families / empty blob / neither ``coverage`` nor ``coverage_floor`` → skip.
    - Nonempty jp/cv blob with coverage keys but missing ``coverage_guarantee_scope``
      → ``coverage_guarantee_scope_missing:<fam>``.
    - Key present but value != ``marginal_exchangeable``
      → ``coverage_guarantee_scope_invalid:<fam>``.
    - NaN coverage/floor still requires scope (key presence only).
    """
    if not isinstance(families, dict):
        return []
    errors: list[str] = []
    for fam in _JP_CV_SCOPE_FAMILIES:
        blob = families.get(fam)
        if not jp_cv_blob_requires_marginal_coverage_scope(blob):
            continue
        assert isinstance(blob, dict)
        if "coverage_guarantee_scope" not in blob:
            errors.append(f"coverage_guarantee_scope_missing:{fam}")
        elif blob.get("coverage_guarantee_scope") != COVERAGE_GUARANTEE_SCOPE_MARGINAL:
            errors.append(f"coverage_guarantee_scope_invalid:{fam}")
    return errors


__all__ = [
    "COVERAGE_GUARANTEE_SCOPE_MARGINAL",
    "H10_EXPECTED_FAMILY",
    "H10_HYPOTHESIS_ID",
    "H11_EXPECTED_FAMILY",
    "H11_HYPOTHESIS_ID",
    "H12_EXPECTED_FAMILY",
    "H12_HYPOTHESIS_ID",
    "H15_EXPECTED_FAMILY",
    "H15_HYPOTHESIS_ID",
    "H16_H18_EXPECTED_FAMILY",
    "H16_H18_PANEL_SPECS",
    "H16_HYPOTHESIS_ID",
    "H17_HYPOTHESIS_ID",
    "H18_HYPOTHESIS_ID",
    "H19_EXPECTED_FAMILY",
    "H19_HYPOTHESIS_ID",
    "H1_EXPECTED_FAMILY",
    "H1_HYPOTHESIS_ID",
    "H20_EXPECTED_FAMILY",
    "H20_HYPOTHESIS_ID",
    "H21_EXPECTED_FAMILY",
    "H21_HYPOTHESIS_ID",
    "H22_EXPECTED_FAMILY",
    "H22_HYPOTHESIS_ID",
    "H23_HYPOTHESIS_ID",
    "H24_HYPOTHESIS_ID",
    "H25_HYPOTHESIS_ID",
    "H26_HYPOTHESIS_ID",
    "H27_HYPOTHESIS_ID",
    "H28_HYPOTHESIS_ID",
    "H29_HYPOTHESIS_ID",
    "H2_EXPECTED_FAMILY",
    "H2_HYPOTHESIS_ID",
    "H30_HYPOTHESIS_ID",
    "H31_HYPOTHESIS_ID",
    "H32_HYPOTHESIS_ID",
    "H33_HYPOTHESIS_ID",
    "H34_HYPOTHESIS_ID",
    "H35_HYPOTHESIS_ID",
    "H36_HYPOTHESIS_ID",
    "H37_HYPOTHESIS_ID",
    "H38_HYPOTHESIS_ID",
    "H39_HYPOTHESIS_ID",
    "H3_EXPECTED_FAMILY",
    "H3_HYPOTHESIS_ID",
    "H40_HYPOTHESIS_ID",
    "H41_HYPOTHESIS_ID",
    "H42_HYPOTHESIS_ID",
    "H43_EXPECTED_FAMILY",
    "H43_HYPOTHESIS_ID",
    "H44_HYPOTHESIS_ID",
    "H45_HYPOTHESIS_ID",
    "H46_HYPOTHESIS_ID",
    "H47_HYPOTHESIS_ID",
    "H48_HYPOTHESIS_ID",
    "H49_HYPOTHESIS_ID",
    "H4B_EXPECTED_FAMILY",
    "H4B_HYPOTHESIS_ID",
    "H4_EXPECTED_FAMILY",
    "H4_HYPOTHESIS_ID",
    "H50_HYPOTHESIS_ID",
    "H51_HYPOTHESIS_ID",
    "H7_EXPECTED_FAMILY",
    "H7_HYPOTHESIS_ID",
    "H8_EXPECTED_FAMILY",
    "H8_HYPOTHESIS_ID",
    "H99_EXPECTED_FAMILY",
    "H99_HYPOTHESIS_ID",
    "H9_EXPECTED_FAMILY",
    "H9_HYPOTHESIS_ID",
    "NORTHSET_H23_H28_SPECS",
    "aci_has_finite_kupiec_p",
    "conformal_aci_blob",
    "conformal_aci_payload",
    "conformal_mondrian_aci_blob",
    "conformal_mondrian_aci_payload",
    "conformal_rank_has_finite_fdr",
    "coverage_guarantee_scope_consistency_errors",
    "coverage_guarantee_scope_is_marginal",
    "crc_has_finite_kupiec_p",
    "cv_plus_has_finite_coverage_and_floor",
    "data_snooping_has_finite_spa_p",
    "evalues_has_finite_e_sup",
    "h10_hypothesis_consistency_errors",
    "h11_hypothesis_consistency_errors",
    "h12_hypothesis_consistency_errors",
    "h15_hypothesis_consistency_errors",
    "h16_h18_panel_kupiec_consistency_errors",
    "h19_hypothesis_consistency_errors",
    "h1_hypothesis_consistency_errors",
    "h20_hypothesis_consistency_errors",
    "h21_hypothesis_consistency_errors",
    "h22_hypothesis_consistency_errors",
    "h2_hypothesis_consistency_errors",
    "h3_hypothesis_consistency_errors",
    "h43_hypothesis_consistency_errors",
    "h4_hypothesis_consistency_errors",
    "h4b_hypothesis_consistency_errors",
    "h7_hypothesis_consistency_errors",
    "h8_hypothesis_consistency_errors",
    "h99_hypothesis_consistency_errors",
    "h9_hypothesis_consistency_errors",
    "hypotheses_include_h1",
    "hypotheses_include_h10",
    "hypotheses_include_h11",
    "hypotheses_include_h12",
    "hypotheses_include_h15",
    "hypotheses_include_h16",
    "hypotheses_include_h17",
    "hypotheses_include_h18",
    "hypotheses_include_h19",
    "hypotheses_include_h2",
    "hypotheses_include_h20",
    "hypotheses_include_h21",
    "hypotheses_include_h22",
    "hypotheses_include_h3",
    "hypotheses_include_h4",
    "hypotheses_include_h43",
    "hypotheses_include_h4b",
    "hypotheses_include_h7",
    "hypotheses_include_h8",
    "hypotheses_include_h9",
    "hypotheses_include_h99",
    "hypotheses_include_id",
    "jackknife_plus_has_finite_coverage",
    "jp_cv_blob_requires_marginal_coverage_scope",
    "mondrian_aci_has_finite_high_x_kupiec_p",
    "mondrian_has_finite_high_x_kupiec_p",
    "northset_h23_h28_consistency_errors",
    "northset_has_finite_book_uncrossed_rate",
    "northset_has_finite_imbalance_p_ic",
    "northset_has_finite_ohlc_identity_rate",
    "northset_has_finite_session_book_vpin_p_ic",
    "oracle_has_finite_ls_p",
    "oracle_has_finite_p_ic",
    "panel_family_has_finite_kupiec_p",
    "rankers_oracle_raw",
    "ranking_data_snooping_blob",
    "ranking_data_snooping_honesty_errors",
    "tail_has_finite_christoffersen_cc_p",
    "tail_has_finite_kupiec_p",
    "volatility_has_finite_dm_p",
    "weighted_conformal_has_finite_kupiec_p",
]
