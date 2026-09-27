"""Blob-shape predicates for the research catalog (``*_has_*`` markers)."""

from __future__ import annotations

from typing import Any

from ._helpers import (
    _finite_scalar,
)


def tail_has_finite_christoffersen_cc_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``christoffersen_cc_p``."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("christoffersen_cc_p"))


def tail_has_finite_kupiec_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``kupiec_p``."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("kupiec_p"))


def volatility_has_finite_dm_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``dm_p``."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("dm_p"))


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


def conformal_aci_blob(conformal: object) -> dict[str, Any] | None:
    """Return ``families['conformal']['aci']`` dict, or None.

    *conformal* is ``families.get("conformal")`` (the conformal family blob),
    not the full families dict.
    """
    if not isinstance(conformal, dict):
        return None
    aci = conformal.get("aci")
    return aci if isinstance(aci, dict) else None


def aci_has_finite_kupiec_p(payload: object) -> bool:
    """True iff *payload* is a dict with finite ``kupiec_p`` (ACI row)."""
    if not isinstance(payload, dict):
        return False
    return _finite_scalar(payload.get("kupiec_p"))


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


mondrian_has_finite_high_x_kupiec_p = mondrian_aci_has_finite_high_x_kupiec_p


def crc_has_finite_kupiec_p(crc: object) -> bool:
    """True iff the CRC family exposes a finite Kupiec p-value."""
    return isinstance(crc, dict) and _finite_scalar(crc.get("kupiec_p"))


def weighted_conformal_has_finite_kupiec_p(wcqr: object) -> bool:
    """True iff the weighted_conformal family exposes a finite Kupiec p-value."""
    return isinstance(wcqr, dict) and _finite_scalar(wcqr.get("kupiec_p"))


def evalues_has_finite_e_sup(evalues: object) -> bool:
    """True iff the evalues family exposes a finite e_sup."""
    return isinstance(evalues, dict) and _finite_scalar(evalues.get("e_sup"))


def jackknife_plus_has_finite_coverage(jp: object) -> bool:
    """True iff the jackknife_plus family exposes a finite coverage."""
    return isinstance(jp, dict) and _finite_scalar(jp.get("coverage"))


def cv_plus_has_finite_coverage_and_floor(cvp: object) -> bool:
    """True iff the cv_plus family exposes finite coverage and coverage_floor."""
    return (
        isinstance(cvp, dict)
        and _finite_scalar(cvp.get("coverage"))
        and _finite_scalar(cvp.get("coverage_floor"))
    )


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


def conformal_rank_has_finite_fdr(topk: object) -> bool:
    """True iff conformal_rank family has finite fdr and is not fixture DGP."""
    if not isinstance(topk, dict):
        return False
    if str(topk.get("dgp") or "") == "fixture":
        return False
    return _finite_scalar(topk.get("fdr"))


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


def northset_has_finite_session_book_vpin_p_ic(blob: object) -> bool:
    """True iff northset exposes finite session_book_vpin_p_ic (multi-snap path)."""
    return (
        isinstance(blob, dict)
        and blob.get("session_book_hypothesis_eligible", True) is not False
        and _finite_scalar(blob.get("session_book_vpin_p_ic"))
    )
