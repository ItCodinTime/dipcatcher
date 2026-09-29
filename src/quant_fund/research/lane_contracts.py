"""Contract checks for lane-specific receipt schemas.

``verify-receipt`` seals prove a receipt was not tampered with after writing;
these checks go further — they re-derive claims *inside* the payload so a
receipt whose numbers were fabricated before sealing still fails:

- ``capacity_overlay.v1``: ``days_to_trade == max_participation /
  participation_cap`` and ``feasible == (max_participation <= cap)`` per row;
  impact and trading days monotone in AUM within each book;
  ``n_rows == len(results)``.
- ``cross_sectional_rankic.v1``: ``n_rows == len(results)``, error rows carry
  a non-empty ``error`` while ok rows carry none, ``|mean_*| <= 1``, and —
  the statistical check — ``p_*`` re-derived from ``t_*`` as the two-sided
  Student-t tail on ``n_dates - 1`` (the writer's ``mean_tstat`` convention).
- ``fast_replay_p42_conformance``: honesty flags, verdict/evidence/gaps
  structure, ``base_commit`` 40-hex, ``code_sha256`` a path→sha256 map whose
  referenced files exist.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from scipy.stats import t as _t_dist

__all__ = ["lane_contract_errors"]

_CAPACITY_SCHEMA = "capacity_overlay.v1"
_RANKIC_SCHEMA = "cross_sectional_rankic.v1"
_P42_RECEIPT = "fast_replay_p42_conformance"

_T_REL_TOL = 1e-6


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _capacity_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return ["results_missing"]
    if payload.get("n_rows") != len(results):
        errors.append("n_rows")
    books: dict[str, list[tuple[float, float, float]]] = {}
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            errors.append(f"results[{index}]_not_object")
            continue
        book = str(row.get("book", "?"))
        cap = row.get("participation_cap")
        max_p = row.get("max_participation")
        days = row.get("days_to_trade")
        feasible = row.get("feasible")
        if not all(_finite(v) for v in (cap, max_p, days, row.get("impact_bps"), row.get("aum"))):
            errors.append(f"results[{index}]_non_finite")
            continue
        cap_f, max_f, days_f = (
            float(cast(float, cap)),
            float(cast(float, max_p)),
            float(cast(float, days)),
        )
        if cap_f <= 0:
            errors.append(f"results[{index}].participation_cap")
            continue
        # participation_cap is a fraction of ADV; trading days to deploy the
        # book is exactly the max participation divided by the daily cap.
        if not math.isclose(days_f, max_f / cap_f, rel_tol=1e-9, abs_tol=1e-12):
            errors.append(f"results[{index}].days_to_trade")
        if feasible not in (0, 1, True, False):
            errors.append(f"results[{index}].feasible_not_bool")
        elif int(bool(feasible)) != int(max_f <= cap_f + 1e-12):
            errors.append(f"results[{index}].feasible")
        books.setdefault(book, []).append((float(row["aum"]), float(row["impact_bps"]), days_f))
    for book, rows in books.items():
        rows.sort(key=lambda r: r[0])
        for earlier, later in zip(rows, rows[1:], strict=False):
            if later[1] < earlier[1] - 1e-9:
                errors.append(f"{book}:impact_not_monotone_in_aum")
            if later[2] < earlier[2] - 1e-9:
                errors.append(f"{book}:days_not_monotone_in_aum")
    return errors


def _rankic_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return ["results_missing"]
    if payload.get("n_rows") != len(results):
        errors.append("n_rows")
    n_error_rows = sum(
        1 for row in results if isinstance(row, dict) and row.get("status") == "error"
    )
    if payload.get("n_error_rows") is not None and payload["n_error_rows"] != n_error_rows:
        errors.append("n_error_rows")
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            errors.append(f"results[{index}]_not_object")
            continue
        status = row.get("status")
        if status == "error":
            if not str(row.get("error", "")).strip():
                errors.append(f"results[{index}].error_empty")
            continue
        if status != "ok":
            errors.append(f"results[{index}].status")
            continue
        if str(row.get("error", "")).strip():
            errors.append(f"results[{index}].error_set_on_ok")
        n_dates = row.get("n_dates")
        if not isinstance(n_dates, int) or n_dates < 3:
            errors.append(f"results[{index}].n_dates")
            continue
        for name in ("mean_pearson", "mean_spearman"):
            value = row.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or abs(value) > 1.0 + 1e-9
            ):
                errors.append(f"results[{index}].{name}")
        # Re-derive two-sided p from the HAC t-stat (writer: mean_tstat,
        # df = n - 1). A fabricated p fails at 1e-6 relative tolerance.
        for t_key, p_key in (("t_spearman", "p_spearman"), ("t_pearson", "p_pearson")):
            t_value, p_value = row.get(t_key), row.get(p_key)
            if t_value is None or p_value is None:
                continue
            if not (_finite(t_value) and _finite(p_value)):
                errors.append(f"results[{index}].{t_key}_non_finite")
                continue
            expected = 2.0 * float(_t_dist.sf(abs(float(t_value)), df=n_dates - 1))
            if not math.isclose(float(p_value), expected, rel_tol=_T_REL_TOL):
                errors.append(f"results[{index}].{p_key}")
    return errors


def _p42_conformance_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("research_only") is not True:
        errors.append("research_only")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim")
    if not str(payload.get("verdict", "")).strip():
        errors.append("verdict")
    if "SYNTHETIC" not in str(payload.get("disclaimer", "")):
        errors.append("disclaimer_synthetic")
    base_commit = payload.get("base_commit")
    if not (
        isinstance(base_commit, str)
        and len(base_commit) == 40
        and all(c in "0123456789abcdef" for c in base_commit)
    ):
        errors.append("base_commit")
    for name in ("evidence", "fixes", "gaps_refused"):
        value = payload.get(name)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            errors.append(name)
    code_map = payload.get("code_sha256")
    if not isinstance(code_map, dict) or not code_map:
        errors.append("code_sha256")
    else:
        for path, digest in code_map.items():
            if not (
                isinstance(digest, str)
                and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest)
            ):
                errors.append(f"code_sha256:{path}_digest")
                continue
            # A sealed digest of a path that does not exist attests nothing.
            if not (Path(__file__).resolve().parents[3] / path).is_file():
                errors.append(f"code_sha256:{path}_missing_file")
    return errors


def lane_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Deep-verify a committed lane receipt; ``[]`` when the schema is unknown."""
    schema = payload.get("schema")
    if schema == _CAPACITY_SCHEMA:
        return _capacity_contract_errors(payload)
    if schema == _RANKIC_SCHEMA:
        return _rankic_contract_errors(payload)
    if payload.get("receipt") == _P42_RECEIPT:
        return _p42_conformance_contract_errors(payload)
    return []
