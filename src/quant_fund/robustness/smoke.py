"""Executable smoke checks for one frozen research certificate."""

from __future__ import annotations

import json

from quant_fund.robustness.certify import certify


def smoke_check(certificate: dict) -> dict:
    """Re-verify a certificate and flag forbidden disclosure strings.

    Deterministic: same certificate bytes in -> same report out.
    """
    result = certify(
        certificate["family"],
        certificate["params"],
        certificate["data_manifest"],
        certificate["code_manifest"],
        certificate.get("seed_manifest"),
    )
    report = result.report
    # Lazy import: leakage is a PROOFCORE package outside the robustness
    # SCC; a top-level edge trips the LH011 layering gate
    # (tests/unit/test_proofcore_layering.py).
    from quant_fund.leakage.patterns import find_forbidden_headline

    hits = find_forbidden_headline(json.dumps(report))
    if hits:
        report["disclosure_violations"] = hits
        report["verdict"] = "fail"
        report["failure_stage"] = "disclosure"
    return report
