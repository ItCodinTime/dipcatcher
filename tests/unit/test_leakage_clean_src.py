"""Clean-src gate (DESIGN.md §6.4): zero error findings at HEAD.

The per-rule allowlists in leakage/rules.py codify the existing legitimate
sites; this test is what forces the allowlist to stay explicit. Warning
findings (LH009 inventory, LH010) are reported but do not fail the gate this
wave (adjudicated: LH009 flips to error only after the call-site migration).
"""

from __future__ import annotations

from pathlib import Path

from quant_fund.leakage import scan_paths

SRC = Path(__file__).resolve().parents[2] / "src" / "quant_fund"


def test_clean_src_scan_zero_errors() -> None:
    report = scan_paths([SRC])
    assert report.scanned_files > 100
    errors = [f for f in report.findings if f.severity == "error"]
    assert errors == [], "error-severity leakage findings at HEAD:\n" + "\n".join(
        f"  {f.rule_id} {f.path}:{f.line} {f.message}" for f in errors
    )


def test_new_packages_pass_layering_rule() -> None:
    """pit/proof/leakage/reality/proofcore must obey LH011 at HEAD."""
    report = scan_paths([SRC], rules={"LH011"})
    assert [f for f in report.findings if f.severity == "error"] == []
