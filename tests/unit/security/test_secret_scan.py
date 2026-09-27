"""Unit tests for scripts/secret_scan.py (staged-diff + tracked-tree modes).

Credential test vectors are built programmatically: this file is itself
tracked, so the literal source lines must never match the scanner's own
patterns (the tracked-tree gate scans tests too).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "secret_scan", Path(__file__).resolve().parents[3] / "scripts" / "secret_scan.py"
)
assert _SPEC is not None and _SPEC.loader is not None
secret_scan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(secret_scan)


@pytest.mark.parametrize(
    "line",
    [
        'api_key = "AKIA' + "Z" * 16 + '"',
        'token = "ghp_' + "a" * 36 + '"',
        'secret = "sk-' + "b" * 24 + '"',
        "-----BEGIN " + "RSA " + "PRIVATE KEY" + "-----",
        'password = "x9Vf2mQp' + "L8sT1rW4yU7" + '"',
        'bearer = "whsec_' + "c" * 24 + '"',
    ],
)
def test_scan_line_flags_credential_shapes(line: str) -> None:
    findings: list[str] = []
    secret_scan._scan_line("f.py", 1, line, findings)
    assert findings, line


@pytest.mark.parametrize(
    "line",
    [
        'api_key = "your-api-key-here"',
        'MOONSHOT_API_KEY = os.environ["MOONSHOT_API_KEY"]',
        'sha256 = "' + "a" * 64 + '"',  # bare digests are committed legitimately
        'config_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"',
        "PLACEHOLDER test key example",
    ],
)
def test_scan_line_ignores_placeholders_and_digests(line: str) -> None:
    findings: list[str] = []
    secret_scan._scan_line("f.py", 1, line, findings)
    assert findings == [], line


def test_tracked_tree_is_clean() -> None:
    """The whole tracked tree must stay credential-free (same gate as CI)."""
    findings = secret_scan.scan(tracked=True)
    assert findings == [], f"credential-shaped strings in tracked files: {findings}"


def test_main_all_flag() -> None:
    assert secret_scan.main(["--all"]) == 0
