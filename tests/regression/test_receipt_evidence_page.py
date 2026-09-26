"""The committed evidence page must match a fresh render of the receipts."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.build_evidence_report import build_report, load_receipt, receipt_paths

from quant_fund.research.catalog import FORBIDDEN_RESEARCH_METRIC_KEYS
from quant_fund.utils.hashing import hash_file

_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _ROOT / "docs" / "evidence" / "index.md"
_VERBATIM = re.compile(
    r"<!-- verbatim-receipt-text -->\n(?P<body>.*?)\n<!-- /verbatim-receipt-text -->",
    re.DOTALL,
)


def _verbatim_bodies(text: str) -> list[str]:
    bodies: list[str] = []
    for match in _VERBATIM.finditer(text):
        lines: list[str] = []
        for line in match.group("body").split("\n"):
            if line == ">":
                lines.append("")
            elif line.startswith("> "):
                lines.append(line[2:])
            else:
                raise AssertionError(f"verbatim line is not a quote: {line!r}")
        bodies.append("\n".join(lines))
    return bodies


def _forbidden_tokens(text: str) -> list[str]:
    stripped = _VERBATIM.sub("\n", text)
    hits: list[str] = []
    for lineno, line in enumerate(stripped.splitlines(), start=1):
        scrubbed = re.sub(r"live_pnl_claim", "", line, flags=re.IGNORECASE)
        tokens = {token.lower() for token in re.split(r"[^A-Za-z]+", scrubbed) if token}
        found = sorted(tokens & FORBIDDEN_RESEARCH_METRIC_KEYS)
        if found:
            hits.append(f"{lineno}: {found}: {line}")
    return hits


def _receipt_strings(root: Path) -> set[str]:
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, str):
            found.add(value)
        elif isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for path in receipt_paths(root):
        walk(load_receipt(path))
    return found


def test_forbidden_scanner_ignores_the_claim_flag_and_verbatim_quotes() -> None:
    assert _forbidden_tokens("ridge date-equal-weight MSE is higher than the zero baseline.") == []
    assert _forbidden_tokens("live_pnl_claim=false") == []
    quoted = (
        "<!-- verbatim-receipt-text -->\n"
        "> Correctness is NAV parity.\n"
        "<!-- /verbatim-receipt-text -->\n"
    )
    assert _forbidden_tokens(quoted) == []
    assert _forbidden_tokens("residual 1.2") == []
    hits = _forbidden_tokens("headline sharpe 1.2")
    assert hits
    assert "sharpe" in hits[0]


def test_evidence_page_matches_receipts_byte_for_byte(tmp_path: Path) -> None:
    text, seal_errors = build_report(_ROOT)
    assert seal_errors == []
    rendered = tmp_path / "index.md"
    rendered.write_text(text, encoding="utf-8", newline="\n")
    assert rendered.read_bytes() == _PAGE.read_bytes()
    for path in receipt_paths(_ROOT):
        assert hash_file(path) in text


def test_evidence_page_has_no_forbidden_headline_metrics() -> None:
    text = _PAGE.read_text(encoding="utf-8")
    assert _forbidden_tokens(text) == []
    quoted = _verbatim_bodies(text)
    assert quoted, "disclaimers and limitations must be copied onto the page"
    known = _receipt_strings(_ROOT)
    missing = [body for body in quoted if body not in known]
    assert missing == []
    assert "Content seals: pass" in text
    assert "Status: FAILED / BLOCKED" in text
    assert "reality_check_p" in text
    assert "spa_consistent_p" in text
    assert "stepm_adjusted_p" in text
    assert "economic_evidence_gate" in text
