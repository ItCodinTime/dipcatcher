"""LH013/LH014 unit tests (ADVERSARIAL §1a warning-severity channels)."""

from __future__ import annotations

from pathlib import Path

from quant_fund.leakage import scan_paths


def _scan(tmp_path: Path, source: str):
    target = tmp_path / "snippet.py"
    target.write_text(source)
    return scan_paths([target])


def test_lh013_comment_headline(tmp_path) -> None:
    report = _scan(tmp_path, "# backtest Sharpe of 2.1\nX = 1\n")
    hits = [f for f in report.findings if f.rule_id == "LH013"]
    assert hits and hits[0].severity == "warning"
    assert "comment" in hits[0].message


def test_lh013_docstring_headline(tmp_path) -> None:
    report = _scan(tmp_path, 'def f():\n    """Sharpe of 2.1."""\n    return 1\n')
    hits = [f for f in report.findings if f.rule_id == "LH013"]
    assert hits and "docstring" in hits[0].message


def test_lh013_fstring_numeric_spec(tmp_path) -> None:
    report = _scan(tmp_path, 'def f(sr):\n    return f"Sharpe was {sr:.2f}"\n')
    hits = [f for f in report.findings if f.rule_id == "LH013"]
    assert hits and "f-string" in hits[0].message


def test_lh013_spelled_out_string(tmp_path) -> None:
    report = _scan(tmp_path, 'NOTE = "The Sharpe, which exceeded two, held"\n')
    hits = [f for f in report.findings if f.rule_id == "LH013"]
    assert hits and "spelled-out" in hits[0].message


def test_lh013_clean_prose_no_finding(tmp_path) -> None:
    report = _scan(tmp_path, 'def f():\n    """Compute the Sharpe ratio."""\n    return 1\n')
    assert [f for f in report.findings if f.rule_id == "LH013"] == []


def test_lh014_helper_call_flagged(tmp_path) -> None:
    source = (
        "def _align(close):\n"
        "    return close.shift(-1)\n"
        "\n"
        "def strategy(close):\n"
        "    return _align(close)\n"
    )
    report = _scan(tmp_path, source)
    lh014 = [f for f in report.findings if f.rule_id == "LH014"]
    assert lh014 and lh014[0].severity == "warning"
    assert "_align" in lh014[0].message and "LH001" in lh014[0].message


def test_lh014_clean_helper_not_flagged(tmp_path) -> None:
    source = (
        "def _lag(close):\n"
        "    return close.shift(1)\n"
        "\n"
        "def strategy(close):\n"
        "    return _lag(close)\n"
    )
    report = _scan(tmp_path, source)
    assert [f for f in report.findings if f.rule_id == "LH014"] == []
