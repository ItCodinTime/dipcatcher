"""quant_fund -> fx1 dependency gate.

The harness gates the model, never the other way: ``quant_fund`` may depend
on ``fx1`` in exactly one place — ``src/quant_fund/__init__.py`` re-exports
``fx1.__version__`` because the canonical package version is the model's
semver (docs/FX1_API_STABILITY.md, pyproject dynamic version). Any other
``import fx1`` inside ``src/quant_fund`` couples the verification harness to
model internals (training, corpus, honesty reward code) and is rejected here.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "quant_fund"


def _fx1_imports(path: Path) -> list[str]:
    """Descriptions of every fx1 import in a file ('module via names @line')."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "fx1":
                    out.append(f"{node.lineno}: import {alias.name}")
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module.split(".")[0] == "fx1"
        ):
            names = ", ".join(a.name for a in node.names)
            out.append(f"{node.lineno}: from {node.module} import {names}")
    return out


def _sanctioned(desc: str) -> bool:
    """True iff the import is `from fx1 import __version__` (names only)."""
    if not desc.split(":", 1)[1].strip().startswith("from fx1 import "):
        return False
    imported = desc.rsplit("import", 1)[1].strip().split(", ")
    return imported == ["__version__"]


def test_quant_fund_imports_fx1_only_for_version() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        for desc in _fx1_imports(path):
            if rel == Path("__init__.py") and _sanctioned(desc):
                continue
            offenders.append(f"{rel}:{desc}")
    assert offenders == [], (
        "quant_fund -> fx1 edge outside the __version__ re-export:\n" + "\n".join(offenders)
    )


def test_fx1_may_import_quant_fund_but_never_its_research_internals() -> None:
    """fx1 may consume quant_fund's data/schemas/utils — not its sealed
    evidence or scoring lanes (research/, proofcore/, pit/, proof/, reality/),
    which would let the model's code reach its own verification machinery."""
    fx1 = SRC.parent / "fx1"
    forbidden = {"research", "proofcore", "pit", "proof", "leakage", "reality"}
    offenders: list[str] = []
    for path in sorted(fx1.rglob("*.py")):
        rel = path.relative_to(fx1.parent)
        for parts in _quant_fund_import_paths(path):
            if len(parts) > 1 and parts[1] in forbidden:
                offenders.append(f"{rel}: {'.'.join(parts)}")
    assert offenders == [], (
        "fx1 -> quant_fund edge reaching sealed/verification internals:\n" + "\n".join(offenders)
    )


def _quant_fund_import_paths(path: Path) -> list[list[str]]:
    """Each quant_fund import as a dotted path ('from quant_fund import x'
    yields ['quant_fund','x'] so submodule imports can't hide)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[list[str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "quant_fund":
                    out.append(parts)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and node.module.split(".")[0] == "quant_fund"
        ):
            base = node.module.split(".")
            for alias in node.names:
                out.append([*base, alias.name])
    return out
