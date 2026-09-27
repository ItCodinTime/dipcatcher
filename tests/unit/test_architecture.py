"""Layering and module-size guards for the quant_fund and fx1 libraries."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path("src")
MAX_MODULE_LINES = 2000
LIBRARY_ROOTS = (SRC / "quant_fund", SRC / "fx1")

# CLI may depend on the library. The library must not depend on the CLI.
CLI_PREFIX = "quant_fund.cli"
# Research and the rest of the core may use execution impact math.
# They must not import the simulated broker or the paper/live loop.
# simtest, pretrade, formal, and parity are the simulation boundary that
# drives the broker; they are not on the research/pipeline/models import path.
# parity replays a session through SimulatedBroker and compares it to the
# backtest. Nothing under research, pipeline, or models imports it.
BROKER_MODULES = ("quant_fund.execution.simulated_broker",)
PAPER_PREFIX = "quant_fund.paper"
BROKER_ALLOWED_PREFIXES = (
    "quant_fund.cli",
    "quant_fund.paper",
    "quant_fund.execution",
    "quant_fund.simtest",
    "quant_fund.pretrade",
    "quant_fund.formal",
    "quant_fund.parity",
)


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    current = _module_name(path)
    parts = current.split(".")
    package = parts if path.name == "__init__.py" else parts[:-1]
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    found.append(node.module)
                continue
            base = package[: len(package) - (node.level - 1)]
            if node.module:
                base = [*base, *node.module.split(".")]
            found.append(".".join(base))
    return found


def _under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def test_modules_stay_under_max_lines() -> None:
    offenders: list[str] = []
    for root in LIBRARY_ROOTS:
        for path in root.rglob("*.py"):
            lines = len(path.read_text(encoding="utf-8").splitlines())
            if lines > MAX_MODULE_LINES:
                offenders.append(f"{path.relative_to(SRC)}:{lines}")
    assert offenders == []


def test_library_does_not_import_cli() -> None:
    offenders: list[str] = []
    for root in LIBRARY_ROOTS:
        for path in root.rglob("*.py"):
            module = _module_name(path)
            if _under(module, CLI_PREFIX):
                continue
            for imported in _imported_modules(path):
                if _under(imported, CLI_PREFIX):
                    offenders.append(f"{module} imports {imported}")
    assert offenders == []


def test_core_does_not_import_broker_or_paper() -> None:
    offenders: list[str] = []
    for root in LIBRARY_ROOTS:
        for path in root.rglob("*.py"):
            module = _module_name(path)
            if any(_under(module, prefix) for prefix in BROKER_ALLOWED_PREFIXES):
                continue
            for imported in _imported_modules(path):
                if _under(imported, PAPER_PREFIX) or any(
                    _under(imported, broker) for broker in BROKER_MODULES
                ):
                    offenders.append(f"{module} imports {imported}")
    assert offenders == []
