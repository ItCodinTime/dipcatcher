"""Docs-vs-CLI drift guard: fx1 documentation may only reference commands,
targets, and import paths that actually exist. An auditor reading the docs
must be able to run what the docs say."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest
from typer.main import get_command

from fx1.cli import app

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOC_PATHS = sorted(
    [_REPO_ROOT / "README.md", *_REPO_ROOT.glob("docs/FX1*.md")],
)

# `fx1 ...` / `make ...` inside inline code or fenced blocks.
_INLINE = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_FX1_CMD = re.compile(r"(?<![/\w])fx1 ([a-z][a-z0-9-]*)(?: ([a-z][a-z0-9-]*))?\b")
_MAKE_TARGET = re.compile(r"\bmake ([a-z0-9][a-z0-9_-]*)")
_IMPORT_FROM = re.compile(r"from (fx1(?:\.[a-z_]+)+) import ([a-z_][a-zA-Z0-9_]*)")
_MODULE_REF = re.compile(r"`(fx1(?:\.[a-z_]+)+)`")


def _code_spans(text: str) -> list[str]:
    spans = _INLINE.findall(text)
    for block in _FENCE.findall(text):
        spans.extend(block.splitlines())
    return spans


def _cli_tree() -> dict[str, object]:
    root = get_command(app)
    tree: dict[str, object] = {}
    for name, cmd in root.commands.items():
        tree[name] = {sub for sub in cmd.commands} if hasattr(cmd, "commands") else set()
    return tree


def _makefile_targets() -> set[str]:
    text = (_REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    return set(re.findall(r"^([a-z0-9][a-z0-9_-]*):(?:\s|##)", text, re.MULTILINE))


@pytest.mark.parametrize("doc", _DOC_PATHS, ids=lambda p: p.name)
def test_documented_fx1_commands_exist(doc: Path):
    tree = _cli_tree()
    text = doc.read_text(encoding="utf-8")
    unknown = []
    for span in _code_spans(text):
        for match in _FX1_CMD.finditer(span):
            first, second = match.group(1), match.group(2)
            if first not in tree:
                unknown.append(f"fx1 {first}")
            elif second and tree[first] and second not in tree[first]:
                unknown.append(f"fx1 {first} {second}")
    assert not unknown, f"{doc.name}: unknown fx1 commands {unknown}"


@pytest.mark.parametrize("doc", _DOC_PATHS, ids=lambda p: p.name)
def test_documented_make_targets_exist(doc: Path):
    targets = _makefile_targets()
    text = doc.read_text(encoding="utf-8")
    unknown = []
    for span in _code_spans(text):
        for match in _MAKE_TARGET.finditer(span):
            if match.group(1) not in targets:
                unknown.append(f"make {match.group(1)}")
    assert not unknown, f"{doc.name}: unknown make targets {unknown}"


@pytest.mark.parametrize("doc", _DOC_PATHS, ids=lambda p: p.name)
def test_documented_import_paths_resolve(doc: Path):
    text = doc.read_text(encoding="utf-8")
    problems = []
    for module_name, symbol in _IMPORT_FROM.findall(text):
        module = importlib.import_module(module_name)
        if not hasattr(module, symbol):
            problems.append(f"{module_name}.{symbol}")
    for module_name in _MODULE_REF.findall(text):
        try:
            importlib.import_module(module_name)
        except ImportError:
            # Not a module — maybe a symbol on its parent (fx1.mrm.compile_dossier).
            parent, _, symbol = module_name.rpartition(".")
            try:
                module = importlib.import_module(parent)
                if not symbol or not hasattr(module, symbol):
                    problems.append(module_name)
            except ImportError:
                problems.append(module_name)
    assert not problems, f"{doc.name}: unresolvable imports {problems}"
