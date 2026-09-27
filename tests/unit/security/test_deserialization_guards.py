"""AST guards: unsafe deserialization must stay out of ``src/``.

These tests scan every ``src/**.py`` file's AST — not the text — so they hold
regardless of comments, strings, or formatting:

- no ``pickle``/``marshal`` imports or ``pickle.load*`` calls (joblib is only
  reachable through the checksum/manifest-verifying loaders in
  ``quant_fund.models.base``);
- no ``yaml.load`` without an explicit safe ``Loader`` (``yaml.safe_load`` and
  ``CSafeLoader`` are fine);
- no bare ``eval``/``exec`` builtin calls (method calls such as polars
  ``list.eval`` or torch ``model.eval()`` are not builtins and are exempt);
- no ``numpy.load``/``np.load`` with ``allow_pickle=True``;
- no ``torch.load`` without ``weights_only=True``.

If a change genuinely needs one of these, the PR must justify it — weaken this
test only with an owner sign-off recorded in docs/SECURITY_REVIEW.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"

# Files allowed to touch joblib.load directly: the verified loaders themselves.
_JOBLIB_BOUNDARY = {Path("quant_fund/models/base.py")}


def _python_files() -> list[Path]:
    assert SRC_ROOT.is_dir(), f"src root missing: {SRC_ROOT}"
    return sorted(SRC_ROOT.rglob("*.py"))


def _calls(tree: ast.AST) -> list[tuple[ast.Call, str]]:
    """(call node, dotted callee name) for every call in the module."""
    out: list[tuple[ast.Call, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                continue
            out.append((node, name))
    return out


def _imported_top_level(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(SRC_ROOT)))
def test_no_unsafe_deserialization(path: Path) -> None:
    rel = path.relative_to(SRC_ROOT)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = _imported_top_level(tree)
    assert "pickle" not in imported, f"{rel}: direct pickle import"
    assert "marshal" not in imported, f"{rel}: marshal import"
    assert "dill" not in imported, f"{rel}: dill import"
    for node, name in _calls(tree):
        if name in {"eval", "exec"} and isinstance(node.func, ast.Name):
            pytest.fail(f"{rel}:{node.lineno}: bare {name}() builtin call")
        if name == "load" and isinstance(node.func, ast.Attribute):
            base = node.func.value
            base_id = base.id if isinstance(base, ast.Name) else ""
            if base_id == "pickle":
                pytest.fail(f"{rel}:{node.lineno}: pickle.load() call")
            if base_id == "marshal":
                pytest.fail(f"{rel}:{node.lineno}: marshal.load() call")
            if base_id == "yaml":
                # yaml.load requires an explicit Loader; yaml.full_load and
                # yaml.unsafe_load are flagged separately below.
                loader_kw = any(kw.arg == "Loader" for kw in node.keywords)
                if not loader_kw:
                    pytest.fail(f"{rel}:{node.lineno}: yaml.load() without Loader=")
            if base_id == "joblib" and rel not in _JOBLIB_BOUNDARY:
                pytest.fail(
                    f"{rel}:{node.lineno}: raw joblib.load() bypasses the "
                    "checksum/manifest boundary — use load_joblib_artifact or "
                    "JoblibMixin.load (quant_fund.models.base)"
                )
            if base_id in {"np", "numpy"}:
                for kw in node.keywords:
                    if (
                        kw.arg == "allow_pickle"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        pytest.fail(f"{rel}:{node.lineno}: numpy load with allow_pickle=True")
        if name in {"unsafe_load", "full_load"} and isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name) and base.id == "yaml":
                pytest.fail(f"{rel}:{node.lineno}: yaml.{name}() call")
        if name == "load" and isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name) and base.id == "torch":
                weights_only = any(
                    kw.arg == "weights_only"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
                if not weights_only:
                    pytest.fail(f"{rel}:{node.lineno}: torch.load() without weights_only=True")
        if name == "loads" and isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name) and base.id in {"pickle", "marshal", "_pickle"}:
                pytest.fail(f"{rel}:{node.lineno}: {base.id}.loads() call")


def test_src_tree_scanned() -> None:
    """Guard against an empty scan silently passing."""
    files = _python_files()
    assert len(files) > 100, f"only {len(files)} src files scanned — tree moved?"
