#!/usr/bin/env python3
"""Architecture boundary guard for ``src/quant_fund`` and ``src/fx1``.

Stdlib-only import-boundary checker. It parses every ``.py`` file under the
source root with :mod:`ast`, resolves absolute and relative imports, and
enforces the declarative rules in ``configs/arch_boundaries.toml``:

* ``[[layers]]``   — an ordered stack of layers. A module in a lower layer may
  not import a module from a higher layer. Layer rules apply to module-scope
  imports only: imports inside function bodies are this codebase's sanctioned
  mechanism for deferring optional/cyclic dependencies, so they are exempt
  from the order check (they are still checked by ``[[deny]]`` rules).
* ``[[deny]]``     — hard "never import" rules evaluated on *every* import
  site, including function-level and ``TYPE_CHECKING`` imports.
* ``[[allow_only]]`` — a whitelist: matching importers may only touch the
  listed module prefixes inside ``target_root``.
* ``[[baseline]]`` — itemized pre-existing violations. Each entry suppresses
  exactly the violations it matches; entries that stop matching become
  ``stale-baseline`` errors so the list cannot silently rot.

Exit codes: 0 clean, 1 violations found, 2 usage/config error.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import os
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path("configs/arch_boundaries.toml")
DEFAULT_SRC = Path("src")
SCANNED_ROOTS = ("quant_fund", "fx1")
# The package-root facade: ``quant_fund/__init__.py`` + ``quant_fund/public.py``
# are the stable public surface. They deliberately wire across every layer and
# are importable by anything, so the *order* check skips them on both sides.
# [[deny]] rules still apply to them.
FACADE_MODULES = frozenset({"quant_fund", "quant_fund.public"})

LAYER_RULE = "layer-order"
UNCLASSIFIED_RULE = "unclassified-package"


# --------------------------------------------------------------------------- #
# Config model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Layer:
    name: str
    packages: frozenset[str]


@dataclass(frozen=True)
class DenyRule:
    name: str
    importer: str
    forbidden: tuple[str, ...]
    allowed: tuple[str, ...] = ()
    except_importers: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class AllowOnlyRule:
    name: str
    importer: str
    target_root: str
    allowed: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class BaselineEntry:
    file: str
    module: str
    rule: str
    note: str = ""


@dataclass(frozen=True)
class Config:
    layers: tuple[Layer, ...]
    denies: tuple[DenyRule, ...]
    allow_onlys: tuple[AllowOnlyRule, ...]
    baseline: tuple[BaselineEntry, ...]
    facade_roots: frozenset[str] = frozenset({"quant_fund"})


# --------------------------------------------------------------------------- #
# Scan results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ImportSite:
    file: str  # posix path relative to the repo root
    line: int
    importer: str  # dotted module name of the importing file
    imported: str  # dotted name of the imported module
    lazy: bool  # inside a function body
    type_checking: bool  # inside ``if TYPE_CHECKING:``


@dataclass(frozen=True)
class Violation:
    site: ImportSite
    rule: str
    detail: str

    def render(self) -> str:
        s = self.site
        lazy = " [lazy import]" if s.lazy else ""
        tc = " [TYPE_CHECKING]" if s.type_checking else ""
        return f"{s.file}:{s.line}: error[{self.rule}]: {self.detail}{lazy}{tc}"


# --------------------------------------------------------------------------- #
# Pattern matching
# --------------------------------------------------------------------------- #
def match(pattern: str, module: str) -> bool:
    """Match a dotted module name against a rule pattern.

    * ``"**"`` matches everything.
    * a trailing ``".**"`` means "this package and everything below it", e.g.
      ``quant_fund.paper.**`` matches ``quant_fund.paper`` and
      ``quant_fund.paper.loop``.
    * ``*`` inside a pattern is :mod:`fnmatch` semantics (it matches across
      dots, so ``quant_fund.*.x`` also matches ``quant_fund.a.b.x``).
    * anything else is an exact match.
    """
    if pattern == "**":
        return True
    if pattern.endswith(".**"):
        base = pattern[: -len(".**")]
        return module == base or module.startswith(base + ".")
    return fnmatch.fnmatchcase(module, pattern)


def _matches_any(patterns: Iterable[str], module: str) -> bool:
    return any(match(p, module) for p in patterns)


# --------------------------------------------------------------------------- #
# Module naming + import extraction
# --------------------------------------------------------------------------- #
def module_name_for(path: Path, src_root: Path) -> str | None:
    """Dotted module name for a file under ``src_root``; None if outside."""
    try:
        rel = path.relative_to(src_root).with_suffix("")
    except ValueError:
        return None
    parts = list(rel.parts)
    if not parts or parts[0] not in SCANNED_ROOTS:
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_parts(path: Path, module: str) -> list[str]:
    """Parts of the *package* a module lives in (for relative imports)."""
    parts = module.split(".") if module else []
    return parts if path.name == "__init__.py" else parts[:-1]


def _in_type_checking(ancestors: list[ast.AST]) -> bool:
    for node in ancestors:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
            return True
        if (
            isinstance(test, ast.Attribute)
            and test.attr == "TYPE_CHECKING"
            and isinstance(test.value, ast.Name)
            and test.value.id == "typing"
        ):
            return True
    return False


def iter_import_sites(
    path: Path,
    file_rel: str,
    importer: str,
    package_parts: list[str],
    known_modules: frozenset[str],
) -> list[ImportSite]:
    """Extract every import of a scanned package from one file.

    ``from X import a`` yields ``X`` and also ``X.a`` when ``X.a`` is a real
    module in the tree (so ``from quant_fund.execution import
    simulated_broker`` is caught by a rule naming the submodule). Relative
    imports are resolved against ``package_parts``.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ValueError(f"{file_rel}: cannot parse ({exc})") from exc

    sites: list[ImportSite] = []

    def emit(imported: str, node: ast.Import | ast.ImportFrom, lazy: bool, tc: bool) -> None:
        if imported.startswith(SCANNED_ROOTS):
            sites.append(ImportSite(file_rel, node.lineno, importer, imported, lazy, tc))

    def visit(node: ast.AST, ancestors: list[ast.AST], in_func: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            in_func = True
        if isinstance(node, ast.Import):
            lazy, tc = in_func, _in_type_checking(ancestors)
            for alias in node.names:
                emit(alias.name, node, lazy, tc)
        elif isinstance(node, ast.ImportFrom):
            lazy, tc = in_func, _in_type_checking(ancestors)
            if node.level:
                base_parts = package_parts[: len(package_parts) - (node.level - 1)]
                if node.module:
                    base_parts += node.module.split(".")
            else:
                base_parts = node.module.split(".") if node.module else []
            if not base_parts:
                base = ""
            else:
                base = ".".join(base_parts)
            if base:
                emit(base, node, lazy, tc)
            # ``from base import a`` may import the submodule ``base.a``.
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidate = f"{base}.{alias.name}" if base else alias.name
                if candidate in known_modules:
                    emit(candidate, node, lazy, tc)
        for child in ast.iter_child_nodes(node):
            visit(child, [*ancestors, node], in_func)

    visit(tree, [], False)
    return sites


def collect_modules(src_root: Path) -> dict[str, Path]:
    """Map every scanned module name to its file."""
    modules: dict[str, Path] = {}
    for path in sorted(src_root.rglob("*.py")):
        name = module_name_for(path, src_root)
        if name:
            modules[name] = path
    return modules


# --------------------------------------------------------------------------- #
# Rule evaluation
# --------------------------------------------------------------------------- #
def _subpackage(module: str) -> str | None:
    """Second-level package, e.g. ``research`` for ``quant_fund.research.x``."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else None


def _layer_of(module: str, layers: tuple[Layer, ...]) -> Layer | None:
    sub = _subpackage(module)
    if sub is None:
        return None
    for layer in layers:
        if sub in layer.packages:
            return layer
    return None


def evaluate(
    sites: list[ImportSite],
    config: Config,
) -> list[Violation]:
    """Apply every rule to every import site; return unsuppressed violations."""
    rank = {layer.name: i for i, layer in enumerate(config.layers)}
    violations: dict[tuple[str, int, str], Violation] = {}

    def add(v: Violation) -> None:
        key = (v.site.file, v.site.line, v.rule)
        prev = violations.get(key)
        if prev is None or len(v.site.imported) > len(prev.site.imported):
            violations[key] = v

    for site in sites:
        importer_root = site.importer.split(".")[0]
        imported_root = site.imported.split(".")[0]

        for deny in config.denies:
            if not match(deny.importer, site.importer):
                continue
            if _matches_any(deny.except_importers, site.importer):
                continue
            if _matches_any(deny.allowed, site.imported):
                continue
            if _matches_any(deny.forbidden, site.imported):
                note = f" — {deny.note}" if deny.note else ""
                add(
                    Violation(
                        site,
                        deny.name,
                        f"{site.importer} imports forbidden module {site.imported}{note}",
                    )
                )

        for rule in config.allow_onlys:
            if not match(rule.importer, site.importer):
                continue
            if imported_root != rule.target_root:
                continue
            if not _matches_any(rule.allowed, site.imported):
                note = f" — {rule.note}" if rule.note else ""
                add(
                    Violation(
                        site,
                        rule.name,
                        f"{site.importer} imports {site.imported}, outside the "
                        f"allowed {rule.target_root} surface "
                        f"({', '.join(rule.allowed)}){note}",
                    )
                )

        # Layer order: module-scope imports only, both ends inside the
        # layered tree, and never the package-root facade.
        if site.lazy:
            continue
        if importer_root != "quant_fund" or imported_root != "quant_fund":
            continue
        if site.importer in FACADE_MODULES or site.imported in FACADE_MODULES:
            continue  # the public facade wires across layers by design
        src_layer = _layer_of(site.importer, config.layers)
        dst_layer = _layer_of(site.imported, config.layers)
        if src_layer is None:
            add(
                Violation(
                    site,
                    UNCLASSIFIED_RULE,
                    f"{site.importer} lives in package "
                    f"'{_subpackage(site.importer)}', which no [[layers]] entry "
                    "claims — add it to a layer in the boundaries config",
                )
            )
            continue
        if dst_layer is None:
            add(
                Violation(
                    site,
                    UNCLASSIFIED_RULE,
                    f"{site.importer} imports {site.imported} from unclassified "
                    f"package '{_subpackage(site.imported)}' — add it to a layer",
                )
            )
            continue
        if rank[dst_layer.name] > rank[src_layer.name]:
            add(
                Violation(
                    site,
                    LAYER_RULE,
                    f"{site.importer} (layer '{src_layer.name}') imports "
                    f"{site.imported} (layer '{dst_layer.name}') — lower "
                    "layers must not depend on higher layers",
                )
            )

    return sorted(violations.values(), key=lambda v: (v.site.file, v.site.line, v.rule))


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def apply_baseline(
    violations: list[Violation],
    entries: tuple[BaselineEntry, ...],
) -> tuple[list[Violation], list[Violation], list[BaselineEntry]]:
    """Split violations into (active, baselined, stale-entries)."""
    used: set[int] = set()
    active: list[Violation] = []
    baselined: list[Violation] = []
    for v in violations:
        hit = None
        for i, entry in enumerate(entries):
            if entry.file != v.site.file:
                continue
            if entry.rule != "*" and entry.rule != v.rule:
                continue
            if not match(entry.module, v.site.imported):
                continue
            hit = i
            break
        if hit is None:
            active.append(v)
        else:
            used.add(hit)
            baselined.append(v)
    stale = [e for i, e in enumerate(entries) if i not in used]
    return active, baselined, stale


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def _str_list(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{where}: expected a list of strings")
    return tuple(value)


def load_config(path: Path) -> Config:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"config not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: invalid TOML ({exc})") from exc

    layers: list[Layer] = []
    seen_packages: set[str] = set()
    for i, item in enumerate(raw.get("layers", [])):
        name = item.get("name")
        packages = _str_list(item.get("packages", []), f"layers[{i}].packages")
        if not isinstance(name, str):
            raise ValueError(f"layers[{i}]: missing string 'name'")
        dup = seen_packages & set(packages)
        if dup:
            raise ValueError(f"layers[{i}]: packages claimed twice: {sorted(dup)}")
        seen_packages |= set(packages)
        layers.append(Layer(name, frozenset(packages)))

    denies = tuple(
        DenyRule(
            name=d.get("name", f"deny[{i}]"),
            importer=d.get("importer", "**"),
            forbidden=_str_list(d.get("forbidden", []), f"deny[{i}].forbidden"),
            allowed=_str_list(d.get("allowed", []), f"deny[{i}].allowed"),
            except_importers=_str_list(
                d.get("except_importers", []), f"deny[{i}].except_importers"
            ),
            note=d.get("note", ""),
        )
        for i, d in enumerate(raw.get("deny", []))
    )

    allow_onlys: list[AllowOnlyRule] = []
    for i, d in enumerate(raw.get("allow_only", [])):
        target_root = d.get("target_root")
        if not isinstance(target_root, str):
            raise ValueError(f"allow_only[{i}]: missing string 'target_root'")
        allow_onlys.append(
            AllowOnlyRule(
                name=d.get("name", f"allow_only[{i}]"),
                importer=d.get("importer", "**"),
                target_root=target_root,
                allowed=_str_list(d.get("allowed", []), f"allow_only[{i}].allowed"),
                note=d.get("note", ""),
            )
        )

    baseline = tuple(
        BaselineEntry(
            file=b.get("file", ""),
            module=b.get("module", "**"),
            rule=b.get("rule", "*"),
            note=b.get("note", ""),
        )
        for i, b in enumerate(raw.get("baseline", []))
        if _require_file(b, i)
    )
    return Config(tuple(layers), denies, tuple(allow_onlys), baseline)


def _require_file(entry: dict, i: int) -> bool:
    if not entry.get("file"):
        raise ValueError(f"baseline[{i}]: missing 'file'")
    return True


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(
    src_root: Path,
    config: Config,
    show_baselined: bool,
    config_label: str = "config",
) -> int:
    if not src_root.is_dir():
        print(f"error: source root {src_root} does not exist", file=sys.stderr)
        return 2

    modules = collect_modules(src_root)
    known = frozenset(modules)
    sites: list[ImportSite] = []
    try:
        for module, path in modules.items():
            rel = Path(os.path.relpath(path)).as_posix()
            sites.extend(iter_import_sites(path, rel, module, _package_parts(path, module), known))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    violations = evaluate(sites, config)
    active, baselined, stale = apply_baseline(violations, config.baseline)

    for v in active:
        print(v.render())
    if show_baselined:
        for v in baselined:
            print(f"{v.site.file}:{v.site.line}: note[{v.rule}/baseline]: {v.detail}")
    for entry in stale:
        print(
            f"{config_label}: error[stale-baseline]: entry "
            f"file={entry.file} module={entry.module} rule={entry.rule} "
            "matches no violation — the debt was fixed or moved; remove the entry"
        )

    n_lazy = sum(1 for s in sites if s.lazy)
    print(
        f"arch-guards: scanned {len(modules)} modules, {len(sites)} imports "
        f"({n_lazy} function-level); "
        f"{len(active)} violation(s), {len(baselined)} baselined, "
        f"{len(stale)} stale baseline entrie(s)",
        file=sys.stderr if active or stale else sys.stdout,
    )
    return 1 if active or stale else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enforce declared import boundaries between quant_fund layers."
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="source root (default: src)")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="boundaries config (default: configs/arch_boundaries.toml)",
    )
    parser.add_argument(
        "--show-baselined",
        action="store_true",
        help="also print violations suppressed by the baseline",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return run(args.src, config, args.show_baselined, args.config.as_posix())


if __name__ == "__main__":
    raise SystemExit(main())
