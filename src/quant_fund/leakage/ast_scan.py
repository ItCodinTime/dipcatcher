"""AST linter engine (stdlib ast only — no third-party parser).

Implements the LH001..LH012 rule pack (DESIGN.md §6.1). Detection is purely
syntactic and deliberately conservative: a rule fires only on the exact
pattern its spec row describes, and per-rule path allowlists in
``leakage/rules.py`` codify the legitimate sites at HEAD.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from pathlib import Path

from quant_fund.leakage.rules import (
    LH009_EXEMPT_GLOBS,
    LH011_LAZY_WHITELIST,
    LH011_PACKAGES,
    LH011_WHITELIST,
    RULE_ALLOWLISTS,
)

_PRICE_NAME_RE = re.compile(r"close|mid|price|bars|nav", re.IGNORECASE)
_PRICE_EXACT_NAMES = frozenset({"close", "mid", "price", "bars", "px", "nav"})
_UNIVERSE_NAME_RE = re.compile(r"universe|members|tickers|tape_ids", re.IGNORECASE)
_FWD_DIFF_NAME_RE = re.compile(r"^(delta|d_|chg_)", re.IGNORECASE)
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,6}$")
_SCALER_NAME_RE = re.compile(r"scaler|standardscaler|rank_gauss", re.IGNORECASE)
_FOLD_FUNC_RE = re.compile(r"fold", re.IGNORECASE)
_PSR_CALL_NAMES = frozenset({"probabilistic_sharpe", "min_track_record_length", "deflated_sharpe"})
_SHARPE_CALL_NAMES = frozenset({"sharpe_ratio"})


def _is_allowlisted(rule_id: str, path_str: str) -> bool:
    globs = RULE_ALLOWLISTS.get(rule_id, frozenset())
    return any(fnmatch.fnmatch(path_str, g) or fnmatch.fnmatch(path_str, f"*/{g}") for g in globs)


def _exempt_by_globs(path_str: str, globs: frozenset[str]) -> bool:
    return any(fnmatch.fnmatch(path_str, g) or fnmatch.fnmatch(path_str, f"*/{g}") for g in globs)


def _neg_int_literal(node: ast.AST) -> bool:
    """True for a negative integer literal like ``-1``."""
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
        and node.operand.value > 0
    )


def _receiver_price_like(receiver: ast.AST) -> bool:
    """True if the `.shift` receiver looks price-like (close/mid/px/...)."""
    if isinstance(receiver, ast.Name):
        name = receiver.id
        return name.lower() in _PRICE_EXACT_NAMES or bool(_PRICE_NAME_RE.search(name))
    if isinstance(receiver, ast.Attribute):
        return bool(_PRICE_NAME_RE.search(receiver.attr))
    if isinstance(receiver, ast.Call):
        # pl.col("close").shift(-1) — inspect the column-name literal.
        func = receiver.func
        func_name = ""
        if isinstance(func, ast.Attribute):
            func_name = func.attr
        elif isinstance(func, ast.Name):
            func_name = func.id
        if func_name == "col" and receiver.args:
            first = receiver.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                return bool(_PRICE_NAME_RE.search(first.value))
            if isinstance(first, ast.Name):
                name = first.id
                return name.lower() in _PRICE_EXACT_NAMES or bool(_PRICE_NAME_RE.search(name))
    return False


def _contains_negative_shift(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == "shift"
        and sub.args
        and _neg_int_literal(sub.args[0])
        for sub in ast.walk(node)
    )


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _enclosing_scopes(
    node: ast.AST, parents: dict[int, ast.AST]
) -> tuple[list[ast.For], list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    fors: list[ast.For] = []
    funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.For):
            fors.append(cur)
        elif isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(cur)
        cur = parents.get(id(cur))
    return fors, funcs


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """ids of string Constants used as module/class/function docstrings."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
            isinstance(body, list) and body
        ):
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                ids.add(id(first.value))
    return ids


class _Finding:
    __slots__ = ("rule_id", "line", "col", "message")

    def __init__(self, rule_id: str, line: int, col: int, message: str) -> None:
        self.rule_id = rule_id
        self.line = line
        self.col = col
        self.message = message


def _check_lh001(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "shift"
            and node.args
            and _neg_int_literal(node.args[0])
            and _receiver_price_like(node.func.value)
        ):
            out.append(
                _Finding(
                    "LH001",
                    node.lineno,
                    node.col_offset,
                    "negative shift on a price-like series reads future bars",
                )
            )
    return out


def _check_lh002(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if not name.startswith("rolling"):
            continue
        for kw in node.keywords:
            if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                out.append(
                    _Finding(
                        "LH002",
                        node.lineno,
                        node.col_offset,
                        "centered rolling window mixes future rows into the window",
                    )
                )
    return out


def _check_lh003(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
        ):
            continue
        receiver = node.func.value
        recv_name = ""
        if isinstance(receiver, ast.Name):
            recv_name = receiver.id
        elif isinstance(receiver, ast.Attribute):
            recv_name = receiver.attr
        if not _SCALER_NAME_RE.search(recv_name):
            continue
        fors, funcs = _enclosing_scopes(node, parents)
        in_fold_loop = any(
            re.search(
                r"fold|split", ast.unparse(f.target) + " " + ast.unparse(f.iter), re.IGNORECASE
            )
            for f in fors
        )
        in_fold_func = any(_FOLD_FUNC_RE.search(f.name) for f in funcs)
        if not in_fold_loop and not in_fold_func:
            out.append(
                _Finding(
                    "LH003",
                    node.lineno,
                    node.col_offset,
                    f"scaler-like `{recv_name}.fit` is not lexically inside a fold/split scope",
                )
            )
    return out


def _check_lh004(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node) == "join_asof"):
            continue
        for kw in node.keywords:
            if (
                kw.arg in ("on", "left_on")
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "event_time"
            ):
                out.append(
                    _Finding(
                        "LH004",
                        node.lineno,
                        node.col_offset,
                        "join_asof keyed on event_time; PIT joins must key on known_at/available_time",
                    )
                )
    return out


def _check_lh005(tree: ast.AST, path_str: str) -> list[_Finding]:
    # tests/ and config files may carry fake ticker lists — EXCEPT the
    # seeded-leak fixture suite, which must stay scannable (DESIGN.md §6.4).
    if "leakage_fixtures" not in path_str and (
        "tests/" in f"/{path_str}" or "config" in Path(path_str).name.lower()
    ):
        return []
    out: list[_Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and _UNIVERSE_NAME_RE.search(target.id)):
            continue
        value = node.value
        if not (isinstance(value, ast.List) and len(value.elts) >= 5):
            continue
        if all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) and _TICKER_RE.match(e.value)
            for e in value.elts
        ):
            out.append(
                _Finding(
                    "LH005",
                    node.lineno,
                    node.col_offset,
                    f"frozen ticker list assigned to `{target.id}` (survivorship-biased universe)",
                )
            )
    return out


def _fwd_diff_value(value: ast.AST) -> bool:
    """True for ``X.shift(-k) - X`` / ``X - X.shift(-k)`` shapes."""
    return (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Sub)
        and (_contains_negative_shift(value))
    )


def _check_lh006(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    for node in ast.walk(tree):
        # (a) ``delta_mid = mid.shift(-1) - mid`` assignment form.
        if isinstance(node, ast.Assign) and _fwd_diff_value(node.value):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and _FWD_DIFF_NAME_RE.match(target.id)
                    and not re.match(r"(?i)^(fwd|lead)", target.id)
                ):
                    out.append(
                        _Finding(
                            "LH006",
                            node.lineno,
                            node.col_offset,
                            f"forward difference assigned to contemporaneous name `{target.id}`",
                        )
                    )
        # (b) polars expression form ``(...shift(-1) - ...).alias("delta_mid")``.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "alias"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            name = node.args[0].value
            if (
                _FWD_DIFF_NAME_RE.match(name)
                and not re.match(r"(?i)^(fwd|lead)", name)
                and _fwd_diff_value(node.func.value)
            ):
                out.append(
                    _Finding(
                        "LH006",
                        node.lineno,
                        node.col_offset,
                        f"forward difference aliased to contemporaneous name `{name}`",
                    )
                )
    return out


def _sharpe_call_unit_safe(call: ast.Call) -> bool | None:
    """True if a sharpe_ratio call is provably per-period; None if not a sharpe call."""
    if _call_name(call) not in _SHARPE_CALL_NAMES:
        return None
    for kw in call.keywords:
        if (
            kw.arg == "periods_per_year"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value in (1, 1.0)
        ):
            return True
        if kw.arg == "irregular" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


def _scope_assignments(
    func: ast.FunctionDef | ast.AsyncFunctionDef | ast.Module,
) -> dict[str, ast.AST]:
    assigns: dict[str, ast.AST] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assigns.setdefault(target.id, node.value)
    return assigns


def _resolve_expr(expr: ast.AST, assigns: dict[str, ast.AST], depth: int = 0) -> ast.AST:
    """Follow name -> assigned expr, unwrapping float(...) and subscripts."""
    if depth > 8:
        return expr
    if isinstance(expr, ast.Name) and expr.id in assigns:
        return _resolve_expr(assigns[expr.id], assigns, depth + 1)
    if isinstance(expr, ast.Call) and _call_name(expr) in ("float", "int") and expr.args:
        return _resolve_expr(expr.args[0], assigns, depth + 1)
    if isinstance(expr, ast.Subscript):
        return _resolve_expr(expr.value, assigns, depth + 1)
    return expr


def _check_lh007(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node) in _PSR_CALL_NAMES):
            continue
        if not node.args:
            continue
        _, funcs = _enclosing_scopes(node, parents)
        scope: ast.AST = funcs[-1] if funcs else tree
        assigns = _scope_assignments(scope)  # type: ignore[arg-type]
        resolved = _resolve_expr(node.args[0], assigns)
        if isinstance(resolved, ast.Call):
            safe = _sharpe_call_unit_safe(resolved)
            if safe is False:
                out.append(
                    _Finding(
                        "LH007",
                        node.lineno,
                        node.col_offset,
                        f"{_call_name(node)} fed a sharpe_ratio result that is not "
                        "provably per-period (needs an explicit per-period convention)",
                    )
                )
    return out


def _check_lh008(tree: ast.AST, path_str: str) -> list[_Finding]:
    from quant_fund.leakage.patterns import find_forbidden_headline

    out: list[_Finding] = []
    docstrings = _docstring_constant_ids(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and len(node.value) > 0
            # Identifier-like literals (dict keys, column names such as
            # "corr_spike_1sigma_pnl") are never prose headlines.
            and any(ch.isspace() for ch in node.value)
        ):
            tokens = find_forbidden_headline(node.value)
            if tokens:
                out.append(
                    _Finding(
                        "LH008",
                        node.lineno,
                        node.col_offset,
                        f"string literal headlines forbidden metric token(s): {', '.join(tokens)}",
                    )
                )
    return out


def _check_lh009(tree: ast.AST, path_str: str) -> list[_Finding]:
    if _exempt_by_globs(path_str, LH009_EXEMPT_GLOBS):
        return []
    out: list[_Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in ("read_parquet", "scan_parquet"):
            out.append(
                _Finding(
                    "LH009",
                    node.lineno,
                    node.col_offset,
                    "direct parquet read outside the data layer bypasses the PIT choke point",
                )
            )
    return out


def _check_lh010(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        is_bfill = name == "bfill"
        is_backward_fill = name == "fill_null" and any(
            kw.arg == "strategy"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value == "backward"
            for kw in node.keywords
        )
        if not (is_bfill or is_backward_fill):
            continue
        _, funcs = _enclosing_scopes(node, parents)
        scope: ast.AST = funcs[-1] if funcs else tree
        try:
            segment = ast.unparse(scope)
        except Exception:
            segment = ""
        if '"event_time"' in segment or "'event_time'" in segment or "event_time" in segment:
            out.append(
                _Finding(
                    "LH010",
                    node.lineno,
                    node.col_offset,
                    "backward fill on a frame with an event_time column in scope",
                )
            )
    return out


def _lh011_package(path_str: str) -> str | None:
    """PROOFCORE package name if path is inside one, else None."""
    parts = Path(path_str).parts
    for i, part in enumerate(parts):
        if part == "quant_fund" and i + 1 < len(parts) and parts[i + 1] in LH011_PACKAGES:
            return parts[i + 1]
    return None


def _check_lh011(tree: ast.AST, path_str: str) -> list[_Finding]:
    package = _lh011_package(path_str)
    if package is None:
        return []
    whitelist = LH011_WHITELIST[package]
    lazy_whitelist = LH011_LAZY_WHITELIST[package]
    parents = _build_parent_map(tree)
    out: list[_Finding] = []
    for node in ast.walk(tree):
        module: str | None = None
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("quant_fund."):
                    module = alias.name
        if module is None or not module.startswith("quant_fund."):
            continue
        sub = module.split(".")[1]
        if sub == package:
            continue  # intra-package imports are always fine
        # Function-level (lazy) imports may use the lazy whitelist.
        cur = parents.get(id(node))
        nested = False
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested = True
                break
            cur = parents.get(id(cur))
        allowed = whitelist | (lazy_whitelist if nested else frozenset())
        if sub not in allowed:
            out.append(
                _Finding(
                    "LH011",
                    node.lineno,
                    node.col_offset,
                    f"quant_fund.{package} imports non-whitelisted quant_fund.{sub} "
                    f"({'lazy ' if nested else ''}import; whitelist: {sorted(allowed) or 'none'})",
                )
            )
    return out


_CHECKERS = {
    "LH001": _check_lh001,
    "LH002": _check_lh002,
    "LH003": _check_lh003,
    "LH004": _check_lh004,
    "LH005": _check_lh005,
    "LH006": _check_lh006,
    "LH007": _check_lh007,
    "LH008": _check_lh008,
    "LH009": _check_lh009,
    "LH010": _check_lh010,
    "LH011": _check_lh011,
}


def scan_file(path: Path, *, rules: set[str] | None = None) -> list[_Finding]:
    """Scan one .py file; never raises on parse errors (LH012 instead)."""
    path_str = path.as_posix()
    enabled = rules if rules is not None else set(_CHECKERS)
    findings: list[_Finding] = []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        line = getattr(exc, "lineno", None) or 1
        return [
            _Finding("LH012", int(line), 0, f"unparseable file: {exc.__class__.__name__}: {exc}")
        ]
    for rule_id, checker in _CHECKERS.items():
        if rule_id not in enabled:
            continue
        if _is_allowlisted(rule_id, path_str):
            continue
        findings.extend(checker(tree, path_str))
    findings.sort(key=lambda f: (f.line, f.col, f.rule_id))
    return findings


def collect_py_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
    # de-duplicate, stable order
    seen: set[str] = set()
    out: list[Path] = []
    for f in files:
        key = str(f)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return sorted(out, key=lambda f: f.as_posix())
