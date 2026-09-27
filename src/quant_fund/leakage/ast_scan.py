"""AST linter engine (stdlib ast only — no third-party parser).

Implements the LH001..LH014 rule pack (DESIGN.md §6.1 + ADVERSARIAL §1a
hardening). Detection is purely syntactic and deliberately conservative: a
rule fires only on the exact pattern its spec row describes, and per-rule
path allowlists in ``leakage/rules.py`` codify the legitimate sites at HEAD.

Residual ceiling (documented, ADVERSARIAL §1a): numpy/pandas index arithmetic
(`px[1:] - px[:-1]`), dict lookups at `dates[i+1]`, single numbers far outside
the proximity window, manually computed Sharpe fed to PSR without a
``sharpe_ratio`` call, cross-file helper indirection, and runtime-only
values remain out of reach of a syntactic tripwire. The runtime watchdog
(leakage/watchdog.py) and the proof layer are the barrier; this linter is
the tripwire.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import io
import re
import tokenize
from pathlib import Path

from quant_fund.leakage.rules import (
    LH008_LITERAL_ALLOWLIST,
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
# ADVERSARIAL §1a-E16: lowercase ticker tuples evade the uppercase-only
# ticker regex; match case-insensitively.
_TICKER_CI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.]{0,6}$")
# ADVERSARIAL §1a-E6: `normalizer`/`preprocessor` attribute names evaded the
# scaler-name regex.
_SCALER_NAME_RE = re.compile(r"scal|rank_gauss|norm|preproc", re.IGNORECASE)
_FOLD_FUNC_RE = re.compile(r"fold", re.IGNORECASE)
_PSR_CALL_NAMES = frozenset({"probabilistic_sharpe", "min_track_record_length", "deflated_sharpe"})
_SHARPE_CALL_NAMES = frozenset({"sharpe_ratio"})
_BLOCKED_IO_NAMES = frozenset({"read_parquet", "scan_parquet"})
# ADVERSARIAL §1a-E13: parquet reads hidden inside SQL strings (duckdb.sql).
_SQL_PARQUET_RE = re.compile(r"read_parquet\s*\(|read_csv\s*\(|\.parquet", re.IGNORECASE)
_FSTRING_NUMSPEC_RE = re.compile(r"\d")


def _is_allowlisted(rule_id: str, path_str: str) -> bool:
    globs = RULE_ALLOWLISTS.get(rule_id, frozenset())
    return any(fnmatch.fnmatch(path_str, g) or fnmatch.fnmatch(path_str, f"*/{g}") for g in globs)


def _exempt_by_globs(path_str: str, globs: frozenset[str]) -> bool:
    return any(fnmatch.fnmatch(path_str, g) or fnmatch.fnmatch(path_str, f"*/{g}") for g in globs)


def _fold_str(node: ast.AST) -> str | None:
    """Constant-fold a string expression: literal or ``+``-concatenation of
    literals (ADVERSARIAL §1a-F4/E12 — string concatenation evasion)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _fold_str(node.left)
        right = _fold_str(node.right)
        if left is not None and right is not None:
            return left + right
    return None


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
    if isinstance(receiver, ast.Subscript):
        key = receiver.slice
        return (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and bool(_PRICE_NAME_RE.search(key.value))
        )
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
        # ADVERSARIAL §1a-E4: a negative shift on a rolling statistic
        # (`rolling_mean(21).shift(k)`, k negative) centers the window on the
        # future regardless of the inner receiver's name.
        if func_name.startswith("rolling"):
            return True
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


def _shift_period_expr(call: ast.Call) -> ast.AST | None:
    """The period argument of a ``.shift`` call: positional or ``periods=``
    keyword form (ADVERSARIAL §1a-E3)."""
    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg == "periods":
            return kw.value
    return None


def _is_negative_shift_period(expr: ast.AST, assigns: dict[str, ast.AST]) -> bool:
    """True if the shift period provably resolves to a negative int.

    Handles literals (``-1``) and single-assignment names (``k = -HALF``,
    ``h = -horizon`` — ADVERSARIAL §1a-E4/E14) via scope-level constant
    resolution. Unresolvable expressions are conservatively clean.
    """
    resolved = _resolve_expr(expr, assigns)
    if _neg_int_literal(resolved):
        return True
    if isinstance(resolved, ast.UnaryOp) and isinstance(resolved.op, ast.USub):
        operand = _resolve_expr(resolved.operand, assigns)
        return (
            isinstance(operand, ast.Constant)
            and isinstance(operand.value, int)
            and not isinstance(operand.value, bool)
            and operand.value > 0
        )
    return False


def _check_lh001(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "shift"
        ):
            continue
        period = _shift_period_expr(node)
        if period is None:
            continue
        _, funcs = _enclosing_scopes(node, parents)
        scope: ast.AST = funcs[-1] if funcs else tree
        # Innermost-scope names shadow module-level constants (E4/E14 define
        # the shift amount at module scope, e.g. `HALF = 10; k = -HALF`).
        assigns = _scope_assignments(tree)  # type: ignore[arg-type]
        assigns.update(_scope_assignments(scope))  # type: ignore[arg-type]
        if _is_negative_shift_period(period, assigns) and _receiver_price_like(node.func.value):
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


def _fit_consumes_fold_param(
    node: ast.Call, func: ast.FunctionDef | ast.AsyncFunctionDef
) -> bool:
    """True if the fit's first argument references one of *func*'s parameters
    (i.e. the caller sliced the fold; ADVERSARIAL §1a-E7 — a function merely
    NAMED `*fold*` that fits a global/frame attribute is not a fold scope).
    """
    if not node.args:
        return False
    params = {a.arg for a in (*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs)}
    names = {n.id for n in ast.walk(node.args[0]) if isinstance(n, ast.Name)}
    return bool(names & params)


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
        # A fold-NAMED function is a safe scope only if the fit consumes one
        # of the function's own parameters (caller-sliced fold data).
        in_fold_func = any(
            _FOLD_FUNC_RE.search(f.name) and _fit_consumes_fold_param(node, f) for f in funcs
        )
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
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node) == "join_asof"):
            continue
        _, funcs = _enclosing_scopes(node, parents)
        scope: ast.AST = funcs[-1] if funcs else tree
        assigns = _scope_assignments(scope)  # type: ignore[arg-type]
        for kw in node.keywords:
            if kw.arg not in ("on", "left_on"):
                continue
            # ADVERSARIAL §1a-E17: the keyword value may hide behind a
            # single-assignment name (`key = "event_time"`).
            resolved = _resolve_expr(kw.value, assigns)
            if isinstance(resolved, ast.Constant) and resolved.value == "event_time":
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
        # ADVERSARIAL §1a-E16: tuples and lowercase tickers are the same leak.
        if not (isinstance(value, (ast.List, ast.Tuple)) and len(value.elts) >= 5):
            continue
        if all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) and _TICKER_CI_RE.match(e.value)
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


def _lh008_allowed(path_str: str, value: str) -> bool:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return any(
        (path_str == path or path_str.endswith(f"/{path}")) and digest in digests
        for path, digests in LH008_LITERAL_ALLOWLIST.items()
    )


def _check_lh008(tree: ast.AST, path_str: str) -> list[_Finding]:
    from quant_fund.leakage.patterns import find_forbidden_headline

    out: list[_Finding] = []
    docstrings = _docstring_constant_ids(tree)
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        value: str | None = None
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and len(node.value) > 0
            # Skip identifier-like keys while checking compact output labels
            # such as "Sharpe:2.1" and "P&L=$4,200".
            and (not node.value.isidentifier())
        ):
            value = node.value
        elif (
            # ADVERSARIAL §1a-F4: constant-fold "a" + "b" string concatenation
            # (top-most Add node only; nested parts are visited as Constants).
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Add)
            and not isinstance(parents.get(id(node)), ast.BinOp)
        ):
            folded = _fold_str(node)
            if folded and not folded.isidentifier():
                value = folded
        if value is None:
            continue
        tokens = find_forbidden_headline(value)
        if tokens and not _lh008_allowed(path_str, value):
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
    # tests/** is exempt EXCEPT the seeded-leak fixture suite, which must stay
    # scannable (same carve-out as LH005, DESIGN.md §6.4).
    if "leakage_fixtures" not in path_str and _exempt_by_globs(path_str, LH009_EXEMPT_GLOBS):
        return []
    out: list[_Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name in ("read_parquet", "scan_parquet"):
            out.append(
                _Finding(
                    "LH009",
                    node.lineno,
                    node.col_offset,
                    "direct parquet read outside the data layer bypasses the PIT choke point",
                )
            )
        # ADVERSARIAL §1a-E12: getattr(pl, "read_" + "parquet")(path).
        elif name == "getattr" and len(node.args) >= 2:
            attr = _fold_str(node.args[1])
            if attr is not None and attr in _BLOCKED_IO_NAMES:
                out.append(
                    _Finding(
                        "LH009",
                        node.lineno,
                        node.col_offset,
                        f"dynamic getattr(.., {attr!r}) parquet read bypasses the PIT choke point",
                    )
                )
        # ADVERSARIAL §1a-E13: duckdb.sql("select * from read_parquet(...)").
        elif name == "sql":
            for arg in node.args:
                folded = _fold_str(arg)
                if folded is not None and _SQL_PARQUET_RE.search(folded):
                    out.append(
                        _Finding(
                            "LH009",
                            node.lineno,
                            node.col_offset,
                            "parquet read embedded in a SQL string bypasses the PIT choke point",
                        )
                    )
                    break
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
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom):
            if node.level:
                parts = Path(path_str).parts
                quant_index = parts.index("quant_fund")
                package_parts = list(parts[quant_index:-1])
                if node.level > len(package_parts):
                    modules = ["quant_fund.__invalid_relative_import__"]
                else:
                    base = package_parts[: len(package_parts) - node.level + 1]
                    modules = [".".join([*base, *([node.module] if node.module else [])])]
            elif node.module:
                modules = [node.module]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        # Function-level (lazy) imports may use the lazy whitelist.
        cur = parents.get(id(node))
        nested = False
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested = True
                break
            cur = parents.get(id(cur))
        allowed = whitelist | (lazy_whitelist if nested else frozenset())
        for module in modules:
            if not module.startswith("quant_fund."):
                continue
            sub = module.split(".")[1]
            if sub == package:
                continue  # intra-package imports are always fine
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


def _check_lh013(tree: ast.AST, path_str: str, source: str) -> list[_Finding]:
    """Warning-severity headline channels (ADVERSARIAL §1a E8/E9/E10/F2/F6):

    - COMMENTS via tokenize (comments are invisible to the AST);
    - DOCSTRINGS (LH008 excludes them, but they render into published docs);
    - f-string templates whose number arrives at runtime
      (``f"Sharpe was {sr:.2f}"`` — a forbidden token plus a numeric format
      spec or spelled-out numeral);
    - spelled-out numerals in any string channel
      (``"Sharpe, which exceeded two"``).
    """
    from quant_fund.leakage.patterns import (
        find_forbidden_headline,
        find_forbidden_token_mentions,
        find_spelled_out_headline,
    )

    out: list[_Finding] = []

    def _fire(line: int, col: int, tokens: list[str], channel: str) -> None:
        out.append(
            _Finding(
                "LH013",
                line,
                col,
                f"{channel} headlines forbidden metric token(s): {', '.join(tokens)}",
            )
        )

    # comments
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type != tokenize.COMMENT:
                continue
            text = tok.string.lstrip("#").strip()
            tokens = find_forbidden_headline(text) or find_spelled_out_headline(text)
            if tokens and not _lh008_allowed(path_str, text):
                _fire(tok.start[0], tok.start[1], tokens, "comment")
    except tokenize.TokenError:
        pass

    docstrings = _docstring_constant_ids(tree)
    for node in ast.walk(tree):
        # docstrings
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) in docstrings
        ):
            tokens = find_forbidden_headline(node.value) or find_spelled_out_headline(node.value)
            if tokens and not _lh008_allowed(path_str, node.value):
                _fire(node.lineno, node.col_offset, tokens, "docstring")
        # spelled-out numerals in non-docstring string literals (digit forms
        # stay with LH008 at error severity)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value
            and not node.value.isidentifier()
            and not find_forbidden_headline(node.value)
        ):
            tokens = find_spelled_out_headline(node.value)
            if tokens and not _lh008_allowed(path_str, node.value):
                _fire(node.lineno, node.col_offset, tokens, "string literal (spelled-out numeral)")
        # f-string templates: forbidden token + numeric format spec
        elif isinstance(node, ast.JoinedStr):
            literal = "".join(
                part.value
                for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            if not literal or not find_forbidden_token_mentions(literal):
                continue
            has_numeric_spec = any(
                isinstance(part, ast.FormattedValue)
                and part.format_spec is not None
                and bool(_FSTRING_NUMSPEC_RE.search(ast.unparse(part.format_spec)))
                for part in node.values
            )
            tokens = (
                find_forbidden_headline(literal)
                or (find_forbidden_token_mentions(literal) if has_numeric_spec else [])
                or find_spelled_out_headline(literal)
            )
            if tokens:
                _fire(node.lineno, node.col_offset, tokens, "f-string template")
    return out


def _check_lh014(tree: ast.AST, path_str: str, findings: list[_Finding]) -> list[_Finding]:
    """Interprocedural-lite (ADVERSARIAL §1a-E2 shape): flag call sites of
    functions whose own body produced a finding. Single-level, single-file:
    cross-file helpers are the documented residual ceiling.
    """
    out: list[_Finding] = []
    functions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    leaky: dict[str, set[str]] = {}
    for func in functions:
        end = func.end_lineno or func.lineno
        hits = {f.rule_id for f in findings if func.lineno <= f.line <= end}
        # the helper's own LH014 call-site findings don't propagate upward
        hits.discard("LH014")
        if hits:
            leaky.setdefault(func.name, set()).update(hits)
    if not leaky:
        return out
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        name = node.func.id
        if name not in leaky:
            continue
        # skip recursive self-calls
        enclosing = [
            f
            for f in functions
            if f.lineno <= node.lineno <= (f.end_lineno or f.lineno) and f.name == name
        ]
        if enclosing:
            continue
        out.append(
            _Finding(
                "LH014",
                node.lineno,
                node.col_offset,
                f"call to helper `{name}` whose body trips {', '.join(sorted(leaky[name]))}; "
                "the leakage travels with the helper",
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

_ALL_RULES = frozenset(_CHECKERS) | {"LH012", "LH013", "LH014"}


def scan_file(path: Path, *, rules: set[str] | None = None) -> list[_Finding]:
    """Scan one .py file; never raises on parse errors (LH012 instead)."""
    path_str = path.as_posix()
    enabled = rules if rules is not None else set(_ALL_RULES)
    findings: list[_Finding] = []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        if "LH012" not in enabled:
            return []
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
    if "LH013" in enabled and not _is_allowlisted("LH013", path_str):
        findings.extend(_check_lh013(tree, path_str, source))
    if "LH014" in enabled and not _is_allowlisted("LH014", path_str):
        findings.extend(_check_lh014(tree, path_str, findings))
    findings.sort(key=lambda f: (f.line, f.col, f.rule_id))
    return findings


def collect_py_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
        else:
            raise ValueError(f"scan target is missing or is not a Python file/directory: {p}")
    # de-duplicate, stable order
    seen: set[str] = set()
    out: list[Path] = []
    for f in files:
        key = str(f)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return sorted(out, key=lambda f: f.as_posix())
