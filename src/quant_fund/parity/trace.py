"""Static import-graph and runtime call-trace comparison.

The shadow and the backtest reference must execute the same strategy
functions. Two checks:

* **Static.** Direct imports of the runner modules (followed only inside
  ``quant_fund.parity``) and the AST callees of each ``decide`` function.
* **Runtime.** ``sys.setprofile`` call events while ``decide`` runs.
  Profiling is independent of ``sys.settrace``, so coverage tracers stay
  in place.

A function that appears on only one side is a hidden branch.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import sys
import textwrap
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any

RUNNER_MODULES: tuple[str, str] = (
    "quant_fund.parity.reference",
    "quant_fund.parity.shadow",
)
_PARITY_PREFIX = "quant_fund.parity"


def digest_calls(calls: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(calls).encode()).hexdigest()


def imports_from_source(source: str) -> set[str]:
    """Module names imported by ``source``. Relative imports keep their tail."""
    tree = ast.parse(source)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def module_source(module_name: str) -> str:
    module = importlib.import_module(module_name)
    file = getattr(module, "__file__", None)
    if not file:
        raise ValueError(f"{module_name} has no source file")
    return Path(file).read_text(encoding="utf-8")


def direct_imports(module_name: str) -> set[str]:
    return imports_from_source(module_source(module_name))


def import_closure(module_name: str, *, follow_prefix: str = _PARITY_PREFIX) -> set[str]:
    """Transitive imports that stay under ``follow_prefix``, plus the root's direct imports."""
    seen: set[str] = set()
    direct: set[str] = set()
    stack = [module_name]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            imports = direct_imports(name)
        except (ValueError, OSError, SyntaxError):
            continue
        if name == module_name:
            direct = set(imports)
        for imported in sorted(imports):
            if imported.startswith(follow_prefix) and imported not in seen:
                stack.append(imported)
    # The comparable graph is the root's direct imports plus parity-internal modules reached.
    reached = {name for name in seen if name != module_name}
    return direct | reached


def static_callees(fn: Callable[..., Any]) -> set[str]:
    """Names and attribute tails called directly in ``fn``'s source."""
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as exc:
        raise ValueError(f"cannot read source for {fn!r}") from exc
    tree = ast.parse(source)
    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            calls.add(func.id)
        elif isinstance(func, ast.Attribute):
            calls.add(func.attr)
    return calls


class CallTracer:
    """Record Python ``call`` events via ``sys.setprofile``.

    The previous profile hook is chained and restored. ``sys.settrace`` is
    left alone so a coverage tracer keeps receiving line events.
    """

    def __init__(self, *, accept_module: Callable[[str], bool] | None = None) -> None:
        self._accept = accept_module or _default_accept
        self.calls: list[str] = []
        self._prev: Any = None

    def __enter__(self) -> CallTracer:
        self.calls = []
        self._prev = sys.getprofile()
        sys.setprofile(self._profile)
        return self

    def __exit__(self, *exc: object) -> None:
        sys.setprofile(self._prev)
        self._prev = None

    def _profile(self, frame: FrameType, event: str, arg: Any) -> None:
        prev = self._prev
        if prev is not None:
            prev(frame, event, arg)
        if event != "call":
            return
        module = str(frame.f_globals.get("__name__", ""))
        if module == "quant_fund.parity.trace":
            return
        if not self._accept(module):
            return
        qual = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        self.calls.append(f"{module}:{qual}")


def _default_accept(module: str) -> bool:
    if module.startswith("quant_fund"):
        return True
    return module.startswith("tests.")


def accept_strategy_modules(*modules: str) -> Callable[[str], bool]:
    """Accept the strategy module plus ``quant_fund`` and ``tests``."""
    extra = tuple(module for module in modules if module)

    def _accept(module: str) -> bool:
        if module in extra or any(module.startswith(item + ".") for item in extra):
            return True
        return _default_accept(module)

    return _accept


def _sorted(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def _flatten(traces: Iterable[Iterable[str]]) -> list[str]:
    out: list[str] = []
    for trace in traces:
        out.extend(trace)
    return out


def compare_import_graphs(left_module: str, right_module: str) -> dict[str, Any]:
    left = import_closure(left_module)
    right = import_closure(right_module)
    only_left = _sorted(left - right)
    only_right = _sorted(right - left)
    return {
        "left_module": left_module,
        "right_module": right_module,
        "equal": not only_left and not only_right,
        "only_left": only_left,
        "only_right": only_right,
    }


def code_path_guard(
    *,
    backtest_fn: Callable[..., Any],
    shadow_fn: Callable[..., Any],
    backtest_traces: Iterable[Iterable[str]],
    shadow_traces: Iterable[Iterable[str]],
    runner_modules: tuple[str, str] = RUNNER_MODULES,
) -> dict[str, Any]:
    """Compare import graphs, static callees, and runtime traces.

    ``same_code_path`` is true only when all three agree. This does not
    prove the strategy is correct. It proves the two origins did not take
    observably different functions on this run.
    """
    runner_graph = compare_import_graphs(runner_modules[0], runner_modules[1])
    bt_module = str(getattr(backtest_fn, "__module__", ""))
    sh_module = str(getattr(shadow_fn, "__module__", ""))
    strategy_graph = (
        compare_import_graphs(bt_module, sh_module)
        if bt_module and sh_module
        else {
            "left_module": bt_module,
            "right_module": sh_module,
            "equal": bt_module == sh_module,
            "only_left": [],
            "only_right": [],
        }
    )
    bt_callees = static_callees(backtest_fn)
    sh_callees = static_callees(shadow_fn)
    only_bt_callees = _sorted(bt_callees - sh_callees)
    only_sh_callees = _sorted(sh_callees - bt_callees)
    bt_seq = _flatten(backtest_traces)
    sh_seq = _flatten(shadow_traces)
    only_bt_runtime = _sorted(set(bt_seq) - set(sh_seq))
    only_sh_runtime = _sorted(set(sh_seq) - set(bt_seq))
    runtime_equal = bt_seq == sh_seq
    static_equal = not only_bt_callees and not only_sh_callees
    imports_equal = bool(runner_graph["equal"] and strategy_graph["equal"])
    return {
        "same_code_path": bool(imports_equal and static_equal and runtime_equal),
        "import_graph_equal": imports_equal,
        "runner_import_graph": runner_graph,
        "strategy_import_graph": strategy_graph,
        "static_callees_equal": static_equal,
        "callees_only_backtest": only_bt_callees,
        "callees_only_shadow": only_sh_callees,
        "runtime_trace_equal": runtime_equal,
        "runtime_only_backtest": only_bt_runtime,
        "runtime_only_shadow": only_sh_runtime,
        "backtest_call_count": len(bt_seq),
        "shadow_call_count": len(sh_seq),
        "live_pnl_claim": False,
        "research_only": True,
        "note": (
            "Equal traces mean these two runs called the same functions. "
            "They are not evidence of economic value."
        ),
    }


def iter_call_names(traces: Iterable[Iterable[str]]) -> Iterator[str]:
    yield from _flatten(traces)
