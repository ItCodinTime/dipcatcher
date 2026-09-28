"""P6.9 deterministic mutation spot-check harness for money paths.

Hand-rolled AST mutator. For each curated money-path target (fill pricing,
fee accrual, NAV update, leverage cap), it enumerates single-token
mutations — flipped comparison operators, Add<->Sub / Mult<->Div swaps on
money terms, dropped ``abs()``/``np.sqrt``/negation, ``min``<->``max``
swaps, ``+=``<->``-=`` swaps, negated returns — applies exactly one at a
time, runs a pytest slice, and records kill/survive.

Each mutant is a position-pinned splice (lineno/col_offset), so results are
deterministic and diff-inspectable. A mutant "survives" only if the full
slice passes; a narrow per-module slice is run first purely as a fast
pre-filter.

Usage:
    uv run python scripts/mutation_spotcheck.py --list          # enumerate
    uv run python scripts/mutation_spotcheck.py                 # run all
    uv run python scripts/mutation_spotcheck.py --only M001 M007
    uv run python scripts/mutation_spotcheck.py --resume        # keep prior kills
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if "MUT_ROOT" in __import__("os").environ:
    REPO = Path(__import__("os").environ["MUT_ROOT"]).resolve()

# Per-module suite slice: every test file that imports the mutated module
# (directly or via the public seam), minus slow/e2e/perf lanes and files with
# baseline failures. Follows the convention in tests/property/mutation_scores.json.
BROKER = "src/quant_fund/execution/simulated_broker.py"
ENGINE = "src/quant_fund/backtest/engine.py"
PERP = "src/quant_fund/backtest/perp_engine.py"
COSTS = "src/quant_fund/execution/costs.py"


MODULE_SLICE = {
    COSTS: [
        "tests/property/test_adversarial_costs.py",
        "tests/unit/backtest/test_execution.py",
        "tests/unit/test_cost_branch_killers.py",
        "tests/unit/backtest/test_p69_mutation_audit.py",
        "tests/unit/backtest/test_cost_allocation.py",
        "tests/unit/backtest/test_cost_sensitivity.py",
    ],
    BROKER: [
        "tests/unit/backtest/test_simulated_broker.py",
        "tests/unit/backtest/test_broker_cash_conservation.py",
        "tests/unit/backtest/test_p69_mutation_audit.py",
        "tests/unit/backtest/test_p61_money_audit.py",
        "tests/unit/backtest/test_execution_determinism.py",
        "tests/unit/backtest/test_paper_loop.py",
        "tests/unit/backtest/test_forward_shadow.py",
        "tests/unit/backtest/test_day_paper_shadow_honesty.py",
        "tests/unit/backtest/test_paper_long_stress.py",
        "tests/unit/backtest/test_paper_resume_promotion.py",
        "tests/unit/backtest/test_paper_multi_challenger_config.py",
        "tests/unit/backtest/test_paper_multi_challenger_e2e.py",
        "tests/unit/backtest/test_paper_wave5.py",
        "tests/formal/test_order_lifecycle.py",
        "tests/formal/test_stateful.py",
        "tests/formal/test_accounting_smt.py",
        "tests/unit/pipeline/test_institutional.py",
        "tests/unit/pretrade/test_shadow.py",
        "tests/regression/test_duplicate_order_submit.py",
        "tests/regression/test_name_cap_float_admits_order.py",
        "tests/unit/risk/test_risk_gate_reject_accounting.py",
    ],
    ENGINE: [
        "tests/unit/backtest/test_backtest_engine.py",
        "tests/unit/backtest/test_p69_mutation_audit.py",
        "tests/unit/backtest/test_p61_money_audit.py",
        "tests/unit/backtest/test_cost_sensitivity.py",
        "tests/unit/backtest/test_fast_replay.py",
        "tests/unit/backtest/test_forward_shadow.py",
        "tests/unit/backtest/test_replay_clock.py",
        "tests/unit/backtest/test_attribution_hook.py",
        "tests/unit/backtest/test_turnover_soft_and_stress.py",
        "tests/regression/test_backtest_turnover_breakdown.py",
        "tests/regression/test_cash_ledger_turnover.py",
        "tests/regression/test_one_period_total_return.py",
        "tests/regression/test_turnover_bps_recorded_on_fills.py",
        "tests/regression/test_split_dividend_conservation.py",
        "tests/property/test_adversarial_backtest.py",
        # excluded: byte-identity IPC assertion fails at baseline on macOS arm64
        # "tests/property/test_fast_replay_byte_identity.py",
        "tests/property/test_nav_resume_identity.py",
        "tests/property/test_net_gross_identity.py",
        "tests/property/test_turnover_identity.py",
        "tests/property/test_capacity_overlay.py",
        "tests/unit/microstructure/test_backtest_causal_liquidity.py",
        "tests/unit/risk/test_risk_gate_backtest.py",
        "tests/unit/risk/test_garch_risk_gate.py",
        "tests/unit/risk/test_realized_garch_risk_gate.py",
        "tests/unit/pipeline/test_overlay.py",
        "tests/unit/pipeline/test_ledger_fail_closed_branches.py",
        "tests/unit/pipeline/test_audit_failclosed_regressions.py",
        "tests/unit/parity/test_replay.py",
        "tests/unit/simtest/test_replay.py",
        "tests/unit/observe/test_hooks.py",
        "tests/unit/models/test_wave6_analytics.py",
        "tests/unit/test_perf_equivalence.py",
        # excluded: allowlist regression asserts a .shift(-1) that no longer exists at HEAD
        # "tests/unit/test_leakage_ast_scan.py",
        # excluded: public-API snapshot is stale at HEAD (run_backtest fast= kwarg)
        # "tests/unit/test_public_api.py",
    ],
    PERP: [
        "tests/unit/backtest/test_perp_engine.py",
        "tests/unit/backtest/test_p69_mutation_audit.py",
        "tests/unit/backtest/test_p61_money_audit.py",
        "tests/unit/backtest/test_carry_engine.py",
        "tests/unit/backtest/test_event_sim.py",
        "tests/property/test_event_sim_invariants.py",
        "tests/property/test_net_gross_identity.py",
        "tests/regression/test_one_period_total_return.py",
        "tests/unit/microstructure/test_backtest_causal_liquidity.py",
        "tests/unit/risk/test_risk_gate_backtest.py",
    ],
}

# Mutation operator sets per target: cmp = comparison flips, boundary =
# inclusive->strict edge, arith = Add<->Sub + Mult<->Div, call =
# abs/sqrt/sign/min/max/isfinite call mutations, aug = +=/<->-=, neg =
# drop unary minus, ret = negate returned expression.
OPS = {"cmp", "boundary", "arith", "call", "aug", "neg", "ret"}


# (file, qualname, (lo, hi) 1-based inclusive line window or None, ops)
TARGETS: list[tuple[str, str, tuple[int, int] | None, set[str]]] = [
    # --- fee accrual ---------------------------------------------------
    (COSTS, "half_spread_cost", None, {"arith", "call"}),
    (COSTS, "commission_cost", None, {"arith", "call"}),
    (COSTS, "sqrt_impact", (47, 52), {"arith", "call", "cmp"}),
    (COSTS, "total_cost", (72, 96), {"arith", "call", "cmp"}),
    # --- fill pricing --------------------------------------------------
    (BROKER, "_limit_fill_price", None, {"cmp", "call", "boundary"}),
    (BROKER, "SimulatedBroker.nav", None, {"arith", "call", "cmp", "ret"}),
    (BROKER, "SimulatedBroker.exposures", None, {"arith", "call", "cmp"}),
    (BROKER, "SimulatedBroker._attempt_fill", None, {"arith", "call", "aug", "cmp", "neg"}),
    (BROKER, "SimulatedBroker.target_to_orders", None, {"arith", "cmp", "call"}),
    (BROKER, "SimulatedBroker.cash_nav_identity", None, {"arith"}),
    # --- NAV update / sizing / borrow ----------------------------------
    (ENGINE, "Book.nav", None, {"arith", "ret"}),
    (ENGINE, "_projected_exposures", None, {"arith", "call"}),
    (ENGINE, "_make_order", None, {"cmp", "call"}),
    (ENGINE, "_run_backtest_event_loop", (383, 404), {"cmp", "call"}),
    (ENGINE, "_run_backtest_event_loop", (413, 503), {"arith", "cmp", "call", "aug"}),
    (ENGINE, "_run_backtest_event_loop", (505, 528), {"arith", "cmp", "call", "aug"}),
    # --- leverage cap / funding accrual / settlement / liquidation -----
    (PERP, "PerpBook.upnl", None, {"arith", "call", "ret"}),
    (PERP, "PerpBook.equity", None, {"arith", "ret"}),
    (PERP, "PerpBook.gross_notional", None, {"arith", "call"}),
    (PERP, "PerpBook.net_notional", None, {"arith"}),
    (PERP, "_funding_by_time", None, {"arith", "cmp"}),
    (PERP, "run_perp_backtest", (278, 309), {"arith", "cmp", "call", "boundary"}),
    (PERP, "run_perp_backtest", (345, 379), {"arith", "cmp", "call", "aug"}),
    (PERP, "run_perp_backtest", (381, 395), {"arith", "cmp", "neg"}),
    (PERP, "run_perp_backtest._mmr_deficit", None, {"arith", "cmp"}),
    (PERP, "run_perp_backtest", (397, 438), {"arith", "cmp", "call", "aug"}),
]

CMP_COMPLEMENT = {
    ast.Lt: ">=",
    ast.LtE: ">",
    ast.Gt: "<=",
    ast.GtE: "<",
    ast.Eq: "!=",
    ast.NotEq: "==",
}
CMP_BOUNDARY = {ast.LtE: "<", ast.GtE: ">"}
BINOP_SWAP = {ast.Add: "-", ast.Sub: "+", ast.Mult: "/", ast.Div: "*"}
AUG_SWAP = {ast.Add: "-=", ast.Sub: "+="}
CALL_ARG = {"abs", "np.sqrt", "sqrt"}
CALL_SWAP = {"min": "max", "max": "min"}
CALL_FLIP = {"np.sign"}  # wrap in negation
CALL_INVERT = {"np.isfinite"}  # wrap in `not`


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        base = _call_name(func.value)
        return f"{base}.{func.attr}" if base else func.attr
    return None


@dataclass
class Mutant:
    mid: str
    file: str
    qualname: str
    line: int
    kind: str
    original: str
    mutated: str
    span: tuple[int, int, int, int]  # lineno, col, end_lineno, end_col
    note: str = ""
    verdict: str = ""  # killed-narrow | killed-full | survived | timeout | invalid
    detail: str = ""
    seconds: float = 0.0


class _Enumerator(ast.NodeVisitor):
    def __init__(self, src: str, qualname: str, window, ops: set[str]):
        self.src = src
        self.qualname = qualname
        self.lo, self.hi = window if window else (0, 10**9)
        self.ops = ops
        self.scope: list[str] = []
        self.mutants: list[Mutant] = []

    # -- scope tracking -------------------------------------------------
    def _push(self, name: str, node) -> None:
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node):
        self._push(node.name, node)

    def visit_FunctionDef(self, node):
        self._push(node.name, node)

    visit_AsyncFunctionDef = visit_FunctionDef

    # -- helpers --------------------------------------------------------
    def _in_target(self, node) -> bool:
        return ".".join(self.scope) == self.qualname and self.lo <= node.lineno <= self.hi

    def _seg(self, node) -> str:
        return ast.get_source_segment(self.src, node) or ""

    def _emit(self, node, kind: str, mutated: str, note: str = "") -> None:
        original = self._seg(node)
        if not mutated or mutated == original:
            return
        self.mutants.append(
            Mutant(
                mid="",
                file="",
                qualname=self.qualname,
                line=node.lineno,
                kind=kind,
                original=" ".join(original.split()),
                mutated=" ".join(mutated.split()),
                span=(node.lineno, node.col_offset, node.end_lineno, node.end_col_offset),
                note=note,
            )
        )

    # -- node handlers --------------------------------------------------
    def visit_Compare(self, node):
        if self._in_target(node) and "cmp" in self.ops and len(node.ops) == 1:
            left, right = self._seg(node.left), self._seg(node.comparators[0])
            comp = CMP_COMPLEMENT.get(type(node.ops[0]))
            if comp:
                self._emit(node, "cmp-complement", f"(({left}) {comp} ({right}))")
            bound = CMP_BOUNDARY.get(type(node.ops[0]))
            if bound and "boundary" in self.ops:
                self._emit(node, "cmp-boundary", f"(({left}) {bound} ({right}))")
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if self._in_target(node) and "arith" in self.ops:
            sym = BINOP_SWAP.get(type(node.op))
            if sym:
                self._emit(
                    node,
                    f"arith-{type(node.op).__name__.lower()}",
                    f"(({self._seg(node.left)}) {sym} ({self._seg(node.right)}))",
                )
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if self._in_target(node) and "neg" in self.ops and isinstance(node.op, ast.USub):
            self._emit(node, "neg-drop", self._seg(node.operand))
        self.generic_visit(node)

    def visit_Call(self, node):
        if self._in_target(node) and "call" in self.ops:
            name = _call_name(node.func)
            if name in CALL_ARG and node.args:
                self._emit(node, "call-drop", self._seg(node.args[0]))
            elif name in CALL_SWAP:
                self._emit(
                    node,
                    "call-swap",
                    CALL_SWAP[name] + self._seg(node)[len(name) :],
                )
            elif name in CALL_FLIP:
                self._emit(node, "call-negate", f"(-{self._seg(node)})")
            elif name in CALL_INVERT:
                self._emit(node, "call-invert", f"(not {self._seg(node)})")
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if self._in_target(node) and "aug" in self.ops:
            sym = AUG_SWAP.get(type(node.op))
            if sym:
                self._emit(
                    node,
                    "aug-swap",
                    f"{self._seg(node.target)} {sym} ({self._seg(node.value)})",
                )
        self.generic_visit(node)

    def visit_Return(self, node):
        if self._in_target(node) and "ret" in self.ops and node.value is not None:
            v = node.value
            numericish = isinstance(
                v, (ast.BinOp, ast.Name, ast.Call, ast.UnaryOp, ast.Attribute, ast.Subscript)
            ) or (isinstance(v, ast.Constant) and isinstance(v.value, (int, float)))
            if numericish:
                self._emit(node, "ret-negate", f"return -({self._seg(v)})")
        self.generic_visit(node)


def _offset(lines: list[str], lineno: int, col: int) -> int:
    """AST (lineno, col_offset utf-8 bytes) -> absolute char offset."""
    base = sum(len(lines[i]) for i in range(lineno - 1))
    return base + len(lines[lineno - 1].encode("utf-8")[:col].decode("utf-8"))


def enumerate_mutants() -> list[Mutant]:
    out: list[Mutant] = []
    seen: set[tuple] = set()
    for file, qualname, window, ops in TARGETS:
        src = (REPO / file).read_text()
        enum = _Enumerator(src, qualname, window, ops)
        enum.visit(ast.parse(src))
        for m in enum.mutants:
            key = (file, m.span, m.mutated)
            if key in seen:
                continue
            seen.add(key)
            m.file = file
            out.append(m)
    for i, m in enumerate(out, 1):
        m.mid = f"M{i:03d}"
    return out


def apply_mutant(m: Mutant) -> str:
    path = REPO / m.file
    original = path.read_text()
    lines = original.splitlines(keepends=True)
    lo = _offset(lines, m.span[0], m.span[1])
    hi = _offset(lines, m.span[2], m.span[3])
    mutated = original[:lo] + m.mutated + original[hi:]
    ast.parse(mutated)  # fail loudly if the splice broke syntax
    path.write_text(mutated)
    return original


def restore(m: Mutant, original: str) -> None:
    (REPO / m.file).write_text(original)


def run_pytest(files: list[str], timeout: int) -> tuple[int, str]:
    n = __import__("os").environ.get("MUT_NWORKERS", "4")
    cmd = [
        "uv",
        "run",
        "pytest",
        "-n",
        n,
        "-x",
        "-q",
        "-m",
        "not network and not slow",
        "-p",
        "no:cacheprovider",
        *files,
    ]
    try:
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        return proc.returncode, tail
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="enumerate and exit")
    ap.add_argument("--only", nargs="*", default=None, help="mutant ids to run")
    ap.add_argument("--resume", action="store_true", help="keep existing verdicts")
    ap.add_argument(
        "--out",
        default="artifacts/p69_mutation_results.json",
        help="results JSON path",
    )
    ap.add_argument("--file", default=None, help="restrict to one target file")
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="run both slices unmutated (pre-flight green check) and exit",
    )
    args = ap.parse_args()

    mutants = enumerate_mutants()
    if args.file:
        mutants = [m for m in mutants if m.file == args.file]
    if args.list:
        for m in mutants:
            print(f"{m.mid} {m.file}:{m.line} [{m.kind}] {m.original} -> {m.mutated}")
        print(f"{len(mutants)} mutants")
        return 0

    if args.baseline:
        files = sorted({m.file for m in mutants})
        for f in files:
            rc, tail = run_pytest(MODULE_SLICE[f], timeout=900)
            print(f"{f}: rc={rc}")
            if rc != 0:
                print(tail)
        return 0 if rc == 0 else 1

    prior: dict[str, dict] = {}
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    if args.resume and out_path.exists():
        prior = {r["id"]: r for r in json.loads(out_path.read_text())["mutants"]}

    results: list[dict] = []
    for m in mutants:
        if args.only and m.mid not in args.only:
            continue
        if m.mid in prior and prior[m.mid].get("verdict", "").startswith("killed"):
            # A kill under any earlier slice remains a kill: the slice only grows.
            results.append(prior[m.mid])
            print(f"{m.mid} {m.file}:{m.line} [{m.kind}] reused {prior[m.mid]['verdict']}")
            continue
        print(f"{m.mid} {m.file}:{m.line} [{m.kind}] {m.original} -> {m.mutated}", flush=True)
        t0 = time.monotonic()
        original = apply_mutant(m)
        try:
            rc, tail = run_pytest(MODULE_SLICE[m.file], timeout=600)
            if rc == -1:
                m.verdict = "timeout"
            else:
                m.verdict = "killed" if rc != 0 else "survived"
            m.detail = tail
        finally:
            restore(m, original)
        m.seconds = round(time.monotonic() - t0, 1)
        print(f"  -> {m.verdict} ({m.seconds}s)", flush=True)
        results.append(
            {
                "id": m.mid,
                "file": m.file,
                "qualname": m.qualname,
                "line": m.line,
                "kind": m.kind,
                "original": m.original,
                "mutated": m.mutated,
                "verdict": m.verdict,
                "seconds": m.seconds,
                "detail": m.detail[-2000:],
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"mutants": results}, indent=2))

    survived = [r for r in results if r["verdict"] == "survived"]
    print(f"\n{len(results)} mutants: {len(survived)} survived")
    for r in survived:
        print(
            f"  SURVIVED {r['id']} {r['file']}:{r['line']} [{r['kind']}] {r['original']} -> {r['mutated']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
