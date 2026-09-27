"""PROOFCORE CI gate helpers (W5): per-package coverage floors + receipts loop.

Layer-4 glue: stdlib only at module level. Gate execution shells out to the
existing console entry points (``python -m coverage``, ``python -m
quant_fund.cli.main verify-research``) so this module never imports the
quant_fund SCC and stays inside the §1.3 layering contract.

Single source of truth for the floors: ``[tool.proofcore.coverage-floors]``
in ``pyproject.toml`` (additive to the global 80% floor in
``[tool.coverage.report]`` — raise, never lower).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path

from quant_fund.proofcore.contracts import ProofcoreError

# The PROOFCORE packages subject to per-package floors (A3 #2). Floors are
# read from pyproject; this dict pins which packages MUST have a floor entry.
REQUIRED_FLOOR_PACKAGES: tuple[str, ...] = ("pit", "proof", "leakage", "reality", "proofcore")

VerifyFn = Callable[[Path], bool]
RunFn = Callable[..., subprocess.CompletedProcess[str]]


def coverage_floors(pyproject_path: Path = Path("pyproject.toml")) -> dict[str, int]:
    """Read ``[tool.proofcore.coverage-floors]`` from pyproject.toml.

    Fail-closed: missing table, missing package entry, or a floor below the
    global 80% ratchet raises ``ProofcoreError``.
    """
    path = Path(pyproject_path)
    if not path.is_file():
        raise ProofcoreError(f"pyproject not found: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    table = data.get("tool", {}).get("proofcore", {}).get("coverage-floors")
    if not isinstance(table, dict):
        raise ProofcoreError("pyproject.toml is missing [tool.proofcore.coverage-floors]")
    floors: dict[str, int] = {}
    for pkg in REQUIRED_FLOOR_PACKAGES:
        floor = table.get(pkg)
        if not isinstance(floor, int) or isinstance(floor, bool):
            raise ProofcoreError(f"[tool.proofcore.coverage-floors] is missing an int entry: {pkg}")
        if floor < 80:
            raise ProofcoreError(
                f"coverage floor for {pkg} is {floor}, below the global 80% ratchet — "
                "floors are raise-never-lower (A3 #7)"
            )
        floors[pkg] = floor
    return floors


def coverage_gate(
    pyproject_path: Path = Path("pyproject.toml"),
    *,
    src_root: str = "src/quant_fund",
    runner: RunFn = subprocess.run,
) -> list[str]:
    """Run ``coverage report --include=<pkg> --fail-under=<floor>`` per package.

    Returns a list of failure descriptions; empty list means every floor held.
    Requires an already-recorded coverage data file (run pytest --cov first).
    """
    failures: list[str] = []
    for pkg, floor in coverage_floors(pyproject_path).items():
        proc = runner(
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                f"--include={src_root}/{pkg}/*",
                f"--fail-under={floor}",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = (proc.stdout + proc.stderr).strip().splitlines()
            failures.append(
                f"{pkg}: below {floor}% floor — {detail[-1] if detail else 'coverage report failed'}"
            )
    return failures


def receipt_paths(receipts_dir: Path) -> list[Path]:
    """Committed receipt files, sorted for deterministic CI logs."""
    return sorted(Path(receipts_dir).glob("*.json"))


def _cli_verifier(path: Path) -> bool:
    """Default verifier: the existing fail-closed receipt verifier CLI.

    Uses ``python -m quant_fund.cli.main`` so no SCC import enters this
    module's import graph (layering contract, DESIGN.md §1.3).
    """
    proc = subprocess.run(
        [sys.executable, "-m", "quant_fund.cli.main", "verify-research", str(path)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def reverify_receipts(receipts_dir: Path, *, verify: VerifyFn | None = None) -> list[str]:
    """Re-verify every committed receipt in ``receipts_dir`` (A3 #4).

    Returns a list of failure descriptions; empty list means all receipts
    re-verify. Fail-closed: an empty receipts directory is itself a failure,
    and a verifier that raises counts as a failure, never a pass.
    """
    verifier: VerifyFn = verify if verify is not None else _cli_verifier
    paths = receipt_paths(receipts_dir)
    if not paths:
        return [f"no receipts found in {receipts_dir}"]
    failures: list[str] = []
    for path in paths:
        try:
            ok = verifier(path)
        except Exception as exc:  # fail-closed: verifier crash == failure
            failures.append(f"{path.name}: verifier raised {exc!r}")
            continue
        if not ok:
            failures.append(f"{path.name}: verification failed")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m quant_fund.proofcore.ci")
    sub = parser.add_subparsers(dest="command", required=True)
    cov = sub.add_parser("coverage-gate", help="per-package coverage floors (A3 #2)")
    cov.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    rev = sub.add_parser("receipts-reverify", help="re-verify committed receipts (A3 #4)")
    rev.add_argument("receipts_dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "coverage-gate":
        failures = coverage_gate(args.pyproject)
    else:
        failures = reverify_receipts(args.receipts_dir)

    if failures:
        for failure in failures:
            print(f"PROOFCORE GATE FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"PROOFCORE GATE OK: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
