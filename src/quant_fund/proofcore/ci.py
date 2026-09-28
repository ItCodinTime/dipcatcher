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
import hashlib
import importlib.metadata
import platform
import shutil
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

    Fail-closed: missing table, missing REQUIRED package entry, a non-int
    entry, or a floor below the global 80% ratchet raises ``ProofcoreError``.
    Dotted keys (``proof.runner``) are per-MODULE floors (wave 2), enforced
    alongside the per-package floors; both are raise-never-lower.
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
    for name, floor in table.items():
        if not isinstance(name, str) or not isinstance(floor, int) or isinstance(floor, bool):
            raise ProofcoreError(f"[tool.proofcore.coverage-floors] bad entry: {name!r}={floor!r}")
        if floor < 80:
            raise ProofcoreError(
                f"coverage floor for {name} is {floor}, below the global 80% ratchet — "
                "floors are raise-never-lower (A3 #7)"
            )
        floors[name] = floor
    for pkg in REQUIRED_FLOOR_PACKAGES:
        if pkg not in floors:
            raise ProofcoreError(f"[tool.proofcore.coverage-floors] is missing an int entry: {pkg}")
    return floors


def _include_pattern(src_root: str, name: str) -> str:
    """Coverage ``--include`` glob: package keys match the tree, dotted keys
    (wave-2 module floors) match exactly one module file."""
    if "." in name:
        return f"{src_root}/{name.replace('.', '/')}.py"
    return f"{src_root}/{name}/*"


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
                f"--include={_include_pattern(src_root, pkg)}",
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


# ---------------------------------------------------------------------------
# WAVE2 §7.2: code-fingerprint fallback outside git worktrees.
# ---------------------------------------------------------------------------

# Repo root derived from this file: src/quant_fund/proofcore/ci.py.
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]

# Process cache: resolved repo root -> fingerprint. Fingerprinting walks the
# whole src tree; caching keeps repeat calls (runner + replay in one process)
# cheap and deterministic.
_FINGERPRINT_CACHE: dict[Path, str] = {}


def _git_revision(root: Path, *, runner: RunFn = subprocess.run) -> str | None:
    """``git rev-parse HEAD`` at ``root``; None when no worktree is available.

    Any failure — git missing, non-zero exit, subprocess error, or empty
    stdout — means "not in a git worktree" and triggers the src-tree fallback.
    """
    git = shutil.which("git")
    if git is None:
        return None
    try:
        proc = runner(
            [git, "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    revision = (proc.stdout or "").strip()
    return revision or None


def src_tree_sha256(src_root: Path) -> str:
    """sha256 over ``src_root/**/*.py`` (WAVE2 §7.2 fallback fingerprint).

    Files are hashed in sorted relative-posix-path order as
    ``relpath`` + NUL + content bytes + NUL; ``__pycache__`` directories are
    excluded. Deterministic for identical trees. Fail-closed: a missing
    ``src_root`` raises ``ProofcoreError``.
    """
    src_root = Path(src_root)
    if not src_root.is_dir():
        raise ProofcoreError(f"src tree not found for fingerprint fallback: {src_root}")
    digest = hashlib.sha256()
    paths = [
        path
        for path in src_root.rglob("*.py")
        if "__pycache__" not in path.relative_to(src_root).parts
    ]
    for path in sorted(paths, key=lambda p: p.relative_to(src_root).as_posix()):
        digest.update(path.relative_to(src_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def code_fingerprint(
    root: Path | None = None,
    *,
    runner: RunFn = subprocess.run,
) -> str:
    """Fingerprint of the quant_fund code (WAVE2 §7.2).

    The git revision (40-hex) when ``root`` is inside a git worktree; else a
    sha256 over ``<root>/src/quant_fund/**/*.py`` (sorted relative posix path
    + content bytes, ``__pycache__`` excluded). Cached per process (keyed by
    resolved root), so repeat calls are deterministic and cheap.
    """
    root = Path(root) if root is not None else _DEFAULT_REPO_ROOT
    key = root.resolve()
    cached = _FINGERPRINT_CACHE.get(key)
    if cached is not None:
        return cached
    revision = _git_revision(key, runner=runner)
    fingerprint = revision if revision is not None else src_tree_sha256(key / "src" / "quant_fund")
    _FINGERPRINT_CACHE[key] = fingerprint
    return fingerprint


def _reset_code_fingerprint_cache() -> None:
    """Drop the process cache (test helper; production code never calls this)."""
    _FINGERPRINT_CACHE.clear()


# ---------------------------------------------------------------------------
# WAVE2 §2.2: environment fingerprint helper (integration reconciliation).
# ---------------------------------------------------------------------------


def quant_fund_version() -> str:
    """The installed distribution version (== ``quant_fund.__version__``, which
    hatch sources from the same ``fx1.__version__``), or ``"dev"``.

    Probed via importlib.metadata because the §1.3 layering gate forbids
    proofcore from importing the quant_fund root package. A missing
    distribution (e.g. PYTHONPATH=src without install) degrades to ``"dev"``;
    both runner and replay probe through THIS helper, so the env gate always
    compares like with like on a given machine.
    """
    try:
        return importlib.metadata.version("fx-1")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def env_fingerprint() -> str:
    """Contracts §2.2 env gate string: ``platform|python version tag|quant_fund version``.

    Single canonical implementation (integration amendment): the proven
    runner (``proof.runner``) mints it into trace/env sidecars and the replay
    engine (``proof.replay``) re-derives it for the env gate. Both sides MUST
    use this helper so the gate compares like with like.
    """
    return f"{platform.platform()}|{platform.python_version()}|{quant_fund_version()}"


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
