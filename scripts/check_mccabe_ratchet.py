"""Fail when any function's McCabe complexity rises above the baseline.

Complexity is measured by ruff C901 (same engine as ``make lint``). The
baseline lists every function currently at or above ``SOFT_THRESHOLD``;
unlisted functions must stay below that soft ceiling. Baseline values may
only decrease — regenerate with ``--write`` after a complexity reduction.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "quality" / "mccabe_baseline.txt"
# New / unlisted functions must stay below this. Listed entries are pinned
# per-function and may only move down.
SOFT_THRESHOLD = 10
_RUFF_MSG = re.compile(r"^(.+) is too complex \((\d+) > (\d+)\)$")


def _qualified_names(path: Path) -> dict[int, str]:
    """Map def lineno → Class.method / outer.inner qualified name."""
    tree = ast.parse(path.read_text(), filename=str(path))
    out: dict[int, str] = {}

    def walk(nodes: list[ast.stmt], prefix: str = "") -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                walk(node.body, f"{prefix}{node.name}.")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{node.name}"
                out[node.lineno] = name
                walk(node.body, f"{name}.")

    walk(tree.body)
    return out


def _ruff_complexities(paths: list[str]) -> dict[tuple[str, str], int]:
    """Return {(repo-relative path, qualified_name): complexity} via ruff C901."""
    # Force every function above complexity 1 into the report so we can
    # compare against the soft threshold without depending on pyproject's
    # hard ceiling.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".toml", delete=False, encoding="utf-8"
    ) as handle:
        handle.write("[lint]\nselect = [\"C901\"]\n[lint.mccabe]\nmax-complexity = 1\n")
        config_path = handle.name
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                *paths,
                "--select",
                "C901",
                "--config",
                config_path,
                "--output-format",
                "json",
                "--no-cache",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        Path(config_path).unlink(missing_ok=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode or 1)
    raw = proc.stdout.strip() or "[]"
    try:
        diagnostics = json.loads(raw)
    except json.JSONDecodeError:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(1) from None

    measured: dict[tuple[str, str], int] = {}
    qmaps: dict[str, dict[int, str]] = {}
    for item in diagnostics:
        message = str(item.get("message", ""))
        match = _RUFF_MSG.match(message)
        if match is None:
            continue
        bare_name = match.group(1)
        complexity = int(match.group(2))
        filename = str(item["filename"]).replace("\\", "/")
        root = ROOT.as_posix()
        if filename.startswith(root + "/"):
            rel = filename[len(root) + 1 :]
        elif filename.startswith("./"):
            rel = filename[2:]
        else:
            rel = filename
        line = int(item["location"]["row"])
        if rel not in qmaps:
            qmaps[rel] = _qualified_names(ROOT / rel)
        qualified = qmaps[rel].get(line, bare_name)
        key = (rel, qualified)
        # Keep the max if ruff ever double-reports (should not happen).
        measured[key] = max(complexity, measured.get(key, 0))
    return measured


def _entries(path: Path) -> dict[tuple[str, str], int]:
    rows: dict[tuple[str, str], int] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or ":" not in parts[0]:
            print(f"malformed mccabe baseline line: {raw!r}", file=sys.stderr)
            raise SystemExit(1)
        loc, complexity_s = parts
        file_part, _, name = loc.partition(":")
        if not file_part or not name:
            print(f"malformed mccabe baseline location: {raw!r}", file=sys.stderr)
            raise SystemExit(1)
        try:
            complexity = int(complexity_s)
        except ValueError:
            print(f"malformed mccabe baseline complexity: {raw!r}", file=sys.stderr)
            raise SystemExit(1) from None
        key = (file_part, name)
        if key in rows:
            print(f"duplicate mccabe baseline key: {file_part}:{name}", file=sys.stderr)
            raise SystemExit(1)
        rows[key] = complexity
    return rows


def _format_baseline(rows: dict[tuple[str, str], int]) -> str:
    header = (
        "# McCabe (ruff C901) per-function ceilings. Values may only decrease.\n"
        f"# Format: <path>:<qualified_name> <max_complexity>\n"
        f"# Soft threshold for unlisted functions: {SOFT_THRESHOLD}\n"
        "# Regenerate after reductions: uv run python scripts/check_mccabe_ratchet.py --write\n"
    )
    lines = [
        f"{path}:{name} {complexity}"
        for (path, name), complexity in sorted(rows.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    return header + "\n".join(lines) + ("\n" if lines else "")


def current_above_threshold(
    measured: dict[tuple[str, str], int] | None = None,
) -> dict[tuple[str, str], int]:
    measured = measured if measured is not None else _ruff_complexities(["src"])
    return {key: value for key, value in measured.items() if value >= SOFT_THRESHOLD}


def check(measured: dict[tuple[str, str], int] | None = None) -> int:
    """Require the baseline to match current complexities (>= soft threshold).

    - A listed function whose complexity rose above its recorded value fails.
    - A new function at or above the soft threshold fails until reviewed.
    - Stale high baseline rows (complexity fell, or the function disappeared)
      also fail so the checked-in file must be lowered via ``--write``.
    """
    measured = measured if measured is not None else _ruff_complexities(["src"])
    baseline = _entries(BASELINE)
    current = current_above_threshold(measured)

    regressions: list[str] = []
    for key in sorted(baseline.keys() | current.keys()):
        path, name = key
        allowed = baseline.get(key)
        got = current.get(key)
        if allowed is None and got is not None:
            regressions.append(
                f"  {path}:{name} complexity {got} is new (>= {SOFT_THRESHOLD}); "
                "reduce it or add via --write after review"
            )
        elif got is None and allowed is not None:
            measured_now = measured.get(key)
            detail = (
                f"now {measured_now}"
                if measured_now is not None
                else "missing (renamed/removed)"
            )
            regressions.append(
                f"  {path}:{name} baseline={allowed} but {detail}; "
                "lower/remove via --write"
            )
        elif got is not None and allowed is not None and got > allowed:
            regressions.append(f"  {path}:{name} complexity {got} > baseline {allowed}")
        elif got is not None and allowed is not None and got < allowed:
            regressions.append(
                f"  {path}:{name} complexity {got} < baseline {allowed}; "
                "lower the baseline via --write"
            )

    expected = _format_baseline(baseline)
    if BASELINE.read_text() != expected:
        regressions.append("  quality/mccabe_baseline.txt must be sorted unique canonical form")

    if regressions:
        print("mccabe complexity ratchet failed:", file=sys.stderr)
        for line in regressions:
            print(line, file=sys.stderr)
        return 1
    print(
        f"mccabe ratchet ok ({len(baseline)} baselined functions, "
        f"soft threshold {SOFT_THRESHOLD})"
    )
    return 0


def write_baseline(measured: dict[tuple[str, str], int] | None = None) -> int:
    measured = measured if measured is not None else _ruff_complexities(["src"])
    current = current_above_threshold(measured)
    if BASELINE.exists():
        previous = _entries(BASELINE)
        raised = {
            key: (previous[key], current[key])
            for key in previous.keys() & current.keys()
            if current[key] > previous[key]
        }
        if raised:
            print("refusing to raise baseline values:", file=sys.stderr)
            for (path, name), (old, new) in sorted(raised.items()):
                print(f"  {path}:{name} {old} -> {new}", file=sys.stderr)
            return 1
        # New keys at/above the soft threshold are allowed on --write only as
        # an explicit acknowledgment (CI ``check`` still fails without it).
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(_format_baseline(current))
    print(f"wrote {BASELINE.relative_to(ROOT)} ({len(current)} functions)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite quality/mccabe_baseline.txt from the current tree "
        "(refuses to raise any recorded value)",
    )
    args = parser.parse_args(argv)
    measured = _ruff_complexities(["src"])
    if args.write:
        return write_baseline(measured)
    return check(measured)


if __name__ == "__main__":
    raise SystemExit(main())
