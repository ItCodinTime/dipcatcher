"""Fail when an allowlisted module regresses under ``mypy --strict``.

Modules absent from ``quality/mypy_strict_modules.txt`` may still report
strict errors. The list only grows; ``tests/unit/test_quality_ratchet.py``
locks the floor.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "quality" / "mypy_strict_modules.txt"


def allowlisted() -> list[str]:
    lines = [
        line.strip()
        for line in ALLOWLIST.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if lines != sorted(set(lines)):
        print("mypy strict allowlist must be unique and sorted", file=sys.stderr)
        raise SystemExit(1)
    missing = [line for line in lines if not (ROOT / line).is_file()]
    if missing:
        print("allowlist paths missing:", file=sys.stderr)
        for line in missing:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(1)
    return lines


def main() -> int:
    modules = set(allowlisted())
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "src/quant_fund"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    if proc.returncode not in (0, 1):
        sys.stderr.write(output)
        return proc.returncode or 1
    root = ROOT.as_posix()
    offenders: set[str] = set()
    for line in output.splitlines():
        if ": error:" not in line:
            continue
        path = line.split(":", 1)[0].replace("\\", "/")
        if path.startswith(root + "/"):
            path = path[len(root) + 1 :]
        if path in modules:
            offenders.add(path)
    if offenders:
        print("mypy --strict regressions in allowlisted modules:", file=sys.stderr)
        for path in sorted(offenders):
            print(f"  {path}", file=sys.stderr)
        return 1
    print(f"mypy strict allowlist ok ({len(modules)} modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
