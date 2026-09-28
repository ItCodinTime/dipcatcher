"""Pin every ``except Exception`` handler under ``src/quant_fund`` per file.

The global ceiling in ``test_quality_ratchet.py`` only caps the total — a new
unjustified handler can trade against an unrelated narrowing and pass. This
manifest binds each handler to its file: ``quality/broad_exceptions.txt`` lists
``path N`` per file, and the check compares the exact per-file counts.

Regenerate after intentionally adding or removing a handler:

    python scripts/update_broad_exceptions.py

``--check`` (default) exits 1 on any mismatch; ``--write`` rewrites the
manifest. The harness prefers narrower catches; expect to justify each entry
in review.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "quality" / "broad_exceptions.txt"
SRC = ROOT / "src" / "quant_fund"

_BROAD = re.compile(r"\s*except\s+Exception\b")


def actual_counts(root: Path = SRC) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        n = sum(1 for line in path.read_text().splitlines() if _BROAD.match(line))
        if n:
            counts[path.relative_to(ROOT).as_posix()] = n
    return counts


def manifest_counts(path: Path = MANIFEST) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rel, _, count_s = stripped.rpartition(" ")
        try:
            count = int(count_s)
        except ValueError:
            raise SystemExit(f"{path}:{lineno}: malformed entry {raw!r}") from None
        if count < 1:
            raise SystemExit(f"{path}:{lineno}: non-positive count {raw!r}")
        if rel in counts:
            raise SystemExit(f"{path}:{lineno}: duplicate path {rel!r}")
        counts[rel] = count
    return counts


def render(counts: dict[str, int]) -> str:
    lines = [
        "# Per-file `except Exception` counts under src/quant_fund.",
        "# Exact pin: a handler added or removed in a file must update its",
        "# entry here — `python scripts/update_broad_exceptions.py`.",
        "# Global ceiling lives in tests/unit/test_quality_ratchet.py.",
    ]
    lines.extend(f"{path} {n}" for path, n in sorted(counts.items()))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="verify manifest matches the tree")
    group.add_argument("--write", action="store_true", help="rewrite the manifest from the tree")
    args = parser.parse_args(argv)

    actual = actual_counts()
    if args.write:
        MANIFEST.write_text(render(actual))
        print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(actual)} files)")
        return 0

    manifest = manifest_counts()
    errors: list[str] = []
    for path in sorted(set(actual) | set(manifest)):
        have, pinned = actual.get(path, 0), manifest.get(path, 0)
        if have != pinned:
            errors.append(f"  {path}: {have} handler(s), manifest pins {pinned}")
    if errors:
        print("broad-exception manifest is stale:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        print("regenerate: python scripts/update_broad_exceptions.py", file=sys.stderr)
        return 1
    print(
        f"broad-exception manifest is fresh ({sum(actual.values())} handlers, {len(actual)} files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
