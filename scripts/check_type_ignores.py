"""Pin every ``type: ignore`` suppression under ``src/`` per file.

A global cap on the suppression count can pass while the distribution drifts:
a new escape in module A trades against an unrelated cleanup in module B.
``quality/type_ignores.txt`` binds each suppression to its file —
``path N`` per file — and this check compares the exact per-file counts.

Regenerate after intentionally adding or removing a suppression:

    python scripts/update_type_ignores.py

``--check`` (default) exits 1 on any mismatch; ``--write`` rewrites the
manifest. Prefer a real type fix over a suppression; expect to justify each
entry in review.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "quality" / "type_ignores.txt"
SRC = ROOT / "src"

_IGNORE = re.compile(r"#\s*type:\s*ignore\b")


def actual_counts(root: Path = SRC) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        n = sum(1 for line in path.read_text().splitlines() if _IGNORE.search(line))
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
        if count <= 0:
            raise SystemExit(f"{path}:{lineno}: non-positive count {count}")
        if rel in counts:
            raise SystemExit(f"{path}:{lineno}: duplicate path {rel!r}")
        counts[rel] = count
    return counts


def render(counts: dict[str, int]) -> str:
    lines = [
        "# Per-file `type: ignore` suppression manifest for src/.",
        "# Drift fails `test_type_ignore_manifest_matches_tree`; regen with:",
        "#   python scripts/update_type_ignores.py",
        "# Prefer a real type fix over adding a suppression.",
    ]
    lines += [f"{path} {n}" for path, n in sorted(counts.items())]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail on drift (default)")
    mode.add_argument("--write", action="store_true", help="regenerate the manifest")
    args = parser.parse_args(argv)

    actual = actual_counts()
    if args.write:
        MANIFEST.write_text(render(actual))
        total = sum(actual.values())
        print(f"wrote {MANIFEST}: {len(actual)} files, {total} suppressions")
        return 0

    expected = manifest_counts()
    if expected == actual:
        total = sum(actual.values())
        print(f"type-ignore manifest is fresh: {len(actual)} files, {total} suppressions")
        return 0
    missing = sorted(set(actual) - set(expected))
    extra = sorted(set(expected) - set(actual))
    drifted = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
    for path in missing:
        print(f"new suppressions unpinned: {path} ({actual[path]})", file=sys.stderr)
    for path in extra:
        print(f"manifest entry stale (file has none now): {path}", file=sys.stderr)
    for path in drifted:
        print(
            f"count drifted: {path} manifest={expected[path]} actual={actual[path]}",
            file=sys.stderr,
        )
    print(
        "run `python scripts/update_type_ignores.py` if the change is intentional", file=sys.stderr
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
