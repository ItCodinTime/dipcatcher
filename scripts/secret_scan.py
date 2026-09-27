"""Staged-diff secret scanner — zero-dependency pre-commit hook.

Scans lines ADDED in the staged diff for credential-shaped strings: known
provider token prefixes, private-key blocks, and ``key = "..."`` assignments
with high-entropy values. Deliberately does *not* flag raw high-entropy blobs
(this repo is full of sha256 digests and asset hashes that are legitimately
committed); it requires a credential context.

Exit 0 = clean. Exit 1 = a likely secret is being committed.
"""

from __future__ import annotations

import re
import subprocess
import sys

# Known credential shapes — prefix is unambiguous, length bounded loosely.
PROVIDER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("aws_secret_key", re.compile(r"(?i)aws(.{0,20})?['\"][0-9a-zA-Z/+=]{40}['\"]")),
    ("github_pat", re.compile(r"\bgithub_pat_[0-9a-zA-Z_]{22,}\b")),
    ("github_token", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[0-9a-zA-Z]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[0-9a-zA-Z-]{10,}\b")),
    ("openai_key", re.compile(r"\bsk-[0-9a-zA-Z]{20,}\b")),
    ("openai_proj_key", re.compile(r"\bsk-proj-[0-9a-zA-Z_-]{20,}\b")),
    (
        "private_key_block",
        re.compile(r"-----BEGIN (RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY( BLOCK)?-----"),
    ),
    ("generic_webhook_secret", re.compile(r"\bwhsec_[0-9a-zA-Z]{20,}\b")),
    ("stripe_key", re.compile(r"\b(sk|pk)_(live|test)_[0-9a-zA-Z]{16,}\b")),
    ("jwt", re.compile(r"\beyJ[0-9a-zA-Z_-]{10,}\.eyJ[0-9a-zA-Z_-]{10,}\.[0-9a-zA-Z_-]{10,}\b")),
]

# key-name context + quoted/assigned high-entropy value.
ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|auth[_-]?token"
    r"|private[_-]?key|client[_-]?secret|password|passwd|bearer)\s*[:=]\s*"
    r"['\"]([0-9a-zA-Z_\-/+=]{16,})['\"]"
)

PLACEHOLDER_HINTS = re.compile(
    r"(?i)(example|placeholder|dummy|changeme|your[_-]?|xxx+|\.\.\.|<[^>]+>|redacted|test[_-]?key)"
)


def _staged_added_lines() -> list[tuple[str, int, str]]:
    """(path, lineno, added-line-content) for staged changes, text files only."""
    proc = subprocess.run(
        ["git", "diff", "--cached", "--unified=0", "--diff-filter=ACMR", "--no-color"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        # Not a git repo or no staged area: nothing to scan.
        return []
    hits: list[tuple[str, int, str]] = []
    path = ""
    new_line = 0
    for line in proc.stdout.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            continue
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            new_line = int(m.group(1)) if m else 0
            continue
        if line.startswith("+"):
            hits.append((path, new_line, line[1:]))
            new_line += 1
        elif not line.startswith("-"):
            new_line += 1
    return hits


def _scan_line(path: str, lineno: int, content: str, findings: list[str]) -> None:
    if PLACEHOLDER_HINTS.search(content):
        return
    for name, pat in PROVIDER_PATTERNS:
        if pat.search(content):
            findings.append(f"{path}:{lineno}: {name}")
            return
    m = ASSIGNMENT.search(content)
    if m and not PLACEHOLDER_HINTS.search(m.group(2)):
        findings.append(f"{path}:{lineno}: credential_assignment({m.group(1)})")


def _skipped(path: str) -> bool:
    if path.endswith((".png", ".jpg", ".parquet", ".npz", ".zip", ".bin")):
        return True
    return path.endswith(".secrets.baseline") or path.endswith("secret_scan.py")


def _tracked_lines() -> list[tuple[str, int, str]]:
    """(path, lineno, line) for every tracked text file — the CI surface."""
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    lines: list[tuple[str, int, str]] = []
    for raw_path in proc.stdout.split(b"\0"):
        if not raw_path:
            continue
        path = raw_path.decode("utf-8", errors="replace")
        if _skipped(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                for lineno, content in enumerate(fh, 1):
                    lines.append((path, lineno, content))
        except (OSError, UnicodeDecodeError):
            continue
    return lines


def scan(*, tracked: bool = False) -> list[str]:
    findings: list[str] = []
    if tracked:
        candidates = _tracked_lines()
    else:
        candidates = [
            (path, lineno, content)
            for path, lineno, content in _staged_added_lines()
            if not _skipped(path)
        ]
    for path, lineno, content in candidates:
        _scan_line(path, lineno, content, findings)
    return findings


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    tracked = "--all" in args or "--tracked" in args
    findings = scan(tracked=tracked)
    if not findings:
        return 0
    scope = "tracked files" if tracked else "staged diff"
    print(f"secret-scan: refusing — likely credentials in {scope}:")
    for f in findings:
        print("  " + f)
    print(
        "If a finding is a deliberate placeholder/test vector, mark it with an"
        " obvious placeholder token (e.g. 'example', 'changeme') and re-stage."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
