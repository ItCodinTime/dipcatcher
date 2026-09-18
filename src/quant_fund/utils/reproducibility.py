"""Shared reproducibility fingerprints for research receipts and gates."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from quant_fund.utils.hashing import hash_bytes


def git_worktree_sha256() -> str:
    """Hash tracked and relevant untracked changes in the current checkout."""
    git = shutil.which("git")
    if git is None:
        return "UNKNOWN"
    try:
        diff = subprocess.run(
            [git, "diff", "HEAD", "--binary"],
            check=True,
            capture_output=True,
            timeout=5,
        ).stdout
        untracked = subprocess.run(
            [git, "ls-files", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
            timeout=5,
        ).stdout
        ignored_prefixes = (
            ".venv/",
            ".pytest_cache/",
            "data/metadata/research/",
            "data/metadata/paper/",
            "htmlcov/",
        )
        additions = bytearray()
        for raw_path in untracked.split(b"\0"):
            if not raw_path:
                continue
            path = raw_path.decode("utf-8")
            if path.startswith(ignored_prefixes):
                continue
            file_path = Path(path)
            if file_path.is_file():
                additions.extend(raw_path)
                additions.extend(b"\0")
                additions.extend(file_path.read_bytes())
        return hash_bytes(diff + bytes(additions))
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"
