"""WAVE2 §7.2 code-fingerprint fallback tests (W8).

``proofcore.ci.code_fingerprint`` returns the git revision inside a worktree;
when ``git rev-parse`` fails or returns nothing it falls back to a sha256
over ``src/quant_fund/**/*.py`` (sorted relative posix path + content bytes,
``__pycache__`` excluded), cached per process and deterministic across calls.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from quant_fund.proofcore import ci
from quant_fund.proofcore.contracts import ProofcoreError


def _failing_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
    """Simulate 'not a git worktree': git exits non-zero."""
    return subprocess.CompletedProcess(args=args, returncode=128, stdout="", stderr="fatal")


def _raising_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
    """Simulate git blowing up (e.g. unreadable worktree)."""
    raise OSError("git executable vanished")


def _empty_stdout_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


def _ok_runner(revision: str) -> object:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=f"{revision}\n", stderr="")

    return run


@pytest.fixture()
def src_tree(tmp_path: Path) -> Path:
    """A fake repo root with a minimal src/quant_fund tree."""
    pkg = tmp_path / "src" / "quant_fund"
    (pkg / "sub" / "__pycache__").mkdir(parents=True)
    (pkg / "a.py").write_text("A = 1\n")
    (pkg / "sub" / "b.py").write_text("B = 2\n")
    (pkg / "sub" / "__pycache__" / "cached.py").write_text("STALE = 3\n")
    return tmp_path


def test_fallback_when_git_fails(src_tree: Path) -> None:
    fp = ci.code_fingerprint(src_tree, runner=_failing_runner)
    assert len(fp) == 64  # sha256 hex, not a 40-hex git sha
    int(fp, 16)  # valid hex


def test_fallback_when_git_raises(src_tree: Path) -> None:
    fp = ci.code_fingerprint(src_tree, runner=_raising_runner)
    assert len(fp) == 64


def test_fallback_when_git_returns_nothing(src_tree: Path) -> None:
    fp = ci.code_fingerprint(src_tree, runner=_empty_stdout_runner)
    assert len(fp) == 64


def test_git_revision_used_inside_worktree(src_tree: Path) -> None:
    revision = "a" * 40
    fp = ci.code_fingerprint(src_tree, runner=_ok_runner(revision))
    assert fp == revision


def test_deterministic_across_calls_and_process_cached(src_tree: Path) -> None:
    calls = []

    def counting_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return _failing_runner(*args, **kwargs)

    fp1 = ci.code_fingerprint(src_tree, runner=counting_runner)
    fp2 = ci.code_fingerprint(src_tree, runner=counting_runner)
    fp3 = ci.code_fingerprint(src_tree, runner=_ok_runner("b" * 40))
    assert fp1 == fp2 == fp3  # process cache: deterministic, git not re-asked
    assert len(calls) == 1


def test_fingerprint_changes_when_a_src_file_changes(src_tree: Path) -> None:
    fp1 = ci.code_fingerprint(src_tree, runner=_failing_runner)
    (src_tree / "src" / "quant_fund" / "a.py").write_text("A = 42\n")
    ci._reset_code_fingerprint_cache()
    fp2 = ci.code_fingerprint(src_tree, runner=_failing_runner)
    assert fp1 != fp2


def test_fingerprint_changes_when_file_set_changes(src_tree: Path) -> None:
    before = ci.src_tree_sha256(src_tree / "src" / "quant_fund")
    (src_tree / "src" / "quant_fund" / "c.py").write_text("C = 3\n")
    after = ci.src_tree_sha256(src_tree / "src" / "quant_fund")
    assert before != after


def test_pycache_excluded(src_tree: Path) -> None:
    before = ci.src_tree_sha256(src_tree / "src" / "quant_fund")
    cached = src_tree / "src" / "quant_fund" / "sub" / "__pycache__" / "cached.py"
    cached.write_text("STALE = 999\n")
    (src_tree / "src" / "quant_fund" / "__pycache__").mkdir()
    (src_tree / "src" / "quant_fund" / "__pycache__" / "x.py").write_text("X = 1\n")
    after = ci.src_tree_sha256(src_tree / "src" / "quant_fund")
    assert before == after


def test_src_tree_hash_is_path_sensitive(src_tree: Path) -> None:
    """Same content under a different relative path -> different hash."""
    before = ci.src_tree_sha256(src_tree / "src" / "quant_fund")
    pkg = src_tree / "src" / "quant_fund"
    (pkg / "a.py").rename(pkg / "renamed.py")
    after = ci.src_tree_sha256(src_tree / "src" / "quant_fund")
    assert before != after


def test_missing_src_tree_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ProofcoreError, match="src tree not found"):
        ci.code_fingerprint(tmp_path, runner=_failing_runner)
    with pytest.raises(ProofcoreError, match="src tree not found"):
        ci.src_tree_sha256(tmp_path / "does-not-exist")


def test_cache_keyed_by_root(src_tree: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    other = tmp_path_factory.mktemp("other")
    pkg = other / "src" / "quant_fund"
    pkg.mkdir(parents=True)
    (pkg / "a.py").write_text("A = 1\n")  # same file content, one file fewer
    fp1 = ci.code_fingerprint(src_tree, runner=_failing_runner)
    fp2 = ci.code_fingerprint(other, runner=_failing_runner)
    assert fp1 != fp2
    # and each root stays cached independently
    assert ci.code_fingerprint(src_tree, runner=_failing_runner) == fp1
    assert ci.code_fingerprint(other, runner=_failing_runner) == fp2
