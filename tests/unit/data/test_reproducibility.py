"""Worktree fingerprint hashes bytes, including through Git LFS."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from quant_fund.utils.hashing import hash_bytes
from quant_fund.utils.reproducibility import git_worktree_sha256


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "reproducibility@example.com")
    _git(repo, "config", "user.name", "reproducibility")
    _git(repo, "config", "commit.gpgsign", "false")


def test_worktree_fingerprint_tracks_real_bytes_not_lfs_clean_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An LFS clean filter must not turn an unmodified blob into UNKNOWN.

    ``git diff HEAD --binary`` on a non-pointer blob with ``filter=lfs`` emits
    the filtered object. After the stat cache is invalidated (``touch``, or a
    racy checkout), that capture exceeds the old 5s timeout on the large
    parquet/npy files and the fingerprint collapses to ``UNKNOWN``. Phase-1
    index verification then rejects ``git_worktree_sha256``.
    """
    lfs = shutil.which("git-lfs")
    if lfs is None:
        pytest.skip("git-lfs is required to reproduce the clean-filter false dirty")
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / ".gitattributes").write_text("payload.bin filter=lfs -text\n", encoding="utf-8")
    payload = repo / "payload.bin"
    payload.write_bytes(b"same-bytes")
    _git(repo, "add", "--", ".gitattributes")
    # Store the raw bytes. A normal ``git add`` would run the LFS clean filter
    # and persist a pointer, which is not what this checkout contains.
    _git(
        repo,
        "-c",
        "filter.lfs.clean=",
        "-c",
        "filter.lfs.smudge=",
        "-c",
        "filter.lfs.process=",
        "-c",
        "filter.lfs.required=false",
        "add",
        "--",
        "payload.bin",
    )
    _git(repo, "commit", "-qm", "init")
    _git(repo, "config", "filter.lfs.process", "git-lfs filter-process")
    _git(repo, "config", "filter.lfs.required", "true")
    # Matching stat information skips the filter. Touching forces the recheck
    # that a same-second checkout leaves racy.
    payload.touch()
    false_dirty = subprocess.run(
        ["git", "diff", "HEAD", "--binary"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    assert false_dirty.stdout, "LFS clean filter did not false-dirty the blob"

    monkeypatch.chdir(repo)
    fingerprint = git_worktree_sha256()
    assert fingerprint == hash_bytes(b"")
    assert len(fingerprint) == 64

    payload.write_bytes(b"same-bytes!")
    changed = git_worktree_sha256()
    assert changed != fingerprint
    assert len(changed) == 64
    assert all(char in "0123456789abcdef" for char in changed)


def _repo_with_file(tmp_path: Path, content: bytes) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    _init_repo(repo)
    target = repo / "f"
    target.write_bytes(content)
    _git(repo, "add", "--", "f")
    _git(repo, "commit", "-qm", "init")
    return repo, target


def _raw_diff(repo: Path) -> bytes:
    return subprocess.run(
        [
            "git",
            "-c",
            "filter.lfs.clean=",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.required=false",
            "diff",
            "--raw",
            "--no-renames",
            "HEAD",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def test_clean_worktree_fingerprint_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _target = _repo_with_file(tmp_path, b"A")
    monkeypatch.chdir(repo)
    first = git_worktree_sha256()
    second = git_worktree_sha256()
    assert first == second == hash_bytes(b"")
    assert len(first) == 64


def test_distinct_unstaged_edits_have_distinct_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``git diff --raw`` prints worktree oid ``0000000`` for every unstaged edit.

    Editing A→B and A→C therefore produces the same raw line. The fingerprint
    has to hash the worktree bytes, including bytes past the first read chunk.
    """
    repo, target = _repo_with_file(tmp_path, b"A")
    monkeypatch.chdir(repo)
    clean = git_worktree_sha256()

    target.write_bytes(b"B")
    raw_b = _raw_diff(repo)
    edited_b = git_worktree_sha256()
    target.write_bytes(b"C")
    raw_c = _raw_diff(repo)
    edited_c = git_worktree_sha256()

    assert b"0000000" in raw_b
    assert raw_b == raw_c
    assert edited_b != edited_c
    assert edited_b != clean
    assert edited_c != clean

    past_chunk_x = b"B" * (1 << 20) + b"X"
    past_chunk_y = b"B" * (1 << 20) + b"Y"
    target.write_bytes(past_chunk_x)
    digest_x = git_worktree_sha256()
    target.write_bytes(past_chunk_y)
    digest_y = git_worktree_sha256()
    assert digest_x != digest_y


def test_deleted_file_changes_worktree_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, target = _repo_with_file(tmp_path, b"A")
    monkeypatch.chdir(repo)
    clean = git_worktree_sha256()
    target.unlink()
    deleted = git_worktree_sha256()
    assert deleted != clean
    assert len(deleted) == 64


def test_smudged_lfs_file_matching_pointer_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filter-off diff lists a smudged LFS file; a matching oid is still clean."""
    if shutil.which("git-lfs") is None:
        pytest.skip("git-lfs is required to smudge a real pointer")
    repo = tmp_path / "repo"
    _init_repo(repo)
    _git(repo, "config", "filter.lfs.process", "git-lfs filter-process")
    _git(repo, "config", "filter.lfs.required", "true")
    (repo / ".gitattributes").write_text("payload.bin filter=lfs -text\n", encoding="utf-8")
    payload = repo / "payload.bin"
    original = b"hello-lfs-content-not-a-pointer"
    payload.write_bytes(original)
    _git(repo, "add", "--", ".gitattributes", "payload.bin")
    _git(repo, "commit", "-qm", "init")
    pointer = _git(repo, "cat-file", "blob", "HEAD:payload.bin").stdout
    assert pointer.startswith(b"version https://git-lfs.github.com/spec/v1\n")
    assert payload.read_bytes() == original
    dirty = subprocess.run(
        [
            "git",
            "-c",
            "filter.lfs.clean=",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.process=",
            "-c",
            "filter.lfs.required=false",
            "diff",
            "--name-status",
            "-z",
            "--no-renames",
            "HEAD",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    assert b"payload.bin" in dirty.stdout

    monkeypatch.chdir(repo)
    assert git_worktree_sha256() == hash_bytes(b"")
    payload.write_bytes(original + b"!")
    assert git_worktree_sha256() != hash_bytes(b"")
