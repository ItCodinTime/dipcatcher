"""Disk/RAM governor for the lab payload (resources.py)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.hedge_lab import resources


def test_lab_root_is_repo_root() -> None:
    assert resources.lab_root().is_dir()
    assert (resources.lab_root() / "pyproject.toml").is_file()


def test_payload_paths_cover_declared_dirs(tmp_path) -> None:
    paths = resources.payload_paths(tmp_path)
    names = {p.name for p in paths}
    assert {"data", "kronos_weights", "hedge_lab"} <= names
    (tmp_path / ".tmp_extra").mkdir()
    assert tmp_path / ".tmp_extra" in resources.payload_paths(tmp_path)


def test_directory_bytes_counts_files(tmp_path) -> None:
    assert resources.directory_bytes(tmp_path / "nope") == 0
    f = tmp_path / "f.bin"
    f.write_bytes(b"x" * 100)
    assert resources.directory_bytes(f) == 100
    sub = tmp_path / "d"
    sub.mkdir()
    (sub / "a").write_bytes(b"y" * 40)
    (sub / "b").write_bytes(b"z" * 60)
    assert resources.directory_bytes(sub) == 100


def test_payload_bytes_sums_payload_dirs(tmp_path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x").write_bytes(b"1" * 10)
    assert resources.payload_bytes(tmp_path) == 10


def test_assert_disk_budget_passes_and_reports() -> None:
    out = resources.assert_disk_budget(extra_bytes=1024)
    assert out["budget_bytes"] == resources.DISK_BUDGET_BYTES
    assert out["projected_bytes"] == out["used_bytes"] + 1024
    assert out["free_budget_bytes"] == out["budget_bytes"] - out["projected_bytes"]


def test_assert_disk_budget_raises_over_cap(tmp_path) -> None:
    with pytest.raises(OSError, match="disk budget"):
        resources.assert_disk_budget(extra_bytes=resources.DISK_BUDGET_BYTES + 1, root=tmp_path)


def test_physical_memory_positive() -> None:
    total, avail = resources.physical_memory()
    assert total > 0
    assert 0 < avail <= total


def test_ram_plan_shape() -> None:
    plan = resources.ram_plan()
    assert plan["headroom_bytes"] >= 8 * 1024**3
    assert 0 < plan["workspace_bytes"] <= plan["total_bytes"]
    assert plan["workspace_gib"] > 0


def test_claim_workspace_small_request() -> None:
    arr, stats = resources.claim_workspace(bytes_wanted=8 * 1024**2)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert stats["claimed_bytes"] == arr.nbytes
    assert stats["claimed_bytes"] <= stats["workspace_bytes"]
    del arr


def test_allocate_float64_workspace_shape() -> None:
    arr = resources.allocate_float64_workspace(4, 3)
    assert arr.shape == (4, 3)
    assert arr.dtype == np.float64


def test_cap_blas_threads_sets_env() -> None:
    n = resources.cap_blas_threads(0.5)
    import os

    assert n == max(1, int((os.cpu_count() or 1) * 0.5))
    assert os.environ["OMP_NUM_THREADS"] == str(n)
    assert os.environ["MKL_NUM_THREADS"] == str(n)


def test_cap_blas_threads_rejects_bad_fraction() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="fraction"):
            resources.cap_blas_threads(bad)


def test_ensure_dirs_creates_nested(tmp_path) -> None:
    a = tmp_path / "a" / "b"
    b = tmp_path / "c"
    resources.ensure_dirs([a, b])
    assert a.is_dir() and b.is_dir()
