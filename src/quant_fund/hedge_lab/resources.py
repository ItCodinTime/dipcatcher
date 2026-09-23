"""Disk and RAM governor for the Artificial Hedge fund lab.

Lab payload under ``D:/dipcatcher`` (data, Kronos weights, hedge-lab artifacts)
must stay ≤ 100 GiB. Process working set targets most of the 128 GiB machine
and always leaves headroom so Windows / Cursor do not OOM.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[3]
DISK_BUDGET_BYTES = 100 * 1024**3
RAM_HEADROOM_BYTES = 28 * 1024**3
RAM_TARGET_FRACTION = 0.72
PAYLOAD_RELATIVE = (
    "data",
    "third_party/kronos_weights",
    "artifacts/hedge_lab",
)


def lab_root() -> Path:
    return LAB_ROOT


def payload_paths(root: Path | None = None) -> list[Path]:
    base = root or lab_root()
    paths = [base / rel for rel in PAYLOAD_RELATIVE]
    paths.extend(sorted(base.glob(".tmp_*")))
    return paths


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    if path.is_file():
        return int(path.stat().st_size)
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            file_path = Path(dirpath) / name
            try:
                total += int(file_path.stat().st_size)
            except OSError:
                continue
    return total


def payload_bytes(root: Path | None = None) -> int:
    return int(sum(directory_bytes(path) for path in payload_paths(root)))


def assert_disk_budget(extra_bytes: int = 0, *, root: Path | None = None) -> dict[str, int | float]:
    used = payload_bytes(root)
    projected = used + max(int(extra_bytes), 0)
    if projected > DISK_BUDGET_BYTES:
        raise OSError(
            f"hedge-lab disk budget exceeded: projected {projected} bytes "
            f"> {DISK_BUDGET_BYTES} (100 GiB payload cap)"
        )
    return {
        "used_bytes": used,
        "projected_bytes": projected,
        "budget_bytes": DISK_BUDGET_BYTES,
        "free_budget_bytes": DISK_BUDGET_BYTES - projected,
        "used_gib": used / 1024**3,
    }


def physical_memory() -> tuple[int, int]:
    """Return ``(total_bytes, available_bytes)`` for physical RAM."""
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        windll = getattr(ctypes, "windll", None)
        if windll is not None and windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
    except Exception:
        pass
    page = int(os.sysconf("SC_PAGE_SIZE")) if hasattr(os, "sysconf") else 4096
    phys = int(os.sysconf("SC_PHYS_PAGES")) if hasattr(os, "sysconf") else 0
    total = page * phys
    return total, total


def ram_plan() -> dict[str, int | float]:
    total, available = physical_memory()
    headroom = min(RAM_HEADROOM_BYTES, max(total // 5, 8 * 1024**3))
    target = int(min(available, total - headroom) * RAM_TARGET_FRACTION)
    target = max(target, 512 * 1024**2)
    return {
        "total_bytes": int(total),
        "available_bytes": int(available),
        "headroom_bytes": int(headroom),
        "workspace_bytes": int(target),
        "total_gib": total / 1024**3,
        "available_gib": available / 1024**3,
        "workspace_gib": target / 1024**3,
    }


def claim_workspace(bytes_wanted: int | None = None) -> tuple[object, dict[str, int | float]]:
    """Commit a large float64 vector up to ``ram_plan`` without pinning the OS.

    Pages are touched so Windows actually charges the working set. Callers must
    drop the array when the lab step finishes.
    """
    import numpy as np

    plan = ram_plan()
    wanted = int(bytes_wanted) if bytes_wanted is not None else int(plan["workspace_bytes"])
    wanted = min(wanted, int(plan["workspace_bytes"]))
    n = max(wanted // 8, 1)
    last_error: BaseException | None = None
    arr = None
    while n >= (1024**2) // 8:
        try:
            candidate = np.empty(n, dtype=np.float64)
            stride = 512
            candidate[::stride] = 0.0
            candidate[-1] = 0.0
            arr = candidate
            break
        except (MemoryError, ValueError) as exc:
            last_error = exc
            n = max(n // 2, 1)
    if arr is None:
        raise MemoryError("could not claim hedge-lab RAM workspace") from last_error
    committed = int(arr.nbytes)
    stats = {
        **plan,
        "claimed_bytes": committed,
        "claimed_gib": committed / 1024**3,
        "n_float64": int(arr.size),
    }
    return arr, stats


def allocate_float64_workspace(n_rows: int, n_cols: int) -> object:
    """Allocate a large float64 workspace, shrinking on MemoryError."""
    import numpy as np

    rows = max(int(n_rows), 1)
    cols = max(int(n_cols), 1)
    last_error: BaseException | None = None
    for _ in range(8):
        try:
            return np.empty((rows, cols), dtype=np.float64)
        except (MemoryError, ValueError) as exc:
            last_error = exc
            rows = max(rows // 2, 1)
    raise MemoryError("could not allocate hedge-lab workspace") from last_error


def cap_blas_threads(fraction: float = 0.6) -> int:
    """Cap BLAS/OpenMP threads at ``fraction`` of logical CPUs (default 60%)."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    n_cpu = int(os.cpu_count() or 1)
    n = max(1, int(n_cpu * float(fraction)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
    except Exception:
        pass
    return n


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
