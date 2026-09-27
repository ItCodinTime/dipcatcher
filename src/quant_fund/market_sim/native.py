"""Compile and load the single-threaded matching core.

The core is C, built on first use with the system compiler and cached
outside the repo. Simulation only: it does not submit orders to a broker.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import cast

_LIB: ctypes.CDLL | None = None
_SOURCE = Path(__file__).with_name("_lob_core.c")


class TradeC(ctypes.Structure):
    _fields_ = [
        ("ts", ctypes.c_int64),
        ("price", ctypes.c_int32),
        ("qty", ctypes.c_int32),
        ("aggressor_side", ctypes.c_int32),
        ("taker_agent", ctypes.c_int32),
        ("maker_agent", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
        ("taker_order_id", ctypes.c_int64),
        ("maker_order_id", ctypes.c_int64),
    ]


class EventC(ctypes.Structure):
    _fields_ = [
        ("ts", ctypes.c_int64),
        ("type", ctypes.c_int32),
        ("side", ctypes.c_int32),
        ("price", ctypes.c_int32),
        ("qty", ctypes.c_int32),
        ("agent", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
        ("order_id", ctypes.c_int64),
    ]


class EventResultC(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("reason", ctypes.c_int32),
        ("filled_qty", ctypes.c_int32),
        ("resting_qty", ctypes.c_int32),
        ("n_trades", ctypes.c_int32),
        ("truncated", ctypes.c_int32),
        ("order_id", ctypes.c_int64),
        ("filled_notional", ctypes.c_int64),
    ]


class UncrossResultC(ctypes.Structure):
    _fields_ = [
        ("price", ctypes.c_int32),
        ("n_trades", ctypes.c_int32),
        ("n_moo_cancelled", ctypes.c_int32),
        ("truncated", ctypes.c_int32),
        ("status", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


class AuditC(ctypes.Structure):
    _fields_ = [
        ("bid_qty", ctypes.c_int64),
        ("ask_qty", ctypes.c_int64),
        ("trade_qty", ctypes.c_int64),
        ("trade_notional", ctypes.c_int64),
        ("checksum", ctypes.c_int64),
        ("n_bid_orders", ctypes.c_int32),
        ("n_ask_orders", ctypes.c_int32),
        ("n_live", ctypes.c_int32),
        ("best_bid", ctypes.c_int32),
        ("best_ask", ctypes.c_int32),
        ("crossed", ctypes.c_int32),
        ("halted", ctypes.c_int32),
        ("n_trades", ctypes.c_int32),
        ("list_ok", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


class BenchResultC(ctypes.Structure):
    _fields_ = [
        ("n_events", ctypes.c_int64),
        ("n_trades", ctypes.c_int64),
        ("checksum", ctypes.c_int64),
        ("elapsed_match_ns", ctypes.c_int64),
        ("elapsed_generate_ns", ctypes.c_int64),
        ("list_ok", ctypes.c_int32),
        ("n_rejects", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


def _cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME")
    base = Path(root) if root else Path.home() / ".cache"
    path = base / "dipcatcher"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compile(source: Path, dest: Path) -> None:
    compiler = os.environ.get("LOB_CORE_CC", "cc")
    cmd = [
        compiler,
        "-O3",
        "-std=c11",
        "-fPIC",
        "-shared",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-o",
        str(dest),
        str(source),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "matching core failed to compile\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def _bind(lib: ctypes.CDLL) -> None:
    lib.lob_version.restype = ctypes.c_char_p
    lib.lob_price_max.restype = ctypes.c_int32
    lib.lob_max_orders.restype = ctypes.c_int32
    for name in (
        "lob_sizeof_trade",
        "lob_sizeof_event",
        "lob_sizeof_result",
        "lob_sizeof_uncross",
        "lob_sizeof_audit",
        "lob_sizeof_bench",
    ):
        getattr(lib, name).restype = ctypes.c_int32
    lib.lob_new.restype = ctypes.c_void_p
    lib.lob_free.argtypes = [ctypes.c_void_p]
    lib.lob_halt.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.lob_order_qty.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.lob_order_qty.restype = ctypes.c_int32
    lib.lob_touch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.lob_depth.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
    lib.lob_depth.restype = ctypes.c_int64
    lib.lob_level_qty.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
    lib.lob_level_qty.restype = ctypes.c_int64
    lib.lob_trades.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
    lib.lob_trades.restype = ctypes.POINTER(TradeC)
    lib.lob_submit.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(EventC),
        ctypes.POINTER(EventResultC),
    ]
    lib.lob_uncross.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.POINTER(UncrossResultC),
    ]
    lib.lob_audit.argtypes = [ctypes.c_void_p, ctypes.POINTER(AuditC)]
    lib.lob_bench.argtypes = [ctypes.c_int64, ctypes.c_uint64, ctypes.POINTER(BenchResultC)]
    lib.lob_bench.restype = ctypes.c_int

    expected = {
        "lob_sizeof_trade": ctypes.sizeof(TradeC),
        "lob_sizeof_event": ctypes.sizeof(EventC),
        "lob_sizeof_result": ctypes.sizeof(EventResultC),
        "lob_sizeof_uncross": ctypes.sizeof(UncrossResultC),
        "lob_sizeof_audit": ctypes.sizeof(AuditC),
        "lob_sizeof_bench": ctypes.sizeof(BenchResultC),
    }
    for name, size in expected.items():
        got = int(getattr(lib, name)())
        if got != size:
            raise RuntimeError(f"{name} is {got} bytes in C and {size} in Python")


def load_library() -> ctypes.CDLL:
    """Return the cached matching library, compiling it if needed."""
    global _LIB
    if _LIB is not None:
        return _LIB
    source = _SOURCE
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    dest = _cache_dir() / f"lob_core_{digest}.so"
    if not dest.is_file():
        fd, tmp_name = tempfile.mkstemp(prefix="lob_core_", suffix=".so", dir=dest.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            _compile(source, tmp)
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink()
    lib = ctypes.CDLL(str(dest))
    _bind(lib)
    _LIB = lib
    return lib


def compiler_command() -> str:
    """Compiler and flags used for the cached core. Methodology, not a benchmark."""
    compiler = os.environ.get("LOB_CORE_CC", "cc")
    return f"{compiler} -O3 -std=c11 -fPIC -shared -Wall -Wextra -Werror"


def core_version() -> str:
    """Version string compiled into the matching core."""
    lib = load_library()
    raw = lib.lob_version()
    if isinstance(raw, bytes):
        return raw.decode()
    return str(raw)


def matching_benchmark(n_events: int = 1_000_000, seed: int = 1) -> dict[str, object]:
    """Time the single-threaded matching core.

    Pass 1 builds a deterministic event tape and matches it. Pass 2 replays
    that tape. The clock covers pass 2 only, after the order arena has been
    prefaulted and the opening ladder posted. ``generate_and_match_events_per_s``
    is pass 1 and includes event generation. A non-zero ``rc`` means the
    replay checksum disagreed or the arguments were rejected.
    """
    if isinstance(n_events, bool) or not isinstance(n_events, int):
        raise ValueError("n_events must be an int")
    if n_events < 1 or n_events > 20_000_000:
        raise ValueError("n_events must be in [1, 20000000]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative int")
    lib = load_library()
    out = BenchResultC()
    rc = cast(int, lib.lob_bench(int(n_events), ctypes.c_uint64(int(seed)), ctypes.byref(out)))
    match_ns = int(out.elapsed_match_ns)
    gen_ns = int(out.elapsed_generate_ns)
    events = int(out.n_events) if int(out.n_events) else int(n_events)
    match_rate = (events / (match_ns / 1e9)) if match_ns > 0 else 0.0
    gen_rate = (events / (gen_ns / 1e9)) if gen_ns > 0 else 0.0
    return {
        "rc": int(rc),
        "n_events": events,
        "n_trades": int(out.n_trades),
        "checksum": int(out.checksum),
        "list_ok": bool(out.list_ok),
        "n_rejects": int(out.n_rejects),
        "elapsed_match_ns": match_ns,
        "elapsed_generate_ns": gen_ns,
        "match_events_per_s": float(match_rate),
        "generate_and_match_events_per_s": float(gen_rate),
        "compiler": compiler_command(),
        "timed_region": "pass-2 lob_submit replay after arena prefault and opening ladder",
        "mix": "15% market, 15% aggressive limit, 30% cancel-at-touch, 40% passive limit",
        "seed": int(seed),
    }
