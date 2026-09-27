"""Session caches for repeated synthetic dataset builds and Northset benches.

The lab suite calls ``bench_northset`` and ``ensure_silver`` from hundreds of
tests with the same synthetic inputs. Those calls are pure with respect to
their inputs (the bench returns a dict; silver/gold materialization is a
function of the config, not of the destination directory). This module caches
them for one pytest session, across xdist workers, and invalidates the cache
when source code or locked dependencies change.

Monkeypatches bypass the cache: a fingerprint of the patched callables is
compared to the fingerprint taken when the wrappers were installed.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import shutil
import sys
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_LAKE_DIRS = ("bronze", "silver", "gold")
_FINGERPRINT_MODULES = (
    "quant_fund.microstructure.book_metrics",
    "quant_fund.microstructure.synthetic_lob",
    "quant_fund.northset.benches",
    "quant_fund.northset.sweep_research",
    "quant_fund.northset.kyle_ofi",
    "quant_fund.northset.estimators",
    "quant_fund.research.agent",
    "quant_fund.pipeline.dataset",
    "quant_fund.data.ingest",
)
_INSTALLED = False
_ORIGINAL_FINGERPRINT: tuple[tuple[str, str, int], ...] | None = None
_BENCH_STATS = {"hit": 0, "miss": 0, "bypass": 0}
_DATA_STATS = {"hit": 0, "miss": 0, "bypass": 0}
_BENCH_MEMORY: dict[str, bytes] = {}
# flock is not reentrant across two opens of the same lock file. This depth
# map lets a materialize hold the data-root lock while ingest, which takes
# that same lock, runs underneath it.
_LOCK_DEPTH: dict[str, int] = {}


def cache_disabled() -> bool:
    return os.environ.get("DIP_DISABLE_SESSION_CACHE") == "1"


def bench_stats() -> dict[str, int]:
    return dict(_BENCH_STATS)


def data_stats() -> dict[str, int]:
    return dict(_DATA_STATS)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _source_stamp() -> str:
    """Separate test runs with different code or locked dependencies."""
    root = _repo_root() / "src" / "quant_fund"
    digest = hashlib.sha256()
    digest.update(b"session-cache-v2\0")
    digest.update(sys.version.encode())
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for name in ("pyproject.toml", "uv.lock"):
        path = _repo_root() / name
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def cache_root() -> Path:
    override = os.environ.get("DIP_SESSION_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".cache" / "dipcatcher"
    path = base / "bench" / _source_stamp()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fingerprint() -> tuple[tuple[str, str, int], ...]:
    items: list[tuple[str, str, int]] = []
    for name in _FINGERPRINT_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        for attr, value in sorted(vars(module).items()):
            if attr.startswith("__") or not callable(value):
                continue
            items.append((name, attr, id(value)))
    return tuple(items)


def fingerprint_is_original() -> bool:
    return _ORIGINAL_FINGERPRINT is not None and _fingerprint() == _ORIGINAL_FINGERPRINT


@contextmanager
def _file_lock(path: Path):
    key = str(path)
    depth = _LOCK_DEPTH.get(key, 0)
    if depth:
        _LOCK_DEPTH[key] = depth + 1
        try:
            yield
        finally:
            _LOCK_DEPTH[key] = depth
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            # msvcrt locks bytes at the current offset. Ensure byte zero exists.
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        locked = True
        _LOCK_DEPTH[key] = 1
        try:
            yield
        finally:
            _LOCK_DEPTH[key] = 0
    finally:
        if locked:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _config_root(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Path:
    config = kwargs.get("config", args[0] if args else None)
    root = getattr(getattr(config, "data", None), "root", "data")
    return Path(root).resolve()


def _lock_for_root(root: Path) -> Path:
    digest = hashlib.sha256(str(root).encode()).hexdigest()[:24]
    return Path(tempfile.gettempdir()) / "dipcatcher-locks" / f"{digest}.lock"


def _wrap_root_lock(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        root = _config_root(args, kwargs)
        with _file_lock(_lock_for_root(root)):
            return fn(*args, **kwargs)

    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _is_synthetic(config: Any) -> bool:
    source = getattr(getattr(config, "data", None), "source", "")
    return str(source).casefold() == "synthetic"


def _config_key(config: Any) -> str:
    payload = config.model_dump(mode="json")
    data = payload.get("data")
    if isinstance(data, dict):
        data["root"] = "<root>"
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# Files a synthetic ingest / gold build actually writes. Restores copy these
# and leave every other path in the data root alone.
_SILVER_ARTIFACTS = (
    "bronze/bars.parquet",
    "bronze/corporate_actions.parquet",
    "bronze/security_master.parquet",
    "silver/bars.parquet",
    "silver/universe.parquet",
)
_GOLD_ARTIFACTS = (
    "gold/features.parquet",
    "gold/labels.parquet",
)


def _hash_files(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _input_token(root: Path) -> str:
    """Identity of silver inputs already on disk.

    An empty root shares a cache entry across directories. A root that already
    has silver is keyed by those bytes, so a hand-edited lake cannot be
    overwritten by the synthetic snapshot.
    """
    silver = (
        root / "silver" / "bars.parquet",
        root / "silver" / "universe.parquet",
    )
    if all(path.is_file() for path in silver):
        return _hash_files(silver)
    return "absent"


def _snapshot_artifacts(root: Path, dest: Path, rels: tuple[str, ...]) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel in rels:
        source = root / rel
        if not source.is_file():
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _restore_artifacts(src: Path, root: Path, rels: tuple[str, ...]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for rel in rels:
        source = src / rel
        if not source.is_file():
            continue
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _silver_ready(root: Path) -> bool:
    return (root / "silver" / "bars.parquet").is_file() and (
        root / "silver" / "universe.parquet"
    ).is_file()


def _pickle_ok(value: Any) -> bytes | None:
    try:
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.loads(blob)
    except Exception:
        return None
    return blob


def _install_dataset_cache() -> None:
    # `quant_fund.data.ingest` is shadowed by the function exported from
    # `quant_fund.data`, so a plain import returns that function.
    ingest_mod = importlib.import_module("quant_fund.data.ingest")
    import quant_fund.data as data_pkg
    import quant_fund.pipeline.dataset as dataset

    orig_ingest = ingest_mod.ingest
    orig_ensure = dataset.ensure_silver
    orig_build = dataset.build_gold
    orig_panel = dataset.panel

    def _slot(kind: str, key: str) -> Path:
        path = cache_root() / kind / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cached_materialize(kind: str, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            config = kwargs.get("config", args[0] if args else None)
            refresh = bool(kwargs.get("refresh", False))
            if (
                cache_disabled()
                or refresh
                or not fingerprint_is_original()
                or not _is_synthetic(config)
            ):
                _DATA_STATS["bypass"] += 1
                return orig(*args, **kwargs)
            root = _config_root(args, kwargs)
            rels = _SILVER_ARTIFACTS if kind == "silver" else _SILVER_ARTIFACTS + _GOLD_ARTIFACTS
            # Root lock first, then the cache slot. build_gold calls
            # ensure_silver while it already holds the root lock; taking the
            # slot lock second cannot deadlock with another worker.
            with _file_lock(_lock_for_root(root)):
                try:
                    key = _config_key(config) + _input_token(root)
                except Exception:
                    _DATA_STATS["bypass"] += 1
                    return orig(*args, **kwargs)
                slot = _slot(kind, key)
                with _file_lock(slot / ".lock"):
                    done = slot / "DONE"
                    blob_path = slot / "return.pkl"
                    lake = slot / "lake"
                    cached: Any = None
                    if done.is_file() and blob_path.is_file() and lake.is_dir():
                        try:
                            cached = pickle.loads(blob_path.read_bytes())
                        except Exception:
                            cached = None
                    if cached is not None:
                        _DATA_STATS["hit"] += 1
                        if kind == "silver":
                            if not _silver_ready(root):
                                _restore_artifacts(lake, root, _SILVER_ARTIFACTS)
                        else:
                            _restore_artifacts(lake, root, _GOLD_ARTIFACTS)
                            if not _silver_ready(root):
                                _restore_artifacts(lake, root, _SILVER_ARTIFACTS)
                        return cached
                    _DATA_STATS["miss"] += 1
                    result = orig(*args, **kwargs)
                    blob = _pickle_ok(result)
                    if blob is None:
                        return result
                    _snapshot_artifacts(root, lake, rels)
                    blob_path.write_bytes(blob)
                    done.write_text("ok\n", encoding="utf-8")
                    return result

        wrapper.__wrapped__ = orig  # type: ignore[attr-defined]
        wrapper.__name__ = getattr(orig, "__name__", kind)
        wrapper.__doc__ = orig.__doc__
        return wrapper

    locked_ingest = _wrap_root_lock(orig_ingest)
    ingest_mod.ingest = locked_ingest
    dataset.ingest = locked_ingest
    if getattr(data_pkg, "ingest", None) is orig_ingest:
        data_pkg.ingest = locked_ingest
    dataset.ensure_silver = _cached_materialize("silver", orig_ensure)
    dataset.build_gold = _cached_materialize("gold", orig_build)
    dataset.panel = _wrap_root_lock(orig_panel)


def _frame_digest(frame: Any) -> str:
    import io

    buffer = io.BytesIO()
    frame.write_ipc(buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _bench_key(bars: Any, config: Any) -> str:
    digest = hashlib.sha256()
    digest.update(_frame_digest(bars).encode())
    payload = config.model_dump(mode="json")
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode())
    book = getattr(getattr(config, "northset", None), "book_panel_path", None)
    if book:
        path = Path(str(book))
        digest.update(str(path).encode())
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def _install_bench_cache() -> None:
    import quant_fund.northset.benches as benches

    orig = benches.bench_northset

    def bench_northset(bars: Any, config: Any) -> dict[str, Any]:
        if cache_disabled() or not fingerprint_is_original():
            _BENCH_STATS["bypass"] += 1
            return orig(bars, config)
        try:
            key = _bench_key(bars, config)
        except Exception:
            _BENCH_STATS["bypass"] += 1
            return orig(bars, config)
        cached = _BENCH_MEMORY.get(key)
        if cached is not None:
            _BENCH_STATS["hit"] += 1
            return pickle.loads(cached)
        slot = cache_root() / "northset" / f"{key}.pkl"
        with _file_lock(slot.with_suffix(".lock")):
            cached = _BENCH_MEMORY.get(key)
            if cached is None and slot.is_file():
                cached = slot.read_bytes()
                _BENCH_MEMORY[key] = cached
            if cached is not None:
                _BENCH_STATS["hit"] += 1
                return pickle.loads(cached)
            _BENCH_STATS["miss"] += 1
            result = orig(bars, config)
            blob = _pickle_ok(result)
            if blob is None:
                return result
            slot.parent.mkdir(parents=True, exist_ok=True)
            temporary = slot.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_bytes(blob)
            temporary.replace(slot)
            _BENCH_MEMORY[key] = blob
            return pickle.loads(blob)

    bench_northset.__wrapped__ = orig  # type: ignore[attr-defined]
    bench_northset.__name__ = orig.__name__
    bench_northset.__doc__ = orig.__doc__
    benches.bench_northset = bench_northset


def install() -> None:
    """Install wrappers. Idempotent; safe to call from conftest import."""
    global _INSTALLED, _ORIGINAL_FINGERPRINT
    if _INSTALLED or cache_disabled():
        return
    # Wrap dataset before importing callers (research.agent, the CLI) so their
    # `from dataset import ensure_silver` bindings see the wrappers.
    _install_dataset_cache()
    _install_bench_cache()
    for name in _FINGERPRINT_MODULES:
        __import__(name)
    _ORIGINAL_FINGERPRINT = _fingerprint()
    _INSTALLED = True


def reset_for_tests() -> None:
    """Clear in-process stats. Does not delete the on-disk cache."""
    _BENCH_STATS["hit"] = _BENCH_STATS["miss"] = _BENCH_STATS["bypass"] = 0
    _DATA_STATS["hit"] = _DATA_STATS["miss"] = _DATA_STATS["bypass"] = 0
    _BENCH_MEMORY.clear()
