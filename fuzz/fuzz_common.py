"""Shared fuzz-driver utilities for the dipcatcher fuzz campaign.

Two modes, selected automatically:

- **atheris mode** (preferred): when `atheris` is importable (Linux CI ships
  its wheel; see ``[dependency-groups] dev`` in ``pyproject.toml``) the harness
  is handed to libFuzzer via ``atheris.Setup``/``atheris.Fuzz`` and the usual
  libFuzzer flags apply (``-max_total_time=N``, corpus dirs,
  ``-artifact_prefix=...``).

- **fallback mode**: everywhere else (notably macOS arm64, where atheris needs
  a non-Apple Clang toolchain) a bounded mutational runner iterates a seeded
  corpus, applies deterministic byte-level mutations, and records any
  *unexpected* exception as a crash artifact under ``fuzz/artifacts/<name>/``.

Contract for every ``test_one_input(data: bytes)``: swallow the exceptions
declared in ``EXPECTED`` (documented rejection paths); let anything else
propagate — in atheris mode that IS the crash signal, in fallback mode the
runner records it and keeps going.

Usage per harness::

    uv run --no-sync python fuzz/fuzz_stooq_csv.py --max-total-time 60
    uv run --no-sync python fuzz/fuzz_stooq_csv.py fuzz/corpus/stooq_csv \
        -max_total_time=60            # atheris-style flags also accepted
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path

FUZZ_ROOT = Path(__file__).resolve().parent
REPO_ROOT = FUZZ_ROOT.parent
CORPUS_ROOT = FUZZ_ROOT / "corpus"
ARTIFACT_ROOT = FUZZ_ROOT / "artifacts"

# Mutator vocabulary: bytes that historically break parsers (format
# metacharacters, float edge spellings, JSON/YAML tokens, path tricks).
INTERESTING: tuple[bytes, ...] = (
    b"",
    b"\x00",
    b"\xff" * 8,
    b"0",
    b"-1",
    b"1e309",
    b"-1e309",
    b"1e-324",
    b"NaN",
    b"nan",
    b"inf",
    b"-inf",
    b"Infinity",
    b"null",
    b"NULL",
    b"true",
    b"false",
    b'"',
    b"'",
    b'"' * 16,
    b",,,,,,",
    b"\n\n\n\n",
    b"\r\n",
    b"\xef\xbb\xbf",  # BOM
    b"{{{{",
    b"[[[[",
    b"{{" + b'"a":' * 32,
    b"- ",
    b"*a",
    b"&a",
    b"<<:",
    b"!python/object",
    b"../../../etc/passwd",
    b"%Y-%m-%d",
    b"9999-12-31",
    b"1970-01-01",
    b"2262-04-12",  # ns timestamp overflow boundary
    b"1677-09-21",
    b"Z",
    b"+00:00",
    b"24:00:00",
    b"receipt_sha256",
    b"would_promote_live",
    b"live_pnl_claim",
    b"\xe2\x98\x83",  # snowman
    b"\xf0\x9f\x92\xa5",
    b"a" * 4096,
    b" " * 1024,
    b"\t" * 64,
)

MAX_INPUT = 128 * 1024  # bound inputs so a single iteration can't hang the run


def _chunks(data: bytes, n: int) -> list[bytes]:
    """Split ``data`` into ``n`` near-equal chunks (order preserved)."""
    if n <= 1:
        return [data]
    step = max(1, len(data) // n)
    out = [data[i : i + step] for i in range(0, len(data), step)][:n]
    return out or [b""]


def mutate(rng: random.Random, data: bytes) -> bytes:
    """One bounded byte-level mutation round (deterministic under ``rng``)."""
    if not data:
        data = rng.choice(INTERESTING)
    buf = bytearray(data[:MAX_INPUT])
    for _ in range(rng.randint(1, 4)):
        op = rng.randrange(7)
        if op == 0 and buf:  # flip random byte
            i = rng.randrange(len(buf))
            buf[i] ^= 1 << rng.randrange(8)
        elif op == 1:  # insert interesting token
            token = rng.choice(INTERESTING)
            at = rng.randrange(len(buf) + 1)
            buf[at:at] = token
        elif op == 2 and buf:  # delete slice
            start = rng.randrange(len(buf))
            del buf[start : start + rng.randint(1, 64)]
        elif op == 3 and buf:  # duplicate slice
            start = rng.randrange(len(buf))
            piece = bytes(buf[start : start + rng.randint(1, 128)])
            at = rng.randrange(len(buf) + 1)
            buf[at:at] = piece
        elif op == 4 and buf:  # overwrite slice with random bytes
            start = rng.randrange(len(buf))
            buf[start : start + rng.randint(1, 32)] = os.urandom(rng.randint(1, 32))
        elif op == 5 and buf:  # arithmetic on a byte (dfuzz-style)
            i = rng.randrange(len(buf))
            buf[i] = (buf[i] + rng.choice((-35, -1, 1, 35, 128))) % 256
        else:  # overwrite with interesting token
            token = rng.choice(INTERESTING)
            at = rng.randrange(len(buf) + 1)
            buf[at : at + len(token)] = token
    return bytes(buf[:MAX_INPUT])


def load_corpus(name: str, extra_dirs: Iterable[Path] = ()) -> list[bytes]:
    """All seed inputs for ``name``: ``fuzz/corpus/<name>/*`` + extra dirs."""
    seeds: list[bytes] = []
    for directory in (CORPUS_ROOT / name, *extra_dirs):
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    seeds.append(path.read_bytes())
    return seeds or [b""]


def classify(
    exc: BaseException, expected: tuple[type[BaseException], ...]
) -> str | None:
    """Return a dedup key for an unexpected exception, else ``None``."""
    if isinstance(exc, expected):
        return None
    tb = traceback.extract_tb(exc.__traceback__)
    anchor = "no_tb"
    for frame in reversed(tb):
        path = frame.filename.replace("\\", "/")
        if "/src/quant_fund/" in path or "/fuzz/" in path:
            anchor = f"{path.rsplit('/', 1)[-1]}:{frame.lineno}:{frame.name}"
            break
    if tb and anchor == "no_tb":
        frame = tb[-1]
        anchor = f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}:{frame.name}"
    return f"{type(exc).__name__}@{anchor}"


def write_artifact(
    name: str, key: str, data: bytes, exc: BaseException, *, runs: int
) -> Path:
    """Persist a crashing input + traceback under ``fuzz/artifacts/<name>/``."""
    out_dir = ARTIFACT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()[:16]
    safe_key = "".join(ch if ch.isalnum() or ch in "@._-" else "_" for ch in key)
    crash_path = out_dir / f"crash-{safe_key}-{digest}.bin"
    crash_path.write_bytes(data)
    trace_path = crash_path.with_suffix(".txt")
    trace_path.write_text(
        "".join(traceback.format_exception(exc))
        + f"\n---\ntarget={name} dedup={key} run_index={runs} sha256={digest}\n"
    )
    return crash_path


def _parse_args(argv: list[str] | None, name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=name)
    parser.add_argument("--max-total-time", type=float, default=60.0)
    parser.add_argument("--max-runs", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0xD1C9)
    parser.add_argument("--jobs", type=int, default=1, help="accepted for parity; runs serially")
    parser.add_argument("--exact-artifact-path", type=Path, default=None)
    parser.add_argument("corpus", nargs="*", type=Path, default=[])
    args, unknown = parser.parse_known_args(argv)
    # Accept libFuzzer-style flags verbatim so one CLI works in both modes.
    for item in unknown:
        if item.startswith("-max_total_time="):
            args.max_total_time = float(item.split("=", 1)[1])
        elif item.startswith("-artifact_prefix="):
            pass  # atheris handles artifacts itself; fallback uses ARTIFACT_ROOT
        elif item.startswith("-jobs="):
            pass
        elif not item.startswith("-"):
            args.corpus.append(Path(item))
    return args


def _child_main(
    conn: object,
    test_one_input: Callable[[bytes], None],
    data: bytes,
    expected: tuple[type[BaseException], ...],
) -> None:
    import os as _os
    import traceback as _tb

    try:
        test_one_input(data)
    except expected:
        _os._exit(0)
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        _os._exit(0)
    except BaseException as exc:  # noqa: BLE001
        try:
            conn.send(  # type: ignore[attr-defined]
                (type(exc).__name__, "".join(_tb.format_exception(exc)))
            )
        except Exception:
            pass
        _os._exit(2)
    _os._exit(0)


def _run_isolated(
    name: str,
    test_one_input: Callable[[bytes], None],
    corpus: list[bytes],
    rng: random.Random,
    args: argparse.Namespace,
    expected: tuple[type[BaseException], ...],
) -> int:
    """Fork-per-input runner: survives Rust-level aborts in native codecs.

    Slower (~ms per run vs µs) but records the input behind SIGABRT/OOM kills
    that an in-process loop cannot observe. Unix-only (fork).
    """
    import multiprocessing

    ctx = multiprocessing.get_context("fork")
    deadline = time.monotonic() + args.max_total_time
    crashes: dict[str, Path] = {}
    runs = 0
    while runs < args.max_runs and time.monotonic() < deadline:
        seed = rng.choice(corpus)
        data = seed if rng.random() < 0.05 else mutate(rng, seed)
        runs += 1
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_child_main, args=(child, test_one_input, data, expected)
        )
        proc.start()
        child.close()
        proc.join(timeout=30)
        info: tuple[str, str] | None = None
        if proc.is_alive():
            proc.terminate()
            proc.join()
            info = ("HangTimeout", f"input exceeded 30s\n{data[:200]!r}")
        elif proc.exitcode == 2 and parent.poll():
            info = parent.recv()
        elif proc.exitcode and proc.exitcode != 0:
            signame = ""
            if proc.exitcode < 0:
                try:
                    import signal as _sig

                    signame = _sig.Signals(-proc.exitcode).name
                except ValueError:
                    signame = f"sig{-proc.exitcode}"
            info = (signame or f"exit{proc.exitcode}", f"process died: {proc.exitcode}")
        if info is None:
            if runs % 64 == 0:
                corpus.append(data)
            continue
        exc_name, detail = info
        # dedup on exception/signal name + first src frame in the traceback
        anchor = exc_name
        for line in detail.splitlines():
            if "/src/quant_fund/" in line or "File \"" in line and "quant_fund" in line:
                anchor = f"{exc_name}@{line.strip()[:120]}"
                break
        if anchor in crashes:
            continue
        fake_exc = RuntimeError(detail)
        path = write_artifact(name, anchor, data, fake_exc, runs=runs)
        crashes[anchor] = path
        print(f"[{name}] NEW CRASH {anchor} at run {runs}: {path}", file=sys.stderr)
    print(
        f"[{name}] isolated runner: {runs} runs, {len(crashes)} unique crashes "
        f"({args.max_total_time:.0f}s budget, seed={args.seed})"
    )
    for key, path in crashes.items():
        print(f"  crash {key} -> {path}")
    return 1 if crashes else 0


def run(
    name: str,
    test_one_input: Callable[[bytes], None],
    *,
    expected: tuple[type[BaseException], ...],
    argv: list[str] | None = None,
    isolate: bool = False,
) -> int:
    """Run ``test_one_input`` against the corpus; return process exit code.

    Prefers atheris when installed (real coverage guidance). Otherwise runs
    the deterministic mutational fallback and exits non-zero if any crash was
    recorded, so CI fails loudly and artifacts land in ``fuzz/artifacts/``.
    ``isolate=True`` forks a child per input so native-level aborts (e.g.
    polars' OOM-on-corrupt-parquet) are recorded instead of killing the run.
    """
    args = _parse_args(argv if argv is not None else sys.argv[1:], name)
    extra_dirs = [p for p in args.corpus if p != CORPUS_ROOT / name]

    try:
        import atheris  # type: ignore[import-not-found]
    except ImportError:
        atheris = None

    if atheris is not None and not os.environ.get("FUZZ_NO_ATHERIS"):
        artifact_dir = ARTIFACT_ROOT / name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        libfuzzer_argv = [sys.argv[0]]
        libfuzzer_argv += [str(p) for p in (CORPUS_ROOT / name, *extra_dirs)]
        libfuzzer_argv.append(f"-max_total_time={int(args.max_total_time)}")
        libfuzzer_argv.append("-rss_limit_mb=2048")
        libfuzzer_argv.append("-print_final_stats=1")
        libfuzzer_argv.append(f"-artifact_prefix={artifact_dir}/")

        def wrapped(data: bytes) -> None:
            try:
                test_one_input(data)
            except expected:
                return

        atheris.Setup(libfuzzer_argv, wrapped)
        atheris.Fuzz()
        return 0

    corpus = load_corpus(name, extra_dirs)
    rng = random.Random(args.seed)

    if isolate and hasattr(os, "fork"):
        return _run_isolated(name, test_one_input, corpus, rng, args, expected)

    # ---- fallback: seeded mutational loop -------------------------------
    deadline = time.monotonic() + args.max_total_time
    crashes: dict[str, tuple[Path, BaseException]] = {}
    runs = 0
    while runs < args.max_runs and time.monotonic() < deadline:
        seed = rng.choice(corpus)
        data = seed if rng.random() < 0.05 else mutate(rng, seed)
        runs += 1
        try:
            test_one_input(data)
        except expected:
            continue
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:  # noqa: BLE001 — crashes are the payload
            # BaseException matters: polars PanicException (Rust panic) is not
            # an Exception subclass and kills processes through `except
            # Exception` guards — exactly the contract violation we hunt.
            key = classify(exc, expected)
            if key is None:
                continue
            if key not in crashes:
                path = write_artifact(name, key, data, exc, runs=runs)
                crashes[key] = (path, exc)
                print(f"[{name}] NEW CRASH {key} at run {runs}: {path}", file=sys.stderr)
        if runs % 512 == 0:
            corpus.append(data)  # cheap in-session corpus growth
    status = (
        f"[{name}] fallback runner: {runs} runs, {len(crashes)} unique crashes "
        f"({args.max_total_time:.0f}s budget, seed={args.seed})"
    )
    print(status)
    for key, (path, _exc) in crashes.items():
        print(f"  crash {key} -> {path}")
    return 1 if crashes else 0


def read_json(data: bytes) -> object:
    """Strict-ish JSON decode used by JSON harnesses (raises ValueError)."""
    return json.loads(io.StringIO(data.decode("utf-8", errors="strict")).read())
