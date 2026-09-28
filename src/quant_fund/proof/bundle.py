"""Proof bundle construction: canonical JSON, self-hash, chain (DESIGN.md §5.3).

Chain layout inside ``bundle_dir`` (fx1 ``CorpusLedger`` pattern — append-only,
one canonical JSON object per line)::

    <bundle_dir>/
      bundles.jsonl              # the chain; prev_bundle_hash links each line
      bundles/<bundle_id>.json   # canonical bundle bytes
      <bundle_id>.signals.parquet    # per-decision signal log sidecar
      <bundle_id>.trades.parquet     # trade log (fills + nav marks) sidecar
      <bundle_id>.metrics.json       # engine metrics, canonical JSON bytes
      <bundle_id>.config.json        # resolved config, canonical JSON bytes

Self-hash: ``bundle_id`` = sha256 over the canonical JSON of the bundle minus
the ``bundle_id``, ``signature`` **and** ``created_utc`` fields. ``created_utc``
is excluded so the double-run identity (DESIGN.md §5.7/§8.1: "identical
bundle_id minus created_utc") is well-defined; wall-clock time is evidence,
not identity. The signature covers the canonical bundle bytes minus the
``signature`` field (``bundle_id`` included).

Trade-log schema: the engine fills frame plus a ``nav`` column carrying the
last mark-to-market NAV at or before each fill (join-asof backward over the
equity curve). This is what lets the verifier recompute every headline metric
from the trade-log bytes alone (A1 F2 fix, DESIGN.md §5.5 check 7).
"""

from __future__ import annotations

import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal

import polars as pl

from quant_fund.proof.sign import NullSigner, Signer
from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    CodeFingerprint,
    DataManifestSummary,
    EnvFingerprint,
    ProofBundleError,
    ProofBundleV1,
    SignatureBlock,
    canonical_json_bytes,
    sha256_hex_bytes,
    sha256_hex_json,
)

__all__ = [
    "BUNDLES_JSONL",
    "METRIC_KEYS",
    "PERIODS_PER_YEAR",
    "SELF_HASH_EXCLUDED_FIELDS",
    "build_bundle",
    "canonical_bundle_bytes",
    "chain_head",
    "code_fingerprint",
    "compute_bundle_id",
    "env_fingerprint",
    "load_chain",
    "recompute_headline_metrics",
    "round_floats",
    "sidecar_paths",
    "signing_payload_bytes",
    "verify_self_hash",
]

BUNDLES_JSONL = "bundles.jsonl"
BUNDLES_SUBDIR = "bundles"

#: Engine bar convention is daily; annualization matches metrics/returns.py defaults.
PERIODS_PER_YEAR = 252.0

#: Fixed keys of ``metrics_recompute`` (DESIGN.md §3 notes).
METRIC_KEYS = ("sharpe_periodic", "sharpe_annualized", "total_return", "max_drawdown", "n_trades")

#: Fields excluded from the bundle_id self-hash preimage.
SELF_HASH_EXCLUDED_FIELDS = frozenset({"bundle_id", "signature", "created_utc"})

_PACKAGE_NAMES = ("dipcatcher", "numpy", "polars", "scipy", "pydantic")


def round_floats(obj: Any, ndigits: int = 12) -> Any:
    """Recursively round floats to the §8.2 determinism precision.

    NaN/inf survive rounding unchanged; numpy-style scalars are coerced via
    ``item()``; datetimes become ISO strings. Mirrors the canonicalization
    policy of ``quant_fund.utils.hashing`` for owned metadata.
    """
    if isinstance(obj, dict):
        return {str(key): round_floats(value, ndigits) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(value, ndigits) for value in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    item = getattr(obj, "item", None)
    if callable(item) and not isinstance(obj, (str, bytes, bool)):
        try:
            return round_floats(item(), ndigits)
        except (TypeError, ValueError):
            pass
    if isinstance(obj, float):
        return round(obj, ndigits)
    return obj


def env_fingerprint() -> EnvFingerprint:
    """Current runtime fingerprint (DESIGN.md §8.1 item 4)."""
    packages: dict[str, str] = {}
    for name in _PACKAGE_NAMES:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "unknown"
    return EnvFingerprint(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        machine=platform.machine(),
        byteorder=sys.byteorder,  # type: ignore[arg-type]
        packages=packages,
    )


def _git_dirty() -> bool:
    """True iff the worktree has uncommitted tracked changes (fail-closed True)."""
    git = shutil.which("git")
    if git is None:
        return True
    try:
        proc = subprocess.run(
            [git, "diff", "--quiet", "HEAD"],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    return proc.returncode != 0


def code_fingerprint() -> CodeFingerprint:
    """Git revision + worktree hash (utils/reproducibility builders)."""
    from quant_fund.utils.reproducibility import git_revision, git_worktree_sha256

    revision = git_revision()
    worktree = git_worktree_sha256()
    dirty = _git_dirty()
    if revision == "UNKNOWN":
        revision = "0" * 40
        dirty = True
    if len(worktree) != 64:
        worktree = GENESIS_HASH
        dirty = True
    return CodeFingerprint(git_revision=revision, worktree_sha256=worktree, dirty=dirty)


def recompute_headline_metrics(trade_log: pl.DataFrame) -> dict[str, float]:
    """Recompute headline metrics from trade-log bytes (A1 F2 independent path).

    Uses ``metrics/returns.py`` on ``nav.pct_change()`` — never the engine's
    own metrics dict. Values are rounded to the §8.2 precision. Fail-closed:
    an empty/degenerate trade log yields honest NaN Sharpe and zero returns.
    """
    import numpy as np

    from quant_fund.metrics.returns import max_drawdown, sharpe_ratio, wealth_index

    n_trades = int(trade_log.height)
    nav_series: list[float] = []
    if n_trades and {"fill_time", "nav"} <= set(trade_log.columns):
        marks = (
            trade_log.select("fill_time", "nav")
            .drop_nulls("nav")
            .unique(subset="fill_time", keep="last", maintain_order=True)
            .sort("fill_time")
        )
        nav_series = [float(v) for v in marks["nav"].to_list()]
    if len(nav_series) >= 2:
        nav = np.asarray(nav_series, dtype=float)
        rets = np.diff(nav) / nav[:-1]
        sharpe_periodic = float(sharpe_ratio(rets, periods_per_year=1.0, irregular=True)["sharpe"])
        sharpe_annualized = float(sharpe_ratio(rets, periods_per_year=PERIODS_PER_YEAR)["sharpe"])
        wealth = wealth_index(rets)
        total_return = float(wealth[-1] - 1.0) if wealth.size else 0.0
        mdd = float(max_drawdown(rets))
    else:
        sharpe_periodic = float("nan")
        sharpe_annualized = float("nan")
        total_return = 0.0
        mdd = 0.0
    return {
        "sharpe_periodic": round(sharpe_periodic, 12),
        "sharpe_annualized": round(sharpe_annualized, 12),
        "total_return": round(total_return, 12),
        "max_drawdown": round(mdd, 12),
        "n_trades": float(n_trades),
    }


def compute_bundle_id(unsigned_payload: dict[str, Any]) -> str:
    """Self-hash over the canonical bundle minus bundle_id/signature/created_utc."""
    preimage = {
        key: value
        for key, value in unsigned_payload.items()
        if key not in SELF_HASH_EXCLUDED_FIELDS
    }
    return sha256_hex_json(preimage)


def signing_payload_bytes(bundle_payload: dict[str, Any]) -> bytes:
    """Canonical bundle bytes minus the signature field (HMAC preimage)."""
    payload = {key: value for key, value in bundle_payload.items() if key != "signature"}
    return canonical_json_bytes(payload)


def canonical_bundle_bytes(bundle: ProofBundleV1) -> bytes:
    """Canonical bytes of the full bundle (what ``bundles/<id>.json`` stores)."""
    return canonical_json_bytes(bundle.model_dump(mode="json"))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """NamedTemporaryFile + flush + fsync + os.replace (models/base.py pattern)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def sidecar_paths(bundle_dir: Path, bundle_id: str) -> dict[str, Path]:
    """Sidecar file locations for one bundle id."""
    return {
        "signal_log": bundle_dir / f"{bundle_id}.signals.parquet",
        "trade_log": bundle_dir / f"{bundle_id}.trades.parquet",
        "metrics": bundle_dir / f"{bundle_id}.metrics.json",
        "config": bundle_dir / f"{bundle_id}.config.json",
    }


def _parquet_bytes(frame: pl.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    return buffer.getvalue()


def chain_head(bundle_dir: Path) -> str:
    """bundle_id of the last line of bundles.jsonl, or GENESIS_HASH."""
    path = Path(bundle_dir) / BUNDLES_JSONL
    if not path.exists():
        return GENESIS_HASH
    head = GENESIS_HASH
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                head = str(json.loads(line)["bundle_id"])
    return head


def load_chain(bundle_dir: Path) -> list[ProofBundleV1]:
    """Parse every line of bundles.jsonl as a ProofBundleV1."""
    path = Path(bundle_dir) / BUNDLES_JSONL
    if not path.exists():
        return []
    bundles: list[ProofBundleV1] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                bundles.append(ProofBundleV1.model_validate(json.loads(line)))
    return bundles


def _append_chain_line(bundle_dir: Path, bundle_bytes: bytes) -> None:
    path = Path(bundle_dir) / BUNDLES_JSONL
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(bundle_bytes + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_bundle(
    *,
    run_kind: Literal["backtest", "research", "paper_shadow"],
    data_manifest: DataManifestSummary,
    config_dump: dict[str, Any],
    seed: int,
    signal_log: pl.DataFrame,
    trade_log: pl.DataFrame,
    engine_metrics: dict[str, Any],
    bundle_dir: Path,
    signer: Signer | None = None,
    created_utc: str | None = None,
    round_config: bool = True,
) -> ProofBundleV1:
    """Mint, sign, chain, and persist a ProofBundleV1 (DESIGN.md §5.2 steps 5-7).

    ``created_utc`` is injectable ONLY for tests; production callers leave it
    None (wall clock). It is evidence, excluded from the self-hash preimage.

    ``round_config`` (wave-2, additive; default preserves wave-1 bytes): when
    True the config sidecar is float-rounded to the §8.2 determinism
    precision. The wave-2 proven runner passes False because
    ``trace.spec_sha256`` commits to the UNROUNDED canonical RunSpec dump —
    rounding would break the mint -> replay hash identity for spec params
    with more than 12 significant digits (canonical_json_bytes already
    encodes non-finite floats deterministically).
    """
    signer = signer if signer is not None else NullSigner()
    bundle_dir = Path(bundle_dir)

    signal_bytes = _parquet_bytes(signal_log)
    trade_bytes = _parquet_bytes(trade_log)
    metrics_bytes = canonical_json_bytes(round_floats(engine_metrics))
    if round_config:
        config_dump = round_floats(config_dump)
    config_bytes = canonical_json_bytes(config_dump)

    # Fail closed at MINT time (ADVERSARIAL §2-H): canonical JSON encodes
    # NaN/inf as null, which the bundle schema (`metrics_recompute:
    # dict[str, float]`) then rejects — a degenerate run (e.g. empty trade
    # log) would otherwise mint a bundle its own verifier cannot parse.
    metrics_recompute = recompute_headline_metrics(trade_log)
    non_finite = sorted(key for key, value in metrics_recompute.items() if not math.isfinite(value))
    if non_finite:
        raise ProofBundleError(
            "metrics contain NaN or non-finite values "
            f"({', '.join(non_finite)}); the run is degenerate (e.g. empty "
            "trade log) and no verifiable bundle can be minted — fix the run "
            "instead of notarizing it"
        )

    unsigned: dict[str, Any] = {
        "schema_version": "proofcore/1",
        "created_utc": created_utc or datetime.now(UTC).isoformat(),
        "run_kind": run_kind,
        "code": code_fingerprint().model_dump(mode="json"),
        "data_manifest": data_manifest.model_dump(mode="json"),
        "config_sha256": sha256_hex_bytes(config_bytes),
        "seed": int(seed),
        "env": env_fingerprint().model_dump(mode="json"),
        "signal_log_sha256": sha256_hex_bytes(signal_bytes),
        "trade_log_sha256": sha256_hex_bytes(trade_bytes),
        "metrics_sha256": sha256_hex_bytes(metrics_bytes),
        "metrics_recompute": metrics_recompute,
        "prev_bundle_hash": chain_head(bundle_dir),
    }
    bundle_id = compute_bundle_id(unsigned)
    signed_payload = dict(unsigned)
    signed_payload["bundle_id"] = bundle_id
    signature_value = signer.sign(signing_payload_bytes(signed_payload))

    bundle = ProofBundleV1(
        **signed_payload,
        signature=SignatureBlock(
            scheme=signer.scheme,  # type: ignore[arg-type]
            key_id=signer.key_id,
            value=signature_value,
        ),
    )

    sidecars = sidecar_paths(bundle_dir, bundle_id)
    _atomic_write_bytes(sidecars["signal_log"], signal_bytes)
    _atomic_write_bytes(sidecars["trade_log"], trade_bytes)
    _atomic_write_bytes(sidecars["metrics"], metrics_bytes)
    _atomic_write_bytes(sidecars["config"], config_bytes)

    bundle_bytes = canonical_bundle_bytes(bundle)
    _atomic_write_bytes(bundle_dir / BUNDLES_SUBDIR / f"{bundle_id}.json", bundle_bytes)
    _append_chain_line(bundle_dir, bundle_bytes)
    return bundle


def verify_self_hash(payload: dict[str, Any]) -> bool:
    """True iff payload's bundle_id equals its recomputed self-hash."""
    return compute_bundle_id(payload) == payload.get("bundle_id")
