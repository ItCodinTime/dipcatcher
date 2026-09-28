"""Cryptographic replay engine (WAVE2.md §5) — bit-exact re-execution check.

Replaces the wave-1 fail-closed stub. The engine NEVER trusts the bundle's
stored metrics or fills (§1.2): it hash-checks the wave-2 sidecars, rebuilds
the :class:`RunSpec` from the config sidecar, gates on the environment
fingerprint, re-executes the run through an injected executor, and compares
every per-decision sha256 field of the stored :class:`DecisionTrace` against
the freshly re-executed one. ``identical`` means bit-exact on every compared
field; anything else is ``diverged`` with a first-divergence record.

Executor injection (W7/W6 decoupling): callers pass
``executor: Callable[[RunSpec, Any, Path], DecisionTrace]`` — signature
``(spec, vault, tmp_bundle_dir)``. The default executor lazily imports the
proven runner (``quant_fund.proof.runner.run_proven``, falling back to the
wave-2 ``run_backtest_proven``), re-runs into a fresh temp bundle dir, and
loads the fresh ``<id>.trace.json``. Unit tests pass a fake executor so this
module never depends on the runner at import time.

Sidecar hash anchoring: wave-2 sidecars (``<id>.trace.json``,
``<id>.env.json``, ``<id>.seeds.json``) are hash-checked against the mint's
sidecar manifest ``<id>.sidecars.json`` (canonical JSON ``{kind: sha256}``,
the §2.5 mechanism), and the config sidecar against ``bundle.config_sha256``
(wave-1 idiom). Because the frozen ``ProofBundleV1`` schema has no wave-2
sidecar slots, the trace is additionally anchored by
``trace.spec_sha256 == sha256(canonical RunSpec dump)`` where the RunSpec
comes from the bundle-anchored config sidecar, and the seeds sidecar by
deterministic re-derivation from the spec seed (§2.5).
"""

from __future__ import annotations

import json
import platform
import tempfile
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from quant_fund.proofcore.contracts import (
    DecisionTrace,
    Divergence,
    ProofBundleV1,
    ProofError,
    ReplayVerdict,
    RunSpec,
    sha256_hex_bytes,
    sha256_hex_json,
)

__all__ = [
    "ReplayExecutor",
    "ReplayUnavailable",
    "ROW_HASH_FIELDS",
    "WAVE2_SIDECAR_KINDS",
    "derive_window_seeds",
    "replay_bundle",
    "window_seed_sha256",
]

#: Executor contract: (spec, vault, tmp_bundle_dir) -> re-executed trace.
ReplayExecutor = Any  # Callable[[RunSpec, Any, Path], DecisionTrace]

#: Wave-2 sidecar kinds, in deterministic check order (§5 step 1).
WAVE2_SIDECAR_KINDS = ("trace", "env", "seeds", "config")

#: Per-row fields compared bit-exactly (§5 step 5), in deterministic order.
ROW_HASH_FIELDS = (
    "data_manifest_sha256",
    "feature_set_sha256",
    "estimator_state_sha256",
    "action_sha256",
    "rng_counter_sha256",
    "prev_row_sha256",
)

#: Dependency pins recorded in the env sidecar (§2.5) and re-checked live.
_PINNED_PACKAGES = ("numpy", "pandas", "polars")

_SHA256_HEX_LEN = 64


class ReplayUnavailable(ProofError):
    """The re-execution path cannot run here (runner missing/failed closed)."""


def window_seed_sha256(seed: int, seq: int) -> str:
    """Window-seed commitment (§2.5): sha256(f"{seed}|{seq}") as hex."""
    return sha256_hex_bytes(f"{seed}|{seq}".encode("utf-8"))


def derive_window_seeds(spec: RunSpec) -> dict[str, str]:
    """Re-derive the seeds sidecar content from the spec's top-level seed."""
    return {
        str(i): window_seed_sha256(spec.seed, i) for i in range(spec.decision_grid.count)
    }


# ---------------------------------------------------------------------------
# Current-environment probes (private; tests monkeypatch these, not the gate).
# ---------------------------------------------------------------------------


def _current_python_tag() -> str:
    return platform.python_version()


def _current_env_fingerprint() -> str:
    """``f"{platform}|{python tag}|{quant_fund.__version__ or 'dev'}"`` (§2.2)."""
    version = "dev"
    try:
        import quant_fund

        version = str(getattr(quant_fund, "__version__", "dev") or "dev")
    except Exception:  # never let a version probe break the gate
        version = "dev"
    return f"{platform.platform()}|{platform.python_version()}|{version}"


def _current_code_fingerprint() -> str:
    """Git revision of quant_fund in a worktree; all-zero fallback (§7.2 is W8)."""
    try:
        from quant_fund.proof.bundle import code_fingerprint

        return code_fingerprint().git_revision
    except Exception:  # fail closed to a deterministic non-match sentinel
        return "0" * 40


def _current_package_pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for name in _PINNED_PACKAGES:
        try:
            pins[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            pins[name] = "unknown"
    return pins


def current_env_sidecar() -> dict[str, Any]:
    """The env-sidecar content this machine would mint now (test/fixture aid)."""
    return {
        "env_fingerprint": _current_env_fingerprint(),
        "code_fingerprint": _current_code_fingerprint(),
        "python_tag": _current_python_tag(),
        "packages": _current_package_pins(),
    }


# ---------------------------------------------------------------------------
# Sidecar IO helpers
# ---------------------------------------------------------------------------


def _sidecar_path(bundle_dir: Path, bundle_id: str, kind: str) -> Path:
    if kind == "config":
        return bundle_dir / f"{bundle_id}.config.json"
    return bundle_dir / f"{bundle_id}.{kind}.json"


def _load_sidecar_manifest(bundle_dir: Path, bundle_id: str) -> dict[str, str] | None:
    """Read ``<id>.sidecars.json`` (kind -> sha256), or None when absent."""
    path = bundle_dir / f"{bundle_id}.sidecars.json"
    if path.is_symlink() or not path.exists():
        return None
    try:
        doc = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    manifest: dict[str, str] = {}
    for kind in ("trace", "env", "seeds"):
        value = doc.get(kind, doc.get(f"{bundle_id}.{kind}.json"))
        if (
            not isinstance(value, str)
            or len(value) != _SHA256_HEX_LEN
            or value.lower() != value
            or any(c not in "0123456789abcdef" for c in value)
        ):
            return None
        manifest[kind] = value
    return manifest


def _hash_check_sidecars(
    bundle: ProofBundleV1, bundle_dir: Path
) -> tuple[dict[str, bytes] | None, str | None]:
    """Step 1: re-hash trace/env/seeds/config sidecars against recorded hashes.

    Returns ``(contents, None)`` on success, ``(None, error)`` with
    ``sidecar_tampered:<which>`` wording on any failure (fail closed).
    """
    manifest = _load_sidecar_manifest(bundle_dir, bundle.bundle_id)
    if manifest is None:
        return None, "sidecar_tampered:manifest"
    expected: dict[str, str] = {
        "trace": manifest["trace"],
        "env": manifest["env"],
        "seeds": manifest["seeds"],
        "config": bundle.config_sha256,
    }
    contents: dict[str, bytes] = {}
    for kind in WAVE2_SIDECAR_KINDS:
        path = _sidecar_path(bundle_dir, bundle.bundle_id, kind)
        if path.is_symlink() or not path.exists():
            return None, f"sidecar_tampered:{kind}"
        try:
            data = path.read_bytes()
        except OSError:
            return None, f"sidecar_tampered:{kind}"
        if sha256_hex_bytes(data) != expected[kind]:
            return None, f"sidecar_tampered:{kind}"
        contents[kind] = data
    return contents, None


# ---------------------------------------------------------------------------
# Env gate (§5 step 3, §1.4: unknown environment -> fail closed)
# ---------------------------------------------------------------------------


def _env_mismatch_key(env_doc: object) -> str | None:
    """First mismatching env-sidecar key vs the live machine, or None."""
    if not isinstance(env_doc, dict):
        return "env_fingerprint"
    stored_env = env_doc.get("env_fingerprint")
    if stored_env is not None and stored_env != _current_env_fingerprint():
        return "env_fingerprint"
    stored_code = env_doc.get("code_fingerprint")
    if stored_code is not None and stored_code != _current_code_fingerprint():
        return "code_fingerprint"
    stored_tag = env_doc.get("python_tag")
    if stored_tag is not None and stored_tag != _current_python_tag():
        return "python_tag"
    stored_pins = env_doc.get("packages")
    if isinstance(stored_pins, dict):
        current = _current_package_pins()
        for name in sorted(stored_pins):
            if current.get(name) != stored_pins[name]:
                return f"packages:{name}"
    return None


# ---------------------------------------------------------------------------
# Default executor: lazy runner import (keeps W7 decoupled from W6)
# ---------------------------------------------------------------------------


def _default_executor(spec: RunSpec, vault: Any, tmp_bundle_dir: Path) -> DecisionTrace:
    """Re-run via the proven runner and load the fresh trace sidecar."""
    try:
        from quant_fund.proof import runner as _runner
    except ImportError as exc:
        raise ReplayUnavailable(f"runner_import:{exc.__class__.__name__}") from exc
    run_proven = getattr(_runner, "run_proven", None)
    try:
        if run_proven is not None:
            ok, result = run_proven(spec, vault=vault, bundle_dir=tmp_bundle_dir)
        else:
            run_backtest_proven = getattr(_runner, "run_backtest_proven", None)
            if run_backtest_proven is None:
                raise ReplayUnavailable("quant_fund.proof.runner has no proven entry point")
            ok, result = run_backtest_proven(spec, vault=vault, bundle_dir=tmp_bundle_dir)
    except ReplayUnavailable:
        raise
    except Exception as exc:  # wave-1 stub raises ProofError; treat as unavailable
        raise ReplayUnavailable(str(exc)) from exc
    if not ok:
        raise ReplayUnavailable(str(result))
    trace_path = Path(tmp_bundle_dir) / f"{result}.trace.json"
    try:
        return DecisionTrace.model_validate(json.loads(trace_path.read_bytes()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ReplayUnavailable(f"fresh_trace_unreadable:{exc.__class__.__name__}") from exc


# ---------------------------------------------------------------------------
# Comparison + metric recomputation (§5 steps 5-6)
# ---------------------------------------------------------------------------


def _first_divergence(stored: DecisionTrace, fresh: DecisionTrace) -> Divergence | None:
    """First bit-level mismatch between stored and re-executed traces."""
    n_stored, n_fresh = len(stored.rows), len(fresh.rows)
    for i in range(min(n_stored, n_fresh)):
        stored_row, fresh_row = stored.rows[i], fresh.rows[i]
        for field in ROW_HASH_FIELDS:
            expected = getattr(stored_row, field)
            actual = getattr(fresh_row, field)
            if expected != actual:
                return Divergence(
                    seq=stored_row.seq,
                    field=field,
                    expected_sha256=expected,
                    actual_sha256=actual,
                )
    if n_stored != n_fresh:
        return Divergence(
            seq=min(n_stored, n_fresh),
            field="row_count",
            expected_sha256=sha256_hex_json(n_stored),
            actual_sha256=sha256_hex_json(n_fresh),
        )
    return None


def _recompute_metrics_from_fills(tmp_bundle_dir: Path) -> dict[str, str]:
    """Step 6: headline metrics from the RE-EXECUTED fills, hashed per name.

    Never reads the stored metrics/trade sidecars (§1.2). When the executor
    minted no trade log (e.g. trace-only fakes), the value set is empty.
    """
    trade_logs = sorted(tmp_bundle_dir.glob("*.trades.parquet"))
    if not trade_logs:
        return {}
    import io

    import polars as pl

    from quant_fund.proof.bundle import recompute_headline_metrics, round_floats

    trade_log = pl.read_parquet(io.BytesIO(trade_logs[0].read_bytes()))
    metrics = recompute_headline_metrics(trade_log)
    return {name: sha256_hex_json(round_floats(value)) for name, value in metrics.items()}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def replay_bundle(
    bundle_path: Path,
    *,
    bundle_dir: Path,
    pit_root: Path | None,
    vault: Any = None,
    executor: ReplayExecutor | None = None,
) -> tuple[bool, str]:
    """Replay one proven bundle bit-exactly (§5).

    Returns ``(identical, verdict_json_or_error)``. ``identical`` is True iff
    the re-executed trace matches the stored trace on every compared sha256
    field of every row. The verdict is evidence about re-executability, not
    live-trading quality. Stored metrics and fills are never trusted.
    """
    del pit_root  # vault is injected by the caller; kept for API stability
    bundle_path = Path(bundle_path)
    bundle_dir = Path(bundle_dir)

    # Step 1 — load the bundle and hash-check every wave-2 sidecar.
    try:
        raw = json.loads(bundle_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"bundle_unreadable:{exc.__class__.__name__}"
    try:
        bundle = ProofBundleV1.model_validate(raw)
    except ValidationError as exc:
        return False, f"bundle_invalid:{exc.error_count()}errors"
    contents, tamper = _hash_check_sidecars(bundle, bundle_dir)
    if tamper is not None or contents is None:
        return False, tamper or "sidecar_tampered:unknown"

    try:
        stored_trace = DecisionTrace.model_validate(json.loads(contents["trace"]))
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        return False, f"trace_invalid:{exc.__class__.__name__}"
    try:
        stored_trace.verify_chain()
    except ProofError as exc:
        return False, f"trace_chain_invalid:{exc}"

    # Step 2 — rebuild the RunSpec from the bundle-anchored config sidecar.
    try:
        spec = RunSpec.model_validate(json.loads(contents["config"]))
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        return False, f"config_spec_invalid:{exc.__class__.__name__}"
    if sha256_hex_json(spec.model_dump(mode="json")) != stored_trace.spec_sha256:
        return False, "spec_mismatch:trace_spec_sha256"

    # Seeds sidecar must reproduce, hash-for-hash, from the spec's seed (§2.5).
    try:
        seeds_doc = json.loads(contents["seeds"])
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, "sidecar_tampered:seeds"
    if seeds_doc != derive_window_seeds(spec):
        return False, "sidecar_tampered:seeds"

    # Step 3 — environment gate (§1.4: unknown environment -> fail closed).
    try:
        env_doc = json.loads(contents["env"])
    except (UnicodeDecodeError, json.JSONDecodeError):
        env_doc = None
    mismatch_key = _env_mismatch_key(env_doc)
    if mismatch_key is not None:
        verdict = ReplayVerdict(
            bundle_id=bundle.bundle_id,
            status="unavailable",
            reason="env_mismatch",
            compared_rows=0,
            recomputed_metrics={},
        )
        return False, verdict.model_dump_json()

    # Step 4 — re-execute through the injected executor on a fresh bundle dir.
    run_executor: ReplayExecutor = executor if executor is not None else _default_executor
    with tempfile.TemporaryDirectory(prefix="proofcore-replay-") as tmp:
        tmp_bundle_dir = Path(tmp)
        try:
            fresh = run_executor(spec, vault, tmp_bundle_dir)
        except ReplayUnavailable as exc:
            return False, f"runner_unavailable:{exc}"
        except ProofError as exc:
            return False, f"replay_reexecute_failed:{exc}"
        except Exception as exc:
            return False, f"replay_reexecute_error:{exc.__class__.__name__}"
        try:
            fresh_trace = DecisionTrace.model_validate(fresh)
        except ValidationError as exc:
            return False, f"executor_trace_invalid:{exc.error_count()}errors"
        recomputed_metrics = _recompute_metrics_from_fills(tmp_bundle_dir)

    # Step 5 — row-by-row bit-exact comparison of all sha256 fields.
    divergence = _first_divergence(stored_trace, fresh_trace)
    if divergence is not None:
        verdict = ReplayVerdict(
            bundle_id=bundle.bundle_id,
            status="diverged",
            reason="trace_divergence",
            first_divergence=divergence,
            compared_rows=divergence.seq,
            recomputed_metrics=recomputed_metrics,
        )
        return False, verdict.model_dump_json()

    # Steps 6-7 — identical: verdict carries the recomputed metric hashes.
    verdict = ReplayVerdict(
        bundle_id=bundle.bundle_id,
        status="identical",
        compared_rows=len(stored_trace.rows),
        recomputed_metrics=recomputed_metrics,
    )
    return True, verdict.model_dump_json()
