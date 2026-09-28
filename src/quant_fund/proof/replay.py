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
proven runner (``quant_fund.proof.runner.run_proven``, W6), re-runs into a
fresh temp bundle dir, and loads the fresh ``<id>.trace.json``. Unit tests
pass a fake executor so this module never depends on the runner at import
time.

Sidecar hash anchoring (W6 integration contract): the wave-2 sidecars
(``<id>.trace.json``, ``<id>.env.json``, ``<id>.seeds.json``) are committed
by the runner inside the config sidecar as
``config["sidecars"]["<kind>_sha256"]``. The config sidecar itself is
anchored by ``bundle.config_sha256`` (frozen wave-1 schema field), so the
commitment chain is bundle -> config -> wave-2 sidecars (§2.5's "existing
sidecar manifest mechanism" — the bundle schema has no wave-2 slots and is
read-only). The trace is additionally pinned by
``trace.spec_sha256 == sha256(canonical RunSpec dump)`` and the seeds
sidecar by deterministic re-derivation from the spec seed (§2.5).
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
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
    "current_env_sidecar",
    "derive_window_seeds",
    "expected_seeds_sidecar",
    "replay_bundle",
    "window_seed_sha256",
]

#: Executor contract: (spec, vault, tmp_bundle_dir) -> re-executed trace.
ReplayExecutor = Any  # Callable[[RunSpec, Any, Path], DecisionTrace]

#: Wave-2 sidecar kinds hash-checked via the config commitment map,
#: in deterministic check order (§5 step 1). ``config`` is anchored by the
#: bundle itself and checked first.
WAVE2_SIDECAR_KINDS = ("trace", "env", "seeds")

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
    return sha256_hex_bytes(f"{seed}|{seq}".encode())


def derive_window_seeds(spec: RunSpec) -> dict[str, str]:
    """Re-derive the per-window seed commitments from the spec's seed."""
    return {str(i): window_seed_sha256(spec.seed, i) for i in range(spec.decision_grid.count)}


def expected_seeds_sidecar(spec: RunSpec) -> dict[str, Any]:
    """The seeds sidecar content the runner must have written for ``spec``."""
    return {"seed": spec.seed, "window_seeds": derive_window_seeds(spec)}


# ---------------------------------------------------------------------------
# Current-environment probes (private; tests monkeypatch these, not the gate).
# The formulas mirror proof.runner (W6), which wrote the stored fingerprints.
# ---------------------------------------------------------------------------


def _current_python_tag() -> str:
    """``implementation + version``, mirroring the runner's env sidecar."""
    return platform.python_implementation() + platform.python_version()


def _current_env_fingerprint() -> str:
    """``platform|python tag|quant_fund-dev`` — mirrors proof.runner (W6)."""
    return f"{platform.platform()}|{platform.python_version()}|quant_fund-dev"


def _current_env_fingerprint_contracts() -> str:
    """Contracts-style variant (§2.2): real ``quant_fund.__version__`` or dev."""
    version = "dev"
    try:
        import quant_fund

        version = str(getattr(quant_fund, "__version__", "dev") or "dev")
    except Exception:  # never let a version probe break the gate
        version = "dev"
    return f"{platform.platform()}|{platform.python_version()}|{version}"


def _current_code_fingerprint() -> str:
    """Git revision of the working tree, else ``"nogit"`` (mirrors W6; the
    src-tree hash fallback is W8 §7.2, integration wave)."""
    git = shutil.which("git")
    if git is None:
        return "nogit"
    try:
        proc = subprocess.run(
            [git, "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    revision = proc.stdout.strip()
    if proc.returncode != 0 or not revision:
        return "nogit"
    return revision


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
# Step 1: sidecar hash-checks, anchored bundle -> config -> wave-2 sidecars
# ---------------------------------------------------------------------------


def _sidecar_path(bundle_dir: Path, bundle_id: str, kind: str) -> Path:
    return bundle_dir / f"{bundle_id}.{kind}.json"


def _valid_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LEN
        and all(c in "0123456789abcdef" for c in value)
    )


def _read_sidecar(path: Path) -> bytes | None:
    if path.is_symlink() or not path.exists():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def _hash_check_sidecars(
    bundle: ProofBundleV1, bundle_dir: Path
) -> tuple[dict[str, bytes] | None, dict[str, Any] | None, str | None]:
    """Re-hash config + trace/env/seeds sidecars against their commitments.

    Returns ``(contents, config_doc, None)`` on success, else
    ``(None, None, error)`` with ``sidecar_tampered:<which>`` wording (fail
    closed, mirroring the wave-1 verify idiom). The config sidecar is checked
    against ``bundle.config_sha256`` FIRST; its ``sidecars`` map then anchors
    the wave-2 sidecars.
    """
    contents: dict[str, bytes] = {}
    config_bytes = _read_sidecar(_sidecar_path(bundle_dir, bundle.bundle_id, "config"))
    if config_bytes is None or sha256_hex_bytes(config_bytes) != bundle.config_sha256:
        return None, None, "sidecar_tampered:config"
    try:
        config_doc = json.loads(config_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, None, f"config_invalid:{exc.__class__.__name__}"
    if not isinstance(config_doc, dict):
        return None, None, "config_invalid:not_a_mapping"
    if "run_spec" not in config_doc and "sidecars" not in config_doc:
        # Wave-1 bundle: minted before causal runs existed, carries no
        # decision trace. Preserve the stub's fail-closed wording (zero
        # public-API/regression breakage).
        return None, None, "runner_unavailable:per_decision_asof_not_implemented"
    commitments = config_doc.get("sidecars")
    commitments = commitments if isinstance(commitments, dict) else {}
    for kind in WAVE2_SIDECAR_KINDS:
        expected = commitments.get(f"{kind}_sha256", commitments.get(kind))
        if not _valid_sha256_hex(expected):
            return None, None, f"sidecar_tampered:{kind}"
        data = _read_sidecar(_sidecar_path(bundle_dir, bundle.bundle_id, kind))
        if data is None or sha256_hex_bytes(data) != expected:
            return None, None, f"sidecar_tampered:{kind}"
        contents[kind] = data
    contents["config"] = config_bytes
    return contents, config_doc, None


# ---------------------------------------------------------------------------
# Step 3: environment gate (§1.4: unknown environment -> fail closed)
# ---------------------------------------------------------------------------


def _env_mismatch_key(env_doc: object) -> str | None:
    """First mismatching env-sidecar key vs the live machine, or None."""
    if not isinstance(env_doc, dict):
        return "env_fingerprint"
    stored_env = env_doc.get("env_fingerprint")
    if stored_env is not None and stored_env not in {
        _current_env_fingerprint(),
        _current_env_fingerprint_contracts(),
    }:
        return "env_fingerprint"
    stored_code = env_doc.get("code_fingerprint")
    if stored_code is not None and stored_code != _current_code_fingerprint():
        return "code_fingerprint"
    stored_tag = env_doc.get("python_tag")
    if stored_tag is not None and stored_tag not in {
        _current_python_tag(),
        platform.python_version(),
    }:
        return "python_tag"
    stored_pins = env_doc.get("packages")
    if isinstance(stored_pins, dict):
        current = _current_package_pins()
        for name in sorted(stored_pins):
            if current.get(name) != stored_pins[name]:
                return f"packages:{name}"
    return None


# ---------------------------------------------------------------------------
# Step 4: default executor — lazy runner import (keeps W7 decoupled from W6)
# ---------------------------------------------------------------------------


def _default_executor(spec: RunSpec, vault: Any, tmp_bundle_dir: Path) -> DecisionTrace:
    """Re-run via the proven runner and load the fresh trace sidecar."""
    try:
        from quant_fund.proof import runner as _runner
    except ImportError as exc:
        raise ReplayUnavailable(f"runner_import:{exc.__class__.__name__}") from exc
    run_proven = getattr(_runner, "run_proven", None)
    if run_proven is None:
        raise ReplayUnavailable("quant_fund.proof.runner has no run_proven entry point")
    try:
        ok, result = run_proven(spec, vault=vault, bundle_dir=tmp_bundle_dir)
    except ReplayUnavailable:
        raise
    except Exception as exc:  # runner fails closed by raising (§4.4)
        raise ReplayUnavailable(str(exc)) from exc
    if not ok:
        raise ReplayUnavailable(str(result))
    trace_path = Path(tmp_bundle_dir) / f"{result}.trace.json"
    try:
        return DecisionTrace.model_validate(json.loads(trace_path.read_bytes()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise ReplayUnavailable(f"fresh_trace_unreadable:{exc.__class__.__name__}") from exc


# ---------------------------------------------------------------------------
# Steps 5-6: comparison + metric recomputation
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

    # Step 1 — load the bundle and hash-check config + wave-2 sidecars.
    try:
        raw = json.loads(bundle_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"bundle_unreadable:{exc.__class__.__name__}"
    try:
        bundle = ProofBundleV1.model_validate(raw)
    except ValidationError as exc:
        return False, f"bundle_invalid:{exc.error_count()}errors"
    contents, config_doc, tamper = _hash_check_sidecars(bundle, bundle_dir)
    if tamper is not None or contents is None or config_doc is None:
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
    run_spec_raw = config_doc.get("run_spec")
    try:
        spec = RunSpec.model_validate(run_spec_raw)
    except ValidationError as exc:
        return False, f"config_spec_invalid:{exc.error_count()}errors"
    if sha256_hex_json(spec.model_dump(mode="json")) != stored_trace.spec_sha256:
        return False, "spec_mismatch:trace_spec_sha256"

    # Seeds sidecar must reproduce, hash-for-hash, from the spec's seed (§2.5).
    try:
        seeds_doc = json.loads(contents["seeds"])
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, "sidecar_tampered:seeds"
    if seeds_doc != expected_seeds_sidecar(spec):
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
