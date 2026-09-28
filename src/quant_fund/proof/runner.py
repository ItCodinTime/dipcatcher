"""Proof runner entry points.

``run_backtest_proven`` remains the wave-1 fail-closed stub — signature and
behavior unchanged (zero public-API breakage, amendment A4).

``run_proven`` (wave 2, WAVE2.md §4) is the causal proven runner: it drives a
``proofcore.scheduler.Scheduler`` over a frozen ``RunSpec``, forces every
feature/label/weight to be computed per-window from ``PitVault.asof`` reads
under the active watchdog, records a hash-chained ``DecisionTrace``, and
mints the wave-1 proof bundle with trace/env/seeds sidecars. Any undeclared
code path fails closed (§4.4) and no partial bundle is ever written: all
fallible computation completes before the atomic commit phase.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import polars as pl

from quant_fund.config.models import AppConfig

# LH011 layering (DESIGN.md §1.3): proof reaches pit/leakage only via LAZY
# (function-level) imports — the wave-1 seam keeps the new stack acyclic.
from quant_fund.proof.bundle import build_bundle
from quant_fund.proof.estimators import ALLOWLISTED_ESTIMATORS, ESTIMATOR_REGISTRY
from quant_fund.proof.recorder import InMemoryRecorder
from quant_fund.proof.sign import HmacSha256Signer, Signer
from quant_fund.proofcore import run_context
from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    DecisionTrace,
    DecisionTraceRow,
    FeatureDecl,
    ProofBundleV1,
    ProofError,
    RunSpec,
    canonical_json_bytes,
    sha256_hex_bytes,
    sha256_hex_json,
    trace_row_hash,
)
from quant_fund.proofcore.scheduler import DecisionWindow, Scheduler

__all__ = [
    "BARS_DATASET",
    "WEIGHTS_DATASET",
    "run_backtest_proven",
    "run_proven",
]

BARS_DATASET = "silver/bars"
WEIGHTS_DATASET = "gold/weights"

#: Determinism precision for feature/action/metric floats (DESIGN.md §8.2).
_FLOAT_NDIGITS = 12

#: Legal vault_window_agg aggregations (WAVE2.md §4, v1).
_WINDOW_AGGS = frozenset({"mean", "std", "min", "max", "last"})

#: Legal prior_decision_state fields.
_PRIOR_STATE_FIELDS = frozenset({"last_action", "last_signal"})


class ProvenVault(Protocol):
    """Structural type of the vault the proven runner reads through.

    A ``pit.vault.PitVault`` satisfies this protocol; the runner stays
    LH011-clean (proof -> pit is a LAZY-only edge, DESIGN.md §1.3) by not
    importing the concrete class at module level.
    """

    def asof(self, name: str, t: datetime, *, columns: list[str] | None = None) -> Any: ...


def run_backtest_proven(
    config: AppConfig,
    *,
    seed: int,
    pit_root: Path,
    bundle_dir: Path,
    replay_engine: Literal["reference", "fast"] = "reference",
    signer: Signer | None = None,
    vault: object | None = None,
    recorder: InMemoryRecorder | None = None,
) -> ProofBundleV1:
    """Reject runs until explicit decision times drive each vault as-of read."""
    raise ProofError(
        "proven run unavailable: explicit per-decision as-of vault reads are not implemented"
    )


# ---------------------------------------------------------------------------
# Wave 2: causal proven runner (WAVE2.md §4) — NEW public entry point.
# ---------------------------------------------------------------------------


def _env_fingerprint() -> str:
    """Narrowed env gate string (WAVE2.md §2.2): platform|python|quant_fund."""
    return f"{platform.platform()}|{platform.python_version()}|quant_fund-dev"


def _code_fingerprint() -> str:
    """Git revision of the working tree, else ``"nogit"``.

    Integration follow-up (W8 §7.2): when not in a git worktree the fallback
    becomes a sha256 over the src tree; flagged for the integration wave.
    """
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


def _require_finite(value: float, *, what: str) -> float:
    """Fail closed (§4.4): any non-finite feature/label/state value aborts."""
    if not math.isfinite(value):
        raise ProofError(f"nan_in_window: {what} is not finite ({value!r})")
    return value


def _read_column_mean(vault: ProvenVault, dataset: str, column: str, t: datetime) -> float:
    """One watchdog-checked vault read; cross-sectional mean of ``column``.

    v1 reference semantics (documented): the frame returned by
    ``vault.asof(t)`` is reduced to the plain-float mean of the requested
    column. The read watermark ``t`` is always <= the active decision time,
    so the returned rows satisfy ``known_at <= t <= decision_time``.
    """
    from quant_fund.leakage.watchdog import LeakageError

    pit_frame = vault.asof(dataset, t, columns=[column])
    # Defense in depth (§4.4): the vault filter + strict watchdog already
    # guarantee max_known_at <= decision_time for an honest PitVault; a
    # corrupted or rogue vault that hands back future-known rows must still
    # abort the run instead of leaking into the trace.
    decision_time = run_context.current_decision_time()
    if decision_time is not None and pit_frame.max_known_at > decision_time:
        raise LeakageError(
            f"leakage: {dataset!r} read at {t.isoformat()} returned rows with known_at up to "
            f"{pit_frame.max_known_at.isoformat()} > decision_time {decision_time.isoformat()}"
        )
    series = pit_frame.frame[column]
    value = float(series.mean())
    return _require_finite(value, what=f"{dataset}.{column} asof {t.isoformat()}")


def _grid_lag_time(window: DecisionWindow, step: timedelta, lag: int) -> datetime:
    """Grid-aligned read time ``decision_time - lag*step``.

    Equals ``prior_times[-lag]`` whenever the lagged point is on-grid
    (``lag <= seq``); for warmup windows the grid extends before ``start``
    through the SAME asof read path (watchdog-checked, recorded) instead of
    failing closed on insufficient history — declared v1 semantics.
    """
    return window.decision_time - lag * step


def _feature_params(decl: FeatureDecl, *required: str) -> dict[str, Any]:
    params = dict(decl.params)
    missing = [key for key in required if key not in params]
    if missing:
        raise ProofError(f"feature {decl.name!r} missing params: {','.join(missing)}")
    return params


def _compute_feature(
    decl: FeatureDecl,
    window: DecisionWindow,
    step: timedelta,
    vault: ProvenVault,
    prior: dict[str, float] | None,
) -> float:
    """Evaluate one declared feature for this window (§4: no user callables)."""
    if decl.kind == "vault_column_lag":
        params = _feature_params(decl, "dataset", "column", "lag")
        lag = params["lag"]
        if not isinstance(lag, int) or isinstance(lag, bool) or lag < 1:
            raise ProofError(f"feature {decl.name!r}: lag must be an integer >= 1")
        t_read = _grid_lag_time(window, step, lag)
        return _read_column_mean(vault, str(params["dataset"]), str(params["column"]), t_read)
    if decl.kind == "vault_window_agg":
        params = _feature_params(decl, "dataset", "column", "window", "agg")
        n = params["window"]
        agg = params["agg"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ProofError(f"feature {decl.name!r}: window must be an integer >= 1")
        if agg not in _WINDOW_AGGS:
            raise ProofError(f"feature {decl.name!r}: unknown agg {agg!r}")
        values = [
            _read_column_mean(
                vault,
                str(params["dataset"]),
                str(params["column"]),
                _grid_lag_time(window, step, k),
            )
            for k in range(1, n + 1)
        ]
        if agg == "mean":
            return statistics.fmean(values)
        if agg == "std":
            return statistics.pstdev(values)  # population std, deterministic
        if agg == "min":
            return min(values)
        if agg == "max":
            return max(values)
        return values[0]  # "last": most recent prior grid point
    if decl.kind == "prior_decision_state":
        params = _feature_params(decl, "field")
        field = params["field"]
        if field not in _PRIOR_STATE_FIELDS:
            raise ProofError(f"feature {decl.name!r}: unknown prior-state field {field!r}")
        initial = float(params.get("initial", 0.0))  # declared warmup default
        if prior is None:
            return _require_finite(initial, what=f"feature {decl.name!r} initial")
        return prior[str(field)]
    raise ProofError(f"feature_kind_undeclared: {decl.kind!r}")  # pragma: no cover


def _label_decl(spec: RunSpec) -> tuple[str, str, int]:
    """Extract the declared label {dataset, column, horizon}; fail closed."""
    raw = spec.estimator_params.get("label")
    if not isinstance(raw, dict):
        raise ProofError("estimator_params_missing_label: declare {'label': {...}}")
    missing = [key for key in ("dataset", "column", "horizon") if key not in raw]
    if missing:
        raise ProofError(f"label declaration missing: {','.join(missing)}")
    horizon = raw["horizon"]
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ProofError("label horizon must be an integer >= 1")
    return str(raw["dataset"]), str(raw["column"]), horizon


def _make_estimator(spec: RunSpec) -> Any:
    """Construct the allowlisted estimator; undeclared names fail closed."""
    if spec.estimator not in ALLOWLISTED_ESTIMATORS:
        raise ProofError(f"estimator_not_allowlisted: {spec.estimator!r}")
    params = {key: value for key, value in spec.estimator_params.items() if key != "label"}
    try:
        return ESTIMATOR_REGISTRY[spec.estimator](**params)
    except TypeError as exc:
        raise ProofError(f"estimator_params_invalid: {exc}") from exc


def _realize_label(
    vault: ProvenVault,
    dataset: str,
    column: str,
    base_time: datetime,
    future_time: datetime,
) -> float:
    """label_i = v(t_{i+horizon}) / v(t_i) - 1 via watchdog-checked reads.

    Computed at the window where it realizes (``decision_time >= t_{i+h}``):
    the late realization is legal because the watchdog checks the returned
    rows against the CURRENT decision time, and the reads land in that
    window's recorder manifest (WAVE2.md §4).
    """
    base = _read_column_mean(vault, dataset, column, base_time)
    future = _read_column_mean(vault, dataset, column, future_time)
    return _require_finite(future / base - 1.0, what="label")


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except ImportError:
        return "not-installed"


def _atomic_write(path: Path, data: bytes) -> None:
    """Write-temp-then-replace inside the target directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            tmp = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _commit_staging(staging: Path, bundle_dir: Path, chain_prefix_lines: int) -> None:
    """Move a fully-minted staging tree into ``bundle_dir`` (commit phase)."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    staging_chain = staging / "bundles.jsonl"
    new_lines = b""
    if staging_chain.exists():
        lines = staging_chain.read_bytes().splitlines(keepends=True)
        new_lines = b"".join(lines[chain_prefix_lines:])
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "bundles.jsonl":
            target = bundle_dir / path.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
    if new_lines:
        chain_path = bundle_dir / "bundles.jsonl"
        with chain_path.open("ab") as handle:
            handle.write(new_lines)
            handle.flush()
            os.fsync(handle.fileno())


def run_proven(
    spec: RunSpec,
    *,
    vault: ProvenVault,
    bundle_dir: Path,
    signing_key: bytes | None = None,
) -> tuple[bool, str]:
    """Run a causal proven backtest (WAVE2.md §4). Returns (True, bundle_id).

    Fail-closed (§4.4): any contract violation raises (ProofError,
    LeakageError, VaultError) BEFORE the commit phase, so a failed run never
    leaves a bundle or sidecar behind in ``bundle_dir``.
    """
    _make_estimator(spec)  # fail closed on undeclared estimator before any IO
    label_dataset, label_column, horizon = _label_decl(spec)
    scheduler = Scheduler(spec)
    recorder = InMemoryRecorder()
    from quant_fund.leakage.watchdog import LeakageWatchdog

    watchdog = LeakageWatchdog(strict=True)

    rows: list[DecisionTraceRow] = []
    feature_history: list[dict[str, float]] = []
    label_cache: dict[int, float] = {}
    signals: list[dict[str, Any]] = []
    prior_state: dict[str, float] | None = None
    prev_hash = GENESIS_HASH

    with run_context.proven_run(recorder, watchdog):
        for window in scheduler:
            with run_context.decision_window(window.decision_time):
                reads_before = len(recorder.reads)
                features = {
                    decl.name: _compute_feature(decl, window, scheduler.step, vault, prior_state)
                    for decl in spec.features
                }
                # Realize the label that matures exactly at this window.
                if window.seq >= horizon:
                    matured = window.seq - horizon
                    label_cache[matured] = _realize_label(
                        vault,
                        label_dataset,
                        label_column,
                        scheduler.window_at(matured).decision_time,
                        window.decision_time,
                    )
                feature_history.append(features)
                # Walk-forward EXPANDING fit: a fresh estimator is fit on the
                # full training set each window (fit is pure; refit-from-scratch
                # keeps EWMA-style stateful estimators deterministic too).
                estimator = _make_estimator(spec)
                train_idx = sorted(label_cache)
                if train_idx:
                    X = np.array(
                        [
                            [feature_history[i][decl.name] for decl in spec.features]
                            for i in train_idx
                        ],
                        dtype=np.float64,
                    )
                    y = np.array([label_cache[i] for i in train_idx], dtype=np.float64)
                    estimator.fit(X, y)
                x_now = np.array([features[decl.name] for decl in spec.features], dtype=np.float64)
                signal = _require_finite(float(estimator.predict(x_now)), what="estimator signal")
                for value in estimator.state_vector():
                    _require_finite(float(value), what="estimator state")
                target_weight = max(-1.0, min(1.0, signal))
                action = {"target_weight": round(target_weight, _FLOAT_NDIGITS)}

                window_reads = recorder.reads[reads_before:]
                data_manifest_sha256 = sha256_hex_json(
                    sorted(read.content_sha256 for read in window_reads)
                )
                feature_set_sha256 = sha256_hex_json(
                    {name: round(value, _FLOAT_NDIGITS) for name, value in features.items()}
                )
                estimator_state_sha256 = sha256_hex_bytes(estimator.state_bytes())
                action_sha256 = sha256_hex_json(action)
                rng_counter_sha256 = sha256_hex_json({"seed": window.seed, "draws_used": 0})
                row = DecisionTraceRow(
                    seq=window.seq,
                    decision_time=window.decision_time,
                    known_at_ceiling=window.decision_time,
                    data_manifest_sha256=data_manifest_sha256,
                    feature_set_sha256=feature_set_sha256,
                    estimator_state_sha256=estimator_state_sha256,
                    action_sha256=action_sha256,
                    rng_counter_sha256=rng_counter_sha256,
                    prev_row_sha256=prev_hash,
                )
                prev_hash = trace_row_hash(row)
                rows.append(row)
                prior_state = {"last_action": target_weight, "last_signal": signal}
                signals.append(
                    {
                        "seq": window.seq,
                        "decision_time": window.decision_time,
                        "signal": signal,
                        "target_weight": target_weight,
                        "feature_set_sha256": feature_set_sha256,
                    }
                )

    # ---- post-loop: all remaining fallible computation happens BEFORE any
    # ---- byte lands in bundle_dir (no partial bundles, §4.4).
    spec_sha256 = sha256_hex_json(spec.model_dump(mode="json"))
    trace = DecisionTrace(
        rows=tuple(rows),
        spec_sha256=spec_sha256,
        code_fingerprint=_code_fingerprint(),
        env_fingerprint=_env_fingerprint(),
        head_row_sha256=prev_hash if rows else GENESIS_HASH,
    )
    trace.verify_chain()
    trace_bytes = canonical_json_bytes(trace.model_dump(mode="json"))
    env_bytes = canonical_json_bytes(
        {
            "env_fingerprint": trace.env_fingerprint,
            "code_fingerprint": trace.code_fingerprint,
            "python_tag": platform.python_implementation() + platform.python_version(),
            "packages": {
                "numpy": np.__version__,
                "pandas": _package_version("pandas"),
                "polars": pl.__version__,
            },
        }
    )
    seeds_bytes = canonical_json_bytes(
        {
            "seed": spec.seed,
            "window_seeds": {
                str(seq): sha256_hex_bytes(f"{spec.seed}|{seq}".encode())
                for seq in range(len(scheduler))
            },
        }
    )

    # Deterministic NAV path from realized labels and emitted actions.
    nav = 1.0
    nav_marks: list[float] = []
    for entry in signals:
        matured = entry["seq"] - horizon
        if matured in label_cache:
            nav *= 1.0 + entry["target_weight"] * label_cache[matured]
        nav_marks.append(nav)
    signal_log = pl.DataFrame(
        {
            "seq": [s["seq"] for s in signals],
            "decision_time": [s["decision_time"] for s in signals],
            "signal": [round(float(s["signal"]), _FLOAT_NDIGITS) for s in signals],
            "target_weight": [s["target_weight"] for s in signals],
            "feature_set_sha256": [s["feature_set_sha256"] for s in signals],
        }
    )
    trade_log = pl.DataFrame(
        {
            "seq": [s["seq"] for s in signals],
            "fill_time": [s["decision_time"] for s in signals],
            "target_weight": [s["target_weight"] for s in signals],
            "nav": [round(mark, _FLOAT_NDIGITS) for mark in nav_marks],
        }
    )
    engine_metrics = {
        "n_windows": float(len(signals)),
        "final_nav": round(nav, _FLOAT_NDIGITS),
    }
    config_dump = {
        "run_spec": spec.model_dump(mode="json"),
        "sidecars": {
            "trace_sha256": sha256_hex_bytes(trace_bytes),
            "env_sha256": sha256_hex_bytes(env_bytes),
            "seeds_sha256": sha256_hex_bytes(seeds_bytes),
        },
    }

    # Mint in a staging sibling; commit atomically only on full success.
    bundle_dir = Path(bundle_dir)
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".run_proven.", dir=bundle_dir.parent))
    try:
        existing_chain = bundle_dir / "bundles.jsonl"
        chain_prefix_lines = 0
        if existing_chain.exists():
            chain_bytes = existing_chain.read_bytes()
            chain_prefix_lines = len(chain_bytes.splitlines())
            (staging / "bundles.jsonl").write_bytes(chain_bytes)
        signer: Signer | None = HmacSha256Signer(signing_key) if signing_key else None
        bundle = build_bundle(
            run_kind="backtest",
            data_manifest=recorder.manifest_summary(),
            config_dump=config_dump,
            seed=spec.seed,
            signal_log=signal_log,
            trade_log=trade_log,
            engine_metrics=engine_metrics,
            bundle_dir=staging,
            signer=signer,
        )
        _atomic_write(staging / f"{bundle.bundle_id}.trace.json", trace_bytes)
        _atomic_write(staging / f"{bundle.bundle_id}.env.json", env_bytes)
        _atomic_write(staging / f"{bundle.bundle_id}.seeds.json", seeds_bytes)
        _commit_staging(staging, bundle_dir, chain_prefix_lines)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return (True, bundle.bundle_id)
