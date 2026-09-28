# PROOFCORE — WAVE 2: Causal Runs & Cryptographic Replay

**Status:** spec of record for wave 2. Supplements `DESIGN.md` (wave 1) and
`docs/proofcore/README.md`. All rules there still bind; this wave is
**additive-only, never weakens a gate, zero public-API breakage**.

## 0. Mission

Wave 1 proved integrity: bundles are signed, hashes recompute, leakage rules
fire. It deliberately left two doors closed with fail-closed stubs:

- `quant_fund.proof.runner.run_backtest_proven` — "proven run unavailable"
- `quant_fund.proof.replay.replay_bundle` — "runner_unavailable:per_decision_asof_not_implemented"

The README states the residual honesty gap: *the verifier "does not establish
that precomputed weights were available to each historical decision"*.

**Wave 2 closes that gap, honestly.** We ship:

1. a **Causal Decision Scheduler** (`proofcore/scheduler.py`) — pure, no IO,
   turns a run spec into a strict, hashed sequence of decision windows;
2. a **Proven Runner** (`proof/runner.py`) — drives the scheduler, forces every
   feature/weight to be computed per-window from vault reads under the
   existing watchdog, records a **decision trace**, and fails closed on any
   undeclared code path;
3. a **Replay Engine** (`proof/replay.py`) — bit-exact re-execution of a
   recorded run from frozen seeds, comparing every per-decision hash,
   reporting the FIRST divergence at decision granularity;
4. an **IO Guard** (`leakage/guard.py`) — runtime interposition that makes raw
   `pl.read_parquet` / `pd.read_parquet` / `open()` illegal inside proven
   windows (redirect, warn+record, or fail-closed by mode), closing the
   "rogue second vault" class at the syscall-ish level;
5. **follow-up fixes** carried from wave-1 delivery: CSCV combo-count guard,
   code-fingerprint fallback outside git worktrees.

When this wave lands, the README's fail-closed sentences for `proof run` and
`verify --replay` are REPLACED with the precise, narrowed scope — the claims
become true, not removed.

## 1. Honesty contract for this wave (binding)

1. The runner supports ONLY a declared, frozen surface. Any estimator or
   feature code outside the declared allowlist → `ProofError`, fail closed.
   No silent fallback to "trust me" paths.
2. The replay engine NEVER trusts original metrics or fills. It recomputes
   every quantity and compares hashes. `identical` means bit-exact on every
   compared field; anything else is `diverged` with a first-divergence record.
3. A replay verdict is evidence about *re-executability*, not about live-trading
   quality. No live-trading claims anywhere (AGENTS.md).
4. Unknown environment → fail closed. Replay requires: same python tag, same
   platform, same quant_fund version fingerprint. Mismatch →
   `ReplayVerdict(status="unavailable", reason="env_mismatch")`.
5. Synthetic/demo data used in tests is labeled synthetic. No real performance
   claims from fixtures.
6. Coverage floors: every NEW module ≥ 90% (scheduler, replay, runner, guard).

## 2. Contracts (SACRED — implement exactly)

All new schemas go in `src/quant_fund/proofcore/contracts.py` as additions
(wave-2 agent may append; must not alter existing schemas). pydantic v2,
frozen where possible, same style as `ProofBundleV1`.

### 2.1 `DecisionTraceRow`

```python
class DecisionTraceRow(BaseModel, frozen=True):
    seq: int                      # 0-based, strictly increasing, gapless
    decision_time: datetime       # tz-aware
    known_at_ceiling: datetime    # tz-aware; == decision_time enforced by runner
    data_manifest_sha256: str     # sha256 over the sorted vault-object hashes read this window
    feature_set_sha256: str       # sha256 over the feature frame's canonical bytes
    estimator_state_sha256: str   # sha256 over estimator state AFTER fit/predict this window
    action_sha256: str            # sha256 over the emitted action/fill payload
    rng_counter_sha256: str       # sha256 over (seed, draws_used) after the window
    prev_row_sha256: str          # sha256 over the previous row's canonical bytes; "" for seq 0
```

Canonical bytes for hashing: `row.model_dump_json(sort_keys=True).encode()`.
Helper `trace_row_hash(row) -> str` in contracts.

### 2.2 `DecisionTrace`

```python
class DecisionTrace(BaseModel, frozen=True):
    rows: tuple[DecisionTraceRow, ...]
    spec_sha256: str        # sha256 of the RunSpec canonical json
    code_fingerprint: str   # git sha of quant_fund when in a worktree;
                            # else src-tree hash fallback (§7.2)
    env_fingerprint: str    # f"{platform}|{python_version_tag}|{quant_fund.__version__ or 'dev'}"
    head_row_sha256: str    # == trace_row_hash(last row); "" when empty

    def verify_chain(self) -> None: ...  # raises ProofError on any break
```

### 2.3 `RunSpec` (scheduler input; frozen pydantic)

```python
class FeatureDecl(BaseModel, frozen=True):
    name: str
    kind: Literal["vault_column_lag", "vault_window_agg", "prior_decision_state"]
    params: dict[str, Any]      # frozen jsonable; hashed into spec_sha256

class DecisionGrid(BaseModel, frozen=True):
    start: datetime             # tz-aware
    step: str                   # e.g. "1d", parsed by scheduler (no dateutil dep beyond stdlib? -> use datetime.timedelta parse helper we ship)
    count: int                  # >0

class RunSpec(BaseModel, frozen=True):
    name: str
    vault_uri: str              # logical, e.g. "vault://main"
    decision_grid: DecisionGrid
    features: tuple[FeatureDecl, ...]
    estimator: str              # allowlist name, see §4.3
    estimator_params: dict[str, Any]
    seed: int                   # top-level; window seeds derived deterministically
```

### 2.4 `ReplayVerdict`

```python
class Divergence(BaseModel, frozen=True):
    seq: int
    field: str                  # one of the DecisionTraceRow hash fields, or "metric:<name>"
    expected_sha256: str
    actual_sha256: str

class ReplayVerdict(BaseModel, frozen=True):
    bundle_id: str
    status: Literal["identical", "diverged", "unavailable"]
    reason: str | None = None               # set iff status != "identical"
    first_divergence: Divergence | None = None
    compared_rows: int
    recomputed_metrics: dict[str, str]      # name -> sha256 of recomputed value set
```

### 2.5 Bundle integration

- The decision trace is stored as a sidecar file `<bundle_id>.trace.json` in
  the bundle dir; its sha256 goes into the bundle's existing sidecar manifest
  (same mechanism as `<id>.trades.json` etc.).
- The runner additionally writes `<bundle_id>.env.json` (env_fingerprint,
  code_fingerprint, dependency pins actually importable: python tag, numpy,
  pandas, polars versions) and `<bundle_id>.seeds.json`
  (`{window_seed_i = sha256(f"{seed}|{i}")}` — deterministic derivation, no
  stored per-window randomness beyond the top-level seed).
- `verify.py` must hash-check the trace/env/seeds sidecars like any other
  (extends the wave-1 config sidecar check pattern).

## 3. Scheduler (`src/quant_fund/proofcore/scheduler.py`)

Pure stdlib + contracts. No IO, no vault imports (layer 1).

```python
@dataclass(frozen=True)
class DecisionWindow:
    seq: int
    decision_time: datetime
    prior_times: tuple[datetime, ...]   # all earlier decision times, in order
    seed: int                           # sha256(f"{spec.seed}|{seq}") -> int

class Scheduler:
    def __init__(self, spec: RunSpec) -> None: ...
    def __iter__(self) -> Iterator[DecisionWindow]: ...
    def window_at(self, seq: int) -> DecisionWindow: ...
```

- `step` parsing: ship `parse_step(s) -> timedelta` supporting `Nd`, `Nh`,
  `Nm`, `Ns` (integer N ≥ 1). Anything else → `ProofError`.
- Grid: `start + i*step` for i in range(count). All tz handling by simple
  arithmetic on the tz-aware start (no calendar awareness claimed — the spec
  is explicit this is a fixed grid; calendar-aware grids are a documented
  follow-up, NOT this wave).
- Determinism: identical spec → identical windows, always.

## 4. Proven Runner (`src/quant_fund/proof/runner.py`)

Replaces the fail-closed `run_backtest_proven` stub. Signature (public API of
wave 1 must keep working — this function was already exported, its stub
returns the unavailable tuple; new behavior below is the intended semantics):

```python
def run_backtest_proven(
    spec: RunSpec,
    *,
    vault: PitVault,
    bundle_dir: Path,
    signing_key: bytes | None = None,
) -> tuple[bool, str]:
    """Run a causal proven backtest. Returns (ok, bundle_id_or_error)."""
```

Behavior per window w (driven by `Scheduler(spec)`):

1. `with proven_run(recorder, watchdog), decision_window(w.decision_time):`
2. Read vault ONLY through `vault.asof(w.decision_time)` (watchdog enforces
   known_at ceiling via existing R1/R2 machinery).
3. Compute declared features for this window ONLY (each `FeatureDecl.kind` has
   one implementation; no user callables in v1 — this is the honesty trade).
4. Fit/predict with the allowlisted estimator using `w.seed`.
5. Emit action payload (sized from estimator output; synthetic sizing rule in
   spec params, default: target weight from signal, clipped).
6. Recorder captures: vault objects read, feature frame canonical bytes,
   estimator state bytes, action payload, rng state summary → trace row.
7. After the last window: build `DecisionTrace`, verify_chain, write
   trace/env/seeds sidecars, mint bundle through the wave-1 recorder/bundle
   path, sign if key given, return (True, bundle_id).

### 4.3 Estimator allowlist (v1)

`{"linear_regression_np": LinearRegressionNumpy, "ewma_signal": EwmaSignal}` —
both are TINY reference estimators implemented IN this wave inside
`quant_fund/proof/estimators.py` (pure numpy, deterministic). Unknown name →
`ProofError("estimator_not_allowlisted")`. The allowlist is a module-level
frozenset so the AST linter and future gates can see it statically.

### 4.4 Hard failures (fail-closed list)

- undeclared estimator / feature kind → ProofError
- any `pl.read_parquet`/`pd.read_parquet`/`open()` call inside a window not
  via the vault → enforced by the IO guard in fail-closed mode
- timezone-naive spec times → ProofError
- vault read with known_at > decision_time → existing LeakageError propagates
- NaN in feature frame / estimator state → ProofError("nan_in_window")
- bundle mint on any exception → no partial bundle written

## 5. Replay Engine (`src/quant_fund/proof/replay.py`)

Replaces the stub. Public signature unchanged:

```python
def replay_bundle(
    bundle_path: Path,
    *,
    bundle_dir: Path,
    pit_root: Path | None,
    vault: Any = None,
) -> tuple[bool, str]:
    """-> (identical, verdict_json_or_error). Never trusts original numbers."""
```

Flow:

1. Load bundle; hash-check trace/env/seeds sidecars (tamper →
   `(False, "sidecar_tampered:<which>")`, mirroring wave-1 wording).
2. Rebuild `RunSpec` from the bundle's config sidecar (it must round-trip;
   the runner wrote it there). spec_sha256 must equal trace.spec_sha256.
3. Env gate: compare current env fingerprint + code fingerprint vs
   `<id>.env.json`. Mismatch → verdict `unavailable` with reason `env_mismatch`
   (returned as `(False, verdict_json)`).
4. Re-execute via the SAME runner path (import `run_backtest_proven` internals
   or factor a shared `_execute_run(spec, vault, seeds) -> DecisionTrace`) on a
   FRESH recorder; deterministic derivation of window seeds from the stored
   top-level seed must reproduce the stored seeds file hash-for-hash.
5. Compare row-by-row with stored trace: same count; for each row compare all
   sha256 fields. First mismatch → `diverged` + `first_divergence`.
6. Recompute headline metrics from the re-executed fills (reuse wave-1
   trade-log recompute) into `recomputed_metrics` (name → value-set sha256).
7. Return verdict json (ReplayVerdict.model_dump_json). Status `identical` →
   `(True, verdict_json)`.

Property tests (hypothesis, `ci` profile like wave 1): for random specs and
synthetic vaults, run → replay is ALWAYS identical (determinism), unless the
vault content is mutated — then replay MUST diverge and the first divergence
must be at a row whose data_manifest covers the mutated object.

## 6. IO Guard (`src/quant_fund/leakage/guard.py`)

Installed by the runner for the duration of a proven run; also usable
standalone (`install_io_guard(mode)` context manager).

- Modes: `"redirect"` (vault-mapped paths route through `vault.asof(decision
  time)`; unknown paths → warn+record), `"enforce"` (any raw parquet/open
  inside a decision window that is not a vault-mapped path → `LeakageError`),
  `"audit"` (record everything, raise nothing). Proven runner uses `"enforce"`.
- Mechanism: `sys.setprofile`/`sys.settrace`-free — use targeted monkey
  interposition on `polars.read_parquet`, `pandas.read_parquet`, and
  `io.open`/`builtins.open` while a proven window is active (check
  `proofcore.run_context.current_decision_time() is not None` and
  `context_is_proven()`), plus an allowlist of paths (tempfiles of the runner
  itself). Must be re-entrant and thread-safe; use the R2 registry so worker
  threads are covered. Guard must NEVER intercept reads when no proven run is
  active (zero overhead claim off, correctness claim on).
- Test: a strategy function that sneaks `pl.read_parquet("/etc/passwd-like
  tmpfile.parquet")` inside a window → `LeakageError` in enforce mode;
  redirect mode routes a vault-mapped path to a recorded vault read; audit
  mode records and continues. Outside a run: untouched behavior.

## 7. Follow-up fixes (same wave, small)

7.1 **CSCV combo guard** (`reality/cscv.py`): cap combinations at 10_000;
   exceeding → `RealityError("cscv_combo_cap:raise the cap explicitly")`.
   Property test: guard fires at cap, no silent truncation.

7.2 **Code-fingerprint fallback** (`proofcore/ci.py`): when not in a git
   worktree, fingerprint = sha256 over `src/quant_fund/**/*.py` sorted path +
   content (exclude `__pycache__`). Deterministic, cached per process.

(7.3 `monotonic_known_at` default flip — DEFERRED, documented only.)

## 8. File ownership (no cross-agent edits)

| Agent | Files |
|---|---|
| W6 runner+scheduler | `src/quant_fund/proofcore/scheduler.py`, `src/quant_fund/proof/runner.py`, `src/quant_fund/proof/estimators.py`, contracts appends (§2.1–2.3 only), `tests/unit/test_scheduler.py`, `tests/unit/test_proven_runner.py`, `tests/unit/test_estimators.py` |
| W7 replay | `src/quant_fund/proof/replay.py`, `tests/unit/test_replay_engine.py`, `tests/property/test_replay_determinism.py` |
| W8 guard+fixes | `src/quant_fund/leakage/guard.py`, `src/quant_fund/reality/cscv.py` (combo guard), `src/quant_fund/proofcore/ci.py` (fingerprint fallback), `tests/unit/test_io_guard.py`, `tests/unit/test_cscv_combo_guard.py`, `tests/unit/test_fingerprint_fallback.py` |

SHARED, DO-TOUCH: `src/quant_fund/proofcore/contracts.py` — W6 appends
§2.1–2.3 schemas ONLY; nobody else edits it. If W7/W8 need a contract, they
define local private helpers and flag it in the PR for integration.

SHARED, NOBODY TOUCHES: `cli/_app.py`, `pyproject.toml`, `Makefile`,
`docs/proofcore/*`, `.github/*`. Integration wave owns them (floors update,
README rewrite of the fail-closed lines, CLI mounts for
`quant proof run` / `quant proof replay`, CHANGELOG).

## 9. Done criteria (whole wave)

- `quant proof run --spec spec.json --bundle-dir ...` executes a causal run,
  mints a signed bundle with trace/env/seeds sidecars.
- `quant proof verify --replay` on that bundle returns `identical` on the same
  machine, `unavailable:env_mismatch` after a forced fingerprint change, and
  `diverged` after a single-byte vault mutation (first divergence points at
  the right row).
- New-module coverage ≥ 90%; wave-1 gates all still green; ruff clean.
- README fail-closed sentences replaced with true, narrowed claims.
- Adversarial round 3 attacks scheduled.


---

## Amendments (lead, pre-flight)

- **A1 (§2.1/§2.2):** chain anchors use `GENESIS_HASH` ("0"*64), not `""`:
  `DecisionTraceRow.prev_row_sha256 == GENESIS_HASH` for seq 0, and
  `DecisionTrace.head_row_sha256 == GENESIS_HASH` for an empty trace. This
  matches the bundle chain convention (`prev_bundle_hash`).
- **A2 (§2.1):** `trace_row_hash` is implemented via the wave-1 helper
  `canonical_json_bytes(row.model_dump(mode="json"))` — same determinism
  policy as every other PROOFCORE hash.
- **A3 (§8):** the contracts appends (§2.1–§2.4) are ALREADY on
  `proofcore/wave2-base` — implemented and import-verified by the lead. All
  agents treat `src/quant_fund/proofcore/contracts.py` as READ-ONLY.
- **A4 (§4):** `run_proven(spec, *, vault, bundle_dir, signing_key=None)` in
  `proof/runner.py` is the wave-2 entry point. The §4 name
  `run_backtest_proven` was taken by the wave-1 fail-closed stub with a
  different signature; zero public-API breakage forbids repurposing it, so
  the stub STAYS fail-closed and the CLI `quant proof run` calls
  `run_proven`.

---

## Amendments (integration wave, post-merge)

Reconciliations across the W6/W7/W8 deliveries, landed on
`proofcore/wave2-base` by the integration agent. Binding as of the
integration PR.

- **I1 (§2.2):** ONE canonical `env_fingerprint` implementation,
  `proofcore.ci.env_fingerprint()`, on both sides (contracts §2.2 formula:
  `platform|python tag|version`; the version is probed via
  `importlib.metadata.version("fx-1")` because the layering gate forbids
  proofcore importing the quant_fund root). The runner mints and replay
  re-derives via this helper. W6's dev literal `...|quant_fund-dev` remains
  ACCEPTED by the replay env gate as a compat shim that expires in wave 3.
- **I2 (§2.5/§5):** ONE canonical `spec_sha256` on both sides:
  `sha256_hex_json(spec.model_dump(mode="json"))` with NO rounding anywhere.
  `build_bundle` gained an additive `round_config: bool = True` parameter;
  the wave-2 runner passes `round_config=False` so the config sidecar carries
  the unrounded spec dump and a >12-significant-digit float param round-trips
  mint → replay bit-exact (e2e-tested).
- **I3 (§6):** the IO guard is wired into `run_proven` in enforce mode. The
  import is lazy and OPTIONAL at import time: an unimportable guard logs one
  warning per process and the run proceeds under the watchdog and
  declared-surface checks (defense in depth, documented). The vault root and
  the runner staging dir are allowlisted (vault-internal manifest-verified
  reads use `open()` inside decision windows).
- **I4 (§2.5):** `verify.py` hash-checks the trace/env/seeds sidecars via the
  config sidecar's `sidecars` commitment map (check 6b, mirroring the wave-1
  config-check pattern). Commitment kinds are restricted to `[a-z0-9_]+` so a
  hostile commitment key cannot escape the bundle dir.
- **I5 (§1.6):** the coverage floors for the new wave-2 modules are
  per-MODULE, all 90: `proofcore.scheduler`, `proof.runner`,
  `proof.estimators`, `proof.replay`, `leakage.guard` — declared in pyproject
  `[tool.proofcore.coverage-floors]`, enforced by `make proofcore-coverage`.
