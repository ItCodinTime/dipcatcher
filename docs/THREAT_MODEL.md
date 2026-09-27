# Threat Model — dipcatcher / fx-1

STRIDE review of the harness (data engine, evaluation, verification) and the
fx-1 model lane. Defensive review only: this document describes surfaces and
residual risk; it is not a penetration test of any live system. The repo has
**no broker connectivity** — `execution/` and `paper/` are simulators — so the
"trading" threats below are about evidence/integrity, not order flow.

Companion document: `docs/SECURITY_REVIEW.md` (finding-by-finding ledger with
severity and status). `SECURITY.md` is the reporting policy.

## Assets

| Asset | Where | Threat if compromised |
|---|---|---|
| `MOONSHOT_API_KEY` | env only; `fx1.serve.backends` (Bearer header, pinned `api.moonshot.ai`) | Paid hosted-API abuse; teacher model impersonation poisons eval traces |
| `FX1_SIGNING_KEY` | env only; `fx1.serve.signing` (HMAC key, never logged) | Forged release signatures → unsigned/swapped checkpoints verify as genuine |
| `QUANT_API_KEY` | env only; API middleware `X-API-Key` (hmac.compare_digest) | Unauthenticated remote access to research endpoints |
| Vendor keys (`FRED_API_KEY`, `BEA_API_KEY`, `VENDOR_HTTP_API_KEY`, `KIMI_API_KEY`, `AGENT_GW_TOKEN`, `QUANT_VENDOR_API_KEY`) | env; query-param auth (FRED/BEA) or env passthrough to plugin subprocesses | Key abuse; leakage into logs/CI output |
| Receipt & corpus integrity | `receipts/`, `data/metadata/research/`, `fx1.data` hash-chained ledgers, sealed code-hash index | Forged/poisoned evidence; research results that cannot be trusted or reproduced |
| Checkpoint authenticity | `release.manifest.json` + `release.sig` HMAC | Serving a tampered model silently |
| Data pipeline correctness | PIT columns (`event_time`/`available_time`/`ingested_time`), schemas | Leakage/bias that looks like alpha (integrity, not just accuracy) |
| CI secrets / `GITHUB_TOKEN` | workflows declare `permissions: contents: read` | Write-capable token abuse; artifact poisoning |
| Supply chain | `uv.lock` (hash-locked), uvx tool pins, GH Actions, Docker base image | Malicious dep/action/image → arbitrary code in dev/CI |

## Trust boundaries

```
                       ┌─────────────────────────────────────────────┐
 remote client ──X-API-Key──► FastAPI app (auth mw, 64 KiB cap,       │
                             configs/ allowlist, honesty stamps)     │
                                                                     │
 fx-1 LLM ──registry-only argv──► Harness ──subprocess(list argv,    │
                                 scrubbed env)──► dipcatcher CLI     │
                                                                     │
 plugin scripts ◄──FX1_PLUGIN_ROOTS (env, operator-trusted)           │
            ◄── env creds only (never argv)                          │
                                                                     │
 public/vendor HTTP APIs ◄──HTTPS GET (scheme-checked, size-capped)── │
 HF Hub datasets ◄──pinned revision sha, HTTPS-only, byte cap        │
                                                                     │
 YAML configs ──safe_load + pydantic──► AppConfig                    │
 artifacts (joblib/npz/parquet) ◄── sha256 sidecar + manifest,       │
                                    joblib ONLY via verified loader  │
                                                                     │
 git object store ──sealed code-hash index (phase1_verify)           │
 MLflow (loopback sqlite / opt-in URI)                               │
 CI runner ◄── uv.lock --frozen + lock --check; uvx pinned tools     │
 Docker ◄── sha256-pinned base image, non-root USER                  │
                       └─────────────────────────────────────────────┘
```

Operator-controlled inputs (config YAML, env vars, `data/` tree, plugin
scripts, `MLFLOW_TRACKING_URI`) are *inside* the trust boundary — the model
and remote networks are outside it.

## Actor model

- **External unauthenticated client** reaching the API surface (loopback-only
  unless `QUANT_API_KEY` is set — fail-closed either way).
- **fx-1 (the model / agent)** — can invoke only registry-listed harness
  commands with `extra_args`; cannot reach `api`/`lab` or arbitrary argv.
- **Compromised or hostile dependency / action / base image** in CI/dev.
- **Tampered data or artifact** — a file under `data/` or a checkpoint dir
  that an attacker with local write access modifies.
- **Malicious datasource/plugin** — scripts under `FX1_PLUGIN_ROOTS` are
  operator-installed code; treated as trusted but output is untrusted.
- **Vendor/MITM** — network responses are schema-validated, not trusted.

## Per-component STRIDE

### 1. API service (`quant_fund.api.app`)

| S | T | R | I | E | D | Assessment |
|---|---|---|---|---|---|---|
| ✓ | | | | | | Spoofing: `X-API-Key` + `hmac.compare_digest`; no key configured ⇒ loopback-only (fail closed). Loopback check trusts `request.client.host` — correct for direct serving; if placed behind a proxy, `client.host` is the *proxy* — document `ProxyHeadersMiddleware` requirement rather than trusting `X-Forwarded-For`. |
| | ✓ | | | | | Tampering: config path allowlisted under `configs/` (resolved, `relative_to`); backtest artifact paths contained under artifact root; receipt reads re-hash the file after verification (TOCTOU guard). |
| | | ✓ | | | | Repudiation: no request audit log — mutating endpoints (`POST /backtest`, `/portfolio/optimize`) have no caller journaling. Accepted for a research service; note for hardening. |
| | | | ✓ | | | Info disclosure: only `/health` public; doctor/diagnostics expose *presence flags* (`set`/`unset`), never values. `_stamp_research_honesty` overwrites claim flags on every metric-bearing response. |
| | | | | ✓ | | Elevation: none — no shelling to user input; config allowlist is the only filesystem reach. |
| | | | | | ✓ | DoS: 64 KiB streamed body cap (declared + actual); backtest bounded to 31 dates; heavy computes (`forecast_asof`, `optimize_asof`) are unbounded CPU/RAM per request — a keyed caller can still exhaust the box. Accepted: research tool, keyed or loopback. |

### 2. Config loader (`quant_fund.config.loader`)

`yaml.safe_load` only (AST-guarded by `tests/unit/security/`); `inherit:`
chains resolve within the config root and fail on cycles/escape; pydantic
`extra="forbid"`-style validation rejects unknown keys on request models.
Residual: YAML bombs mitigated by 64 KiB API cap for HTTP callers; CLI callers
are trusted operators.

### 3. Public-source adapters (`quant_fund.data.sources`)

| S | T | R | I | E | D | Assessment |
|---|---|---|---|---|---|---|
| | | | ✓ | | | **Was a finding (S1, fixed):** `HttpClient` embedded the full URL — including FRED `api_key`/BEA `UserID` query params — in `SourceError` text → keys reached stderr/logs. Now `redact_url` masks credential params before any URL enters an error string. |
| | ✓ | | | | | Tampering: responses are schema/PIT-validated; fail-closed on dups, non-finite values, impossible time chains. HTTPS enforced for dataset downloads; scheme allowlist on `HttpClient`. |
| | | | | | ✓ | DoS: `max_bytes` response cap, bounded retries, timeouts. `query` params (e.g. GDELT/CFTC `$query`) are caller-built; no injection risk server-side beyond the upstream API's own parsing. |

### 4. fx-1 datasource plugins (`fx1.data.sources`)

Subprocess argv is a list (no shell); child env is scrubbed to
PATH/HOME/LANG/LC_ALL/PYTHONPATH **plus declared credential names only** —
credentials never appear on argv or in logs; stdout is size-capped.
`FX1_PLUGIN_ROOTS` controls which scripts run → env is a trusted input by
design (documented boundary). `request.params` are serialized onto argv —
plugin CLIs receive untrusted key/value strings; plugins are
operator-installed (inside the boundary), so residual risk is a hostile
plugin, not hostile params.

### 5. Artifact store (`joblib`, `np.load`, parquet)

| S | T | R | I | E | D | Assessment |
|---|---|---|---|---|---|---|
| | ✓ | | | | | joblib (pickle-family) is only reachable through `load_joblib_artifact`/`JoblibMixin.load` (`quant_fund.models.base`) — sha256 sidecar + manifest verification when present, atomic `os.replace` writes. **Fixed (S3):** `forecast.py`'s paper-challenger stamp used raw `joblib.load`; it now uses the verified loader. **Residual (N1):** sidecars are optional and *unkeyed* — they catch corruption, not a writer who can also write the sidecar. If artifacts cross hosts, add a keyed MAC (same shape as `release.sig`). |
| | ✓ | | | | | `np.load` default `allow_pickle=False` everywhere; AST test forbids `allow_pickle=True` and any `pickle`/`marshal`/`dill`/`torch.load` without `weights_only=True` under `src/`. |
| | | | | | ✓ | Parquet reads via polars/pyarrow (memory-safe parsers); no `tarfile`/`zipfile` extraction of untrusted archives under `src/`. |

### 6. Receipts & verification (`quant_fund.research`, `fx1.train.receipts`, `phase1_verify`)

Tampering is the dominant threat: receipts bind git revision, config hash,
dataset content hash, worktree hash, and sealed code hashes re-pulled from
`git show <rev>:<path>`; verification is fail-closed and receipts are
immutable by convention. `env_fingerprint` hashes env-var **names** only —
values can never enter a receipt. Residual: integrity rests on the local git
object store; a hostile operator can mint self-consistent receipts (accepted —
receipts are evidence, not authorization).

### 7. fx-1 release signing (`fx1.serve.signing`, `serve.backends`)

HMAC-SHA256 detached signature over a per-file sha256 manifest;
`FX1_SIGNING_KEY` env-only; `LocalFx1Backend` refuses unsigned/invalid
checkpoints whenever the key is configured (fail-closed).
**Fixed (S2):** verification now requires the on-disk file set to *equal* the
signed manifest — files added post-signing (a dropped `sitecustomize` shim or
extra weight file) invalidate the release. Residual: `FX1_SIGNING_KEY` unset ⇒
signing not enforced (by design — document "no key ⇒ no verification"); HMAC
is symmetric, so any holder of the key can sign — acceptable at tier 1 of the
attestation ladder; keyless OIDC is the documented upgrade path.

### 8. fx-1 harness (`fx1.harness`)

Registry-only commands; `dipcatcher` argv is a list (no shell); `config=` is
contained to `configs/`. Residual: `extra_args` are unvalidated flags to a
registered command (the model could pass e.g. `--out` redirects a command
accepts) — accepted as the tool-use surface; containment of `config` is
CWD-relative (`Path.cwd()/configs`), so run the harness from the repo root
(documented as N2). The `api` and `lab` commands are deliberately absent from
the registry.

### 9. Paper / execution simulators (`quant_fund.paper`, `execution`)

Listed for owner review, not modified (hard rule). JSON-only state;
`_safe_run_id` sanitizes run ids; `validate_ledger_schema` audits
`broker_state.json` shape + resume fingerprint. Residual if these ever back a
real broker: state files are unsigned — tampering = resume poisoning (L1).
Kill switch is fail-closed on unknown states; flatten requires human
authorization.

### 10. CI/CD (`.github/workflows/`)

| S | T | R | I | E | D | Assessment |
|---|---|---|---|---|---|---|
| | | | ✓ | | | No `${{ secrets.* }}` is used anywhere — nothing to leak; `GITHUB_TOKEN` is `contents: read` in both workflows. New `security.yml` follows the same posture. |
| | ✓ | | | | | `uv sync --frozen` + `uv lock --check` → lockfile is hash-verified; `uv export`-fed `pip-audit --strict` in ci.yml; tag-pinned actions (`@v7`) — recommend SHA pins for third-party actions (listed, N4). No `pull_request_target`, no script injection surfaces (no `${{ github.event.* }}` interpolation into `run:`). |
| | | | | | ✓ | Concurrency groups cancel stale runs; timeouts bound every job. |

### 11. Supply chain

All deps resolve from PyPI into `uv.lock` with recorded hashes — no direct
git/path/registry deps, no extra index → dependency-confusion exposure is
limited to name squatting on *future* additions (mitigated by the lockfile +
review). uvx tool invocations (`bandit==1.9.4`, `pip-audit==2.10.1`) are
version-pinned. `.cursor/install.sh` bootstraps uv via `curl | sh` over HTTPS
— standard but unsigned; dev-env only (N6). Dockerfile pins the base image by
sha256 and `uv==0.11.23`; runtime is non-root with no pip/uv.

### 12. MLflow registry (`quant_fund.registry.mlflow_store`, docker-compose)

Default tracking is local sqlite; `MLFLOW_TRACKING_URI`/compose server are
opt-in and loopback-bound, unauthenticated — accepted dev risk (N7). Champion
alias requires an approved promotion receipt (`promotion_is_approved` is
fail-closed on every field). Residual: a remote tracking URI would export run
metadata off-host — operator decision.

## Findings delta from this review

See `docs/SECURITY_REVIEW.md` — summary: 3 fixed (URL credential redaction,
release-manifest extras check, verified joblib loading), 3 listed for owner
(live/paper-adjacent), ~115 existing bandit findings captured in the new
checked-in baseline (dominated by intentional `assert` validators and
argv-list subprocess usage), 0 committed secrets, 0 lockfile CVEs.

## Assumptions / out of scope

- The operator's filesystem, shell env, and git object store are trusted.
- No multi-tenant serving; the API is single-user research infrastructure.
- `paper`/`execution` are simulators; no real order flow exists to attack.
- Git history was not rewritten-audited; `secret_scan.py --all` covers the
  tracked tree, not history (periodic full-history scan recommended, N9).
