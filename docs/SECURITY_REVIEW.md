# Security Review — findings ledger

Companion to `docs/THREAT_MODEL.md` (STRIDE analysis, trust boundaries). This
page records what the k3/security-review pass found, what was fixed, what was
deliberately listed instead of fixed, and how the new CI gates work.

Method: full read of the credential/network/deserialization surfaces, repo's
own `secret_scan` patterns plus extended regexes over all 2,582 tracked files,
bandit 1.9.4 over `src/` + `scripts/` (all severities), `pip-audit` 2.10.1
over the full `uv.lock` export (all groups + extras), manual review of every
`subprocess`, `os.environ`, `urlopen`, `np.load`, `joblib.load`, `yaml.*`,
`eval`/`exec`, `marshal`, `torch.load` usage. No committed secrets were found.

## Findings

| # | Severity | File / surface | Finding | Status |
|---|----------|----------------|---------|--------|
| S1 | Medium | `src/quant_fund/data/sources/base.py` | `HttpClient` embedded the raw request URL in `SourceError` text. FRED (`api_key=`) and BEA (`UserID=`) authenticate via query params → keys leaked into stderr/logs on any failed request. | **Fixed** — `redact_url()` masks credential-valued params (`SENSITIVE_QUERY_PARAMS`) in every error path; non-HTTP URLs are never echoed. Test: `tests/unit/security/test_url_redaction.py`. |
| S2 | Medium | `src/fx1/serve/signing.py` | `verify_release` checked only manifest-listed artifacts — a file *added* to a signed checkpoint dir (shim `.py`, extra weight, config override) still verified. | **Fixed** — `_artifact_paths()` requires the on-disk file set to equal the manifest exactly (fail-closed, matching repo posture). Test: `tests/unit/security/test_release_manifest_extras.py`. Caveat: serving stacks that write runtime files into the checkpoint dir must write elsewhere. |
| S3 | Low | `src/quant_fund/pipeline/forecast.py` `_paper_challenger_stamp` | Raw `joblib.load` on `ranker_*.joblib` bypassed the checksum-sidecar + manifest verification every other artifact uses. | **Fixed** — now uses `load_joblib_artifact`; identical exception contract (callers already swallow `TypeError`/`ValueError`/`OSError`/`AttributeError`), behavior unchanged when no sidecar exists. Enforced by AST guard test. |
| L1 | Low (latent) | `src/quant_fund/paper/ledger.py` `load_broker_state`, `latest_run_id`, `promotion_dry_run` | `broker_state.json` / `latest_run.json` are unsigned JSON — integrity rests on filesystem trust + post-hoc `validate_ledger_schema`. JSON parsing itself is safe; the latent risk is *state poisoning* if these files ever feed a real order path. | **Listed — paper/sim path, owner review.** Recommendation: hash-chain or sign broker state at write time before any real-order wiring. |
| L2 | Info | `src/quant_fund/execution/simulated_broker.py` | Orders/fills are internal dataclasses serialized via `model_dump` — no untrusted deserialization found. This is the seam a real broker adapter would occupy. | **Listed — paper/sim path.** No change required while simulated; flagging the seam. |
| L3 | Info | `src/quant_fund/paper/sim_live.py` | `np.load` on a quantile-panel cache keyed by sha256 of inputs, `allow_pickle=False` default. Cache poisoning can only corrupt its own input key's output. | **Listed — paper/sim path.** Acceptable; optional `allow_pickle=False` kwarg for explicitness. |
| N1 | Low | `src/quant_fund/models/base.py` | Artifact sha256 sidecars/manifests are **optional and unkeyed** — they catch corruption, not a writer who can also rewrite the sidecar. | **Listed (hardening).** If `data/` artifacts cross a trust boundary, sign sidecars with a keyed MAC (same shape as `release.sig`). Owner decision — changing the contract breaks existing artifacts. |
| N2 | Low | `src/fx1/harness.py` `Harness.run` | Config containment compares against `Path.cwd()/configs` — if the harness runs outside the repo root the allowlist anchors to a wrong (possibly nonexistent) dir. `extra_args` are unvalidated flags to registry commands (the intended tool-use surface). | **Listed.** Recommend anchoring to the repo root (`Path(__file__).parents[N]`) when the harness ships; document "run from repo root" meanwhile. |
| N3 | Info | `src/fx1/data/sources/registry.py` `FX1_PLUGIN_ROOTS` | Env var selects executable plugin scripts — env-controlled code execution by design; child env is scrubbed to PATH/HOME/LANG + declared creds. | **Documented boundary.** Env vars are trusted input; never expose this knob to untrusted callers. |
| N4 | Low | `.github/workflows/*.yml` | Actions are tag-pinned (`actions/checkout@v7`, `astral-sh/setup-uv@v7`), not SHA-pinned; dependabot tracks them weekly. | **Listed.** Recommend SHA-pinning third-party actions. Not changed — repo convention + "do not edit ci.yml" constraint. |
| N5 | Info | `src/fx1/serve/backends.py` | `HostedK3Backend(api_url=...)` accepts an endpoint override; the API key rides in `Authorization` to whatever URL is configured. Constructor input is operator-controlled. Response body is read unbounded. | **Listed (info).** Keep `api_url` operator-only; add a response size cap if the surface grows. |
| N6 | Info | `.cursor/install.sh` | `curl https://astral.sh/uv/install.sh \| sh` bootstrap — TLS-protected but unsigned installer; dev-agent environment only. | **Listed (info).** Consider pinning the installer by version/checksum. |
| N7 | Info | `docker-compose.yml`, `quant_fund.registry.mlflow_store` | Optional mlflow server is unauthenticated; loopback-bound only. `MLFLOW_TRACKING_URI` honored by the client — a remote URI exports run data off-host. | **Accepted dev risk; documented.** Keep loopback binding; do not point the URI at untrusted hosts. |
| B1 | Medium | `scripts/fetch_*.py` (6 files) | Bandit B310: `urlopen` audit flag. All URLs are hardcoded `https://` constants — no user-controlled scheme. Dev-only fetchers. | **Baselined** (`security/baselines/bandit.json`). |
| B2 | Medium | `scripts/fetch_binance_vision_carry.py` | Bandit B314/B405: `xml.etree.ElementTree.fromstring` on the Binance S3 listing response. Python 3.12 ElementTree does not resolve external entities; residual = DoS-ish parse surface on a dev script. | **Baselined.** If this code moves into `src/`, switch to `defusedxml` first. |
| B3 | Low | `src/` + `scripts/` (≈100 findings) | Bandit Lows: `B101` asserts (deliberate fail-closed validation style), `B404/B603` subprocess module/list-argv calls (no `shell=True` anywhere), `B311` `random` (non-crypto use), `B105` false positive on the string `live_pnl_claim`, `B110/B112` except-pass/continue in dev scripts, `B607` partial executable paths (`git`, `dipcatcher` resolved via PATH — operator-trusted). | **Baselined** — new findings at ANY severity now fail `security.yml`. |

## Secrets & env-var audit summary

- **No committed secrets**: `secret_scan` provider patterns + extended
  high-entropy scans over all tracked files → 0 hits. `.env` is gitignored;
  `.env.example` lists the full env surface.
- **No secrets on argv/logs**: plugin creds pass via child env only
  (`fx1.data.sources.base.default_runner` scrubs to an allowlist); doctor
  reports `set`/`unset` flags; receipts hash env-var *names*; HTTP error
  paths now redact credential query params (S1).
- **CI hygiene**: no `${{ secrets.* }}` in use (nothing needs one);
  `GITHUB_TOKEN` is `contents: read`; no `pull_request_target`; no
  `github.event` interpolation into `run:` blocks.
- **Env surface** (all optional, all env-only): `QUANT_API_KEY`,
  `QUANT_VENDOR_API_KEY`, `VENDOR_HTTP_API_KEY`, `FRED_API_KEY`,
  `BEA_API_KEY`, `MOONSHOT_API_KEY`, `FX1_SIGNING_KEY`, `KIMI_API_KEY`,
  `AGENT_GW_TOKEN`, `MLFLOW_TRACKING_URI`, `HF_OHLCV_1M_CACHE`,
  `FX1_PLUGIN_ROOTS`, `QUANT_DATA_ROOT`.

## Unsafe-deserialization audit summary

No `pickle.load(s)`, bare `yaml.load`, `eval`/`exec`, `marshal`, or
`allow_pickle=True` exists under `src/` — now enforced by an AST test
(`tests/unit/security/test_deserialization_guards.py`, 468 files scanned).
`joblib` loads only through the checksum/manifest loaders in
`quant_fund.models.base`. Kronos loads local-only artifacts with optional
pinned sha256 digests.

## Supply-chain / dependency-confusion audit

All dependencies resolve from PyPI into `uv.lock` (recorded hashes; `uv sync
--frozen` + `uv lock --check` in CI). No git deps, no extra indexes → no
private-index confusion vector. `pip-audit` over the full export (all groups
+ extras, 2026-09-27): **0 known vulnerabilities**. uvx tools are
version-pinned; Docker base image is sha256-pinned. `scripts/` ships in the
sdist — dev fetchers are part of the distributed source; keep them
network-explicit.

## The new CI gate — `.github/workflows/security.yml`

Three jobs, all `uvx`/stdlib (no env sync needed):

| Job | Command | Gate |
|---|---|---|
| `secrets` | `uv run --no-project python scripts/secret_scan.py --all` | Any credential-shaped string in any tracked file fails. (New `--all` mode of the pre-commit scanner.) |
| `sast` | `uvx --from bandit==1.9.4 bandit -q -r src scripts -f json` → `scripts/security_baseline_diff.py --tool bandit --baseline security/baselines/bandit.json` | Any bandit finding not in the baseline fails — at *any* severity (stricter than ci.yml's medium+). |
| `deps` | `uv export … | uvx --from pip-audit==2.10.1 pip-audit --format json` → `security_baseline_diff.py --tool pip-audit --baseline security/baselines/pip-audit.json` | Any new advisory fails. Baseline starts empty (clean lockfile). |

Fingerprints exclude line numbers (refactor-safe) but include the flagged
code snippet's hash, so touching a flagged line re-triggers review. A weekly
cron (`0 6 * * 1`) re-runs everything so newly published CVEs against
unchanged pins are caught.

### Updating a baseline (risk-acceptance workflow)

```bash
# after triaging a new finding as accepted:
uv run --no-project python scripts/security_baseline_diff.py \
  --tool bandit --report bandit-report.json \
  --baseline security/baselines/bandit.json --update
# commit the diff — the baseline file is the reviewable record
```

## Verification commands (what was run for this review)

```bash
# secrets sweep (repo scanner + extended patterns, all tracked files)
uv run --no-project python scripts/secret_scan.py --all        # → 0 findings

# bandit full scan + baseline
uvx --from bandit==1.9.4 bandit -q -r src scripts -f json -o bandit-report.json
uv run --no-project python scripts/security_baseline_diff.py \
  --tool bandit --report bandit-report.json --baseline security/baselines/bandit.json

# dependency audit
uv export --format requirements.txt --no-hashes --no-emit-project \
  --all-extras --all-groups \
  | uvx --from pip-audit==2.10.1 pip-audit --format json -r /dev/stdin \
    -o pip-audit-report.json        # → "No known vulnerabilities found"

# unit tests
uv run --no-sync pytest tests/unit/security/ -q                # → 517 passed
```

## Limitations

- Manual review covered secrets/deserialization/network/exec surfaces; it is
  not exhaustive over all 468 source files' logic.
- Git *history* is not scanned (tracked tree only); run a history scanner
  (gitleaks/trufflehog) periodically — recommended follow-up.
- Baseline diffing trusts that baselined findings were triaged honestly; the
  baseline file itself is the audit trail.
- Bandit covers common smells, not dataflow; consider semgrep `p/python` as a
  future second opinion (noted, not installed to keep the workflow zero-dep).
