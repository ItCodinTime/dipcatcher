# P4.6 security evidence — audit receipts, secrets scan, deserialization and input validation

Scope per ULTRAPLAN P4.6: `uv audit`/`pip-audit` receipt, secrets scan
(gitleaks), no-`eval`/no-`pickle-load` audit, input-validation matrix.
Evidence collected 2026-09-28 on commit `2ed19c2` + this PR's dep change;
raw artifacts under `data/metadata/reports/deps_hygiene_20260928/`
(digest-pinned by `MANIFEST.json` and the sealed receipt
`receipts/deps_security_hygiene_<hash>.json`).

## Dependency vulnerability receipts

| Tool | Invocation | Result |
|---|---|---|
| uv 0.12.19 (OSV) | `uv audit --output-format json` | `{"audited_packages": 206, "vulnerabilities": 0, "adverse_statuses": 0}` — `uv_audit.json` |
| pip-audit 2.10.1 (PyPI advisory DB) | `make audit` — `uv export --all-extras --all-groups` piped to `pip-audit --strict` | `No known vulnerabilities found` — `pip_audit.json` (182 audited deps, 0 vulns) |
| bandit 1.9.4 | `make security` (medium severity / medium confidence) | 0 findings over 158k LOC — `bandit_report.json` |

## Secrets scan

- **Local run**: gitleaks 8.30.1 (darwin-arm64 tarball, sha256-verified
  against the release `checksums.txt` before execution — the same version
  CI installs for linux-x64). Command matched CI exactly:
  `gitleaks detect --source . --redact --no-banner --config .gitleaks.toml --log-opts=--full-history`.
  Result: **408 commits / 49.31 MB scanned, no leaks found**
  (`gitleaks_report.json`, empty findings array).
- **CI gate**: `.github/workflows/secret-scan.yml` runs gitleaks 8.30.1 on
  every push to `main` and every PR, sha256-pinned install
  (`551f6fc8…2470eb`), `fetch-depth: 0` + `--full-history` so the scan
  covers the commits reachable from the branch, findings redacted in logs.
  The allowlist is `.gitleaks.toml` — four narrowly-scoped rules bound to
  exact paths and line shapes (public model-revision pins, an mkdocs nav
  line, committed `.dsh-24x7` evidence digests, one synthetic test canary
  `sk-test-SECRET-123`). No blanket hex-string allowlists.
- Also enforced pre-push: the repo's secret-scan pre-commit hook
  (staged-diff) per `SECURITY.md`.

## `eval`/`exec`/unsafe-deserialization audit

- `eval(` / `exec(` over `src/`: **0 hits**.
- `pickle.loads` / `joblib.load` / `torch.load`: only inside
  `src/fx1/forecast/artifacts.py`, behind a fail-closed gate —
  `allow_unsafe_deserialization=true` **and** a 64-hex
  `trusted_checkpoint_sha256` whose digest must `compare_digest`-match the
  file bytes, else `UntrustedArtifactError`. Default load paths (JSON,
  ONNX, `weights_only` torch) never reach `pickle.loads`.
- `yaml.load`/`yaml.unsafe_load`: not used; config loads go through
  `yaml.safe_load` paths.

## Input-validation matrix (`quant_fund.api.app`)

| Surface | Control | Location |
|---|---|---|
| Auth | `X-API-Key` must equal `QUANT_API_KEY` when set; when unset, **loopback-only** (remote unauthenticated access refused) | `api_auth_middleware` |
| Request size | 64 KiB cap enforced twice: declared `Content-Length` (400 invalid / 413 oversize) and streamed body bytes (`_RequestBodyTooLarge`) | `api_auth_middleware` |
| Config path | `resolve_allowed_config_path` — resolves under `configs/` only; traversal/symlink escapes get 400, missing files 404 | `resolve_allowed_config_path` |
| Request models | pydantic `extra="forbid"`, `config_path` `max_length=512` | `OptimizeRequest`, `BacktestRequest` |
| Outbound honesty | `research_only=True` / `live_pnl_claim=False` force-stamped on metric-bearing payloads | `_stamp_research_honesty` |
| Response headers | `nosniff`, `DENY`, `no-referrer`, CSP `default-src 'self'`, `no-store`, HSTS on https | `api_auth_middleware` |

## Residuals / follow-ups

- `uv audit` is still flagged experimental by uv (preview schema); the
  `make audit` pip-audit lane is the stable cross-check — both are green.
- gitleaks local run used the darwin-arm64 artifact (CI runs linux-x64);
  version, config, and scan scope are identical.
- License inventory and deptry findings are documented in
  `docs/AUDIT_P610_DEPS.md`; `cryptography`/`threadpoolctl` are now
  declared direct deps and `pydantic-settings`/`python-dotenv` removed.

Rerun everything: `make audit && make security` plus the commands listed in
`docs/AUDIT_P610_DEPS.md`.
