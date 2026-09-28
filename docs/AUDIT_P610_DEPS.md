# P6.10 dependency hygiene audit — pin audit, uv audit, license scan, dead deps

Scope per ULTRAPLAN P6.10: pin audit, `uv audit` receipt, license scan,
dead-dep removal. Run on `macOS arm64`, Python 3.12.14, uv 0.12.19 against
the locked environment (`uv.lock`, resolved 207→206 packages after the
hygiene change below). Raw artifacts: `data/metadata/reports/deps_hygiene_20260928/`;
the sealed receipt `receipts/deps_security_hygiene_<hash>.json` binds the
sha256 of each artifact. See `docs/SECURITY_EVIDENCE.md` (P4.6) for the
secrets-scan and deserialization controls.

## Verdicts

| Check | Command | Result |
|---|---|---|
| Lock consistency | `uv lock --check` | clean — `uv.lock` matches `pyproject.toml`, `--frozen` sync passes |
| Single-version pin | `uv export --all-extras --all-groups` | 206 pinned rows, **0 duplicated package names** (numpy duplication fixed in #191 stays resolved) |
| Vulnerability audit (lockfile-native) | `uv audit --output-format json` | **0 vulnerabilities, 0 adverse statuses** across 206 packages (OSV service) |
| Vulnerability audit (PyPI DB cross-check) | `uv export … \| uvx --from pip-audit==2.10.1 pip-audit --strict` | **No known vulnerabilities found** over the full pinned export |
| License scan | `importlib.metadata` over the installed env + PyPI metadata for platform-only pins | **0 GPL/AGPL/LGPL** runtime deps; 4 weak-copyleft `MPL-2.0` (certifi, hypothesis, pathspec, tqdm); 6 blank metadata fields |
| Static security analysis | `make security` (bandit 1.9.4, medium+/medium+) | 0 findings (158k LOC scanned) |
| Dead/misdeclared deps | `deptry 0.25.1 src` | fixed below |

## Dead-dep removal + declaration fixes (this PR)

- **Removed** `pydantic-settings` — zero imports anywhere in the repo (src,
  tests, scripts, web, deploy). Dropped from the locked env.
- **Removed** `python-dotenv` from direct deps — zero `load_dotenv`/`dotenv`
  usage repo-wide; env config is documented via `.env.example` and read
  straight from `os.environ`. It remains in `uv.lock` only as a transitive
  dep of `mlflow`.
- **Declared** `cryptography>=44` — `quant_fund.audit.signing` imports it at
  module level (Ed25519 receipt signing) but previously relied on a
  transitive pull: if the provider had dropped it, `import signing` would
  break. Now an explicit direct dep.
- **Declared** `threadpoolctl>=3.5` — `research/receipt_v2.py` imports it
  (guarded) for the BLAS-pool environment fingerprint; previously
  transitive via scikit-learn.

No vulnerability-driven version bump was needed: both audit tools report
zero findings, so no direct dep required a fix-version bump. Pin discipline
unchanged — all requirements keep their `>=` floor form
(`exchange-calendars==4.13.2` remains the single exact pin, intentional).

## License detail

206 locked pins covered: 184 installed packages scanned via dist metadata,
plus 22 platform-only pins (NVIDIA CUDA wheels, triton, pywin32, waitress,
greenlet — not installed on darwin-arm64) resolved via PyPI metadata.

- GPL-family (GPL/AGPL/LGPL/SSPL/CC-BY-SA): **none**. No distribution
  blocker in any runtime dep.
- `MPL-2.0` (weak file-level copyleft, safe for library use): certifi
  2026.7.22, hypothesis 6.168.0 (dev), pathspec 1.1.1, tqdm 4.70.1.
- Blank license metadata (6): `huey` — metadata empty but bundled
  `dist-info/licenses` is MIT text; `cuda-toolkit`, `nvidia-cuda-runtime`,
  `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `nvidia-nvshmem-cu13` — all
  NVIDIA proprietary EULA packages (siblings declare
  `LicenseRef-NVIDIA-Proprietary`), platform-conditional only.
- Distribution of the wheel bundles only permissive licenses (MIT/BSD/
  Apache-2.0/ISC/PSF dominant) — no attribution-only surprises.

## Remaining deptry notes (accepted, documented)

- `statsmodels` DEP002: not imported under `src/`; used by
  `tests/unit/core/test_regression.py` and `scripts/sota_eval_kronos.py`.
  Kept in runtime deps — the SOTA script is shipped tooling; a dev-group
  move is a packaging decision left for the owner.
- `clarabel` DEP002: never imported by name — it is the cvxpy solver the
  portfolio layer selects (`solver="CLARABEL"` in
  `research_allocators.py`, `cost_allocation.py`) and version-stamps into
  receipts. Required runtime dep; deptry cannot see solver-string usage.
- DEP001 optional integrations (`sigstore`, `huggingface_hub`, `ray`,
  `quant_core`, third-party `kronos` `model` modules): all behind
  `import_optional(...)` / lazy loading — deliberately optional, not
  missing declarations.

## Reproduce

```bash
uv lock --check
uv audit --output-format json
uv export --format requirements.txt --no-hashes --no-emit-project \
  --all-extras --all-groups | uvx --from pip-audit==2.10.1 pip-audit --strict -r /dev/stdin
make security
uv run --with deptry==0.25.1 deptry src
```

The full license inventory (`licenses_all.json`), both audit JSON reports,
the bandit JSON report, the deptry output, and a `MANIFEST.json` of sha256
digests live in `data/metadata/reports/deps_hygiene_20260928/`.
