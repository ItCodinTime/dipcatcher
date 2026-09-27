# Research API — read-only evidence service

`quant_fund.api.research_api` is a standalone FastAPI app that serves the
lab's research evidence over HTTP for local tooling (the TypeScript client in
`clients/typescript/`, notebooks, dashboards). It is **read-only**: every route
is `GET`, and there are no trading, order, broker, execution, or write
endpoints anywhere in the app. That property is enforced by the route audit in
`tests/unit/api/test_research_api.py` (`test_no_trading_or_write_routes`), not
just by convention.

It is separate from `quant_fund.api.app` (the forecast/optimize service): this
app never runs a model, never touches a config-driven data pipeline, and never
mutates evidence — it only reads sealed artifacts and runs the repo's own
verifiers on copies of them.

## What it serves

| Evidence | Source | Routes |
|---|---|---|
| Research run notebooks | `<data_root>/metadata/research/runs/<run_id>.json` (+ `.md` twins, `latest.json`) | `/runs`, `/runs/latest`, `/runs/{run_id}`, `/runs/{run_id}/markdown`, `/runs/{run_id}/verification` |
| Committed receipts | `<repo>/receipts/*.json` | `/receipts`, `/receipts/{id}`, `/receipts/by-hash/{sha256}`, `/receipts/{id}/verification` |
| Sealed benchmark runs | `<data_root>/metadata/<kind>/<name>/` dirs with `manifest.json` (`real_benchmark`, `net_tournament`, `cost_aware_tournament`, …) | `/results`, `/results/{kind}/{name}`, `/results/{kind}/{name}/verification` |
| Verifier acceptance history | `<repo>/verifier/vN/acceptance.md`, `<repo>/verifier/runs/*.md` | `/verifier/versions`, `/verifier/versions/{v}`, `/verifier/runs`, `/verifier/runs/{id}` |
| Committed artifacts | `<repo>/artifacts/**/*.json` | `/artifacts`, `/artifacts/{id}` (nested ids keep `/`) |

All responses are Pydantic models (`RunListResponse`, `ReceiptDetail`,
`VerificationResult`, …). Receipts and notebooks are returned **verbatim** —
they are immutable evidence; the API never rewrites their fields. Honesty
flags (`claim: "research_only"`, `live_pnl_claim: false`) live on the envelope
models, not injected into evidence documents.

## Verification semantics

`{…}/verification` routes run the repo's own fail-closed verifiers and report
what they say — no editorializing:

- `kind: "research_run"` → `quant_fund.research.verify.verify_research_artifact`.
  Run receipts are the *immutable twins*; their `artifacts.immutable_*`
  pointers are relative to `metadata/research/`, so the API stages the
  declared layout (`latest.json` + `runs/<id>.{json,md}`) in a temp dir before
  verifying (`context: "staged_declared_layout"`). Verifying a twin file in
  place would misreport `artifact_missing` context errors.
- `kind: "benchmark_run"` → `quant_fund.research.phase1_verify.verify_phase1_run`
  over the sealed run directory. `state` may be `"blocked"` for a frozen
  tournament (see `docs/RECEIPT_VERIFICATION.md`).
- `kind: "receipt"` → a deliberately narrow honesty check
  (`verifier: "receipt_honesty_flags"`): the committed receipts are
  schema-tagged evidence documents, not research notebooks, so the check is
  *valid JSON object + `schema` tag + `research_only == true` +
  `live_pnl_claim == false`*.

## Run it

```bash
# repo checkout; real lab data lives in ./data (gitignored)
uv run --no-sync python -m quant_fund.api.research_api
# equivalent explicit form:
uv run --no-sync uvicorn quant_fund.api.research_api:app --host 127.0.0.1 --port 8010

# or point at another evidence root
RESEARCH_API_DATA_ROOT=/path/to/data \
  uv run --no-sync python -m quant_fund.api.research_api
```

`python -m quant_fund.api.research_api` reads `RESEARCH_API_HOST` /
`RESEARCH_API_PORT` (defaults `127.0.0.1:8010`) and refuses to bind a
non-loopback host when `RESEARCH_API_KEY` is unset — same guard as
`dipcatcher api`.

Environment variables:

| Var | Default | Meaning |
|---|---|---|
| `RESEARCH_API_DATA_ROOT` | `$QUANT_DATA_ROOT`, else `<repo>/data` | metadata tree root (research runs + benchmark dirs) |
| `RESEARCH_API_RECEIPTS_DIR` | `<repo>/receipts` | committed receipt dir |
| `RESEARCH_API_VERIFIER_DIR` | `<repo>/verifier` | acceptance versions + run logs |
| `RESEARCH_API_ARTIFACTS_DIR` | `<repo>/artifacts` | committed artifact dir |
| `RESEARCH_API_KEY` | unset | when set, all non-`/health` routes require `X-API-Key` |
| `RESEARCH_API_HOST` / `RESEARCH_API_PORT` | `127.0.0.1` / `8010` | bind address for `python -m quant_fund.api.research_api` |

## Security posture

- Binds `127.0.0.1` by default. When `RESEARCH_API_KEY` is **unset** the app
  refuses non-loopback clients outright (secure default, same posture as the
  main API). When set, non-`/health` routes require `X-API-Key`.
- **Never expose this service with trading credentials** — it has none and
  cannot reach any — and never bind it non-loopback without `RESEARCH_API_KEY`.
- Evidence paths are containment-checked, including enumerated files and
  Markdown twins. Escaping symlinks are rejected. Keep mounted evidence roots
  read-only; this service is not a sandbox for concurrently hostile writers.
- JSON parse caches use current content hashes. Run verification rechecks both
  the JSON receipt and Markdown twin on each request.
- Security headers: `nosniff`, `no-store`, `no-referrer`.

## Docker

```bash
docker build -f docker/research-api.Dockerfile -t dipcatcher-research-api .
docker run --rm -p 127.0.0.1:8010:8010 \
  -e RESEARCH_API_KEY \
  -v /path/to/data:/app/data:ro dipcatcher-research-api
```

Committed evidence (`receipts/`, `verifier/`, `artifacts/`) is baked into the
image; `data/metadata` is dockerignored, so mount a data root read-only as
above. Export `RESEARCH_API_KEY` in the launching shell first. The container
runs as a non-root user and binds all container interfaces so the loopback
host port mapping works. Its entry point refuses to start without the key.

## TypeScript client

`clients/typescript/` contains the committed `openapi.json`, a generated
`schema.d.ts` (openapi-typescript), and a zero-dependency typed `client.ts`.
See `clients/typescript/README.md`. Regenerate the spec with:

```bash
uv run --no-sync python scripts/export_research_api_openapi.py
```

## Tests

```bash
uv run --no-sync pytest tests/unit/api -q
```

Contract tests validate every route's response against its model, check the
OpenAPI schema is self-consistent (all `$ref`s resolve), audit the route table
for forbidden trading/write surfaces, exercise auth (loopback default, API-key
mode), path-traversal rejection, corrupted-evidence fail-closed behavior, and
a Hypothesis sweep asserting arbitrary path params never produce a 500.
