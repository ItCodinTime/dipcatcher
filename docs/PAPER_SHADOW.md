# Paper / Shadow (Phase 17)

Simulated paper loop with optional shadow challenger. **Not live broker connectivity.**
SYNTHETIC bars ⇒ research / infrastructure diagnostics only (`live_pnl_claim=false`).

## Quick start

```bash
uv run dipcatcher paper --config configs/paper.yaml --max-steps 20
```

Ledger lands under `data/metadata/<paper.ledger_subdir>/<run_id>/` (default
`data/metadata/paper/<run_id>`): orders, equity, positions, cash ledger,
`broker_state.json`, `promotion_dry_run.json`, `analytics_export.json`, and meta.
`paper.ledger_subdir` must be a safe relative path and is also used for resume
and latest-run discovery.

## CLI flags (`dipcatcher paper`)

| Flag | Meaning |
|------|---------|
| `--config` | Config path (default `configs/paper.yaml`) |
| `--max-steps` | Cap replay steps (accelerated paper) |
| `--shadow` / `--no-shadow` | Enable/disable shadow challenger (no capital) |
| `--wall-clock` | Use wall clock instead of bar replay |
| `--halt` | Force kill switch `HALT_NEW_ORDERS` for this run |
| `--clear-halt` | Force kill switch `ENABLED` (pairs with `--resume`) |
| `--resume` | Resume from `broker_state.json` for `--run-id` / latest |
| `--run-id` | Paper run id (new or resume target) |
| `--from-start` | Prefer earliest decision dates (multi-day grind) |

Resume example:

```bash
uv run dipcatcher paper --config configs/paper.yaml --resume --run-id paper-abc --max-steps 30
uv run dipcatcher paper --halt --resume --run-id paper-abc   # halt mid-grind
```

## Promotion dry-run honesty

`promotion_dry_run.json` / metrics `promotion_dry_run`:

- Newly generated receipts include the paper ledger `run_id`; ledger validation
  rejects a receipt copied from a different run (`promotion_run_id_mismatch`).
- Newly generated receipts also include a canonical self-excluding
  `receipt_sha256`; any mutation is rejected as
  `promotion_receipt_sha256_mismatch`.
- Receipt validation also rejects impossible promotion claims, including
  inconsistent mean/max divergence, insufficient steps, threshold breaches,
  and promotion after a kill switch.
- A no-shadow run may carry `NaN` divergence only with
  `would_promote_paper=false` and the explicit `missing_divergence` reason;
  this diagnostic remains resumable without weakening promotion gates.
- Broker state persists a deterministic prefix/config fingerprint; malformed
  fingerprints fail ledger validation before resume.
- Broker state also persists the runtime kill-switch state. A resumed run therefore
  remains blocked after a prior `HALT_NEW_ORDERS` trip; the resumed promotion
  receipt also preserves the kill-trip veto. Legacy state without this field
  keeps the configured initial state for backward compatibility.
- `would_promote_paper` — infrastructure gate (steps, L1 divergence, kill mid-run).
- **`would_promote_live` is always `false`** — paper dry-run never touches live capital or aliases.
- SYNTHETIC evidence always appends `synthetic_evidence_not_live_promotable`.
- `live_pnl_claim=false`, `research_only=true`.

Do **not** treat paper Sharpe / vol / DD as live P&L. Execution diagnostics are
labeled `role=1` / research-only.

## Schema validation

```python
from quant_fund.paper import validate_ledger_schema
report = validate_ledger_schema("data/metadata/paper/<run_id>")  # default subdir
assert report["ok"]
```

Fails closed on `live_pnl_claim=true` or `would_promote_live=true` in on-disk artifacts.

Promotion dry-run receipt validation (`validate_promotion_dry_run_receipt`) also
**errors** when honesty flags are missing or wrong (`would_promote_live≠false`,
`live_pnl_claim≠false`, `research_only≠true`).

## Multi-challenger shadow

Beyond the primary `--shadow` challenger (no capital), the paper loop accepts:

- `challenger_weights: dict[str, DataFrame]` — named static weight panels
- `challenger_fns` — callables producing challenger weights per step

**Config / CLI (Wave 17):** optional `paper.challenger_scales: list[float]` in
`configs/paper.yaml` (default `[]`). The `dipcatcher paper` CLI builds extra scaled
panels via `build_scaled_challenger_weights(champion, scales)` into
`challenger_weights` named `scale_<g>` (duplicate scales get a `_i` suffix).
The primary shadow (`enable_shadow` / `shadow_scale` × 0.85) stays separate for
ledger and `promotion_dry_run` compatibility.

Rolling L1 divergence vs champion is aggregated by
`rolling_challenger_metrics(...)` and embedded in `promotion_dry_run` schema v2
(`rolling_l1`, challenger snapshot, `risk_accounting`, `kill_tripped_mid_run`).
Closest/farthest challenger labels are research diagnostics only — never a live
promotion signal. All paper metrics keep `live_pnl_claim=false`.

## Analytics export + validation

Paper writes `analytics_export.json` under the run ledger and validates it
in-loop:

```python
from quant_fund.metrics import validate_analytics_export
report = validate_analytics_export(metrics["analytics_export"])
assert report["ok"]  # fail-closed on live_pnl_claim=true / research_only=false / missing keys
```

Shared schema: `ANALYTICS_SCHEMA_KEYS` (exposure, execution, equity,
drawdown_duration, var_es, stress, stress_report, …). Backtest
`export_backtest_metrics_json(...)` uses the same keys. Both force
`live_pnl_claim=false`.

Loop metrics also surface `analytics_export_validation` and
`analytics_export_ok`.

## Long resume stress honesty (200-step)

Unit stress (`test_paper_200_step_resume_analytics_export`): 110+100 resume →
`n_steps≥200`, `analytics_export.json` present + `validate_analytics_export` ok,
`would_promote_live=false`, `live_pnl_claim=false`. This proves **infrastructure
persistence**, not live edge. Do not quote NAV/vol/DD from that stress as live
P&L.

## What this is not

- Not a vendor market-data adapter
- Not a live broker / real fills
- Not permission to claim live Sharpe from SYNTHETIC paper smoke
- Not unsafe parallel causal dates (`w_prev` remains sequential)
