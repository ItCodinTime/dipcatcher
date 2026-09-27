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

Deterministic replay and seeded fault injection of this loop live in
`docs/DETERMINISTIC_SIMULATION.md`. That harness is simulation-only.

## Phase 1 paired forward paper protocol

`dipcatcher paper --forward-stage ...` is an **opt-in simulated-only** adapter for
the published Phase 1 momentum-20 and equal-weight comparison. It does not
turn the earlier `--shadow` slot into a capitalized challenger. Four isolated
paper books give both strategies identical initial capital and the frozen
commission, spread, participation, risk limit, borrow, financing and impact
inputs; the other two books double the impact coefficient. They use the same
causal rank/common-universe functions as the retrospective tournament.

The existing retrospective `net_replay` and the paper `SimulatedBroker` have
**distinct execution gates**: the latter can reject a risk-reducing sell while
the post-trade book remains over its hard exposure limit; the former allows
that sell to reduce the breach. The modeled paper result is therefore a
prospective comparison under the explicitly frozen paper execution contract,
not bit-for-bit parity with the earlier backtest. Both use close-time target
quantities, later next-session opens, lagged dollar ADV/volatility, the same
participation cap and transaction-cost formula. Neither observes venue fills.

First commit this adapter, the pre-collection power analysis in
`docs/FORWARD_SHADOW_POWER.md`, and all configuration changes. In a **clean
checkout**, request the protocol commitment:

```bash
uv run dipcatcher paper --forward-stage commitment
```

Obtain an independently recorded freeze file containing exactly
`recorded_at` (offset-aware timestamp), `issuer`, `reference`, and the printed
`commitment_sha256`. The commitment binds the exact Git commit and clean
worktree digest, both printed for review with the schedule hash. The record
must follow every historical warmup event,
availability, and ingestion timestamp, including late ingestions on older
rows.
Then create a new, exclusive run directory:

```bash
uv run dipcatcher paper --forward-stage freeze \
  --forward-attestation /path/to/external-freeze.json \
  --forward-run data/metadata/paper_forward/<run_id>
```

No freeze or forward record is supplied with this adapter. The phase-one
2025/2026 historical snapshot is **warmup only**; the first accepted decision
must occur on a *later market date* than the recorded freeze. The program
checks the published benchmark and validation receipts through the sealed
historical evidence index, clean Git state, the
exact `configs/net_tournament.json` slate, source file hashes, the power-plan
hash, and the static 424-name universe from the final warmup session. No other
strategy or comparator can be selected through this mode.

The protocol also commits an offline `exchange_calendars==4.13.2` XNYS session
schedule from 2026-09-18 through 2034-12-31. It requires exact scheduled close
and next-session open instants, including DST and early closes. The first
decision must use the first scheduled session after freeze; later close packets
cannot skip scheduled sessions. A claimed closure on a scheduled session fails
verification. This schedule is a planning input, not independent exchange
attestation. Published NYSE hours currently extend only through 2028; the
2029–2034 calendar is projected and needs comparison with each new official
NYSE publication. Any discrepancy stops the current protocol version and
requires a new freeze. `official_comparison_verified=false` and no forward
evidence is accepted by this adapter.

The indexed historical runs were produced with a recorded runtime that may
differ from the current environment. Archive verification checks the seals,
source commit, config and dataset links without claiming to replay those runs
under today's dependency versions. Standalone research-run verification retains
its strict current-runtime check. Forward verification requires the exact clean
Git checkout named by the freeze. Keep a dedicated clean checkout at that
commit for restart and verification. Even unrelated later docs or test edits
block verification in the frozen checkout until restored; a later commit cannot
silently reinterpret the frozen protocol.

For each genuinely new session, supply a close packet with `kind` set to
`forward_shadow_close`, `source` set to `yahoo`, all frozen names in `bars`,
and `external_attestation` containing `issuer`, `reference`, and
`recorded_at`. Each bar has `security_id`, `event_time`, `available_time`,
`ingested_time`, `close`, and `volume`. Timestamps have UTC offsets. Any
scheduled exchange closure between the previous close and current close must appear in
`missed_sessions` as `{ "date": "YYYY-MM-DD", "reason": "market_closed" }`.
Feed downtime, a missing name, or a skipped scheduled session cannot be hidden
as a market closure; interrupt the protocol version. These calendar claims
require independent review.

```bash
uv run dipcatcher paper --forward-stage decide \
  --forward-run data/metadata/paper_forward/<run_id> \
  --forward-packet /path/to/new-close.json
```

The close receipt records decision observation time, eligible names, target
weights and every intended signed quantity. It has **no open price or fill**.
The corresponding separately captured open packet has
`kind: forward_shadow_open`, the same source, all frozen names, and offset-aware
`event_time`, `available_time`, `ingested_time`, and `open` per bar. If an
exchange weekday is closed between decision and open, list it in
`market_closures` with `reason: market_closed`.

```bash
uv run dipcatcher paper --forward-stage execute \
  --forward-run data/metadata/paper_forward/<run_id> \
  --forward-packet /path/to/next-open.json
uv run dipcatcher verify-research data/metadata/paper_forward/<run_id>
```

Only an open strictly after the **observed** close decision can fill orders.
Each event is exclusively numbered and SHA-256 sealed to the prior event.
Restart reconstructs the cursor, pending orders and four brokers from the
whole chain; verifier checks source timing, attestation fields, complete
frozen-name coverage, skipped dates, and every close decision against the
frozen strategy, packet and prior broker state. It recomputes target weights,
intended orders and the entire close-state snapshot, then checks fills,
rejects, cash events, position deltas, participation, costs, NAV and paired
net differences.
`paired_sessions` advances only after both books in both impact scenarios
complete. The power plan requires **1,400 eligible paired sessions** before
its single planned analysis; this adapter makes no success or promotion claim.
If CLI close/open input validation fails, the command writes an interruption
event with the input packet digest and error and blocks the run. Accidental
repeated phase commands do not interrupt. If a corrupted prior journal prevents
the interruption receipt from being written, the CLI prints
`interruption_not_recorded` and the run cannot be accepted. When the feed is
missing or downtime is known before a packet arrives, record it immediately:

```bash
uv run dipcatcher paper --forward-stage interrupt \
  --forward-run data/metadata/paper_forward/<run_id> \
  --forward-reason no_feed
```

An interrupted run can be verified but cannot resume collection. A future
protocol version needs a new freeze and new run; it cannot erase this failure.
Unobserved missed days still cannot be detected automatically. A complete
power-plan evidence claim needs independent session/calendar monitoring and
external attestation as well as the counted paired events.

**External feed blocker:** the frozen tournament has zero decision delay.
Consequently, a new close and its volume must be available no later than the
exact close `event_time`; a delayed Yahoo daily feed will fail this cutoff.
Changing it requires a newly committed, powered, externally recorded protocol
before collecting outcomes. The frozen 424-name complete-feed requirement can
also halt ordinary delistings, missing opens, and partial sessions. This is a
bounded fail-closed prototype awaiting a qualifying prospective feed, not a
general vendor connector. Supplied attestation references and market-closure
claims are **not independently authenticated** (`external_attestation_verified=false`).
The local hash chain detects accidental changes but can be resealed by anyone
who controls all local files unless receipts are anchored outside the run.
Close decisions are deterministically replayed using the same frozen ranking
implementation as collection. A resealed change to weights, orders, cash or
positions that conflicts with the packet and prior state is rejected. This is
not an independent implementation of the strategy
(`independent_strategy_replay=false`). The verifier does not replay every
broker risk rejection.
An external reviewer must verify source timestamps, exchange calendars,
closures and independently anchored packet hashes before treating a session
as forward evidence. Neither a valid receipt nor an external anchor
constitutes live broker evidence.
