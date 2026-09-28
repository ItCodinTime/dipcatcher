# Pre-trade risk surface audit — `src/quant_fund/pretrade/`

Scope: every line of `pretrade/{codes,config,state,kernel,engine,spec,shadow,
bench}.py` plus its call graph (`execution/simulated_broker.py`,
`paper/loop.py`, `tests/property/test_pretrade_risk.py`). Hunted: look-ahead,
fail-open paths, unchecked bounds, check/exec TOCTOU, NaN/inf injection,
units errors (bps vs frac), clock misuse, and fail-closed behavior under
degenerate inputs (empty book, zero qty, negative prices, missing marks).
Verdicts: `correct`, `fixed`, `documented` (trust boundary, not a defect).
Regression tests live in `tests/unit/pretrade/{test_engine,test_shadow,
test_config}.py` (synthetic data only).

## Bugs fixed

| fix commit | change |
|---|---|
| `4a7792b` | engine: finite initial nav/cash, finite `set_symbol` pos, finite fill fields incl. `pos_before` and `ts_ns`, finite unsettled proceeds, forward-only session-floor rebase, midnight-close rollover, audit pending trips on reset/internal latch; state+kernel: `ref_ts`/`mark_ts` staleness; kernel: Rule-201 missing-bid deny; shadow: weakref-keyed broker state, pending-fill pop, real `reason_bits`; config: `allow_inf_nan=False` on unbounded float limits |

### `engine.py` — state feeds and cold paths

| claim checked | verdict | evidence / fix |
|---|---|---|
| `__init__` rejects bad initial state (L86) | **fixed** | `x != x` caught NaN only; `initial_cash=inf` → `book.settled=inf` → `qty*px > settled` never true → BUYING_POWER dead (kernel L318). `initial_nav=inf` → `dd_floor=inf` → self-latched kill on first check. Now `math.isfinite` on both. |
| `set_symbol` position (L147) | **fixed** | `pos` was unvalidated; `pos=NaN` makes `new_pos` NaN → POSITION_QTY/NOTIONAL/GROSS/NET/CONCENTRATION comparisons all False → every exposure check vacuous. Now raises. |
| `update_account` mark staleness (L173-182) | **fixed** | only an upper bound (`deadline = stamp + max_age`) was stored; a mark dated in the future kept `mark_deadline` ahead of `eff` forever → STALE never fired. Now `book.mark_ts` is stored and the kernel requires `eff >= mark_ts` (kernel L262). |
| `refresh_session` day roll (L209) | **fixed** | `local != self._session_date` rebased `session_start` and the loss floors on ANY date change, including backward ones — a backdated order ts could reset the daily-loss floor mid-session. Now `local >` only; `_session_date` still tracks the observed date so the next forward day still rolls. Regression test pins `session_start`/`daily_floor` unchanged after a backdated check. |
| `refresh_session` close time (L222) | **fixed** | `start.replace(hour=close_h)` raised `ValueError` for `close_minute=1440` (allowed by the schema) — an exception inside `check` latches INTERNAL|KILL, so a legal config bricked the engine. Now wraps to next-day 00:00. |
| `note_fill` field validation (L325-335) | **fixed** | checked `x != x` (NaN) but not ±inf, and ignored `pos_before`/`ts_ns` entirely; inf `qty`/`px`/`fee` or a NaN `pos_before` poisoned lots/settled/PDT memory silently. Now `isfinite` on every float field plus `ts_ns >= 0`. |
| `_enqueue_unsettled` (L513) | **fixed** | no finiteness check — `note_fill` sell with `qty*px=inf` enqueued inf, which matured into `settled=inf` → BUYING_POWER vacuous. Now returns False → caller raises → INTERNAL latch. |
| `reset` / `_latch_internal` audit (L260, L540) | **fixed** | both zeroed `book.pending_trip` before the audit flush — a circuit-breaker trip latched by a direct `hot_check` call (shadow/tests) left no `trip` record. Now `_flush_pending_trip` runs first. |
| `_set_ref` deadline (L380-395) | **fixed** | same future-dated hole as marks: `deadline = ref_ts + max_age` with no lower bound. `sym.ref_ts` recorded; kernel denies `eff < ref_ts` (kernel L260). |
| `_apply`/`note_fill` lot math, `_consume_lots`, `_mature`, `_push_day_trade` | correct | degenerate sell-below-position, full lot table (STATE_FULL), unsettled FIFO, and PDT window eviction all verified fail-closed; `restore_lot` full table raises (existing test). |
| `check` exception path (L284-288) | correct | any kernel/apply exception → INTERNAL|KILL latch + deny; cancel-block mask then covers INTERNAL (codes.py) so even cancels stop. Existing test covers; kept. |
| `trip`/`reset` actor+reason required | correct | `_require_actor` rejects empty/whitespace; latch persists until manual reset (docs/PRETRADE_RISK.md semantics). |

### `kernel.py` — the hot check

| claim checked | verdict | evidence / fix |
|---|---|---|
| staleness window (L257-263) | **fixed** | see engine rows: now lower-bounded by `ref_ts`/`mark_ts`, upper-bounded by `deadline`/`mark_deadline`; `ref_ok == 0` still STALE. |
| NON_FINITE early return skips rate accounting (L234-245) | correct | intentional: a malformed order must not consume rate budget nor be counted — deny without side effects. |
| NON_MONOTONIC clamps `eff` to `last_ts` | correct | check still runs against last-seen clock; deny bit set; cancels blocked. |
| Rule 201 / REG_SHO (L308-317) | **fixed** | `px + _EPS < sym.bid` is vacuous when `bid <= 0` (no quote ever compares below a non-positive bid) — sho-flagged shorts passed with no market reference. Now `bid <= 0` denies. |
| BUYING_POWER / GFV / PDT | correct | `settled + _CASH_EPS` tolerance documented; GFV only on unsettled-funded lots; PDT counts cross-session round trips and tightens below the equity threshold. All verified with existing + new KATs. |
| `_hash_remove`/`_expire_ring`/`_push_ring`/`_note_duplicate` | correct | ring eviction O(1), duplicate fingerprints expire on their own window, capacity overflow → STATE_FULL bit. Traced by hand + property parity. |
| trip latch (L384-397) | correct | first trip writes `pending_trip` for the audit flush; subsequent trips OR into `kill_reason`. KILL bit applied only for orders (or INTERNAL cancels). |
| units | correct | `collar_frac = bps / 10_000` (state L161); all times ns; concentration = fraction × nav. No bps/frac mixups found. |
| check/exec TOCTOU | correct | `hot_check` mutates book state atomically within the call and `_apply` runs in the same `engine.check` invocation; single-threaded by contract (docs state the caller serializes). No interleaved window exists. |

### `config.py` / `state.py`

| claim checked | verdict | evidence / fix |
|---|---|---|
| float limit bounds (L24-45) | **fixed** | `Field(gt=0.0)` accepted `inf` — an infinite `max_order_quantity`/`max_order_notional`/etc. loads fine and disables the check. `allow_inf_nan=False` on every unbounded float field. |
| `SessionConfig` | correct after fix | `close_minute` up to 1440 is now handled (midnight rollover); `close > open` still enforced. |
| `sign_config` / HMAC / `extra="forbid"` | correct | canonical JSON + sha256 + HMAC; unknown keys and foreign schema versions rejected (existing tests). |
| `__slots__` preallocation | correct | fixed-capacity rings/tables; no hot-path allocation. |

### `shadow.py` — broker observation

| claim checked | verdict | evidence / fix |
|---|---|---|
| per-broker engine keying (L64-66, L148-179) | **fixed** | `_engines`/`_pending` were keyed by `id(broker)` forever; after GC a new broker can reuse the id and bind to a stale book. Now a `weakref` finalizer drops all state for a dead broker; `_pending` entries are popped on first use instead of leaking. |
| `_bare` deny records (L330+) | **fixed** | internal-error/unknown-symbol records emitted `reason_bits=0` while `reasons` said otherwise — the JSONL evidence contradicted the bitset. Now carries INTERNAL/UNKNOWN_SYMBOL. |
| fail-closed wrapping | correct | observer exceptions are logged and emit an `internal_error` deny record without touching the broker result (existing test + strengthened `reason_bits` assert). |

### `codes.py` / `spec.py` / property parity / `bench.py`

| claim checked | verdict | evidence |
|---|---|---|
| `HARD_LIMIT_MASK` / `CANCEL_BLOCK_MASK` / `decision_allowed` | correct | cancels blocked exactly on KILL-with-INTERNAL, STALE, NON_FINITE, MESSAGE_RATE, INTERNAL, STATE_FULL, UNKNOWN_SYMBOL, NON_MONOTONIC — matches docs/PRETRADE_RISK.md. |
| `spec.py` ↔ kernel parity | correct | all 8 predicates agree under the hypothesis sweep in `tests/property/test_pretrade_risk.py`; unchanged by this fix (spec predicates cover the exposure checks, which were already consistent). |
| `bench.py` latency gate | correct | p50 < 5µs / p99 < 20µs self-check; allocation-free hot path confirmed by reading `hot_check` (no allocs on the allow path). |

## Trust boundaries (documented, not fixed)

- **Caller clock is authoritative.** `refresh_session` uses order `ts_ns`; a
  fabricated *forward* date still rolls the session and rebases the daily
  floor (by design — a new trading day resets the floor). A fabricated
  *backward* date no longer rebases it. The floor is only as strong as the
  timestamp feed; `TRAILING_DD` (peak-anchored, cannot ratchet down) is the
  real backstop.
- **Kill latch** persists until an explicit `reset(actor, reason)`; nothing
  in the module clears it implicitly.
- **Holidays** are not modeled (docs/PRETRADE_RISK.md); weekends are closed.
- **Shadow mode only** — the engine never submits; `simulated_broker.py` and
  `loop.py` carry no `quant_fund.pretrade` import (enforced by test).
