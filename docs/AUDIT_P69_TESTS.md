# P6.9 test-quality audit — mutation spot-checks on money paths

Scope: `src/quant_fund/execution/costs.py` (`total_cost` fee accrual),
`src/quant_fund/execution/simulated_broker.py` (`_limit_fill_price`, `nav`,
`exposures`, `_attempt_fill`, `cash_nav_identity`),
`src/quant_fund/backtest/engine.py` (overlay scale, missing-mark flatten,
participation cap, cash guard),
`src/quant_fund/backtest/perp_engine.py` (leverage cap, participation cap,
settlement/vwap block, fee debit, liquidation victim selection, metric
accumulators).

## Method

A deterministic hand-rolled AST mutator (`scripts/mutation_spotcheck.py`)
enumerates position-pinned mutants over the targeted functions: comparison
complements (`<=`↔`>` etc.), boundary flips (`<=`↔`<`), arithmetic swaps on
money terms (`+`↔`-`, `*`↔`/`), augmented-assign sign flips, `min`↔`max`,
`np.sign`/`np.isfinite` negation, and `abs()`/`np.sqrt` argument drops. Each
mutant is applied as a byte-precise source splice, `ast.parse`-verified, run
against the module's covering test slice (per the `mutation_scores.json`
convention), then reverted. Kill = pytest exits nonzero; survive = suite
green under the mutant.

**279 mutants** seeded across 25 target functions: 26 (costs), 85
(simulated_broker), 66 (engine), 102 (perp_engine). Baseline runs verified
the slices green on clean code first; three pre-existing baseline failures
on macOS were excluded from the engine slice (`test_fast_replay_byte_identity`
byte-IPC diff, `test_leakage_ast_scan` stale allowlist,
`test_public_api` stale API snapshot — all unrelated to money paths).

**Initial run: 192 killed, 87 survived.** After adding the regression tests
in `tests/unit/backtest/test_p69_mutation_audit.py` (32 tests), every real
gap is killed. **Final matrix: 271 killed, 8 surviving — all 8 provably
equivalent mutants** (see "Equivalent mutants" below).

## Assertion gaps found (surviving mutants, now killed)

Root causes, in order of prevalence:

1. **Bounds asserted, values never asserted.** Tests checked `gross <= cap`
   but never `gross == cap`; `liquidation_count >= 1` but never the exact
   fee or post-liquidation NAV. Any arithmetic slip that still satisfies the
   bound survived (perp cap headroom, fee formula, turnover accumulator).
2. **Short positions never exercised.** `nav()`, `exposures()`,
   `cash_nav_identity()`, participation caps, and perp settlement were only
   ever tested with longs, so every `abs()` drop and sign flip on the sell
   side survived.
3. **Flat-price fixtures.** With `open == close == entry`, `price - entry`
   is identically zero and settle/flip/vwap arithmetic is unobservable.
   Pyramid/partial-close and price-moving fixtures were needed.
4. **Counters and metric accumulators unasserted.** `reject_count`,
   `risk_gate_rejects`, `mean_turnover`, the `commission`/`spread`/`impact`
   buckets, and the perp `net` column had no assertions; sign flips on the
   accumulators survived.
5. **Boundary/touch conditions untested.** Limit fills at exactly
   `low == limit` / `high == limit`, exact dust-filter thresholds, and
   `nav == 0` guards.
6. **Guard-path coverage.** The insufficient-cash-by-fees margin, the
   missing-mark flatten for held positions, the invalid-NAV reject, and
   liquidation victim ordering were never directly observed.

## Surviving mutants → killing tests

| Mutant | File:line | Mutation | Why the suite missed it | Killed by |
|---|---|---|---|---|
| M020 | costs.py:82 | `abs(quantity)` → `quantity` | `total_cost` never called with a signed/negative qty | `test_total_cost_symmetric_for_signed_qty` |
| M023 | costs.py:86 | `abs(notional)` → `notional` | same — sell-side fee symmetry unasserted | `test_total_cost_symmetric_for_signed_qty` |
| M028 | simulated_broker.py:41 | `lo <= limit` → `lo < limit` | exact-touch boundary never tested | `test_buy_limit_fills_on_exact_low_touch` |
| M031 | simulated_broker.py:44 | `h >= limit` → `h > limit` | same, sell side | `test_sell_limit_fills_on_exact_high_touch` |
| M034 | simulated_broker.py:121 | `abs(float(quantity))` → `float(quantity)` | `nav()` never called holding a short | `test_nav_requires_mark_for_short_position` |
| M042 | simulated_broker.py:139 | `nav == 0.0` → `nav != 0.0` | zero-nav book never tested | `test_exposures_zero_nav_returns_zero_tuple` |
| M043 | simulated_broker.py:142 | gross: `abs(q*p)` → `q*p` | only longs held; gross == net → invariant | `test_exposures_values_for_mixed_book` |
| M044 | simulated_broker.py:142 | `q*p` → `q/p` | same | `test_exposures_values_for_mixed_book` |
| M045 | simulated_broker.py:143 | net: `q*p` → `q/p` | net exposure value never asserted | `test_exposures_values_for_mixed_book` |
| M046 | simulated_broker.py:144 | `gross / nav` → `gross * nav` | `exposures()` values never asserted | `test_exposures_values_for_mixed_book` |
| M047 | simulated_broker.py:144 | `net / nav` → `net * nav` | same | `test_exposures_values_for_mixed_book` |
| M052 | simulated_broker.py:428 | cap: `abs(exec_qty)` → `exec_qty` | participation cap never bound on a sell | `test_participation_cap_binds_buy_and_sell` |
| M054 | simulated_broker.py:429 | `exec_qty > 0` → `<= 0` | cap direction never tested | `test_participation_cap_binds_buy_and_sell` |
| M055 | simulated_broker.py:429 | `-1.0` → `1.0` | same | `test_participation_cap_binds_buy_and_sell` |
| M059 | simulated_broker.py:440 | `reject_count += 1` → `-= 1` | invalid-nav reject counter unasserted | `test_missing_mark_reject_increments_reject_count` |
| M061 | simulated_broker.py:444 | `current_shares * price` → `/` | name-gate `current_w` units untested | `test_name_gate_uses_current_weight_units` |
| M064 | simulated_broker.py:448 | projected gross: `abs(q*p)` → `q*p` | gross gate never tested with a held short | `test_gross_gate_sees_short_plus_long_book` |
| M065 | simulated_broker.py:448 | `q*p` → `q/p` | gross-gate units unasserted | `test_gross_gate_units_are_shares_times_price` |
| M066 | simulated_broker.py:449 | net: `q*p` → `q/p` | net-gate units unasserted | `test_net_gate_uses_signed_exposure_units` |
| M068 | simulated_broker.py:450 | participation `q*p` → `q/p` | participation gate never tripped | `test_order_rejected_when_participation_breaches_gate` |
| M078 | simulated_broker.py:481 | `exec_qty > 0` → `<= 0` | cash-check path untested on buys | `test_buy_rejected_when_cash_covers_notional_but_not_fees` |
| M080 | simulated_broker.py:481 | `notional + costs` → `-` | same | same |
| M081 | simulated_broker.py:482 | `reject_count += 1` → `-= 1` | counter unasserted | same |
| M093 | simulated_broker.py:519 | `is_partial`: `abs(exec_qty)` → `exec_qty` | a *full* sell fill never asserted `is_partial is False` | `test_partial_sell_fill_keeps_residual` |
| M094 | simulated_broker.py:519 | `abs(requested_signed)` → `requested_signed` | same | same |
| M097 | simulated_broker.py:527 | residual `abs(requested_signed)` → `requested_signed` | residual qty unasserted on sells | same |
| M098 | simulated_broker.py:527 | residual `abs(exec_qty)` → `exec_qty` | same | same |
| M108 | simulated_broker.py:588 | `cash_nav_identity` mv: `q*p` → `q/p` | identity checked only for all-long book | `test_cash_nav_identity_position_market_value` |
| M127 | engine.py:394 | `overlay_scale != 1.0` → `==` | overlay halt observed via nav only, fills unasserted | `test_overlay_scale_applies_to_positions` |
| M139 | engine.py:434 | `adv / price` → `adv * price` | participation cap never bound in engine | `test_participation_cap_bounds_engine_fill_qty` |
| M141 | engine.py:435 | `abs(delta)` → `delta` | sell side uncapped untested | same |
| M142 | engine.py:436 | `np.sign(delta) * max_qty` → `/` | cap magnitude unasserted | same |
| M143 | engine.py:436 | `np.sign(delta)` → `-np.sign` | same | same |
| M146 | engine.py:450 | participation `q*p` → `q/p` | engine participation gate never tripped | `test_participation_gate_rejects_oversized_order` |
| M148 | engine.py:451 | `order_seq += 1` → `-= 1` | order ids on fills unasserted | engine fill tests in this file |
| M186 | perp_engine.py:83 | net col: `qty*mark` → `/` | `net` column never asserted | `test_perp_net_notional_signs_with_position` |
| M194 | perp_engine.py:282 | dust: `abs(delta)*price` → `/` | small-but-executable delta untested | `test_perp_dust_filter_allows_small_executable_delta` |
| M203 | perp_engine.py:293 | `cap - others_gross` → `+` | multi-asset headroom untested | `test_perp_leverage_cap_headroom_accounts_for_other_positions` |
| M207 | perp_engine.py:295 | `sign*room` → `sign/room` | cap-to-exact-room unasserted | same / `test_perp_leverage_cap_reaches_cap_not_below` |
| M208 | perp_engine.py:295 | `np.sign` → `-np.sign` | same | same |
| M209 | perp_engine.py:296 | `capped_delta == 0` → `!=` | zero-room reject unasserted | cap headroom test |
| M214 | perp_engine.py:300 | post-cap dust `q*p` → `q/p` | cap→dust interplay untested | `test_perp_participation_cap_bounds_fill_qty` |
| M217 | perp_engine.py:303 | `adv / price` → `adv * price` | perp participation cap never bound | same |
| M218 | perp_engine.py:304 | `abs(delta)` → `delta` | sell-side cap untested | same |
| M219 | perp_engine.py:305 | `sign*max_qty` → `/` | cap magnitude unasserted | same |
| M220 | perp_engine.py:305 | `np.sign` → `-np.sign` | same | same |
| M221 | perp_engine.py:346 | notional `delta*price` → `/` | turnover/metric accumulators unasserted | `test_perp_metrics_turnover_and_cost_buckets` |
| M228 | perp_engine.py:353 | `cash += realized` → `-=` | flip settle: flat prices made `p-e ≡ 0` | `test_perp_flip_settles_realized_pnl` |
| M230 | perp_engine.py:353 | `price - old_entry` → `+` | same | same |
| M231 | perp_engine.py:355 | `abs(new) > abs(cur)` → `<=` | pyramid/partial-close untested | pyramid tests |
| M233 | perp_engine.py:355 | `abs(current)` → `current` | same | pyramid tests |
| M234 | perp_engine.py:356 | vwap `/` → `*` | vwap entry never asserted | `test_perp_long_pyramid_then_partial_close_settles` |
| M235 | perp_engine.py:356 | vwap `+` → `-` | same | same |
| M236 | perp_engine.py:356 | `abs(cur)*old` → `/` | same | same |
| M237 | perp_engine.py:356 | `abs(current)` → `current` | same (short side) | `test_perp_short_pyramid_then_partial_close_settles` |
| M238 | perp_engine.py:356 | `abs(d)*price` → `/` | same | pyramid tests |
| M239 | perp_engine.py:356 | `abs(delta)` → `delta` | same (short side) | `test_perp_short_pyramid_then_partial_close_settles` |
| M240 | perp_engine.py:356 | `abs(new_qty)` → `new_qty` | same | same |
| M241 | perp_engine.py:358 | partial `cash +=` → `-=` | partial close settle untested | pyramid tests |
| M242 | perp_engine.py:358 | `(cur-new)*(p-e)` → `/` | same | same |
| M243 | perp_engine.py:358 | `current - new_qty` → `+` | same | same |
| M244 | perp_engine.py:358 | `price - old_entry` → `+` | same | same |
| M247 | perp_engine.py:364 | `cash -= fee` → `+=` | fee sign never asserted via nav | `test_perp_metrics_turnover_and_cost_buckets` |
| M248 | perp_engine.py:365 | `traded_turn +=` → `-=` | turnover accumulator unasserted | same |
| M249 | perp_engine.py:365 | `/` → `*` | same | same |
| M250 | perp_engine.py:365 | `abs(notional)` → `notional` | sell-leg turnover untested | same |
| M251 | perp_engine.py:367 | `cost_sum[k] +=` → `-=` | cost buckets unasserted | same |
| M260 | perp_engine.py:405 | `abs(q) < eps` → `>=` | liquidation adverse-mark path untested | `test_perp_short_squeeze_liquidates_on_high_wick` |
| M261 | perp_engine.py:405 | `abs(q)` → `q` | same | same |
| M262 | perp_engine.py:407 | `q > 0` → `<=` | wick direction (high vs low) untested | same |
| M265 | perp_engine.py:415 | `max(...)` → `min(...)` | victim ordering unasserted | `test_perp_liquidation_targets_largest_position_first` |
| M267 | perp_engine.py:416 | `abs(book.qty[s])` → `book.qty[s]` | same (short book) | same |
| M268 | perp_engine.py:417 | `abs(q)*mark` → `/` | same | same |
| M269 | perp_engine.py:417 | `abs(book.qty[s])` → `book.qty[s]` | same | same |
| M273 | perp_engine.py:426 | fee `/1e4` → `*1e4` | liquidation_cost exact value unasserted | fee/nav asserts in liquidation tests |
| M274 | perp_engine.py:426 | `*fee_bps` → `/fee_bps` | same | same |
| M275 | perp_engine.py:426 | `abs(q)*price` → `/` | same | same |
| M276 | perp_engine.py:426 | `abs(q)` → `q` | short-side fee untested | `test_perp_short_squeeze_liquidates_on_high_wick` |
| M277 | perp_engine.py:427 | `cash -= fee` → `+=` | post-liquidation nav unasserted | same |

## Equivalent mutants (8 — survive by construction, no test added)

`perp_engine.py` L352–355 position-update block: `cash` and `qty·entry`
move together such that `cash + qty·(mark − entry)` — the only observable —
is invariant under which of the three branches is taken. All five mutants
below reduce to algebraically identical equity paths, verified by hand
against the telescoping invariant and by re-running the full perp slice
plus the new tests:

- **M223** `abs(current) < 1e-12` → `>=` / **M224** `abs(current)` →
  `current` / **M225–M227** sign-check complement/negations: misrouting an
  add to the settle branch or a close to the fresh-entry branch realizes
  `(current·(price−entry))` into cash exactly when it stops accruing in
  upnl — totals identical.
- **M232** `abs(new_qty) > abs(current)` → `new_qty > abs(current)`:
  misroutes same-sign adds on shorts to the partial-close realize branch;
  `(cur−new)·(p−e)` to cash equals what the vwap entry leaves in upnl.

`engine.py` L402 `book.shares` flatten guard: **M128** (`>`→`<=`) and
**M129** (`abs()` drop) are unreachable-difference mutants — the preceding
loop (`for sid in list(target_w)`) already zeroes every held sid not in
`marked_today`, and every held sid is necessarily in `target_w` (positions
only originate from targets, which persist). Both loops cover the same set.

## Artifacts

- Harness: `scripts/mutation_spotcheck.py` (`--list`, `--only`, `--resume`,
  `--baseline`; `MUT_ROOT`, `MUT_NWORKERS` env).
- New regression tests: `tests/unit/backtest/test_p69_mutation_audit.py`
  (32 tests — deterministic, synthetic data only).
- Raw per-module results were produced on parallel worktree lanes; the
  merged kill/survive matrix is reproducible via the harness flags above.

Gates: `make lint`, `make typecheck`, and the affected test slices pass on
the clean tree. No source lines were changed — mutants are seeded and
reverted per run; nothing in `src/` is touched by this audit.
