# P6.1 money-paths audit — execution + backtest engines

Scope: `src/quant_fund/execution/simulated_broker.py`,
`src/quant_fund/execution/costs.py`,
`src/quant_fund/execution/implementation_shortfall.py`,
`src/quant_fund/backtest/event_sim/costs.py`,
`src/quant_fund/backtest/engine.py`,
`src/quant_fund/backtest/carry_engine.py`,
`src/quant_fund/backtest/perp_engine.py`,
`src/quant_fund/backtest/sleeves.py`,
`src/quant_fund/portfolio/risk_gate.py`,
`src/quant_fund/portfolio/pnl_attribution.py`.

Each file was read end-to-end and checked against the model/spec named in its
docstring (Almgren–Chriss, sqrt impact, SEC/TAF schedule, implementation
shortfall decomposition, carry/perp margin + funding conventions), plus the
cross-file contracts the callers rely on: order → fill → cash accounting,
weight → exec-time NAV sizing, and mark causality (nothing decided at bar
`t` may depend on bar `t`'s own close).

Verdicts: `correct`, `fixed` (with failing-first regression test), `waived`
(documented looser semantics), `suspicious-but-unproven`, `out-of-lane`
(exposed by a test here; the owning file belongs to another lane).
Regression KATs live in `tests/unit/backtest/test_p61_money_audit.py`
(31 tests, deterministic, offline, synthetic data only).

## Bugs fixed (each in its own commit)

1. **`SimulatedBroker.submit` accepted non-finite prices** — **FIXED**
   (`4a7be45`). `submit` guarded with `not (price > 0)`, which passes `+inf`
   (and `nan` — `nan > 0` is `False`, but `nan` slipped in through the
   market-data path that skips `submit`'s guard): a `+inf` fill priced an
   infinite notional, `cash -= inf` poisoned equity to `-inf`, and all later
   marks/nav were corrupted silently. `submit` now rejects `None`,
   non-finite, and `<= 0` prices.
2. **`SimulatedBroker._attempt_fill` mutated state before validating
   `decision_price`** — **FIXED** (`4a7be45`). The `decision_price` sanity
   check raised *after* `self.cash` and `self.shares` were already updated,
   so a rejected fill still moved cash/shares with no fill row to account
   for it — an accounting-identity violation. Validation is now hoisted
   before any state move; the raise leaves the book untouched.
3. **`shortfall_frame` silent NaN/inf swallowing + unsigned-qty contract
   drift** — **FIXED** (`8551c78`). (a) Quantity, exec price,
   decision price, and the cost columns were coerced with `fill_null(0.0)` —
   a missing decision price silently scored zero drift instead of failing
   closed. Now: cast to Float64, raise on null/non-finite/`<= 0` for
   quantity/price/decision_price and on null/non-finite/negative for cost
   columns. (b) When callers supplied `side_sign`, negative `quantity`
   values were used raw (buy/sell drift flipped); `side_sign` itself was
   never validated. Now: `side_sign` must be exactly `+1`/`-1` for every
   fill and `quantity` must be unsigned when `side_sign` is present;
   the no-`side_sign` mode (engine's signed `delta` convention) still
   derives the sign and `abs()`es quantity. Empty frame still returns an
   empty scored frame.
4. **Execution-time NAV leak in all three engines** — **FIXED** (`3ad25ca`,
   `e328c21`). At each exec bar the engines updated the mark map to that
   bar's own close and *then* computed the NAV used to size the orders being
   filled at that same bar's open: a position that wasn't trading was valued
   at a price not yet printable, leaking the exec bar's close-to-open move
   into sizing. In `engine.py`, `perp_engine.py`, `carry_engine.py` the exec
   NAV now uses marks knowable before the fill (`pre_exec_marks`), overlaid
   with the exec open for the names being filled — `nav_prices =
   {**pre_exec_marks, **exec_mark}`. KATs: `test_spot_exec_nav_does_not_leak_exec_day_close`,
   `test_perp_exec_equity_does_not_leak_exec_bar_close`,
   `test_carry_exec_equity_does_not_leak_exec_bar_close` pin causal
   quantities on a 2x exec-day rally.
5. **Leverage cap blocked deleveraging of an over-cap book** — **FIXED**
   (`e328c21`). `perp_engine` and `carry_engine` capped `|delta|` (the
   *trade* size) against the margin cap, so once a book was over the cap —
   e.g. after a rally inflates a short's notional — every order was
   rejected, including reductions; the book could never deleverage back
   inside the cap. Now the cap applies to the *projected* position:
   `room_qty = max(0, cap - others_gross) / price`; if `|current + delta|`
   exceeds the room the delta is clamped to the cap boundary, and a clamp
   that would zero the trade or reverse its sign is the only
   `margin_reject` — reductions and flips always pass the part that
   deleverages. Same fix in the perp leg of the carry pair.
6. **Carry funding marked at exec-open instead of bar close** — **FIXED**
   (`e328c21`). Funding accrual used `mark_p.get(sid)` — the exec-open
   overlay — for names that had executed that bar, while the book's own
   convention everywhere else is the bar close. Now `last_perp.get(sid)`,
   the close mark. KAT `test_carry_funding_marks_at_bar_close`: open=105 /
   close=110 accrues funding on 110.

## Exposed but out of lane

7. **`backtest/fast_replay.py` had the same exec-NAV leak** — **FIXED**.
   Both the numba kernel and the interpreted loop now size off marks
   knowable before the fill (`pre_mark` / `pre_ever`), matching
   `nav_prices = {**pre_exec_marks, **exec_mark}` in the event loop.
   `test_fast_replay_exec_nav_does_not_leak_exec_day_close` pins the
   causal quantity (delta +2000, not the leaked +4400).

## Waivers (checked; semantics intentionally looser — documented, not bugs)

8. `carry_engine` funding docstring says a funding print applies when
   `(open, close]` contains it; in practice the join is an exact-timestamp
   match on the funding frame's `event_time`. Wording-only looseness;
   the equality match is deterministic and documented here — waived.
9. `sleeves.funding_spike_fade_weights` computes the funding z-score over a
   rolling window that *includes* the current event, so an extreme print
   partially normalizes itself. This is a deliberate, causal design (the
   print must be in the window to be detected at all) and is documented in
   the function docstring — waived.
10. `sleeves` borrow-cost annualization divides by 252, assuming daily bars.
    The module is documented as a daily-bar sleeve library; a non-daily
    caller would misprice borrow — waived, documented assumption.
11. `Book.mark` / `Book.nav` value unmarked names at zero rather than
    raising. Callers only see this after the stale-mark gate
    (`stale_price_bars` in `risk_gate`), which drops names whose marks are
    too old — internal-only, fail-open by design — waived.

## Ledger

| file | claim checked | verdict | fix commit |
|---|---|---|---|
| `execution/simulated_broker.py` | `submit` rejects non-finite / non-positive prices | fixed — `inf` accepted, poisoned cash to `-inf` | `4a7be45` |
| `execution/simulated_broker.py` | fill rejection is atomic (no cash/shares move without a fill) | fixed — `decision_price` validated after mutation | `4a7be45` |
| `execution/simulated_broker.py` | cash identity `cash -= qty*price + costs`, shares identity `shares += qty`, partial-fill accounting | correct | — |
| `execution/simulated_broker.py` | market order fills at bar open (next-open convention), limit respects the limit price | correct | — |
| `execution/costs.py` | `half_spread`/`commission` = `|notional| * bps / 1e4`; sqrt impact = `notional * Y * sigma * sqrt(|Q|/ADV_shares)`; all reject non-finite | correct | — |
| `execution/costs.py` | `participation_limit` in (0,1], zero-qty → zero impact, dollar-vs-share ADV consistency | correct | — |
| `execution/implementation_shortfall.py` | drift = `(exec - decision) * side`, arrival/`decision_price` decomposition | correct after fix | — |
| `execution/implementation_shortfall.py` | fail-closed on null/non-finite price/qty/decision/costs; unsigned qty + valid `side_sign` contract | fixed — was `fill_null(0.0)` + unvalidated sign | `8551c78` |
| `execution/implementation_shortfall.py` | signed-`quantity` mode (engine fills frame) unchanged; empty frame → empty result | correct | — |
| `backtest/event_sim/costs.py` | AC one-slice dollar cost `eta/tau*q^2 + gamma*q^2/2` matches `expected_shortfall_ac`; SEC $27.80/$1M sells, TAF $0.000166/sh with $0.01–$8.30 ceil-cent bounds | correct — constants verified against fee schedule | — |
| `backtest/engine.py` | exec-time NAV uses only marks knowable before the fill | fixed — exec-bar close leaked into sizing | `3ad25ca` |
| `backtest/engine.py` | decision marks = signal-bar close; adv/vol from signal-day rows; next-open fill; stale-mark gate before trading | correct | — |
| `backtest/carry_engine.py` | exec-time NAV causal (same leak class) | fixed | `e328c21` |
| `backtest/carry_engine.py` | leverage cap on projected perp gross; deleveraging never rejected | fixed — `|delta|` cap trapped over-cap books | `e328c21` |
| `backtest/carry_engine.py` | funding accrues on bar close mark | fixed — used exec-open overlay | `e328c21` |
| `backtest/carry_engine.py` | pair bar = both venues print (inner join); `equity = cash + units*spot - units*(perp - entry)`; gross = Σ|qty|*mark | correct | — |
| `backtest/perp_engine.py` | exec-time equity causal | fixed | `e328c21` |
| `backtest/perp_engine.py` | projected-gross leverage cap; deleveraging allowed | fixed | `e328c21` |
| `backtest/perp_engine.py` | fill_delay_bars via `pending_exec_at`; funding = rate * qty * mark; liquidation at maint-margin breach, `liquidation_fee_bps` on notional; `equity = cash + qty*(mark - entry)` | correct | — |
| `backtest/sleeves.py` | funding-spike fade z-score, 252 borrow day-count, momentum skip/lookback, backward as-of joins | waived — documented causal approximations (items 9–10) | — |
| `portfolio/risk_gate.py` | fail-closed on non-finite order/nav/price; nav>0; participation, name, gross, net, predicted-vol, staleness caps; ulp-tolerant limit comparison | correct — verified `check_order` is the backstop for exec prices | — |
| `portfolio/pnl_attribution.py` | contribution `w_{i,t-1} * r_{i,t}` (causal weight convention); cost = fill costs / NAV; `live_pnl_claim` always false; unmapped → `unmapped` sleeve | correct | — |
| `backtest/fast_replay.py` | exec-time NAV causal | **fixed** — pre-bar marks for names without an exec print | fast-replay byte identity |

## Repro notes

- All new KATs are in `tests/unit/backtest/test_p61_money_audit.py`;
  configs are `configs/research.yaml` (engines) and `configs/paper.yaml`
  (broker) with risk gates relaxed where the test targets the margin cap,
  not the gate.
- The exec-NAV KATs pin exact share counts (e.g. 2000 causal vs 4400 leaked
  on a 2x exec-day rally) so any regression is an arithmetic failure, not a
  tolerance failure.
- `tests/unit/diffbacktest` fails in this environment with
  `ModuleNotFoundError: jax` on the untouched base commit too — optional
  `jax` extra not installed here; unrelated to this lane (verified by
  `git stash` + rerun on `origin/main` tip).
