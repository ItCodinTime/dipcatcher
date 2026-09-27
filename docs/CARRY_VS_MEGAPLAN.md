# Carry lanes vs the megaplan proof gate — reconciliation

Two lanes measured different Sharpe numbers for what looks like the same
strategy family (delta-neutral funding carry). Both results are true; they
are different engines, universes, and protocols. This doc reconciles them.

## The two lanes

### Megaplan lane — `scripts/eval_carry_book.py`

- Engine: `CarryBook` (`src/quant_fund/backtest/carry_engine.py`) —
  per-symbol spot+perp pair, delta-neutral by construction, next-open fills,
  real taker fees, cross-margin maintenance check, `StaleValuationError`
  fail-closed; weights via `basis_carry_hysteresis_weights`
  (`rebalance_band=1.5` bounding fixed-unit drift).
- Universe: 89 currently-listed Binance perp+spot pairs (22 perp-only
  excluded, 1 coverage-excluded). Survivorship disclosed: misses delisted
  symbols, returns upward-biased.
- Receipts: `.dsh-24x7/evidence-carry-{1d,1h}.json`:

  | interval | segment | Sharpe | total | max DD | funding net |
  |---|---|---:|---:|---:|---:|
  | 1d | dev (n=2055)    | 1.74  | +7.21%  | −0.74% | +$121,435 |
  | 1d | holdout (n=513) | −0.70 | −0.18%  | −0.25% | +$629     |
  | 1h | dev (n=49311)   | 2.07  | +59.2%  | −5.7%  | +$651,006 |
  | 1h | holdout (n=12329)| −0.11| −0.17%  | −0.99% | +$996     |

### Devin carry lane — `scripts/carry_research.py` + expansion

- Engine: `run_one` → same `basis_carry_hysteresis_weights` generator with
  tuned params (champion `enter=1.5bp/d, exit=0, lb=9, nw=0.11, mx=60,
  band=1.3`) plus regime/rate scaling knobs the megaplan eval does not
  exercise.
- Universe: 355 Binance perp+spot pairs (incl. `data/binance_carry_extra/`,
  ~294 extra symbols fetched from Binance vision) + a Hyperliquid book —
  ~4–5× wider than the megaplan universe.
- Receipt: `artifacts/carry_champion_expanded.json` (committed):

  | segment | Sharpe | CAGR | total | max DD | funding net | liqs |
  |---|---:|---:|---:|---:|---:|---:|
  | dev 2020–24 (283 elig.)    | 6.86 | 38.2% | +404% | −1.56% | $4.21M | 0 |
  | holdout 2025–26            | 1.68 | 4.0%  | +6.9% | −2.23% | $0.08M | 0 |
  | full 2020–26 (57 elig.)    | 6.26 | 28.0% | +425% | −1.62% | $4.58M | 0 |

- Cross-venue arb sleeve (`scripts/build_arb_book.py`, 119 HL∩Binance
  perp-perp pairs both directions, `data/arb_carry_book/`): dev 7.90 /
  holdout **4.78** / DD −0.55% — the smoothest book measured, and the
  closest any lane has come to the >5 holdout bar.

## Why the numbers differ

1. **Universe width is the dominant term.** The megaplan universe is 89
   symbols; the expansion book dilutes squeeze risk across 283 dev-eligible
   names — [carry_expansion_2026_09.md](carry_expansion_2026_09.md) shows the same config is
   liquidated to ruin at `nw ≥ 0.15` on the wide book but safe at
   `nw ≤ 0.12` / `mx ≈ 50–60`. Concentrated books show both higher dev
   Sharpe and fatter liquidation tails.
2. **The protocols answer different questions.** The megaplan eval is a
   deliberately narrow, locked audit of one engine on one universe; the
   research lane swept the hysteresis/membership/rate-scaling space and
   picked a champion — selection bias included, disclosed as such.
3. **Both lanes agree on the load-bearing fact:** post-2025 funding
   compression degrades the edge on locked holdout (megaplan −0.70,
   expansion +1.68, arb +4.78). The disagreement is in magnitude and
   window coverage, not direction.

## Verdict

- The headline records (+425% / 28.0% CAGR / Sharpe 6.26 full-window /
  DD −1.62% / 0 liquidations) are real, committed receipts — on the
  **dev and full windows**.
- `PROOF.md ## Strategy performance` stays `NOT PROVEN` because the
  megaplan gate ([MEGAPLAN_SHARPE5.md](MEGAPLAN_SHARPE5.md) Phase E) requires a
  **frozen-config holdout** clearing Sharpe > 5 / DD < 5 %. No lane has
  produced that receipt: carry holdout is 1.68, arb holdout is 4.78.
- The honest gap to close: a frozen config whose 2025–26 holdout clears
  the bar. Arb-only and arb-weighted blends are the closest direction
  (4.78) — roughly 5 % short — while outright-carry concentration is the
  known failure mode.
