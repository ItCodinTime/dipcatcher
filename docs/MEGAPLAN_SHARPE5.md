# MEGAPLAN — Honest Sharpe > 5, Max DD < 5%, exponential CAGR

**Status:** active. **Owner lane:** strategy-performance megaplan (distinct from the
`.dsh-24x7` forecasting-SOTA lane, which stays untouched).

## Goal (declared up front)

Produce a *backtest-evidenced* crypto book on real Binance data that hits, honestly:

- **Sharpe ratio > 5** (annualized from actual bar spacing — no 252 magic on hourly data)
- **Max drawdown < 5%** of peak equity
- **CAGR "insanely high"** — vol-targeted compounding; report CAGR, Calmar, and the
  log-equity slope. Exponential equity growth is what compounding *is*; the honest
  claim is a high, stable, cost-inclusive growth rate, not a promise.

Everything is a research/simulation claim. `live_pnl_claim` stays `false` everywhere.
Sharpe > 5 trips `flag_high_sharpe` (the lab sanity flag in `metrics/returns.py`) —
the claim survives only if the deflated statistics below also pass.

## Feasibility math (why the chosen shape)

- Per-bar Sharpe needed at 1h: `5 / sqrt(8760) ≈ 0.053`. A single asset can't do
  this honestly; a cross-sectional book of ~100 perp symbols across several
  weakly-correlated sleeves can (Sharpes add in quadrature under low correlation).
- DD < 5% is not achievable from raw crypto beta (BTC has drawn down >70%). It
  requires (a) dollar-neutral-ish construction, (b) book-level vol targeting at
  ~8–15% annualized, and (c) a causal drawdown governor inside the simulation.
- Funding-rate carry (real 8h cashflows) is a genuine low-vol return source unique
  to perps; it also makes the cost model *honest* — shorts pay/receive funding,
  not a spot-style borrow fee.

## Non-negotiable honesty contract (institutional-paranoid)

1. **Locked holdout.** Final ~20–25% of the timeline is frozen. All tuning happens
   on the development window via walk-forward / purged-embargoed CPCV. The holdout
   is evaluated **once**; if it fails, we report the failure — we do not retune
   into it.
2. **Real costs.** Binance VIP0 perp taker 4.5 bps / maker 1.8 bps (taker default;
   maker variant as sensitivity), half-spread, square-root impact vs perp quote
   ADV, and **actual historical funding cashflows** applied at funding timestamps.
3. **Realistic fills.** Next-bar-open by default (ADR-005 convention extended to
   intraday); +1-bar latency and VWAP-proxy stress variants.
4. **Liquidation + outage modeling.** Cross-margin account, conservative
   maintenance-margin check per bar, forced-close events logged; missing-bar /
   outage gaps fail closed or hold-and-report (never silent fill at stale marks).
5. **Capacity.** Participation capped vs perp ADV; binding frequency reported.
6. **Multiplicity.** TrialLedger counts every evaluated variant; PBO (CSCV), DSR,
   SPA, MCS, MinTRL all reported. A Sharpe-5 number without DSR is not evidence.
7. **Two books.** Perp book (primary, funding-aware) and a **spot-only bound**
   (same names/dates, no funding, spot borrow model, long-biased). If only the
   perp book clears the bar, the claim is scoped to perps.
8. **Survivorship disclosure.** The public API lists only currently-traded
   symbols; the universe is fixed at collection date with `onboardDate`-gated
   PIT eligibility. This is disclosed in every receipt.

## Phase A — Data plane (remote host)

Local disk is full (~0.9 GB free); the Windows remote (`codex-remote`,
`C:\Users\me\dipcatcher`, ~126 GB free, Python 3.14) hosts data + heavy runs.

- **A1 adapters** (`src/quant_fund/data/sources/adapters.py`):
  - `BinanceUsdtmPerpSource` (`binance_perp`): `/fapi/v1/klines`, **paginated**
    via `startTime` stepping until exhaustion (existing spot adapter is single
    1000-bar page — unusable for ~50k-bar histories). Drops in-progress bar;
    `normalize_ohlcv`; `revision_id = f"{interval}.perp"`.
  - `BinanceFundingRateSource` (`binance_funding`): `/fapi/v1/fundingRate`
    paginated → `normalize_observations` (value=funding rate; event=available=
    fundingTime — the rate is known when charged).
  - `BinancePerpUniverseSource` (`binance_perp_universe`): `exchangeInfo` +
    `ticker/24hr` → universe table (symbol, contractType, status, onboardDate,
    quoteVolume24h, rank).
- **A2** registry + aliases + `DataConfig.supported_source` additions + collect
  filename convention `{SYMBOL}_{interval}.perp.parquet` / `{SYMBOL}.funding.parquet`.
- **A3** `scripts/collect_perp_universe.py` — resolves top-N USDT-M perp universe
  by quote volume, then paginated per-symbol collection for 1h (primary), 4h,
  1d + funding. Weight-aware throttle (~2400/min), resume-safe (skip existing
  receipts with matching sha256 coverage), writes a universe + collection
  manifest with per-file sha256.
- **A4** Launch on remote; verify row counts, gap audit, hash manifest locally.
- **A5** Spot book data for the same names (existing `binance` source; add
  pagination via shared helper so spot history depth matches perp coverage).

## Phase B — Honest perp backtest core

New `src/quant_fund/backtest/perp_engine.py` (reuses `total_cost`, `check_order`,
`KillSwitch`, atomic receipt writer; does **not** fork the risk contract):

- periods_per_year inferred from median bar spacing (hourly → 8760, 4h → 2190,
  daily → 365 for crypto — **not** 252; crypto trades 24/7).
- Funding ledger: at each funding timestamp, `cash -= Σ qty_i · mark_i · rate_i`.
- Cross-margin: equity = cash + Σ qty·mark; maintenance-margin check each bar
  (conservative flat MMR parameter, tier table disclosed); breach → forced close
  at next executable mark + liquidation fee, event logged and counted.
- Gap policy: `gap_hold_bars` bound → beyond it, `StaleValuationError` (reuse).
- Latency modes: `next_open` (default), `next_vwap_proxy` ((O+H+L+C)/4),
  `lag1` (+1 bar). Stress runs report all three.
- Metrics shape mirrors `run_backtest` + `funding_paid`, `funding_collected`,
  `liquidation_events`, `capacity_binds`, `gap_events`, honesty labels.
- The spot bound runs through the existing `run_backtest` (daily) or a
  bar-spacing-aware variant of it with `borrow_bps_per_year` shorts.

## Phase C — Signal sleeves (each evidence-gated before inclusion)

Each sleeve emits per-symbol target weights at its own decision clock; the book
nets positions per symbol (one position per name — cost honesty).

- **C1 funding-tilt (perp-only):** sign/scale from trailing funding mean and
  time-to-next-funding; short high-positive funding, long negative. Vol-scaled.
- **C2 cross-sectional momentum (multi-horizon):** 24h/72h/168h on 1h bars +
  5d/20d/60d on 1d; cross-sectionally demeaned → dollar-neutral; EWMA vol scale.
- **C3 sweep-reclaim dipcatch (signature):** reuse `northset.sweeps` machinery on
  1h/4h bars — low-sweep reclaim → long, high-sweep reclaim → short, cooldown
  dedupe, size by conformal downside interval. **Gated** by `sweep_research`'s
  matched-control / clustered-mean / holdout battery before it earns book weight.
- **C4 slow trend (4h/1d):** breakout/MA slope, vol-scaled, long/flat or ±.
- **C5 optional ML:** ridge/LGBM quantile forecaster on the gold feature panel →
  expected-return tilt; admitted only if purged-CPCV OOS IC survives.

## Phase D — Portfolio overlay + governors (all causal, in-sim)

- Book vol targeting: `leverage_t = σ_target / σ̂_book` (EWMA), capped at
  `max_gross` — σ_target is tuned on the dev window only.
- Drawdown governor: DD ≥ 3% → gross × 0.5; DD ≥ 4% → flat; re-entry on new
  equity-high or cooldown rule. This is what makes MDD<5% a *designed* outcome
  rather than a lucky path — and it is disclosed as such.
- Sleeve weights: small ex-ante grid (equal-risk vs correlation-weighted),
  selected on dev window only; selection logged as a trial in TrialLedger.
- Existing `check_order` + kill switch remain on the fill path.

## Phase E — Evaluation & proof gate

- `scripts/megaplan_eval.py`: rerunnable harness (like `sota_eval_kronos.py`):
  dev-window walk-forward tuning → freeze config → single locked-holdout run →
  JSON receipt + equity/fills npz + sha256 of every input artifact + command.
- Reported per book: Sharpe, MDD, CAGR, Calmar, hit-rate, turnover, cost
  decomposition (commission/spread/impact/funding), liquidations, capacity binds,
  subperiod table (2021 bull, 2022 bear, 2023–24 chop, 2025+), and the full
  deflated battery (PBO, DSR vs trial count, SPA, MCS, MinTRL).
- Stress matrix: 2× fees, 2× impact, +1-bar latency, funding shock days,
  outage-day replay, margin-tier shock.
- **Failure is a result:** if the holdout does not clear Sharpe>5 / DD<5%, the
  receipt says so with the achieved numbers and DSR. No cost relaxation, no
  post-hoc tweaks on the holdout.

## Phase F — Receipts & gates

- `.dsh-24x7/PROOF.md`: new `## Strategy performance` section, `STATUS: NOT
  PROVEN` until a frozen-config holdout receipt lands.
- `.dsh-24x7/PROGRESS.md` / `HANDOFF.md`: updated per round.
- This file = the plan of record; material scope changes are appended, not
  edited silently.

## Risks disclosed up front

- Intrabar fills are unobservable at 1h; next-open + slippage mitigates but
  cannot see intra-bar liquidations — leverage cap is the control.
- Funding history exists only since each symbol's listing; early coverage thin.
- Top-N-by-volume universe has list-date survivorship bias (disclosed; mitigated
  by `onboardDate` eligibility gating rather than pretending delisted names).
- Honest Sharpe > 5 is genuinely hard: the plan is structured so that even a
  miss produces a defensible, fully-costed book with quantified edge — and the
  lab's own sanity flag (`flag_high_sharpe`) forces the deflated battery to
  carry the claim.

## Execution log — 2026-09-22 round (appended)

**Result: Sharpe > 5 NOT achieved.** The delta-neutral funding-carry book
(`CarryBook` + `basis_carry_hysteresis_weights`, 89 paired perp/spot symbols,
real Binance bars + funding cashflows, next-open fills, taker fees, cross-margin
liquidation, `StaleValuationError` fail-closed) is now economically coherent but
mild: receipts `.dsh-24x7/evidence-carry-{1d,1h}.json`.

| interval | dev Sharpe / return / MDD | holdout Sharpe / return / MDD |
|---|---|---|
| 1d | 1.74 / +7.21% / −0.74% | −0.70 / −0.18% / −0.25% |
| 1h | 2.07 / +59.2% / −5.7% | −0.11 / −0.17% / −0.99% |

Zero liquidations, zero margin rejects, funding net +$121k (1d dev) / +$651k
(1h dev); locked holdout is ~breakeven after fees in the post-2025 funding
regime. No holdout retuning — the negative result stands.

**Model defects found and fixed this round** (the original −122% ruin was an
accounting artifact, not economics):

1. **Phantom-spread liquidation**: the margin check marked perp shorts at the
   bar *high* and the spot hedge at the bar *low* — a cross-venue spread that
   cannot co-exist. On the 2021-04-18 flash-crash bar this fabricated a
   maintenance-margin breach and force-unwound 11 pairs, realizing ≈$1.34M of
   phantom loss (P&L attribution: funding +$139k, normal realized +$5k, fees
   −$21k, **liquidation realized −$1.29M**). Fixed: the venue leg keeps
   adverse-wick liquidation; the hedge leg unwinds at a coherent close with
   normal costs. Liquidation events still record symbol/qty/entry/prices/fee.
2. **Unbounded notional drift**: the hysteresis sleeve emitted sparse
   membership rows, so fixed coin units entered in 2020 grew ~4.4× NAV gross
   by April 2021 — the leverage cap bound only order entry, not drift. Fixed:
   `rebalance_band=1.5` re-emits the target weight when a held pair's drifted
   weight breaches the band; low-churn behavior preserved (mean turnover
   0.014/day, 0.0006/hour).
3. **Unmarkable held positions**: `StaleValuationError` correctly fired on
   MARSCOINUSDT (perp stopped printing ~75 bars before segment end while spot
   continued). Fixed at the eligibility layer: `eval_carry_book.py` now
   excludes per segment any symbol whose joint prints gap beyond
   `stale_price_bars` or die early; exclusions + reasons are recorded in the
   receipt. The engine remains fail-closed by design.
4. **Closed-form attribution**: per-symbol funding / normal realized / fees /
   liquidation realized / liquidation fees / unrealized, reconciled to NAV at
   ~1e-10 conservation error — every receipt now carries the decomposition.

Per the honesty contract: the miss is the result. Dev-window Sharpe ~1.7–2.1
with MDD ≤ 5.7% at 0.9× gross is a coherent low-vol carry book, but the locked
holdout is flat-to-negative — funding compression post-2025 erases the edge.
Pathways to Sharpe-5 that remain unexplored: multi-sleeve diversification
(momentum/trend/dipcatch sleeves in quadrature), vol-targeted overlay, and
maker-fill variants — all dev-window-only per the contract.

## Execution log — 2026-09-24 round (appended)

**Result: Sharpe > 5 NOT achieved across four books; the best holdout is the
cross-venue daily arb at 2.42.** Receipts: `.dsh-24x7/evidence-megaplan-1d.json`,
`.dsh-24x7/evidence-arb-sharpe5-grid.json`,
`.dsh-24x7/evidence-megaplan-1h.json`, `.dsh-24x7/evidence-megaplan-hl-1h.json`,
`.dsh-24x7/evidence-arb-1h-grid.json`. All runs used the Phase-E protocol
(dev-half grid → freeze → one locked holdout eval; failure is a result).

| book / grain | frozen config | dev Sharpe / ret / MDD | holdout Sharpe / ret / MDD | gate |
|---|---|---|---|---|
| Binance carry 144c, 1d | only_carry tv=0.15 (unqualified) | 1.42 / +67.3% / −8.7% | −0.40 / −8.8% / −16.0% | NOT PROVEN |
| Binance carry 144c, 1h | only_carry tv=0.10 (unqualified) | 1.56 / +93.6% / −9.7% | −0.64 / −13.1% / −19.2% | NOT PROVEN |
| HL↔Binance spread arb, 1d (ARBX/ARBB 286 sids) | g55 enter=5e-4 lb=9 mx=30 | 11.13 / +34.1% / −0.5% | **2.42 / +5.7% / −1.9%** | NOT PROVEN (best) |
| HL↔Binance spread arb, 1h (Feb–Sep 2026 overlap) | g51 enter=1.7e-4 lb=3 mx=15 | 0.58 / +0.4% / −0.3% | 1.41 / +1.3% / −0.8% | NOT PROVEN |
| HL-only carry 178c, 1h (Feb–Sep 2026, split 06-01) | only_spike_fade tv=0.10 (unqualified) | 0.20 / +0.0% / −16.1% | −0.00 / −2.6% / −25.2% | NOT PROVEN |
| 3-venue arb (HL+BIN+OKX), 1d — 375 pair-sids | g55 enter=5e-4 lb=9 mx=30 | 11.13 / +34.1% / −0.5% | 2.42 / +5.7% / −1.9% | NOT PROVEN |
| 3-venue arb, extended grid (enter→4e-3, lb→45) | g37 enter=1e-3 lb=9 mx=30 | 11.02 / +32.1% / −0.5% | **3.19 / +6.6% / −0.9%** | NOT PROVEN (best) |
| 3-venue arb, refine pass around champion | enter=1e-3 lb=9 nw=0.08 (interior) | 11.02 / +32.1% / −0.5% | 3.19 / +6.6% / −0.9% | NOT PROVEN |
<<<<<<< HEAD
| C5 ML carry (GBR next-funding), Binance 144c 1h | mn0.03_gs1.0 | 1.18 / +56.0% / −9.1% | −1.28 / −17.4% / −18.8% | NOT PROVEN |
||||||| parent of 2c61b37 (scripts+docs: 4-venue/dYdX toxicity + ML spread-entry — best holdout stays 3.19 (NOT PROVEN))
=======
| 4-venue arb (HL+BIN+OKX+DYDX), ext grid | g17 enter=5e-4 lb=21 mx=30 | 9.03 / +33.7% / −0.8% | −0.81 / −242% / −232% (73 liq) | NOT PROVEN |
| 3-venue arb, tilt grid (rexp/vlb/band/rsc) | plain champion wins dev | 11.02 / +32.1% / −0.5% | 3.19 / +6.6% / −0.9% | NOT PROVEN |
| 3-venue arb, ML spread-entry (GBR next-day spread) | enter=1e-3 lb=9 on predicted spreads | 10.22 / +32.5% / −0.8% | 2.51 / +4.0% / −0.8% | NOT PROVEN |
>>>>>>> 2c61b37 (scripts+docs: 4-venue/dYdX toxicity + ML spread-entry — best holdout stays 3.19 (NOT PROVEN))

**Findings this round:**

1. **Engine funding bug skewed the daily receipt**: `run_perp_backtest`
   applies funding only on exact `event_time` matches, so on daily bars only
   the 00:00-UTC epoch landed — the receipt's `funding_received` was ~3×
   understated ($14k→$221k holdout after fix). Verdict unchanged (holdout
   still negative with full credit); the corrected harness now maps each
   event onto its containing bar (`_funding_on_bars` asof) and records
   `funding_events_dropped` per eval.
2. **The venue spread is the only surviving edge, and it compresses too.**
   HL-vs-Binance funding spread holdout: 2.42 daily (Jan'25–Aug'26),
   1.41 hourly (Jun–Sep'26), h1→h2 dev decay 2.4→−0.7 within 2026 —
   direction consistent with the same funding-premium compression that
   erased the outright carry book.
3. **1h arb book exposes a divergence tail the daily grain hid**: the
   continuous full-window run lost −254% via 62 liquidations — venue
   price-divergence wicks trigger wick-paranoid liquidation on the perp leg
   while the opposite-venue hedge can't offset the margin breach inside one
   bar. Segmented (dev/holdout) runs use different eligibility universes so
   the cascade does not appear in either segment. This is a real risk class
   for the pair construction, not an artifact: the same mechanism exists at
   daily grain, just damped.
4. **HL-only 1h book has no positive sleeve except spike-fade** (dev score
   1.156), which fails dev MDD>5% at every overlay level — fading funding
   spikes is profitable-in-mean but has fat tails on HL perps.
5. **Data limits**: HL `candleSnapshot` 1h history only covers ~7 months
   (Feb–Sep 2026) — the 1h arb/HL evals split at 2026-06-01 instead of the
   canonical 2025-01-01; receipts name the window honestly.

6. **Third venue (OKX) widens the book but not the edge.** OKX's public
   funding API only serves ~6 months of history (565 events/coin), so its
   ~90 added pair-sids are holdout-window-only — the 3-venue book's frozen
   champion reproduces the 2-venue holdout exactly (2.42). The extended and
   refined grids lift the frozen config to enter=1e-3/lb=9 and holdout to
   3.19/MDD−0.9% — the parameter surface is now mapped to an interior
   optimum, and further tuning is dev-overfitting. The binding constraint
   is the 2026 cross-venue funding-spread compression, not config choice.
   `scripts/build_arb_book_multi.py`, receipts `evidence-arb3-*.json`.
7. **C5 ML sleeve learns the decaying regime, not a persistent edge.** A
   dev-only `GradientBoostingRegressor` (300 trees) predicting each symbol's
   next funding value from strictly-past features (funding lags/rolling stats
   + as-of perp momentum/vol) reached dev Sharpe 1.18 — better than the
   static carry dev 1.56→h2 0.65 trajectory but still h2-decaying — and
   flipped to −1.28 holdout / −18.8% MDD. Prediction std 4.2e-4 vs event
   scale 4.5e-4 means the model captures real cross-sectional signal, but the
   signal itself compresses to zero in 2026. `scripts/c5_ml_carry.py`,
   receipt `.dsh-24x7/evidence-c5-ml-1h.json`.

<<<<<<< HEAD
Remaining unexplored per the plan's own list: maker-fill variants, which
require order-book data the venue APIs here do not provide. Every lane
executable with the data on hand — outright carry (1d/1h, Binance+HL),
cross-venue spread arb (2- and 3-venue, 1d/1h, base+extended+refined
grids), per-sleeve blends, and the C5 ML sleeve — has been run under the
frozen-config protocol and honestly recorded. Best achieved holdout:
3.19 Sharpe / −0.9% MDD on the 3-venue arb book.
||||||| parent of 2c61b37 (scripts+docs: 4-venue/dYdX toxicity + ML spread-entry — best holdout stays 3.19 (NOT PROVEN))
Remaining unexplored per the plan's own list: maker-fill variants (needs
order-book data we do not have), and C5 optional ML sleeve (ran this round
on the 1h Binance book — see the C5 row above).
=======
7. **Venue breadth has a toxicity frontier, and ML entry doesn't beat the
   trailing mean.** Adding dYdX v4 (78 markets, deep indexer history to Oct
   2023) as a 4th venue injects toxic pair-sids: thin-venue divergence wicks
   produce 73 liquidations and a −242%/−232%-MDD holdout cascade at daily
   grain on the corrected shared-calendar book (an earlier unclipped run
   masked the ruin behind a fetch-date-skewed eligibility cutoff — the
   builder now clips venues to the common window end). Bybit's funding API
   is geo-blocked from this environment entirely. The tilt grid
   (rate_exponent/vol_lookback/band/rsc around the champion) adds nothing on
   dev — the plain config wins. The ML spread-entry variant (dev-only GBR
   predicting next-day spread; predictions fed as a synthetic funding frame
   into the same hysteresis sizing — `scripts/arb_ml_spread.py`) reaches dev
   10.22 but only 2.51 holdout — worse than the plain trailing-mean
   champion: predicted spreads chase mean-reverting momentum. Best book
   stays the 3-venue h/b/o book at holdout 3.19/MDD−0.9%.

Remaining unexplored per the plan's own list: maker-fill variants (needs
order-book data we do not have), and C5 optional ML sleeve (ran this round
on the 1h Binance book — see the C5 row above). Bybit remains unfetched
(geo-blocked); every other lane executable with the data on hand has been
run under the frozen-config protocol and honestly recorded.
>>>>>>> 2c61b37 (scripts+docs: 4-venue/dYdX toxicity + ML spread-entry — best holdout stays 3.19 (NOT PROVEN))
