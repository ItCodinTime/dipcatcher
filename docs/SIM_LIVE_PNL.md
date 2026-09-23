# Simulated Live-PnL — `sim-live` lane

**SIMULATED ONLY — no live-PnL claim.** Every number below is a causal
replay of real collected Binance bars through the paper loop /
`run_backtest` with modeled fees (2bps commission + 3bps half-spread +
impact, 10% participation cap), the risk gate, and the kill switch. No
broker connectivity exists on this path.

Pipeline: causal quantile panels (`fhs|evt|garch_t|egarch_l|empirical|
ewma_emp{,94,99}|vincent(a+b)|agree(a,b)`, `@hN` horizon variants) →
`QuantilePolicy` (mu-bps or edge-z gate with hysteresis, `edge`/`risk`
sizing, book-vol target, tail gate, entry/exit persistence, leadership
gate, breadth-scaled gross, signal-Sharpe meta-gate, accel gate,
rebalance cadence, market-disp breaker, top-k concentration, target
EWMA, deadband carry, caps) → sparse weight panel → event loop
(next-open fills). Receipts in `.dsh-24x7/lane-simlive/`.

## Headline book (daily, 5 majors, 2020-08→2026-09, shared calendar)

`ewma_emp` long-flat, `gate_on=edge z≥0.15`, `sizing=risk`
(`w = κ·edge/σ`, κ=0.15), deadband 1%, name cap 25%, gross ≤1.0:

| measure | bench (`run_backtest`) | paper loop (ledger) |
|---|---|---|
| total return | +2,628.8% | +2,494.2% |
| Sharpe (simulated) | +1.785 | +1.769 |
| max drawdown | −22.1% | −22.2% |
| ann vol | 33.4% | 33.2% |
| fills | 1,721 | 1,721 |

Second flagship — `vincent(ew94+97+99)` + xp3 + bvt=0.03 g2 (no leader):
paper loop +2,769.8% / SR 1.850 / DD −27.0% / 1,947 fills / 1,014 gate
rejects / 0 halts vs bench +2,752.5% / SR 1.849 — bench↔loop parity
holds within participation-cap noise on the leveraged book too.

Third flagship — full stack `vincent(3-lam)` + xp3 + **BTC-lead** +
bvt=0.03 g2: paper loop **+2,923.5% / SR 1.970 / DD −27.0%** / 1,893
fills / 857 gate rejects / 0 halts vs bench +2,905.2% / SR 1.970.

Benchmarks on identical bars/costs: equal-weight buy-hold 5 majors
+2,201% at Sharpe 1.16, **maxDD −72.3%**, vol 59.5%. The book beats
buy-hold on Sharpe (1.79 vs 1.16) and drawdown (−22% vs −72%), not on
raw return scale.

## Out-of-sample (panels on full history; book runs last 800 days only)

| book | OOS return | OOS Sharpe | OOS maxDD |
|---|---|---|---|
| **xp3 + BTC-lead + breadth-gross** | **+50.5%** | **+1.04** | −15.2% |
| **vincent(3-lam) + xp3 + BTC-lead + bvt** | +49.5% | +1.01 | **−12.8%** |
| champion + xp=3 + BTC-lead | +51.1% | +1.05 | −15.3% |
| ewriskz (champion) | +45.4% | +1.03 | −12.2% |
| vincent(3-lam) + xp=3 + bvt=0.03 | +48.2% | +0.99 | −12.8% |
| champion + xp=3 | +46.5% | +0.97 | −15.3% |
| BTC-lead only | +40.4% | +0.97 | −12.2% |
| ewma_emp mu-gate 10bps | +17.3% | +1.05 | −4.9% |
| vincent(ewma_emp+empirical) risk | +25.7% | +0.89 | −9.3% |
| persist=2 + bvt=0.02 | +30.0% | +0.81 | −14.5% |

In-sample Sharpe 1.79 → OOS 1.03: real degradation, disclosed. The
signal survives but does not replicate the 2020-21 regime's magnitude.

## Return/drawdown frontier (bench, same bars/costs)

`book_vol_target` scales the whole book toward a per-bar vol target
(both directions; gross cap bounds the upside). `persist_bars` requires
consecutive gate-passing dates before (re-)entry.

| config | return | Sharpe | maxDD | ann vol |
|---|---|---|---|---|
| ewma_emp@h5 (5d horizon) | +3,486.1% | +1.62 | −39.8% | 41.7% |
| **xp3 + BTC-lead + breadth-gross** | **+3,309.8%** | +1.85 | −27.0% | 34.4% |
| **vincent(3-lam) + xp3 + BTC-lead + bvt=0.03 g2** | +2,905.2% | **+1.97** | −27.1% | 30.7% |
| champion + xp=3 + BTC-lead | +2,942.1% | +1.71 | −27.7% | 36.5% |
| champion + xp=3 | +2,882.4% | +1.71 | −30.7% | 36.3% |
| vincent(3-lam) + xp=3 + bvt=0.03 g2 | +2,752.5% | +1.85 | −27.1% | 32.5% |
| champion (no bvt) | +2,628.8% | +1.79 | −22.1% | 33.4% |
| vincent(ew94+97+99) bvt=0.03 g2 | +1,543.7% | +1.83 | −21.9% | 27.1% |
| bvt=0.02 | +672.8% | +1.82 | −16.6% | 19.4% |
| persist=2 + bvt=0.02 | +630.2% | +1.81 | **−13.2%** | 19.0% |
| bvt=0.015 | +447.7% | +1.85 | **−13.1%** | 15.7% |

Vol targeting is a clean efficient frontier — Sharpe holds ~1.8 while
drawdown halves at bvt 0.015–0.02; chasing return past ~1,500% costs
drawdown faster than it buys Sharpe.

`exit_persist` (slow exit — hold through a single gate-failing bar)
lifts both return and Sharpe at scale: xp=3 on the vincent blend is the
best risk-adjusted large book (SR 1.85), at the cost of deeper DD
(−27.1%). `leader_sid=BTCUSDT` (alts trade only while BTC's edge > 0)
stacks on xp=3 for the best raw return, **+2,942%**, and the best OOS
slice yet (+51.1% / SR 1.045 / DD −15.3%). `top_k` concentration
(2–3 names) lowers return and Sharpe — diversification across the
5-name book is load-bearing. `@hN` multi-horizon specs amplify edge by
~√N but *only* through a looser effective gate — at matched selectivity
(z·√N) h3/h5/h10 are all worse than the daily champion (SR ≤1.70), so
the horizon axis is disclosed as rejected. `mkt_edge_min` (flat book
when mean cross-asset edge sags) and `w_alpha` (target EWMA) are both
~Sharpe-neutral: the edge filter cuts into trend profits, and smoothing
inflates turnover 6× (10.6k fills) without DD benefit — rejected.
`breadth_gross` (gross cap × fraction of names with positive edge)
*stacks*: on xp3+BTC-lead it lifts return to +3,310% at SR 1.85 —
breadth-proportional exposure is the rare soft market filter that pays.
Rejected in the same sweep: `rebal_every` cadence (SR ≤1.75, turnover
down but exposure timing worse), `meta_min` signal-Sharpe gate (DD cut
but lower return; meta=0.5 starves the book), `accel_min` (negative),
`gate_out` hysteresis (SR 1.39 — loose exits hold losers).

## Rejected experiments (honest negatives)

- **`tail_gate`** (enter only when the forecast's own adverse-tail bound
  is benign): too strict — crypto left tails are fat; kills the book.
- **`mkt_disp_cut`** (market-wide vol breaker; flat when median disp
  > cut): every cut level *lowers* Sharpe — the biggest trend-follow
  gains occur inside high-vol regimes. Feature retained, disclosed.
- **Leverage gross 2–3× without vol targeting**: +2,938% at SR 1.46 /
  DD −42.6% in-sample; **OOS collapses to SR 0.16** — leverage amplifies
  chop, not edge.
- **4h interval**: costs × 2,191 bars/yr compress the edge to +10.8%
  (SR 0.60) at best.

## Regime dependence (the honest limit)

| window | book | Sharpe |
|---|---|---|
| 2020-08→2026-09 (5 majors) | champion | 1.79 |
| last 800d only | champion | 1.03 |
| 2018-05→2026-09 (BTC/ETH/XRP, 8.3y) | champion | ~1.01 |
| 998d breadth book (9 majors) | champion | 1.04 |

The edge is real but regime-dependent: ~1.0 Sharpe outside the trending
2020-24 regime — the +2,629% headline is regime compounding, not a
universal multiplier.

## Per-asset attribution (same policy, single-name books)

BNB +195% (SR 0.95) · BTC +112% (0.99) · ETH +100% (0.75) ·
SOL +274% (1.48) · XRP +69% (0.61). Broad-based — no single-name luck.
The multi-asset book exceeds the naive sum because gross normalization
reallocates capital to whichever names currently pass the gate.

## Robustness checks

- **EWMA memory (the real axis — `window` saturates for lam=0.97)**:
  lam 0.94 → SR 1.59–1.74; 0.97 → 1.79; 0.99 → 0.87–1.17. Degrades
  gracefully, not knife-edge.
- **Deadband** 1%→3%: +2,488% / SR 1.775 (turnover drag only).
- **Name cap** 15%: +691% SR 1.68 · 40%: +567% SR 1.20 — 25% optimal.
- **4h interval** (harder: 2191 bars/yr × costs): best book
  `ewma_emp` z0.2/risk +10.8% SR 0.60 over ~2y; fhs 15bps conviction
  book +0.83% SR 0.97 on 154 fills. Daily dominates 4h.
- **Fail-closed verified**: stale-mark `StaleValuationError`,
  fingerprint-bound resume (80→160-step continuation, NAV seam parity,
  cumulative fills, no duplicate orders), monotone-quantile rejection.

## What this does NOT establish

- Live profitability. Slippage, venue latency, borrow/funding, and
  market-impact regimes past the modeled caps are unmeasured.
- Cross-venue or cross-class generality — Binance USDT spot majors only.
- The OOS window overlaps the strongest trend regimes; a flat market
  would compress these numbers further.
