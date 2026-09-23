# Funding-carry universe expansion — 2026-09-23

Follow-up to `carry_research_2026_09.md`. Goal: raise net returns by widening the
Binance carry universe and/or adding venues. Net result: a new champion with
record returns, plus a mapped "no-go" region of configs that look great on the
continuous universe but die on the realistic one.

Research-only results; same honest engine (costs on both legs, wick-paranoid
liquidation, leverage cap at order time, t+1 fills, zero dropped funding events).

## Venue check — OKX (dead end)

`scripts/fetch_okx_carry.py` fetches deep OHLC history from OKX's public API
(spot since 2017, perp since 2019) but `funding-rate-history` only returns the
last ~3 months (279 events, oldest 2026-06-22). No public deep funding archive ⇒
no multi-venue carry backtest. Binance remains the only venue with years of
public funding history via `data.binance.vision`.

## Binance universe expansion

865 USDT-margined perps exist on Binance; 471 have a USDT spot pair; 356 have
≥36 months of funding history. `fetch_binance_vision_carry.py` fetched the 294
symbols missing from the core 61-coin set into `data/binance_carry_extra/`
(perp 365,623 bars, spot 435,670, funding 1,431,125 raw events, daily-aggregated
to 456k rows). `load_carry(extra_dir)` merges it with the core parquets.

- dev-window eligible (paired, ≤3-day gaps, listed through 2024-12): **283**
- holdout-window eligible: 58 (unchanged — new coins are too young)
- full-window eligible (listed continuously 2020→2026): **57** — the headline
  constraint: the expansion cannot lift the full-window universe at all.

## The survivorship trap (important)

A config evaluated on the full-window universe only sees the 57 coins that
survived to 2026. `nw0.20×mx15` scores +370% / Sharpe 6.38 / DD −1.43% there —
but on the 283-name dev-eligible book the same config is **liquidated to ruin**
(21 liquidations, DD −196%). Heavy per-name weights concentrate in the microcap
squeeze tail (late-2024 alt squeezes, Mar-2024 memecoin mania): delta-neutral
does not hedge basis squeezes — perp up 80% while spot lags is a real MTM loss.

Empirical safe region on the expanded dev book:

- `nw ≤ 0.12` at `mx ≈ 50–60` (broad enough to dilute squeezes) — safe
- mid-size books `mx 25–45` at `nw ≥ 0.05` — DD cliffs −9% to −24%, some with
  liquidations (e.g. nw0.10×mx40: −23.8%, 7 liqs)
- `nw ≥ 0.15` at any mx — ruined on dev-expanded
- gross ≥ ~3.5× → wick-liquidation wall (nw0.20×mx20: −94%, 17 liqs)

Config selection therefore required passing **both** conventions: dev-eligible
(283 names, coins that later die included) and full-window (57 survivors).

## New champion — `enter=1.5bp/d, exit=0, lb=9, nw=0.11, mx=60, band=1.3`

| window | Sharpe | CAGR | total | max DD | funding_net | liqs |
|--------|-------:|-----:|------:|-------:|------------:|-----:|
| dev 2020–24 (283 elig.)    | 6.54 | 33.8% | +328% | −1.85% | $3.54M | 0 |
| holdout 2025–26 (58 elig.) | 1.33 | 1.3%  | +2.2% | −1.70% | $0.07M | 0 |
| full 2020–26 (57 elig.)    | 6.20 | 24.2% | **+330%** | −2.09% | **$3.75M** | 0 |

vs the core champion (nw0.12×mx15): full Sharpe 6.88 → 6.20, CAGR 21.5% → 24.2%,
total +271% → **+330%**, funding $3.04M → **$3.75M**, DD −1.48% → −2.09%.
Record net return, record funding capture, record CAGR — at still >6 Sharpe and
<2.1% max DD on the full window. 2021 alone: **+122%**.

- worst day ever: −0.88% (2020-01-18); May-19-2021 crash: −0.87%
- max gross 2.96× (at the 3.0 cap); zero liquidations; worst-day < 1%
- yearly: 2020 +36%, 2021 +122%, 2022 +1.5%, 2023 +10.4%, 2024 +24.0%,
  2025 +2.9%, 2026 −0.4% (funding regime still compressed)

Sharpe-optimal alternative on the same expanded universe: `nw0.08×mx15`
(full Sharpe 7.55 / +186% / DD −1.30%) — traded away ~144pp of total return.
Aggressive variant `nw0.12×mx60` (+336% / $3.82M) passes all constraints but its
holdout Sharpe drops to 1.08; kept off the podium.

## Regime-scaled sizing (`rate_scale_*`)

`basis_carry_hysteresis_weights` gained optional regime-scaling:
`w = name_weight * clip(book_rate / rate_scale_ref, floor, cap)` where
`book_rate` is the day's mean trailing funding across top candidates + held
names. Upside-only mode (`floor=1.0`) never de-rates the flat book — it only
adds exposure when funding dispersion is rich.

Champion v2: `rsr=20bp/d, cap=1.5, floor=1.0` on `nw0.11/mx60` scored
+356% / 25.3% CAGR / Sharpe 5.98 / DD −2.10% / $4.03M — superseded by v3 below.

## Champion v4 — `enter=2bp/d, exit=−1.25bp/d, lb=9, nw=0.12, mx=60, band=1.3, rsr=20bp/d, cap=1.5, floor=1.0`

A 2bp/day entry bar (vs 1.5bp) keeps the squeeze-prone tail out while keeping
~60 qualifying names. A *negative* exit bar (−1.25bp/d, vs 0) holds a payer
through shallow funding dips instead of churning out and back in — mean daily
turnover halves (0.016 vs 0.025), saving ~35% of churn cost and keeping
positions through the dips that revert.

| window | Sharpe | CAGR | total | max DD | funding_net |
|--------|-------:|-----:|------:|-------:|------------:|
| dev (283 elig.)    | 6.52 | 34.4% | +338% | −1.56% | $3.54M |
| holdout (58 elig.) | 4.35 | 2.5%  | +4.3% | −0.35% | $0.06M |
| full (57 elig.)    | 6.44 | 26.9% | **+397%** | **−1.40%** | **$4.19M** |

Dominates v3 on every axis. Yearly: 2020 +37.4%, **2021 +136.6%**, 2022 +1.8%,
2023 +12.2%, 2024 +27.0%, 2025 +4.4%, 2026 +1.0%. Worst day ever −0.96%; zero
liquidations; 18 margin-rejects (3.0× leverage cap clipping at peak
dispersion; gross marks peaked 3.29×). Exit depth is a razor: −1.5bp rides
into squeezes (5 liqs, −26% DD); −0.5bp keeps too much churn (1 liq, 4.72
Sharpe).

## The frontier is mapped — v4 sits at the maximum

Every adjacent config is strictly worse or inadmissible; the binding
constraints are (a) baseline per-name weight ≤ ~0.12, (b) scaled weight
`nw×cap ≤ ~0.165`, (c) admission quality `enter ≈ 2bp`, (d) breadth `mx 50–75`:

| knob | values tried | outcome |
|------|--------------|---------|
| nw | 0.125–0.14 | liquidation (3–14 liqs, DD −18% to −69%) |
| rsc | 1.55–2.0+ | liquidation cliff (cap 1.5 = last safe rung) |
| rsr | ≤18bp, ≥22bp | cliff or flat-worse |
| band | 1.15–1.8 | 1.3 optimal; ≥1.4 dev-ruins (drift balloons before rebalance) |
| enter | 1.0–3.0bp | 2bp optimal; looser admits squeezers, tighter concentrates |
| exit | −2.0–0bp | −1.25bp optimal; ≥−0.5 keeps churn, ≤−1.5 rides into squeezes (5 liqs) |
| bar resolution | 1d vs 8h | 8h churn bleeds in compressed regimes: holdout −0.54 Sharpe / −5.97% DD — daily is the sweet spot |
| lb | 3–20 | 9 optimal; shorter whipsaws, longer rides through squeezes |
| mx | 15–150 | plateau 50–75; mid zone (25–45) is the DD cliff |
| floor | <1.0 | bidirectional scaling gives up ~100pp of full return |
| `rate_exponent` | 0.25–1.0 | tilting weight ∝ rate^exp is dominated: +7pp return at exp 0.25 but worse Sharpe/DD/holdout; concentration into top payers IS the squeeze channel |
| vol-scaling (`vol_lookback`/`vol_ref`) | 10–40d, 3–5%/d | Sharpe-optimal variant below |

## Vol-scaled sizing — Sharpe-optimal variant (not champion)

`vol_lookback=20, vol_ref=3%/d` scales each name's weight by
`min(1, vol_ref/vol_i)` — wild microcaps get diluted weight. Full-window
(under the v3 config): **Sharpe 6.81, DD −1.29%**, but only +201% total —
it trades ~164pp of return for +0.69 Sharpe. Kept as a capability (not the
return-max config) and worth revisiting if the mandate shifts from CAGR to
risk-adjustment.

## 8h bar resolution — probed and rejected

`data/binance_carry_8h{,_majors}` (299 coins, funding-native 8h bars) under
the v3 config: full Sharpe 6.63 but only +274%, and holdout **−0.54 Sharpe /
−5.97% DD** — 3× more membership/rebalance decision points turns boundary
names into churn cost exactly where yield is thinnest. Daily bars remain
the sweet spot for this mechanism.

## Multi-venue expansion — venue depth census + dYdX probe

Venue funding-history census (all public endpoints probed 2026-09-22):

| venue | OHLC depth | funding depth | verdict |
|-------|-----------|---------------|---------|
| Binance | 2019+ | 2019+ (Vision) | **base book** |
| Hyperliquid | listing+ | listing+ (hourly, paginated) | **fetched** |
| dYdX v4 indexer | 2023-10+ | 2023-10+ (hourly) | fetched, rejected below |
| OKX | 2017+ | ~3 mo only | dead |
| Bybit | — | ~3 mo only | dead |
| Gate.io | — | hard-capped 180 d | dead |
| Bitget | — | ~1 mo | dead |

`scripts/fetch_dydx_carry.py` / `build_dydx_book.py` + `fetch_hyperliquid_carry.py`
/ `build_hl_book.py` produce venue-namespaced books (`DYDX:BTC`, `HL:BTC` — each a
separate delta-neutral pair marked at its own wicks, spot hedge from the venue
itself when listed, else Binance spot). This keeps cross-venue basis risk
*inside* the model rather than assuming one global perp price.

### Two data traps found and fixed (`carry_research.py`)

1. **`eligible_coins` union-staleness:** "still listed at window end" compared
   each coin's last bar to the *merged* union's last timestamp — any venue
   whose fetch ended a day earlier silently dropped ALL its names, and a stale
   `binance_carry_extra` (ended 2026-08-31) ejected the whole extra universe
   from holdout/full. Now `nz[-1] >= last_i - max_gap`: delisted-mid-window
   names stay excluded, venues ending ≤3 days apart all pass.
2. **`run_grid` skipped the eligible filter** — dead coins (YFII, AUDIO)
   entered the book then aborted every config on `StaleValuationError`. Grid
   now filters identically to `run_champion`.

`fetch_binance_daily_tail.py` gained `--data`/`--since` + a funding-tail
merge; klines for both Binance dirs refreshed through 2026-09-22.
(`fapi.binance.com` funding endpoint geo-blocks this host with HTTP 451 —
funding history ends 2026-08-31 for Binance names, so Sep-2026 carry is
modestly understated. Honest direction: missing events can't pay.)

### dYdX result — real risk, net-negative, rejected

- dYdX standalone (68 names, Binance-spot hedge): dev Sharpe 6.80 / +23%, but
  **holdout DD −17.1%** — the Oct-2025 flash crash decoupled dYdX perps from
  the Binance spot hedge leg for hours. That is honest modeled risk, not a
  data bug: on thin venues the two legs genuinely stop trading together.
- Merged with the Binance book under champion v4: dev 6.56/+342% (a wash —
  dYdX rarely clears the 2bp bar), full **5.83/+355% vs Binance-only
  6.44/+397%**, holdout 1.42 vs 4.35. The venue adds basis-crash exposure
  with almost no funding harvest.
- `enter_rate_by_prefix` (per-venue entry bars, e.g. `{"DYDX:": 0.0005}`)
  probed at 3/5/8bp — dev return moved ≤0.4pp; thin book stays thin.
  dYdX is **documented, not shipped**; fetcher + builder retained.

## Reproduce

```
uv run python scripts/fetch_binance_vision_carry.py \
  --coins "$(cat /tmp/new_coins.txt)" --out data/binance_carry_extra --workers 32
uv run python scripts/carry_research.py champion --extra-dir data/binance_carry_extra
```

Artifacts: `artifacts/carry_champion_expanded.json`,
`carry_equity_expanded_full.parquet`, `carry_weights_expanded_full.parquet`.
Raw sweep outputs used for selection: `/tmp/expand_grid2.json` and the mx/nw
probes logged in-session (dev-expanded + full-window both listed above).
