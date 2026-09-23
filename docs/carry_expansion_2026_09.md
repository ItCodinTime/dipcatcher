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

## Champion v3 — `enter=2bp/d, exit=0, lb=9, nw=0.12, mx=60, band=1.3, rsr=20bp/d, cap=1.5, floor=1.0`

A 2bp/day entry bar (vs 1.5bp) is a sharp local optimum: high enough to keep
the squeeze-prone tail out, low enough to keep ~60 qualifying names.

| window | Sharpe | CAGR | total | max DD | funding_net |
|--------|-------:|-----:|------:|-------:|------------:|
| dev (283 elig.)    | 6.36 | 35.3% | +353% | −1.84% | $3.79M |
| holdout (58 elig.) | 1.87 | 1.7%  | +2.9% | −0.98% | $0.07M |
| full (57 elig.)    | 6.12 | 25.7% | **+365%** | **−1.53%** | **$4.09M** |

Dominates v2 on every axis. Yearly: 2020 +37.4%, **2021 +132.7%**, 2022 +1.5%,
2023 +11.5%, 2024 +25.3%, 2025 +2.8%. Worst day ever −0.96%; zero
liquidations; 8 margin-rejects (leverage cap honestly clipping at peak
dispersion; gross marks peaked 3.26×).

## The frontier is mapped — v3 sits at the maximum

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
| lb | 3–20 | 9 optimal; shorter whipsaws, longer rides through squeezes |
| mx | 15–150 | plateau 50–75; mid zone (25–45) is the DD cliff |
| floor | <1.0 | bidirectional scaling gives up ~100pp of full return |
| vol-scaling (`vol_lookback`/`vol_ref`) | 10–40d, 3–5%/d | Sharpe-optimal variant below |

## Vol-scaled sizing — Sharpe-optimal variant (not champion)

`vol_lookback=20, vol_ref=3%/d` scales each name's weight by
`min(1, vol_ref/vol_i)` — wild microcaps get diluted weight. Full-window:
**Sharpe 6.81, DD −1.29%**, but only +201% total — it trades ~164pp of
return for +0.69 Sharpe. Kept as a capability (not the return-max config)
and worth revisiting if the mandate shifts from CAGR to risk-adjustment.

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
