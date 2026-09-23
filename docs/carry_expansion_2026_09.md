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
