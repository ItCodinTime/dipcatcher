# Delta-Neutral Funding-Carry Research — 2026-09

Research-only results. Simulated fills at next bar open, costs on both legs,
wick-paranoid liquidation. No live-PnL claim.

## The trade

`basis_carry_hysteresis_weights` + `run_carry_backtest`: long spot + short
perp per coin (directional PnL cancels), harvesting positive funding while the
trailing realized rate stays above threshold. Sparse membership book — pairs
are entered/exited on hysteresis bands, not rebalanced daily.

## Data

- Source: Binance public archive (`data.binance.vision` S3) — no auth needed.
- `data/binance_carry/`: 61 USDⓈ-M perp coins, daily perp + spot klines +
  funding events. 2020-01 → 2026-09-22. ~110k perp / 127k spot / 366k funding rows.
- Funding is aggregated to the daily bar (sum of the day's 8h settlements) —
  `funding_events_dropped = 0`.
- Note: monthly archive files use millisecond epochs; daily-granularity files
  (needed for spot 2025+) use **microseconds** — the fetcher auto-detects.
- Universe caveat: continuously-listed names only. 5 coins with >3-bar paired
  gaps in holdout (FTM, EOS, MKR, ONDO, TON) are excluded per-window — a held
  pair can't be marked while delisted, and the engine aborts on stale marks.
  Mild survivorship tilt is inherent to the archive (current-listed names).

## Champion config (dev-selected)

`enter=1.5bp/day`, `exit=0`, `lookback=9 events`, `name_weight=12%`,
`max_names=15`, `rebalance_band=1.3`, `max_leverage=3`, costs default
(1bp comm + 5bp half-spread + sqrt impact), risk gate relaxed for the pair
book (name cap / predicted-vol gates are equity-book calibrations).

| window | Sharpe | CAGR | total ret | max DD | funding net |
|--------|--------|------|-----------|--------|-------------|
| dev 2020–2024 | **8.15** | **29.8%** | +268% | −1.1% | $2.87M |
| holdout 2025–2026 | 2.04 | 2.1% | +3.6% | −1.3% | $78k |
| full 2020–2026 | **6.88** | **21.5%** | +271% | −1.5% | $3.04M |

Yearly: 2020 +35%, 2021 +92%, 2022 +1.5%, 2023 +10.6%, 2024 +22%, 2025 +3.1%, 2026 +0.1%.
Worst daily NAV move: −0.96%. Zero liquidations, zero margin rejects,
gross stayed ≤2.1× (band-bounded drift).

## Robustness

- **Funding haircut**: ×0.7 funding → Sharpe 7.99; ×0.5 → 6.92; breaks 5 only at ×0.3.
- **BookRiskOverlay** (vol target + 5% DD halt): identical metrics — never
  needed to fire (book vol ~1.6%/yr). Free insurance available.
- **Wick-liquidation safety**: the earlier flat book at band=None drifted to
  3.26× gross and was force-liquidated on 2021-05-19 (−180%). With
  band=1.3 + nominal gross ≤1.8× the same bar is −0.66%. Rebalance band is
  load-bearing — do not remove.
- **Grid stability**: Sharpe >7.6 across all 36 hysteresis configs — the
  result is structural (yield harvesting), not a parameter fit.
- **Tiered (rate-scaled) sizing**: CAGR up to 32.6% dev but Sharpe ~7; flat
  name weight kept for risk-adjusted quality.

## The honest regime caveat

2025–2026 funding compressed ~10× vs 2020–24 (mean daily 1.9bp → −12bp in
2026). The book's holdout Sharpe is 2.0 — still positive, still <1.3% DD,
but it mostly sits in cash waiting for the funding cycle. The full-period
Sharpe 6.9 / CAGR 21.5% is regime-mixed. This cannot be tuned away honestly.

## Equity-MN cross-check (why carry, not equity alpha)

`scripts/eq_mn_sweep.py` on the 424-name survivorship tape: every reversal
AND continuation signal loses money after costs (rev1 −1.59, cont1 −3.02;
turnover ~2.5 NAV/day bleeds ~12bps/day). Best book = mom12_1 at ~0.3 Sharpe.
Daily equity MN on this tape cannot pay 5bps/side at required turnover — the
carry lane is the only structure in reach of the targets.

## Reproduce

```
uv run python scripts/fetch_binance_vision_carry.py --workers 8   # monthly bulk
uv run python scripts/fetch_binance_daily_tail.py --workers 30    # daily tail (spot 2025+)
uv run python scripts/carry_research.py grid                      # dev sweep
uv run python scripts/carry_research.py champion                  # dev/holdout/full eval
```

Artifacts: `artifacts/carry_champion.json`, `carry_equity_full.parquet`,
`carry_weights_full.parquet`, `carry_dev_grid.json`, `carry_tiered_dev.json`.
