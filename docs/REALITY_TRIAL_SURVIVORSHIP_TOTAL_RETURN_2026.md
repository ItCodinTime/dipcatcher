# Survivorship trial: quote close vs total return

Research diagnostic only. Same point-in-time S&P membership grid as
`docs/REALITY_TRIAL_SURVIVORSHIP_2026.md`, scored twice on one Yahoo
fetch: once on quote closes (default), once after cash-dividend
reinvestment through `prepare_bars(..., actions)`. Reality-filter
thresholds and the sealed ledger were not edited. Not a live-trading
claim.

Comparison receipt sha256: `3669d2f6cf7e2049d54cdc3d292d58a3f1f29d060c1cea73d6ea7628c2fa82f3`.
Cash-dividend rows on the fetch: 18166.
Fetch failures: 149.

## Equal-weight pit baseline

| return basis | validation compounded | holdout compounded | validation annualized | holdout annualized |
|---|---:|---:|---:|---:|
| quote_close | 0.0756639 | 0.236106 | 0.337252 | 1.64571 |
| total_return | 0.105134 | 0.26149 | 0.418527 | 1.71939 |

## Every cell: holdout compounded return

| strategy | params | published quote | recomputed quote | total return | delta (TR − quote) |
|---|---|---:|---:|---:|---:|
| xs_momentum | `{"gross_scale": 1.0, "lookback": 63, "max_name": 0.05, "quantile": 0.2, "rebalance_bars": 21, "skip": 21}` | -0.117396 | -0.113467 | -0.115681 | -0.00221409 |
| st_reversal | `{"gross_scale": 1.0, "lookback": 21, "max_name": 0.05, "quantile": 0.2, "rebalance_bars": 21, "skip": 1}` | 0.00326659 | 0.00746731 | 0.0165312 | 0.00906391 |
| dip_regime | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 10, "lookback": 40, "max_name": 0.05, "regime_lookback": 63}` | 0.0813172 | 0.0631418 | 0.115853 | 0.0527115 |
| dip_regime | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 10, "lookback": 20, "max_name": 0.05, "regime_lookback": 63}` | 0.103002 | 0.0961774 | 0.117969 | 0.021792 |
| dip_regime | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 21, "lookback": 20, "max_name": 0.05, "regime_lookback": 63}` | 0.155046 | 0.154938 | 0.18022 | 0.025282 |
| xs_momentum | `{"gross_scale": 1.0, "lookback": 126, "max_name": 0.05, "quantile": 0.2, "rebalance_bars": 21, "skip": 21}` | -0.0485293 | -0.0439046 | -0.0461893 | -0.00228472 |
| low_vol | `{"quantile": 0.2, "rebalance_bars": 21, "vol_window": 21}` | 0.0571605 | 0.0571605 | 0.100705 | 0.0435442 |
| dip_regime | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 21, "lookback": 40, "max_name": 0.05, "regime_lookback": 63}` | 0.133744 | 0.138408 | 0.162177 | 0.0237688 |
| equal_weight_pit | `{"name_cap": "risk_gate.max_name", "net_cap": "risk_gate.max_net"}` | 0.235979 | 0.236106 | 0.26149 | 0.0253844 |
| low_vol | `{"quantile": 0.2, "rebalance_bars": 21, "vol_window": 126}` | 0.0752294 | 0.0752294 | 0.101017 | 0.0257876 |
| low_vol | `{"quantile": 0.2, "rebalance_bars": 21, "vol_window": 63}` | 0.100446 | 0.100446 | 0.081161 | -0.0192846 |
| xs_momentum | `{"gross_scale": 1.0, "lookback": 252, "max_name": 0.05, "quantile": 0.2, "rebalance_bars": 21, "skip": 21}` | -0.0130461 | -0.0100688 | -0.0215185 | -0.0114498 |
| st_reversal | `{"gross_scale": 1.0, "lookback": 5, "max_name": 0.05, "quantile": 0.2, "rebalance_bars": 5, "skip": 1}` | -0.120275 | -0.116559 | -0.11618 | 0.00037838 |

## Notes

- Published quote figures are the sealed survivorship receipt
  (quote closes, dividends omitted).
- Recomputed quote and total-return figures share one Yahoo download
  with `events=div,split`, so the delta isolates dividend reinvestment.
- Fills stay at the open on the same price basis as the mark.
- Annualized ratio is mean/std × √252 on net simple returns; it is an
  overfitting diagnostic here, not a promotion.

