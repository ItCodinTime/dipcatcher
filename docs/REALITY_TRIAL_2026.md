# Reality-filter trial

Research diagnostic on the pre-registered Yahoo daily sweep.
Annualized ratio means Sharpe ratio: per-period mean over sample
standard deviation, times the square root of the periods-per-year count.
It is not a promotion and not a live-trading claim.
Proper-score research elsewhere in this repo is unchanged.
Reality-filter thresholds were not edited.

Verdict: `deflated`.
Trials recorded: 29.
Gate best ledger row: `7cef5aafdc0eeabe0e7b5b6ce3aa3b2dfdc206df6c958fe03fb746ef65a8f249`.
Effective trials (cluster count): 3.0.
Gate deflated probability: 0.6835483771834405.
Gate PSR of the best ledger row: 0.8009901688519906.
Gate PBO: None (unset when the JSONL ledger has no return series).
CSCV PBO on train-then-validation returns: 0.12152292152292152.
Deflated probability using the raw trial count: 0.48311665018613564.
Selected trial: `7cef5aafdc0eeabe0e7b5b6ce3aa3b2dfdc206df6c958fe03fb746ef65a8f249` (equal_weight_long).
Dataset sha256: `1eea9658c5b0665be9889a78f48ab8faa5d11c3faf53c50bf697332fa35bfb32`.
Receipt sha256: `514ccac702d5fa657e415ceb5f6fdf95642dc1741ea24e526da8848208135845`.
Audit entry hash: `257730ce7caeca772953563dda88b540f7779ea3b85165a63672b3fe14f47bb7`.

## Selected trial

| window | n | compounded net return | per-period ratio | annualized ratio | CI low | CI high | max drawdown | mean turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1006 | 0.417719 | 0.0740094 | 1.17486 | 0.187483 | 2.20158 | -0.154105 | 0 |
| validation | 756 | 0.277651 | 0.0308356 | 0.4895 | -0.484299 | 1.53192 | -0.333424 | 0 |
| holdout | 502 | 2.20888 | 0.137267 | 2.17904 | 0.908711 | 3.54141 | -0.197809 | 0 |

Holdout was summarized once, after every trial had been inserted.
Selection used the validation per-period ratio only.

## Every trial

| strategy | params | validation annualized | holdout annualized | validation compounded | holdout compounded |
|---|---|---:|---:|---:|---:|
| sweep_reclaim | `{"decay": 0.75, "gross_scale": 1.0, "hold_bars": 10, "lookback": 10, "max_name": 0.05}` | -0.107215 | -0.707735 | -0.0103762 | -0.010496 |
| sweep_reclaim | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 5, "lookback": 10, "max_name": 0.05}` | -0.431205 | -0.888469 | -0.040478 | -0.0181151 |
| sweep_reclaim | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 21, "lookback": 10, "max_name": 0.05}` | -0.339626 | -0.49927 | -0.0599176 | -0.0241402 |
| slow_trend | `{"fast_bars": 50, "gross_scale": 1.0, "max_name": 0.05, "slow_bars": 100, "vol_window": 20}` | -0.504677 | -2.79067 | -0.0965853 | -0.260578 |
| sweep_reclaim | `{"decay": 0.75, "gross_scale": 1.0, "hold_bars": 21, "lookback": 20, "max_name": 0.05}` | -0.0200125 | -0.64689 | -0.00242785 | -0.00927383 |
| slow_trend | `{"fast_bars": 20, "gross_scale": 1.0, "max_name": 0.05, "slow_bars": 200, "vol_window": 20}` | -0.398431 | -3.21044 | -0.0743165 | -0.275977 |
| sweep_reclaim | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 5, "lookback": 20, "max_name": 0.05}` | -0.156041 | -0.828712 | -0.0140839 | -0.0157949 |
| sweep_reclaim | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 10, "lookback": 20, "max_name": 0.05}` | -0.49397 | -0.630111 | -0.0689045 | -0.0199492 |
| sweep_reclaim | `{"decay": 0.75, "gross_scale": 1.0, "hold_bars": 10, "lookback": 40, "max_name": 0.05}` | 0.062194 | -0.516317 | 0.00362976 | -0.00679922 |
| slow_trend | `{"fast_bars": 20, "gross_scale": 1.0, "max_name": 0.05, "slow_bars": 200, "vol_window": 60}` | -0.50563 | -3.42374 | -0.0926376 | -0.290822 |
| slow_trend | `{"fast_bars": 20, "gross_scale": 1.0, "max_name": 0.05, "slow_bars": 100, "vol_window": 20}` | -0.551739 | -3.02124 | -0.10458 | -0.27738 |
| sweep_reclaim | `{"decay": 0.75, "gross_scale": 1.0, "hold_bars": 5, "lookback": 40, "max_name": 0.05}` | 0.176282 | -0.546308 | 0.0105271 | -0.00649796 |
| slow_trend | `{"fast_bars": 50, "gross_scale": 1.0, "max_name": 0.05, "slow_bars": 100, "vol_window": 60}` | 0.044851 | -2.62411 | 0.00251597 | -0.258508 |
| slow_trend | `{"fast_bars": 50, "gross_scale": 1.0, "max_name": 0.05, "slow_bars": 200, "vol_window": 20}` | -0.326712 | -3.18445 | -0.0666265 | -0.290614 |
| equal_weight_long | `{"name_cap": "risk_gate.max_name", "net_cap": "risk_gate.max_net"}` | 0.4895 | 2.17904 | 0.277651 | 2.20888 |
| sweep_reclaim | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 5, "lookback": 40, "max_name": 0.05}` | 0.0219696 | -0.723947 | 0.000708339 | -0.0128502 |
| sweep_reclaim | `{"decay": 0.75, "gross_scale": 1.0, "hold_bars": 5, "lookback": 20, "max_name": 0.05}` | 0.0601073 | -0.698874 | 0.00328739 | -0.0088979 |
| sweep_reclaim | `{"decay": 0.75, "gross_scale": 1.0, "hold_bars": 5, "lookback": 10, "max_name": 0.05}` | -0.0912832 | -0.804812 | -0.00741118 | -0.011 |
| sweep_reclaim | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 21, "lookback": 40, "max_name": 0.05}` | -0.0117582 | -0.692268 | -0.00613287 | -0.0306233 |
| sweep_reclaim | `{"decay": 0.75, "gross_scale": 1.0, "hold_bars": 21, "lookback": 40, "max_name": 0.05}` | 0.110702 | -0.509249 | 0.00692334 | -0.00681926 |
| sweep_reclaim | `{"decay": 0.75, "gross_scale": 1.0, "hold_bars": 10, "lookback": 20, "max_name": 0.05}` | -0.068837 | -0.646462 | -0.00625073 | -0.00913008 |
| slow_trend | `{"fast_bars": 20, "gross_scale": 1.0, "max_name": 0.05, "slow_bars": 100, "vol_window": 60}` | -0.17682 | -2.99635 | -0.0385642 | -0.285196 |
| slow_trend | `{"fast_bars": 50, "gross_scale": 1.0, "max_name": 0.05, "slow_bars": 200, "vol_window": 60}` | -0.269838 | -2.94469 | -0.0576137 | -0.27296 |
| slow_trend | `{"fast_bars": 100, "gross_scale": 1.0, "max_name": 0.05, "slow_bars": 200, "vol_window": 20}` | -0.481252 | -2.17941 | -0.0986439 | -0.190771 |
| sweep_reclaim | `{"decay": 0.75, "gross_scale": 1.0, "hold_bars": 21, "lookback": 10, "max_name": 0.05}` | -0.1484 | -0.677929 | -0.0132608 | -0.0101635 |
| sweep_reclaim | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 10, "lookback": 40, "max_name": 0.05}` | -0.239015 | -0.598936 | -0.0319272 | -0.0176076 |
| sweep_reclaim | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 21, "lookback": 20, "max_name": 0.05}` | -0.346719 | -0.737373 | -0.0585989 | -0.0348532 |
| slow_trend | `{"fast_bars": 100, "gross_scale": 1.0, "max_name": 0.05, "slow_bars": 200, "vol_window": 60}` | -0.0517449 | -2.4116 | -0.0170831 | -0.217202 |
| sweep_reclaim | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 10, "lookback": 10, "max_name": 0.05}` | -0.388922 | -0.479223 | -0.0574955 | -0.0158489 |

## Data

Included symbols: SPY, AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, JNJ, XOM, UNH, V, PG, HD, MA, KO.
Excluded symbols: [].
Prices stayed out of git. The return panel hash is in the receipt.
US_LIQUID is a present-day list. Quote OHLC is the adapter series;
the total-return mark equals that close.
