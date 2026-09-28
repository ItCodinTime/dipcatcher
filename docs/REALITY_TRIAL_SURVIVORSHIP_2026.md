# Survivorship-aware reality trial

Research diagnostic. Annualized ratio means a Sharpe ratio:
per-period mean over sample standard deviation, times the square
root of the periods-per-year count. It is not a promotion and not
a live-trading claim. Reality-filter thresholds were not edited.
Round-1 ledger rows were not rewritten.

Verdict: `deflated`.
Trials recorded: 42.
Effective trials (cluster count): 8.0.
Gate best ledger row: `3a945e54c0ea626f60649898e09df6d7eb80031cdb606c4aa9fca44202e5a748`.
Gate deflated probability: 0.6095974644672961.
Gate PSR of the best ledger row: 0.8467776487651919.
Gate PBO: None (unset; the JSONL ledger has no return series).
CSCV PBO on the 13 new train-then-validation paths: 0.626961926961927.
Deflated probability using the raw trial count: 0.4584882964082586.
New-study winner: `3a945e54c0ea626f60649898e09df6d7eb80031cdb606c4aa9fca44202e5a748` (dip_regime).
Dataset sha256: `dba802984c115ca978fd54ed3d69b0e9225457d51f69cc332c04eb520fbbdef0`.
Receipt sha256: `599476ae565b14155ed1b7ab9d6d1531c458a10b3ba86d14f2f28a57322eb13b`.

## Survivorship and the equal-weight baseline

Both books use the frozen net cap of 0.2 and the same cost model.
Round-1 figures are copied from that receipt. The point-in-time
figures are this study's equal-weight cell.

| book | universe | validation compounded | holdout compounded | validation annualized ratio | holdout annualized ratio |
|---|---|---:|---:|---:|---:|
| equal_weight_long | present-day US_LIQUID | 0.277651 | 2.20888 | 0.4895 | 2.17904 |
| equal_weight_pit | point-in-time S&P 500 members | 0.0750745 | 0.235979 | 0.335048 | 1.64612 |

## Every new trial

| strategy | params | validation annualized | holdout annualized | validation compounded | holdout compounded |
|---|---|---:|---:|---:|---:|
| xs_momentum | `{"gross_scale": 1.0, "lookback": 63, "max_name": 0.05, "quantile": 0.2, "rebalance_bars": 21, "skip": 21}` | -0.283593 | -0.937345 | -0.112508 | -0.117396 |
| st_reversal | `{"gross_scale": 1.0, "lookback": 21, "max_name": 0.05, "quantile": 0.2, "rebalance_bars": 21, "skip": 1}` | 0.0732744 | 0.0575035 | 0.00610762 | 0.00326659 |
| dip_regime | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 10, "lookback": 40, "max_name": 0.05, "regime_lookback": 63}` | 0.371697 | 0.823504 | 0.0835727 | 0.0813172 |
| dip_regime | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 10, "lookback": 20, "max_name": 0.05, "regime_lookback": 63}` | 0.595073 | 1.06805 | 0.132112 | 0.103002 |
| dip_regime | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 21, "lookback": 20, "max_name": 0.05, "regime_lookback": 63}` | 0.340721 | 1.23994 | 0.0835394 | 0.155046 |
| xs_momentum | `{"gross_scale": 1.0, "lookback": 126, "max_name": 0.05, "quantile": 0.2, "rebalance_bars": 21, "skip": 21}` | -0.0345873 | -0.326075 | -0.0356942 | -0.0485293 |
| low_vol | `{"quantile": 0.2, "rebalance_bars": 21, "vol_window": 21}` | 0.25737 | 0.751537 | 0.0463502 | 0.0571605 |
| dip_regime | `{"decay": 1.0, "gross_scale": 1.0, "hold_bars": 21, "lookback": 40, "max_name": 0.05, "regime_lookback": 63}` | 0.301938 | 1.16236 | 0.0694204 | 0.133744 |
| equal_weight_pit | `{"name_cap": "risk_gate.max_name", "net_cap": "risk_gate.max_net"}` | 0.335048 | 1.64612 | 0.0750745 | 0.235979 |
| low_vol | `{"quantile": 0.2, "rebalance_bars": 21, "vol_window": 126}` | 0.261808 | 1.02598 | 0.0456322 | 0.0752294 |
| low_vol | `{"quantile": 0.2, "rebalance_bars": 21, "vol_window": 63}` | 0.293278 | 1.2796 | 0.0541672 | 0.100446 |
| xs_momentum | `{"gross_scale": 1.0, "lookback": 252, "max_name": 0.05, "quantile": 0.2, "rebalance_bars": 21, "skip": 21}` | -0.168092 | -0.0461532 | -0.0947902 | -0.0130461 |
| st_reversal | `{"gross_scale": 1.0, "lookback": 5, "max_name": 0.05, "quantile": 0.2, "rebalance_bars": 5, "skip": 1}` | -0.0856318 | -1.02337 | -0.0527252 | -0.120275 |

## Coverage

| anchor | session | members | with a bar | coverage |
|---|---|---:|---:|---:|
| 2016-01-04 | 2016-01-04 | 509 | 401 | 0.787819 |
| 2020-01-02 | 2020-01-02 | 507 | 453 | 0.893491 |
| 2023-01-03 | 2023-01-03 | 504 | 480 | 0.952381 |

Fetch failures: 148.
Delisting-exit bars (copied last close, not a print): 0.
Names with at least one real bar: 577.

Every fetched series runs through the end of the sample, so the exit-bar
rule never fired. Names that left the index and then left Yahoo are the
148 failures, not positions that were flattened. That list includes SIVB,
FRC, TWTR, CELG, ATVI, and BBBY. They are absent from the book. The
point-in-time membership is real; the price panel still misses delisted
names Yahoo does not serve. `make reality-gate` prints the same deflated
verdict and exits 1, which is the gate's non-pass status, not a crashed run.
The selected cell has the best validation ratio. Its holdout compounded
return is not the best holdout in the new family.

## Caveats

- The change log is Wikipedia's selected-changes table, not CRSP. Event dates can be announcement dates. The source validated price coverage from 2016-06-30, not the economic identity of every earlier event. Train begins 2016-01-04, so January-June 2016 membership is inside the source's unvalidated tail.
- Renames without a remove/add pair keep the snapshot's later ticker for the whole span (META, BKNG). Yahoo's series under that ticker is what gets fetched. The historical ticker is not reconstructed.
- Names Yahoo does not serve are absent. That is residual survivorship, counted in the receipt, not corrected by substitution.
- The delisting exit copies the last close. Wipeouts that halted above the recovery are not marked to zero.
- Quote closes omit dividends. A high-yield name is understated relative to a total-return index.
- The same frozen net cap of 0.2 applies. This is not a fully invested S&P book. The round-1 baseline used that cap too, so the compounded-return comparison is on the same risk budget.

Holdout was summarized after the new trials were appended.
Selection used the validation per-period ratio only.
The CSCV figure covers the 13 new paths. It does not include
the original 29, whose return series were not stored.
The gate deflates the best row in the full ledger.
