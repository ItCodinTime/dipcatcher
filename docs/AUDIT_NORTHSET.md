# Northset deep audit (P6.x addendum)

Scope: `src/quant_fund/northset/` — estimator pack, candle geometry,
sweep-research benches, data view. `kyle_ofi.py` was verified correct
against Cont–Kukanov–Stoikov in a prior pass and excluded here.

## Defects fixed

1. **`volume_over_range` fail-open clip** — `volume / clip(high−low, 1e-12)`
   turned a flat or invalid bar into a finite `volume × 1e12` liquidity
   score. Sibling functions (`amihud_illiquidity`, `queue_imbalance`)
   already emit null on degenerate denominators; now this does too.
   (`estimators.py`)

2. **`candle_geometry` same class, wider blast radius** — every
   range-derived column and all six pattern flags divided by the clipped
   `1e-12` denominator. A corrupt flat bar with `close` outside its
   (zero-width) range flagged `candle_marubozu`; `close_location_value`
   and wick fractions exploded to ±1e12-scale values. Now all
   range-derived columns and flags are null unless the bar is valid
   (finite, positive, `high ≥ low`, positive range); `candle_gap` also
   requires a finite positive `prev_close`, and `candle_engulfing`
   requires a previous bar. (`candles.py`)

3. **`_volume_clock_vpin` dropped the bucket-crossing remainder** — a
   row whose volume crossed a bucket boundary had its overflow
   discarded, so buckets were *at least* `bucket_volume` (variable
   volume) rather than exactly `bucket_volume` as the Easley volume
   clock requires; a single large row collapsed to one bucket instead
   of several. Now carries the excess into the next bucket, split by
   the row's own buy/sell mix, in a `while` loop (a row can complete
   multiple buckets). (`estimators.py`)

## Verified correct (line-by-line)

- `kyle_lambda`, `roll_spread` (sign convention + positive-γ NaN),
  `ohlc_variance_frame` (Parkinson / Garman–Klass / Rogers–Satchell /
  close-to-close), `yang_zhang_variance` (`k` weight correct),
  `corwin_schultz_spread` (PIT two-day pairing, negative→zero before
  averaging), `amihud_illiquidity`, `order_flow_imbalance` (CKS legs),
  `session_realized_variance`, `session_bipower_jump` (BNS π/2
  coefficient), `session_vpin`, `queue_imbalance`, `abdi_ranaldo_spread`,
  `dm_range_vs_park` / `dm_split_vs_park` (date-level QLIKE + DM),
  `yang_zhang_vs_close_to_close` (expanding PIT forecast — no pooled
  in-sample constant), `true_range_frame`, `lag1_corr`,
  `overnight_share`, `realized_semivariance`.
- `sweeps.py`, `sweep_research.py`, `data_view.py`, `identities.py`,
  `benches.py`: `ohlc_ok` gating pattern used consistently; `fill_null(0)`
  on flag columns is defined absence semantics; entry/exit causality
  (`entry_ok & prev bar`) correct.

## Tests

`tests/unit/microstructure/test_northset_failclosed_hygiene.py` — 10
regression tests pinning the three fixes plus valid-path behavior.
`test_p65_exec_microstructure_audit.py::test_vpin_buy_minus_sell_equals_ofi`
updated: it encoded the buggy "one row = one bucket" assumption; under
the true volume clock a row of volume V completes V unit buckets.
