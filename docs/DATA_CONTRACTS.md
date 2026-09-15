# Data contracts

## Point-in-time fields

Every stored observation includes:

| Field | Meaning |
|---|---|
| `event_time` | When the economic event occurred (bar close, print, filing timestamp) |
| `available_time` | When a market participant could know it |
| `ingested_time` | When this system ingested it |
| `source` | Adapter / vendor id |
| `security_id` | Immutable internal id |
| `symbol` | Ticker at `event_time` (not a join key) |
| `revision_id` | Vintage of restated data |

`available_time <= decision_time` is mandatory for every feature. For daily bars, default `available_time = event_time` (close). Ingestion lag is recorded but does not relax availability.

## Bars (bronze)

Columns: `security_id`, `symbol`, `event_time`, `available_time`, `ingested_time`, `source`, `revision_id`, `open`, `high`, `low`, `close`, `volume`, `currency`, `session`.

OHLC are **raw** (unadjusted). Volume is share volume.

## Corporate actions (bronze)

`action_type` in `{split, cash_dividend, special_dividend, delist, ticker_change}`. Splits store `factor` (e.g. 2.0 for 2-for-1). Dividends store `amount` in the listing currency and `ex_date` as `event_time`. `available_time` is the announcement time when known; otherwise ex-date (conservative: do not assume earlier knowledge).

## Adjusted series (silver)

Computed, never replacing raw:

- `close_split_adjusted`
- `close_total_return` (splits + cash dividends reinvested)
- `open/high/low` split-adjusted with the same cumulative split factor

**Label mapping**

| Target | Series |
|---|---|
| Alpha / ranking / excess return | simple total-return |
| Distribution quantiles | log total-return |
| Realized volatility / HAR-RV | log total-return (or RV if intraday exists) |
| Liquidity / Amihud | raw dollar volume with simple raw return |

## Security master

`security_id`, `ticker`, `name`, `exchange`, `currency`, `sector`, `industry`, `valid_from`, `valid_to`. Ticker lookup is as-of dated.

## Universe membership

`effective_from`, `effective_to`, `security_id`, `symbol`, `sector`, `industry`, `exchange`. Membership at \(t\) uses only information with `available_time <= t` (trailing ADV, price, history). Today's index constituents are never applied historically. Delisted names remain in history through `effective_to`.

## Feature frames (gold)

Must include `decision_time`, `max_source_available_time`, `security_id`, feature columns, and `feature_set_version`. Building a frame with `max_source_available_time > decision_time` raises `PointInTimeError`.

## Forecasts

See `AssetForecast` in `quant_fund.schemas.forecast`. Horizon keys are strings `1d`, `5d`, `20d`. All engines write `model_version`. Optional conformal fields `interval_lo`, `interval_hi`, `interval_alpha`, and `interval_method` (`mondrian_cqr` / `split_cqr`) are coverage-guaranteed sets and do not replace raw quantile PIT, pinball, or CRPS.

## Synthetic data

Rows produced by `SyntheticMarketProvider` set `source="synthetic"` and `revision_id="SYNTHETIC"`. Reports and CLI output prefix `SYNTHETIC` when any input row is synthetic. Synthetic results are not evidence of live edge.

## Missingness

Do not `fillna(0)` by default. Policies: train-period median, missing indicator, model-native NA, drop feature, drop row. Policy is config `missing.policy`.
