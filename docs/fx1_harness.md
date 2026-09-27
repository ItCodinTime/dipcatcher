# fx-1 forecast harness

Dipcatcher can host an **external** price/return model named fx-1. This
repository does not implement that model, does not ship weights, and does not
train it. `fx1.forecast` is plumbing: load a checkpoint, build point-in-time
features from an existing market-data adapter, run inference, and score the
forecasts.

`dummy-zero` and `dummy-momentum` are reference baselines for tests and smoke
runs. They are not fx-1.

The command surface is `fx1 infer` and `fx1 backtest`. Neither command places
orders or talks to a broker.

## What fx-1 must implement

Subclass `fx1.forecast.Fx1Model` (a `fx1.forecast.ForecastModel`) **outside**
this repo:

```python
from fx1.forecast import Fx1Model


class YourFx1Model(Fx1Model):
    name = "fx-1"
    version = "fx-1.v0"

    def load(self, checkpoint_path=None, config=None):
        # Read the local checkpoint. config["horizon_bars"] is the label length.
        ...

    def predict(self, features):
        # features is a polars DataFrame. Return the forecast schema below.
        ...
```

Register it in the harness config:

```yaml
model:
  name: fx-1
  entrypoint: your_package.module:YourFx1Model
  checkpoint_path: /secure/local/fx1.pt
  checkpoint_format: auto   # torch loads weights only by default
  version: null             # optional override; artifact version wins when present
```

`fx1 infer` resolves `model.name`. The names `fx-1` and `fx1` fail closed
unless `model.entrypoint` is `module:attr` and that object is an `Fx1Model`.
There is no in-repo fallback that pretends to be the model.

`load` may call `fx1.forecast.load_artifact`. Backends are import-guarded, so
missing optional dependencies do not break import. JSON and ONNX load normally;
torch defaults to `weights_only=True`. Pickle, joblib, and full-module torch
can execute code and are disabled unless the operator sets both
`allow_unsafe_deserialization: true` and `trusted_checkpoint_sha256` to a digest
obtained from a trusted source. The helper checks the digest before loading and
deserializes those same bytes. A digest proves byte identity, not safety of an
unknown source. The loader never downloads a checkpoint.

An external `model.entrypoint` is itself executable Python supplied by the
operator. Custom `load` implementations must enforce their own trust policy;
the harness checks the configured digest before calling them and rejects a
checkpoint that changes during loading.

Every loaded artifact is stamped with:

- `sha256` of the file bytes
- `format`
- `version`, from a sibling `<path>.version` file, else a `version` field on
  a dict/object payload, else `model.version` in config

## I/O contract

These are the assumptions to confirm before a real fx-1 checkpoint is wired up.

**Features in.** One row per `(event_time, security_id)`. Polars is the
canonical frame; a pandas frame returned by `predict` is accepted at the
boundary.

| Column | Meaning |
|---|---|
| `event_time` | Decision timestamp `t` (UTC) |
| `security_id` | Instrument id |
| `close` | Close at `t`, observable at `t` |
| `available_time` | When the bar became available |
| `ret_1` | `close[t] / close[t-1] - 1` |
| `mom_k` | `close[t] / close[t-k] - 1` for each configured lookback other than 1 |
| `vol_w` | Trailing standard deviation of `ret_1` over `w` bars |

The default pipeline is `fx1.forecast.OhlcvFeaturePipeline`. Replace it with
`features.entrypoint` (`module:attr` exposing `build` and `feature_columns`)
without changing the data adapter.

Label-like columns are rejected (`target_*`, `future_*`, `fwd_*`, `label_*`,
`realized_return`, `next_open`, `planted_signal`, and the same family). The
synthetic adapter's `planted_signal` oracle never enters the feature frame.

**Forecasts out.** One row per `(event_time, security_id, horizon_bars)`.

| Column | Required | Meaning |
|---|---|---|
| `event_time` | yes | Decision time `t` the features were known |
| `security_id` | yes | Instrument id |
| `horizon_bars` | yes | Integer horizon `h >= 1` in bars of the feature panel |
| `predicted_return` | one of these | Simple return forecast for `t → t+h` |
| `predicted_price` | one of these | Price forecast; if return is null, the score is `price / close[t] - 1` |
| `confidence` | no | Null or a value in `[0, 1]` |
| `q_0.1`, `q_0.5`, ... | no | Optional return quantiles. Tau is the suffix and must lie in `(0, 1)`. On a row where every quantile is present, values must be nondecreasing in tau. Quantiles are validated and passed through; they are not scored yet. |

**Target.** The realized outcome used only at evaluation time is the simple
return `close[t+h] / close[t] - 1` on the same (optionally resampled) bar
panel. `t+h` is `h` bars later for that `security_id`, not a wall-clock
offset. Rows without a future bar are dropped from the score.

**Point in time.** A feature at `t` is built from bars with `event_time <= t`
and `available_time <= t`. When the frame has the dipcatcher PIT columns,
filtering goes through `quant_fund.data.point_in_time.filter_available`.
`inference.mode=walk_forward` (the default) calls `predict` once per decision
time on that truncated frame, so the model cannot see a later row.
`inference.mode=batch` computes the same causal features but passes the whole
window in one call and is only valid for a row-local model. Batch mode
refuses panels where any `available_time` is after `event_time`; use
walk-forward for late revisions.

`data.start` / `data.end` bound decision times. History before `start` is
still loaded so lookbacks are defined. `data.end` is also the last bar pulled
from the adapter.

## Data adapters

Bars come from `get_bars(start, end, security_ids)`:

| `data.provider` | Source |
|---|---|
| `parquet` | `quant_fund.data.adapters.ParquetMarketProvider` rooted at `data.root` |
| `synthetic` | `quant_fund.data.adapters.SyntheticMarketProvider` (labeled SYNTHETIC) |
| `entrypoint` | `module:attr` implementing the same `get_bars` signature |

No dataset is special-cased. A new OHLCV source should show up as an adapter
with `get_bars`, then be named from `data.entrypoint`.

`src/quant_fund/data` does not expose a resample function. `data.resample`
(a polars duration such as `1d` or `1h`) aggregates adapter bars inside the
harness. Each bucket is labeled at its **last** print, with open = first,
high = max, low = min, close = last, volume = sum, and `available_time` = max
availability in the bucket. Mixed sources in one bucket are rejected.

## Walk-forward evaluation

`fx1 backtest` does not fit a model. It reuses
`quant_fund.validation.walk_forward.walk_forward` to cut decision times into
expanding or rolling folds with the configured embargo and with purging at
`features.horizon_bars`. Each fold is checked with `assert_no_label_overlap`.
Forecast metrics and the signal diagnostic are computed on **test** times
only. Train rows are not used for fitting.

## How to run

Smoke path (reference model, SYNTHETIC bars, offline):

```bash
fx1 infer --config configs/fx1_harness.example.yaml
fx1 backtest --config configs/fx1_harness.example.yaml
```

`fx1 infer` writes `inference.output_parquet` and a JSON sidecar. The parquet
file also stores that JSON under the Arrow schema metadata key `fx1_harness`.
`fx1 backtest` reads the parquet (`--forecasts` overrides the path), reloads
bars through the same adapter, and prints a JSON report.

An external checkpoint:

```yaml
model:
  name: fx-1
  entrypoint: your_package.module:YourFx1Model
  checkpoint_path: /secure/local/fx1.pt
  checkpoint_format: torch
  # Use these only for a trusted full-module checkpoint. Omit them for a
  # weights-only torch checkpoint.
  allow_unsafe_deserialization: true
  trusted_checkpoint_sha256: <64-character lowercase SHA-256 from a trusted source>
```

## Metrics

Forecast diagnostics are reported per `horizon_bars` on test-fold rows that
have a realized return. These point diagnostics are not a proper-score
promotion receipt for fx-1; a real research comparison needs its declared
target and a proper scoring contract:

| Field | Definition |
|---|---|
| `ic` | Pearson IC via `quant_fund.metrics.scoring.pearson_ic` |
| `rank_ic` | Spearman rank IC via `quant_fund.metrics.scoring.rank_ic` |
| `mean_cross_sectional_ic` | Mean of per-timestamp Pearson IC on dates with at least 3 names |
| `hit_rate` | Fraction of rows where `sign(score) == sign(realized)`, zeros excluded. When at least 20 such rows exist, the rate and `pt_stat` come from `quant_fund.metrics.direction.pesaran_timmermann` |
| `mae`, `rmse` | Error of the score against the realized simple return |
| `folds.ic_stability` | `quant_fund.validation.walk_forward.fold_ic_stability` on per-fold IC |

A constant score (the zero baseline) yields a null IC. That is an honest
undefined correlation, not a zero.

The signal section is a **placeholder**, marked `PLACEHOLDER_NOT_A_STRATEGY`.
It is not a strategy and it does not submit orders.

| Mapping | Signal |
|---|---|
| `sign` | Sign of the score |
| `threshold` | Sign when `abs(score) > signal.threshold`, else 0 |
| `rank` | Cross-sectional average rank at `t`, scaled to `[-1, 1]`. One name maps to 0 |

Weights at each test timestamp are the signals divided by the sum of absolute
signals (0 when every signal is 0). `signal_diagnostics` reports turnover
against the previous selected weight (starting from flat) and the assumed
one-way cost drag in basis points (`cost_bps * mean_turnover`). When
`horizon_bars > 1`, the diagnostic samples every `horizon_bars` test timestamps.
It reports no Sharpe, drawdown, NAV, or P&L. This is not a promotion gate or
evidence of an edge. The report sets
`research_only: true`, `live_pnl_claim: false`, and `orders_submitted: false`.
`data_label` is `SYNTHETIC` when the bar source is synthetic.

## Layout

| Piece | Module |
|---|---|
| Protocol | `fx1.forecast.protocol` |
| Registry | `fx1.forecast.registry` |
| Checkpoint loader | `fx1.forecast.artifacts` |
| Schemas | `fx1.forecast.schema` |
| Features and resample | `fx1.forecast.features` |
| Reference models | `fx1.forecast.dummy` |
| Placeholder signals | `fx1.forecast.signals` |
| Scores | `fx1.forecast.evaluate` |
| Runners | `fx1.forecast.runner` |
| Example config | `configs/fx1_harness.example.yaml` |
