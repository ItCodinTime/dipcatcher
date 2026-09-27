# Prospective distribution forecast journal

Status: **protocol and software only**. No future forecasts or sealed labels have
been collected under this journal. Historical SOTA panels, including the
inspected Binance panels in `EVAL_REPORT_SOTA.md`, cannot become prospective
observations by copying them here. A valid local chain does not establish an
independent timestamp, a canonical external model run, or model promotion.

## Frozen question and success rule

At each declared UTC daily origin, submit one-step close-to-close return
distributions for every fixed asset. `event_time` denotes the **completed close
boundary**, not the bar open. `available_time` and `ingested_time` for every
history close must be at or before the forecast observation time. The future
label's availability must be after that forecast observation time.

The three fixed columns are a new candidate, the in-house `dip_fhs` baseline,
and one named published model. `dip_fhs` uses the existing GJR-GARCH(1,1,1)
normal fit to the last **750** historical returns, with standardized residuals
rescaled by next-step volatility. The 750-return default matches the
retrospective `scripts/sota_eval_kronos.py` baseline; the journal recomputes it
from the forecast packet and seals the exact empirical sample. The candidate
and published-model samples are submitted with the model, artifact-bundle, and
adapter hashes frozen in the protocol. All three distributions use the same
`crps_empirical` proper score. A target day's score is the equal-weight mean
across fixed assets. An asset is never treated as an independent date.

The **single primary test** compares the candidate's daily CRPS against both
fixed comparators on the same dates. Define each signed reduction as comparator
CRPS minus candidate CRPS. At exactly the preregistered number of fully paired
dates, use a one-sided normal lower confidence bound for each mean reduction,
with Bartlett HAC variance using the frozen lag count. Split the total alpha
equally across the two fixed contrasts (Bonferroni). The predeclared score rule
passes only if **both** lower bounds exceed zero. The journal emits no effects,
intervals, or decision before that final date. No adaptive extension, new
comparator, favorable subset, or additional look is permitted under this run.

Before `prepare`, provide a positive planning CRPS effect, a conservative
long-run standard deviation for the **larger of the two date-level paired
differences**, a digest of the prior planning source, alpha, power, and fixed
HAC lags. The declared minimum paired date count must satisfy

`N >= ceil(((z_(1-alpha/2) + z_power) * planning_long_run_sd / planning_effect)^2)`.

These inputs have to come from a documented prior source; the program checks
their types and formula, but cannot judge whether the historical variance is
credible. No actual planning variance for this exact three-model prospective
comparison is supplied here, so **a real run cannot responsibly freeze yet**.
The first missed origin, asset, model distribution, failed FHS fit, or late label
requires an `interrupt` receipt. An interruption permanently blocks the run;
it cannot quietly append replacement dates to reach the planned N. Verification
reports attempted dates, fully paired dates, and coverage.
For an active run, it also compares the wall clock with the frozen schedule and
reports due dates that have no recorded attempt. Merely omitting an
`interrupt` call cannot make an overdue run appear fully covered.

## Prepare, forecast, settle, verify

Create a JSON protocol with these exact keys (the values below are **field
descriptions**, not a ready-to-freeze research plan):

| Field | Required content |
|---|---|
| `schema` | `prospective_sota_v1` |
| `source_id`, `asset_ids` | Fixed source and sorted, unique, nonempty assets |
| `bar_interval_seconds`, `history_returns` | Exactly `86400`, `750` in this version |
| `first_origin_time` | Future daily close boundary, offset-aware ISO timestamp |
| `candidate` | `model_id`, `artifact_sha256`, `adapter_sha256` |
| `published` | Same three fields plus `canonical_reference` identifying the publication and exact model variant |
| `minimum_paired_origins` | Fixed fully paired target dates; must meet the power formula |
| `planning_effect_crps`, `planning_long_run_sd_crps`, `planning_source_sha256` | Preregistered planning values and prior-source digest |
| `alpha`, `power`, `hac_lags` | Total family alpha, planning power, final Bartlett HAC lag count |
| `coverage_rule`, `primary_metric` | `all_assets_all_models_or_block`, `equal_weight_asset_crps_by_target_date` |

The artifact digest must cover the full runnable model bundle, including
weights, tokenizer, configuration, and preprocessing where applicable. The
adapter digest must cover the horizon/return conversion and sample-generation
code. Those are obligations on the submitting model owner; the journal cannot
prove that the declared bundle actually generated submitted samples.

Run from the repository environment:

```bash
python -m quant_fund.research.prospective_sota commitment protocol.json
```

The output contains `commitment_sha256`, protocol and source digests, and
runtime versions. Obtain an independent timestamp for that commitment **before
the first origin**. Supply `anchor.json` with exactly `recorded_at`, `issuer`,
`reference`, and `commitment_sha256`. The local verifier checks the digest and
the declared time relationship; it does **not** authenticate the issuer or
retrieve the reference.

```bash
python -m quant_fund.research.prospective_sota prepare run/ protocol.json anchor.json
python -m quant_fund.research.prospective_sota forecast run/ forecast.json
python -m quant_fund.research.prospective_sota settle run/ label.json
python -m quant_fund.research.prospective_sota verify run/
```

`prepare` creates a new run directory before the fixed first origin. Forecast
packets have `origin_time` and a sorted `assets` list. Each asset has
`asset_id`, `bars` (751 consecutive completed daily closes, oldest first), and
`candidate`/`published` submissions. A close bar has `event_time`,
`available_time`, `ingested_time`, `close`, and `source_id`. Each submission has
`model_id`, `artifact_sha256`, `adapter_sha256`, and an array of 20–512 finite
return samples. All assets and model submissions must be present. The CLI uses
the current UTC clock for the forecast receipt and requires it to precede the
next target close boundary.

Settlement packets have `target_time` and sorted `assets`, each with
`asset_id` and one close `bar` using the same bar fields. Settlement occurs only
after the pending target bar is available and ingested. It recomputes the
one-step return and every CRPS. Any missing target bar or a failed scheduled
forecast must be recorded with `interrupt run/ "reason"` (optionally
`--packet-sha256 ...`). The interruption is terminal. A later forecast must
otherwise use the previous target as its exact origin; it cannot skip a day.
Its first 750 closes must exactly match the last 750 closes of the prior
forecast, and its final close must be the prior sealed label. A source revision
therefore blocks this run and requires a separately declared new protocol.

Every write uses a new numbered event file and checks the full existing local
chain first. `verify` replays the FHS fit from the stored pre-target bars,
recomputes scores from the stored later label, and compares all derived fields
and state. The manifest binds the protocol, journal/FHS/scoring source files,
and Python/arch/NumPy/SciPy versions. Preserve the exact environment and
source tree to verify old receipts. An operator with write access can rewrite
and reseal the entire local directory, so independently archive the forecast
receipt hashes before labels arrive, then verify them against that archive.

`promotion_state` remains `pending_external_attestation_and_replication` even
if the proper-score rule eventually passes. The `external_timestamp_authenticity_verified`,
`external_model_inference_verified`, and `forward_evidence_independently_attested`
flags remain false until a separate verifier authenticates the external archive
and the model inference path. Trading costs, latency, capacity, and net results
would require a separately frozen secondary study; this journal makes no
economic or live-trading claim.
