# Scenario Monte Carlo (`quant_fund.mc_engine`)

Research simulation of strategy and portfolio paths. The engine estimates
tail risk under a scenario model. It does not submit orders, and a run is
not market evidence.

Built-in generators are synthetic. A plug-in generator is only as real as
the data source it declares. The report sets `research_only: true`,
`live_pnl_claim: false`, and `market_evidence: false` on every run.

## What a run estimates

Loss is the drop of a unit numeraire: `1 - terminal level`. Positive means
the path finished lower. The report does not carry Sharpe, Sortino, Calmar,
P&L, or NAV keys.

- Expected shortfall at 97.5% and 99%, with a batch-means interval when the
  design is independent Monte Carlo and the losses were retained.
- The distribution of maximum drawdown across paths.
- Probability of ruin (wealth at or below `ruin_level`), with a Wilson
  interval on unweighted counts.
- Time to recovery from the maximum-drawdown trough. Paths that never get
  back to the peak that was in force are censored and are left out of the
  conditional mean.
- Peaks-over-threshold generalized Pareto fit, with diagnostics that fail
  closed. The Kolmogorov-Smirnov p-value is reported and is not a pass/fail
  gate.

Intervals are Monte Carlo error under the scenario design. They are not
sampling error of a historical market estimate. Quasi-Monte Carlo does not
get an independence-based interval.

## Scenario generator interface

Bootstrap, copula, and HMM engines plug in here. Implement:

```text
generate(path_indices, *, shocks, seed) -> ScenarioBatch
```

`ScenarioBatch.returns` is a float64 array of shape `(n_paths, n_steps)` of
simple portfolio returns, in the same order as `path_indices`.

- `accepts_external_shocks = True`: consume `shocks`. Do not draw again.
  Antithetic variates, scrambled Sobol, and importance sampling are applied
  by the engine before your generator sees the shocks.
- `accepts_external_shocks = False`: only `shock_mode="crude"` is legal. Draw
  with `philox_normals` or `philox_uniforms`, keyed by `path_indices` and
  `seed`, and use `stream_id >= USER_STREAM_ID_MIN` (16). Stream ids 1–15
  belong to the engine.

`spec_dict()` must be JSON-stable. It is part of the checkpoint fingerprint.
The object must be picklable if you use the process or Ray backend.

Built-ins:

- `GbmPortfolioGenerator` — static long-only weights on correlated GBMs.
- `VolTargetStrategyGenerator` — one asset, position from trailing realized
  volatility, causal in the past returns only.
- `IdentityShockGenerator` — reference map from factor-0 shocks to returns,
  for tests.

`python -m quant_fund.mc_engine help-interface` prints the same contract.

## Reproducibility

NumPy's Philox generator is counter-based (Salmon et al., 2011). For each
path the engine writes:

| counter word | contents |
| --- | --- |
| 0 | starts at 0 and advances as that path is drawn |
| 1 | path index, or the antithetic base index |
| 2 | stream id |
| 3 | 0 |

Normals are Box-Muller on `(raw + 1/2) / 2^64`. That spends a fixed number
of words per normal. `Generator.standard_normal` uses ziggurat rejection and
a data-dependent word count, so it is not used.

Paths are split into chunks of a fixed `chunk_size`. Workers may finish in
any order. The reduction always folds chunks in chunk-id order. Same seed,
chunk size, generator, and platform arithmetic produce the same fingerprint
at 1 worker and at N workers. A test locks that down, including a 100,000-path
run marked `slow`.

Changing `chunk_size` changes the association of floating-point merges and
the t-digest compression schedule. The fingerprint is not claimed to be
invariant to chunk size.

## Variance reduction

Techniques are measured separately. Factors are not multiplied. A factor
below 1 is reported below 1.

| technique | what the factor is |
| --- | --- |
| Antithetic pairs | variance of an i.i.d. mean over the variance of the mean of pair averages, same path count. Exact cancellation is reported as infinite, not as a made-up finite number. |
| Control variate | `b` is fit on every `pilot_every`-th path. The factor is the ratio of evaluation-sample variances. Pilot paths are not reused. |
| Importance sampling | mean shift on factor 0 only. Default shift is `-1/sqrt(n_steps)` so the squared norm is 1. The factor compares `mean(weight * loss)` with a same-seed unshifted sample. That crude sample is not part of the headline distribution. If the effective sample size is under 10% of the path count, the ratio is shown and `claim` is false. |
| Scrambled Sobol | one sequence has no internal variance, so the factor is null until `n_scrambles >= 2`. The factor then compares one scramble of `n` paths with a crude sample of `n` paths. Extra scrambles are the measurement instrument and are not pooled into the tails. |

Sobol points are balanced at powers of two. Other lengths still run; the
report says `balance_power_of_two: false` instead of pretending otherwise.

## Aggregation

Each chunk streams its path into a Welford accumulator and a merging
t-digest, then (in exact mode) keeps the scalar outcomes. The path tensor
itself is not retained past the chunk.

- Welford / Chan moments merge by the parallel formula, left to right in
  chunk-id order.
- The t-digest merge concatenates centroids in that same order and
  compresses. Quantiles are the centroid step function and are labeled
  approximate. In exact mode the headline quantile is empirical and the
  absolute sketch error is reported beside it.
- P² (Jain and Chlamtac, 1985) is a single stream. It has no merge. Exact
  mode streams retained losses in path order. Sketch mode leaves P² null
  rather than gluing per-chunk markers together.

`memory_mode="sketch"` drops per-path arrays. Expected shortfall is then the
t-digest figure, approximate, and without an interval. Peaks-over-threshold
in sketch mode needs an absolute `--evt-threshold` so every exceedance can
be retained. A truncated exceedance buffer refuses the fit.

## Extremes

The GPD fit uses `scipy.stats.genpareto` with location fixed at 0.
`diagnostics_ok` requires at least 50 exceedances, finite positive scale,
shape below 1 (so expected shortfall is finite), and mean-excess relative
error at most 0.25. Shape movement across the 0.90 / 0.95 / 0.97 thresholds
is a warning, not a silent pass. Standard errors come from a numerical
Hessian and assume i.i.d. excesses; that assumption is flagged off for
antithetic and Sobol designs. Importance sampling does not get a GPD fit,
because the engine does not claim a weighted peaks-over-threshold estimator.

## Checkpoint, progress, CLI

```bash
python -m quant_fund.mc_engine run \
  --paths 100000 --steps 252 --workers 4 --chunk-size 4096 \
  --checkpoint data/mc-engine/run1 --seed 1

python -m quant_fund.mc_engine resume --checkpoint data/mc-engine/run1 --workers 4

python -m quant_fund.mc_engine bench --paths 50000 --steps 64 --workers 1,2,4 --repeats 2
```

`mc-engine` is the same CLI. Progress lines go to stderr. The report is
JSON on stdout. `--no-progress` keeps stdout as a single document.

A checkpoint is a directory of chunk archives plus `manifest.json`. The
manifest fingerprint covers the generator spec and every numeric input that
changes the paths. Worker count and backend are not part of it. Resume
refuses a different generator. An incomplete run returns `status:
"incomplete"` and does not invent tail numbers from the finished prefix.

Ray is optional. `backend="ray"` imports `ray` and raises `ImportError` with
a plain message if it is not installed. This environment does not vendor Ray,
so the Ray path is not part of the measured benchmark.

## Scaling benchmark

`scaling_benchmark` times real calls. `paths_per_s_using_min_elapsed` is
`n_paths / min(elapsed_s)` across repeats. `measured_speedup_vs_first_row`
is the ratio of those minimum elapsed times and is allowed to be below 1.
The limitation string in the JSON says this is wall time on the host,
including pool startup, and not a linear-scaling claim.

With `backend="process"`, the one-worker row still starts a process pool, so
the comparison is pool-versus-pool. `backend="serial"` never starts a pool.
The pool uses the `spawn` start method. Run it from an importable module
(the CLI, pytest, or a file). `python -c` and stdin scripts cannot re-import
`__main__`, and the pool fails closed instead of falling back to `fork`.

## Tests

`tests/unit/mc_engine/`. The 100,000-path worker-count identity test is
marked `slow`. `make mc-engine-smoke` runs the rest plus a 1,500-path CLI
call. CI job `mc-engine-smoke` does the same with two processes and
antithetic shocks.
