# Differentiable research backtest

Research tooling only. The book is the delay-1 close-to-close weight identity
used by `hedge_lab.directional` and `research.net_replay._weights`. It is not
`backtest.engine` and not the share/cash ledger. Nothing here submits orders
or writes a research-catalog score.

JAX is an optional extra (`uv sync --extra jax`). The NumPy core imports
without it. CPU is enough. `JAX_PLATFORMS=cpu` is what CI sets.

## Pipeline

Prices `P` are strictly positive. Simple returns `R_0 = 0` and
`R_t = P_t / P_{t-1} - 1`. A weight `w_t` is a function of prices through
`t - delay` only (`delay` is an integer ≥ 1). It earns `R_t`.

```
net_t = w_t · R_t − c ‖Δw_t‖_1 − y Σ_i σ_{i,t} |Δw_{i,t}|^{1.5}
```

`c = one_way_cost + (commission_bps + half_spread_bps) / 1e4`. Turnover on
bar 0 is 0. `σ` is a causal sample standard deviation (ddof = 1). The impact
term is the weight-space planning cost, not dollar square-root impact. It is
off when `impact_y = 0`.

NAV is `initial * cumprod(1 + net)` with `initial = 1`. Terminal P&L is the
last NAV minus `initial`. The sample ratio is `metrics.returns.sharpe_ratio`
(ddof = 1, annualized by `periods_per_year`, default 252). Drawdown is
`metrics.returns.max_drawdown` (≤ 0, running peak includes the starting 1).

`delay` and `rebalance_every` stay integers. They are causal controls, not
relaxed parameters. `periods_per_year` is a unit convention.

## Strategies

| Name | Hard book |
|---|---|
| `tsmom` | sign of the skipped lookback sum, size `target_vol / (σ √252)`, long-only clip, gross cap only if gross exceeds `max_gross` |
| `topk` | equal weight on the top-k formation winners (`sma` and crash filter off) |
| `antonacci` | dual momentum on two columns, cash when both are non-positive |
| `risk_parity` | inverse-vol, lookback clamped to ≥ 8, gross 1 unless `max_gross < 1` |
| `momentum`, `reversal` | `net_replay._weights` rank book. Count is `int(N * fraction)` (truncation) |
| `equal_weight` | constant size from the delay bar |

The rebalance clock holds the last target between scheduled bars. A no-trade
band `b` applies the proximal map
`held + sign(Δ) max(|Δ| − b, 0)`. `b = 0` is the identity.

## Three JAX modes

* **hard.** Same decisions as the NumPy core, float64. No relaxation.
  Absolute tolerance `HARD_PARITY_ATOL = 1e-8` against NumPy.
* **smooth.** Softplus absolute value, soft-threshold band, tanh in place of
  sign, soft top-k, softmax dual-momentum, soft gross cap, soft warmup gates.
  Exact reverse-mode gradients of this relaxation.
* **ste.** Forward pass uses the hard weights. The backward pass uses the
  smooth surrogate (`stop_gradient(hard − smooth)`). Discrete trades
  (sign, top-k, dual momentum) get a straight-through estimator. Costs stay
  smooth, so the STE net series is not the hard net series.

Pairwise rank comparisons use an extra factor `_RANK_SCALE = 1e3` (per unit
of return) so a sharpened `beta` resolves a few basis points of formation
return instead of freezing a fractional rank. Inclusion still sharpens with
`beta` alone. The soft-threshold is rescaled so a zero band is exactly the
identity at every finite `beta`.

`PARITY_BETA = 128` is the sharpened check. On the fixtures in
`tests/unit/diffbacktest/test_jax_parity.py` (N ≤ 4, `one_way_cost = 1e-3`,
band 0) the max absolute gap between smooth and hard net returns is below
`SMOOTH_NET_ATOL = 5e-4`, and that gap is smaller than the gap at `beta = 8`.
The leading leftover at exact-zero turnover is the soft-abs gap
`one_way_cost * N * 2 * log(2) / beta` per bar. `GRAD_BETA = 8` is the
finite-difference check, where a saturated tanh would otherwise have a
numerically zero slope.

## Gradients

`objective_gradients` differentiates terminal P&L, the sum of net returns,
the sample ratio, and drawdown with respect to every active parameter
(including cost coefficients) and with respect to the price path.

Smooth drawdown replaces the running peak with `soft_max` and the minimum
with `−logsumexp(−β dd) / β`. Hard drawdown is the exact minimum. Hard sign
and hard absolute value are not differentiable at the kinks; the hard-mode
gradient is a subgradient (zero on flat discrete regions).

One identity that does not depend on the relaxation: in hard mode the sum of
net returns has derivative `−Σ turnover` with respect to `one_way_cost`, and
the same quantity over `1e4` with respect to `commission_bps` and
`half_spread_bps`, because weights do not depend on those coefficients.

## Flat objective and walk-forward

The regularizer, on the smooth book, is

```
J(θ) = sample_ratio(θ) − λ ‖∇_θ sample_ratio‖₂
```

The norm is over every active parameter. Adam (learning rate 0.05, z clipped
to [−6, 6]) steps only the free parameters, mapped from an unconstrained z
by a sigmoid into `BOXES`. The best iterate is kept. Free parameters are the
same names the grid is allowed to change, so the comparison is not a larger
search.

The grid maximizes the hard-book sample ratio on `net[warmup:]`. Tie break is
grid order. A grid with no finite score returns the input parameters and
status `degenerate`.

Walk-forward is expanding. At cursor `train, train+test, ...` both selectors
see only `prices[:cursor]`. Out-of-sample scores use the hard book on the
causal prefix `prices[:cursor+test]`, sliced to `[cursor, cursor+test)`.
Pooled diagnostics concatenate those slices and compound across the fold
boundary. Neither selector is chosen after seeing the test fold. There is no
claim that the flat objective wins out of sample.

## Adversarial radius

`σ` is the full-sample per-name return standard deviation, floored at `1e-4`,
and it is frozen. A perturbation `ε` is in those volatility units:

```
R' = clip(R + ε ⊙ σ, −0.8, 2),  R'_0 = 0,  P_0 fixed
```

so `1 + R' ≥ 0.2`. Projected gradient minimizes the smooth terminal P&L with
an L∞ step inside `[−ρ, ρ]`, then the hard NumPy book certifies the path.
Binary search lowers the upper bound only when that certification says
terminal P&L ≤ 0. The reported radius is that upper bound. If the unperturbed
book is already non-positive, the radius is 0. If the attack never certifies
a hit inside `ρ_max`, the radius is `None`. `None` means the attack failed,
not that the book is robust.

## Small case

`run_small_case` uses the constants in `quant_fund.diffbacktest.spec`
(`SMALL_CASE_*`). The path is i.i.d. Gaussian simple returns, seed 0,
T = 96, N = 3, two folds of 64/16 and 80/16. Those constants were fixed
before the out-of-sample numbers existed. Do not retune them to flatter the
table below. Two folds have no power to claim an edge.

```bash
JAX_PLATFORMS=cpu uv run python -c "from quant_fund.diffbacktest.study import run_small_case; import json; print(json.dumps(run_small_case(), default=str, indent=2))"
```

### SYNTHETIC diagnostic measurements

These are mechanics checks on the fixed synthetic path. They are not
research-catalog scores and not a live P&L claim. The sample ratio below is
`metrics.returns.sharpe_ratio` on the concatenated hard-book out-of-sample
net returns.

Measured by `run_small_case()` on this branch (seed 0, β = 8, λ = 0.1,
12 Adam steps, ρ_max = 1.5, 8 PGD steps, 6 bisections). Pooled rows
concatenate the two out-of-sample slices (32 bars) and compound across the
fold boundary. Sample ratio in the table is the annualized ratio from
`metrics.returns` (the same function the hard book already delegates to).
The radius is for the default parameters on the full
96-bar path, not for the walk-forward selection.

| Strategy | Selector | Sample ratio | Terminal P&L | Sum of net | Max drawdown |
|---|---|---:|---:|---:|---:|
| tsmom | grid | 0.230522 | 0.001757 | 0.002058 | −0.017644 |
| tsmom | flat | 0.228804 | 0.003126 | 0.004755 | −0.040845 |
| momentum | grid | −3.865098 | −0.048564 | −0.049126 | −0.054745 |
| momentum | flat | −0.685815 | −0.007475 | −0.007094 | −0.026178 |
| risk_parity | grid | −3.161225 | −0.022412 | −0.022465 | −0.028441 |
| risk_parity | flat | −2.859789 | −0.039221 | −0.039263 | −0.048653 |

Flat is not uniformly better on this path. TSMOM is a toss-up. Momentum's
flat book loses less. Risk parity's flat book loses more. In sample, the
hard sample ratio did not rise: the regularizer improved because the gradient
norm fell. On the full path the smooth sample-ratio gradient norm is about
1535 (tsmom), 1794 (momentum), and 1596 (risk parity). The largest component
is the derivative with respect to `one_way_cost` (weights do not depend on
that coefficient, and the mean net return is near zero, so a cost shift moves
the ratio a lot). λ = 0.1 therefore dominates J. That is what this fixed
setting does. It was not changed after seeing the table.

Adversarial radius (volatility units, attack-found upper bound):

| Strategy | Status | Radius | Unperturbed terminal P&L | Attacked terminal P&L |
|---|---|---:|---:|---:|
| tsmom | certified | 0.0234375 | 0.021499 | −0.003520 |
| momentum | certified | 0.046875 | 0.012693 | −0.004241 |
| risk_parity | already non-positive | 0 | −0.038209 | −0.038209 |

Risk parity's radius is 0 because the default book already loses on this
path. TSMOM's reported radius is the smallest of the six bisection probes;
every probe certified a destroying path, so the true radius may be smaller.


## How to run

```bash
uv sync --extra jax --group dev
make diffbacktest
```

CI job `diffbacktest` installs the `jax` extra, sets `JAX_PLATFORMS=cpu`, and
runs `tests/unit/diffbacktest`. The heavy small case is marked `slow`. The
main test job installs every extra, so the same tests run there too. The
container image does not install JAX.

## Limitations

`quant_fund.diffbacktest.spec.limitations` is the contract text:

- Synthetic paths are a correctness check, not market evidence.
- Not a live P&L claim and not a broker, order router, or execution simulator.
- Not `backtest.engine` and not the share/cash ledger.
- Smooth gradients are exact for the relaxation. Straight-through gradients
  are surrogates. Hard sign, absolute value, and top-k are not differentiable
  at the kinks.
- The adversarial radius is an attack-found upper bound, not a certificate.
- The CI path cannot support an edge claim.
- `delay` and `rebalance_every` stay integers.
