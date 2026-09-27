# Robustness certification

Research diagnostic for a strategy that maps a finite array to a discrete
decision. Nothing in this package routes orders, talks to a broker, or
turns a simulated path into a live-performance claim.

The implementation lives in `quant_fund.robustness`. A scorecard can be
stamped onto a research notebook as an optional extension. Sealed files
under `receipts/` are not rewritten.

## What is proven, and what is not

| Object | Status | Where it does not apply |
|---|---|---|
| L2 radius of the Gaussian-smoothed decision | Proven for the population probability (Cohen, Rosenfeld, Kolter, ICML 2019, Theorem 1). A Monte Carlo run is a high-probability lower bound via a Clopper–Pearson interval (Clopper and Pearson, Biometrika, 1934), not a deterministic proof. | Missing bars, stale prints, timing jitter, cost shocks, and distribution shift. The ball is in the coordinates that were noised. |
| L-infinity radius derived from that L2 ball | Corollary of `\|\|z\|\|_2 <= sqrt(d) \|\|z\|\|_inf`. Not a tight infinity-norm certificate. | Same exclusions as the L2 certificate. |
| Minimal L2 perturbation of a linear margin | Proven: `\|w·x+b\| / \|\|w\|\|`. Equals the population smoothing radius. | Nonlinear or non-differentiable rules. |
| One-coordinate spike of a linear margin | Proven: `\|margin\| / max_i \|w_i scale_i\|`. | Rules without that closed form. |
| Black-box search (CMA-ES or TPE) and gradient search (PGD) | Empirical. A checked flip is an upper bound on the minimal perturbation. Failure to flip is not a certificate. | — |
| Worst-case mean inside a Wasserstein ball | Proven for every reference law and every order `p >= 1`: `mean - radius`, attained by a translation. | Not a confidence interval for an unknown true law. |
| Worst-case mean-to-scale ratio | Gelbrich (1990) moment disk. Tight when the reference law is univariate Gaussian. For any other law it is a lower bound, and it is vacuous when the disk minimum is unbounded. | Not a finite-sample guarantee. Plug-in moments are labeled as plug-in. |
| Cost-rate increase that makes one path's net excess non-positive | Algebra on that path's gross sum and turnover. | Not a market result. |
| Missing bars, stale prints, timing jitter | Exact enumeration on one finite array (`exact_on_path`). | Not a theorem about other paths. |

SYNTHETIC toys check these formulas. They are correctness tests, not market evidence.

Cohen's radius equals the linear distance in exact arithmetic. In float64, `norm.ppf(norm.cdf(z))` drifts from `z` once the standardized margin is large. The property test checks that identity to `1e-8` for standardized margins at most `5.5`. A unit test pins the larger tail, where the gap is inversion error.

The mean-to-scale ratio is the population form of a risk-adjusted return. The lab does not stamp that headline name into research receipts. The field is `worst_case_ratio`, with an explicit proof status.

## Threat models

- **Volatility-scaled path.** A vector `eta` with bounded `l2` or `linf` norm is applied as `sample + scale * eta`. On prices, the same shock is added to log increments and the path is rebuilt. `scale` defaults to ones.
- **Missing bars.** Selected coordinates are set to zero (recorded as no move). The fill is a modeling choice.
- **Stale prints.** A window is replaced by the previous value. The first bar has no previous print and is filled with zero.
- **Spikes.** One coordinate is shifted by `magnitude * scale`.
- **Timing jitter.** The window is rolled by an integer number of bars. The roll is circular so the threat stays inside the window. It is not a causal filter.
- **Cost shock.** An additive increase of the cost rate on `gross - cost * turnover` for the strategy's position path.
- **Regime shift.** A Wasserstein ball around the outcome law. This is not a path edit.

## Attacks

`black_box_attack` binary-searches the radius. Inside each ball, CMA-ES (Hansen, arXiv:1604.00772, 2016) or Optuna TPE (Akiba et al., KDD 2019) proposes a direction. When the strategy exposes a margin, the search minimizes that score. Otherwise it sees only the discrete decision. Every reported success is rechecked by calling `decision`.

`gradient_attack` asks a `GradientBackend` for the margin and its gradient. A differentiable backtester, including one written in JAX, implements that protocol and is registered with `register_gradient_backend`. This package does not import JAX. If the strategy itself implements `margin_and_grad`, certification uses it. Finite differences are opt-in (`gradient="finite_difference"`) and are labeled numerical. With no backend and no margin, the gradient attack is `unavailable`.

L2 gradient steps are Newton steps on the linearized margin, exact for a linear margin. L-infinity steps are the signed updates of Madry, Makelov, Schmidt, Tsipras, and Vladu (ICLR 2018).

## Randomized smoothing

For isotropic Gaussian noise of standard deviation `sigma`, if the top class has probability at least `p_lower` and every other class has probability at most `p_runner_upper`,

```
R = sigma / 2 * (Phi^{-1}(p_lower) - Phi^{-1}(p_runner_upper))
```

The binary case `p_runner_upper = 1 - p_lower` is `R = sigma * Phi^{-1}(p_lower)`. A top-class lower bound of at most one half abstains (`R = 0`).

A linear margin has the closed form `P(positive) = Phi(margin / (sigma \|\|w\|\|))`. Substituting it recovers `R = \|margin\| / \|\|w\|\|`.

The Monte Carlo certificate uses a Bonferroni split of `alpha` between a lower Clopper–Pearson bound on the top class and an upper bound on the runner-up. The guarantee is over the draws: with probability at least `1 - alpha` the radius is valid. It can be too large with probability up to `alpha`.

## Distributional bounds

Worst-case mean, ground cost `|u - v|`, any Wasserstein order `p >= 1`:

```
inf { E_Q[X] : W_p(Q, P) <= radius } = E_P[X] - radius
```

The translation by `radius` attains it. The mean cannot move by more than `W_1`, and `W_1 <= W_p`. This is the linear case of Mohajerin Esfahani and Kuhn (Mathematical Programming, 2018) and the Kantorovich–Rubinstein theorem.

An L-Lipschitz score moves by at most `radius * L` over a W1 ball of that radius. The caller supplies `L`. The certifier does not invent one.

The mean-to-scale ratio minimizes `μ' / σ'` over the Gelbrich disk `(Δμ)^2 + (Δσ)^2 <= radius^2`. For `radius` strictly inside the disk and not equal to the reference scale,

```
t = (μ σ - radius * sqrt(μ^2 + σ^2 - radius^2)) / (σ^2 - radius^2)
```

Boundary cases are in the docstring of `gelbrich_worst_case_ratio`. When the disk can push the scale to zero while the mean stays negative, a Gaussian reference has an unbounded ratio. For a non-Gaussian reference the same disk is only an outer relaxation, so an unbounded disk minimum is reported as vacuous rather than as a fact about that law.

## Receipt extension

Parent `schema_version` stays 1. Notebooks that omit `robustness` and `extensions_schema_version` still verify. A stamped notebook sets `extensions_schema_version` to 1 and adds a `robustness` object with the same schema version, `claim: robustness_diagnostic_only`, `research_only: true`, and `live_trading_claim: false`.

`migrate_robustness_view` returns a copy. If the extension is absent it attaches `metrics_status: legacy_uncomputed` and does not recompute anything. Parent schema version is unchanged, including a parent schema of 2, so this stamp composes with an overfitting block that uses parent schema 2. Sealed inputs are not mutated. `write_stamped_notebook` refuses any path under a `receipts` directory.

`dipcatcher verify-research` checks the extension when it is present and ignores it when it is absent.

## Smoke

```bash
uv run pytest -q tests/unit/robustness tests/property/test_robustness_bounds.py
uv run python -m quant_fund.robustness
```

CI runs the same two commands in the `robustness-smoke` job.
