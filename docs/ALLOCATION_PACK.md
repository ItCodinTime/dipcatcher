# Allocation pack — causal weight engines and honest evaluation

`quant_fund.research.allocation` is a research-only portfolio-construction
pack. Engines fit weights on a **trailing returns window only** (strictly
causal — no lookahead), pass through a shared constraint layer, and are
scored on **risk-targeting quality**: realized-vol-vs-target tracking
error, weight turnover/stability, and equal-risk-contribution residuals.
No Sharpe, P&L, or NAV is computed anywhere; the honesty contract is
enforced at receipt time by the catalog's forbidden-key check.

## Layout

| Module | Contents |
|---|---|
| `constraints.py` | `AllocationConstraints` (long-only, `leverage_cap`, `min_weight`, `max_weight`), `apply_constraints`, `weights_satisfy` audit, `validate_covariance`, `DegenerateCovarianceError` |
| `engines.py` | `inverse_volatility_weights`, `risk_parity_weights`, `kelly_weights`, `volatility_target_scale`, `fit_weights` (causal driver), `estimate_covariance`, `portfolio_vol`, `risk_contributions` |
| `evaluation.py` | `run_walk_forward`, `compare_engines`, `AllocationEvaluation` |
| `receipt.py` | `build_receipt`, `write_receipt`, `verify_allocation_receipt` (`allocation_evaluation.v1` JSON, hash-bound) |

## Engines

All engines take a trailing window `returns[t-window:t]` (via
`fit_weights`) or a validated covariance directly (layer-1 functions).

| Engine | Definition |
|---|---|
| `inverse_volatility` | `w_i = (1/σ_i) / Σ_j (1/σ_j)` — equalizes marginal vol contribution |
| `risk_parity` | Maillard–Roncalli–Teiletche ERC: `w_i(Σw)_i = σ_p²/n`, solved by damped Newton on the convex log-barrier problem `min 0.5 y'Σy − cΣlog y_i` (Roncalli formulation); convergence measured on the relative RC spread, with a best-iterate fallback at a documented relaxed bound for float-precision limit cycles |
| `kelly` | Continuous-time Kelly `w* = f·Σ⁻¹μ` (Thorp 2006) with `fraction` in `(0,1]`; near-singular covariance fails closed |
| `vol_target` | Scales `base_engine` weights by `target_vol / sqrt(w'Σw)` (per-period units); the gross cap bounds realized leverage |

## Constraints

`apply_constraints` is feasible-by-construction, applied in order:
long-only clip → per-asset `max_weight` clip → gross rescale down to
`leverage_cap` (never up) → `min_weight` floor (drops positions strictly
below the floor; never lifts — lifting could violate the cap). Final books
satisfy `|w_i| ≤ max_weight`, `Σ|w_i| ≤ leverage_cap`, and every nonzero
position `|w_i| ≥ min_weight`. Degenerate covariance — non-square,
non-finite, asymmetric, non-PSD, or zero-variance — raises
`DegenerateCovarianceError`; nothing is repaired or jittered.

## Evaluation metrics

`run_walk_forward(returns, engine, window=…, step=…, horizon=…)` — per
decision `t`: fit on `[t-window, t)`, measure on the forward block
`[t, t+horizon)`:

- `predicted_vol` — ex-ante `sqrt(w'Σw)` from the fitting window;
- `realized_vol` — RMS of forward per-period portfolio returns (valid at
  horizon 1);
- `vol_rmse` / `vol_mae` / `vol_bias` — realized vs predicted calibration;
- `target_vol_rmse` / `target_vol_mae` / `target_vol_bias` — realized vs
  target (vol-target overlay only);
- `initial_turnover`, `mean_turnover`, `max_turnover`,
  `mean_turnover_one_way` — `Σ|Δw|` weight stability;
- `erc_residual_mean` / `erc_residual_max` — max `|RC share − 1/n|`;
- `constraint_violations` — post-hoc audit count (expected 0).

## Receipts

`build_receipt(evaluation, synthetic=…)` produces an
`allocation_evaluation.v1` dict (`claim: research_only`, explicit
`synthetic` flag, metrics block, `weights_sha256`, `payload_sha256` over
the canonical payload). `write_receipt` serializes atomically;
`verify_allocation_receipt` re-derives the hash. This is **additive** to
the sealed research-receipt machinery — it never touches `receipts/` or
`verify-research` paths.

## Commands

```bash
# gates (scoped)
uv run --no-sync ruff check src/quant_fund/research/allocation tests/unit/research/allocation tests/property/test_allocation_properties.py
uv run --no-sync ruff format --check src/quant_fund/research/allocation tests/unit/research/allocation tests/property/test_allocation_properties.py
uv run --no-sync mypy src/quant_fund/research/allocation
PYTHONPATH=src uv run --no-sync pytest tests/unit/research/allocation tests/property/test_allocation_properties.py -q
```

`PYTHONPATH=src` is needed inside a worktree because the shared `.venv`
editable-install points at the main checkout's `src/`.

Minimal example:

```python
import numpy as np
from quant_fund.research.allocation import (
    AllocationConstraints, build_receipt, run_walk_forward,
)

returns = ...  # T x N simple per-period returns
ev = run_walk_forward(
    returns, "vol_target", window=60, step=20, target_vol=0.02,
    constraints=AllocationConstraints(leverage_cap=2.0),
)
print(ev.metrics)                      # calibration/stability stats only
receipt = build_receipt(ev, synthetic=True, label="example")
```

## Relationship to `quant_fund.portfolio.allocators`

`quant_fund.portfolio.allocators` holds single-shot covariance→weight
functions (inv-vol, ERC via SLSQP, Kelly, CVaR, HRP, Black–Litterman, vol
scaling) used elsewhere in the harness. This pack adds what those lack for
research evaluation: a **trailing-window causal contract**, a unified
constraint layer with a feasibility audit, an iterative ERC solver with an
explicit convergence/residual contract, walk-forward calibration metrics,
and hash-bound receipts — deliberately separate modules so nothing in the
existing portfolio paths changes.
