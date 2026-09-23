# ML and RL capabilities

Dipcatcher exposes supervised ML and contextual RL through the same causal
gold-panel workflow. All outputs are research-only; none of these commands
authorizes live execution or creates a live-P&L claim.

## Supervised ranking

```bash
quant train ranking --model auto
quant train ranking --model composite
quant train ranking --model ridge
quant train ranking --model elasticnet
quant train ranking --model neural
quant train ranking --model xgboost
quant train ranking --model lightgbm
quant train ranking --model lambdarank
quant train ranking --model xendcg
quant train ranking --model ensemble
```

The ensemble combines Ridge, Elastic Net, and the deterministic neural MLP.
Tree rankers remain available as standalone models because their native
runtimes are isolated from the default ensemble. Artifacts are written below
`data/metadata/` as `ranker_<model>.joblib` with a SHA-256 sidecar. Auto-selection
also writes `ranker_auto.joblib`, which forecast discovery prioritizes. Ranking
artifacts retain the exact training feature order; forecast loading validates
that contract before scoring.

`auto` evaluates Ridge, Elastic Net, the neural MLP, and the ensemble on the
same purged walk-forward folds, then selects the highest finite mean IC. It
returns the selected artifact and candidate diagnostics; selection is not a
live-performance claim.

## Contextual RL

```bash
quant train reinforcement --model auto
quant train reinforcement --model linucb
quant train reinforcement --model thompson
quant train reinforcement --model quantile_thompson
quant train reinforcement --model policy_gradient
```

The policies operate on cross-sectional feature rows grouped by decision date.
They update or train in chronological order and compare policy rewards with
uniform/oracle diagnostics. Artifacts are:

- `rl_linucb.joblib`
- `rl_thompson.joblib`
- `rl_quantile_thompson.joblib`
- `rl_policy_gradient.joblib`

RL artifacts are written atomically with SHA-256 sidecars and are rejected if
their checksum does not match.

`auto` evaluates all four policies on the same chronological panel and selects
the highest finite mean advantage versus uniform. It writes `rl_auto.joblib`
and records the selected policy and all candidate diagnostics.

Forecasting loads the strongest available compatible RL artifact only when no
supervised ranker artifact is present. It validates the stored feature
contract and fails closed when required columns are absent.

RL diagnostics are chronological: LinUCB, linear Thompson, and quantile
Thompson update after each realized date; policy-gradient evaluation trains on
prior dates and scores the current date before adding it to history.

## Other ML families

The existing training dispatcher also exposes causal probability calibration:

```bash
quant train calibration --model auto
quant train calibration --model isotonic
quant train calibration --model platt
```

Calibration uses the bounded momentum score and the sign of the configured
ranking label, evaluates only chronological out-of-sample folds, and persists
research-only calibrator artifacts. It is not silently applied to forecast
confidence until a matching runtime score contract is configured. The
dispatcher also exposes distribution, volatility, regime, tail, alpha,
covariance, and liquidity families. Distribution and volatility families
include tree-backed variants; conformal and risk modules provide uncertainty
and interval controls around forecasts.

## Evidence boundary

Synthetic panels and fixture DGPs validate implementation behavior only.
Walk-forward metrics are diagnostics, not proof of deployability or SOTA
performance. Promotion remains fail-closed on missing provenance, leakage
evidence, incomplete folds, synthetic data, or non-finite metrics.
