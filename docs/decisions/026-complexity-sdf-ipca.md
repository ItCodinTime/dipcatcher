# ADR-026: Virtue of Complexity, SDF ridge, IPCA rankers

## Status

Accepted

## Date

2026-09-19

## Context

The hedge-lab overlay caps drawdown; it cannot mint alpha. Public-feature
ridge remains the champion. The user asked to find heavy research papers
and implement them as causal engines. Three open (or NBER-equivalent)
papers fill a gap ridge/trees do not cover: nonlinear random-feature
complexity, an SDF that shrinks low-eigenvalue PCs, and a latent-factor
map from characteristics to expected return.

Wiley served Kelly–Malamud–Zhou, *Journal of Finance* 2024
(DOI 10.1111/jofi.13298) as open access. Kozak–Nagel–Santosh (JFE 2020)
and Kelly–Pruitt–Su (JFE 2019) were read from NBER w24070 and w24540.

## Decision

1. Add ranking catalog names `rff`, `sdf_ridge`, and `ipca` in
   `quant_fund.models.asset_pricing`. Train them with the same purged
   walk-forward as ridge. Public CS features only. Oracle columns stay
   dropped on public cards.
2. Formulas and deviations live in MATH_SPEC. RFF uses JoF (20) sin/cos
   pairs with \(\gamma=2\), train-column standardized \(S\), and the
   paper's \(z\) ridge (dual when \(P>T\)). SDF ridge is KNS (22) on
   date-level managed portfolios. IPCA is ALS on FOCs (6)–(7) with
   \(\Gamma'\Gamma=I_K\), default \(K=3\).
3. `forecast_asof` still loads `ranker_ridge.joblib`. These models are
   challengers until a **non-SYNTHETIC** public-feature card beats ridge
   on date IC / pinball / CRPS / DM. `blend_weight` stays 0.
   `live_pnl_claim` stays false. Metadata must not carry Sharpe.
4. Risk overlays, hold-until-rebalance, and pyRisk diagnostics are
   unchanged. A better ranker does not relax the 5% drawdown budget.

## Consequences

- `dipcatcher train ranking --model rff` (or `sdf_ridge` / `ipca`) writes
  `ranker_<name>.joblib` under the data root. That file does not replace
  the ridge cache used by `forecast_asof`.
- SYNTHETIC IC on planted features is a correctness test, not a promotion
  ticket. Hedge-lab paper-book Sharpe stays in `hedge_lab_analytics`.
- Complexity \(P/T\) on a stacked panel is not the paper's monthly market
  timing \(c=P/T\). Do not quote JoF Sharpe ratios as Dipcatcher results.
