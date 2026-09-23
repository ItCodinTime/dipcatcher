# ADR-023: robinhood+ K-line foundation engine (Kronos)

## Status

Accepted

## Date

2026-09-19

## Context

ADR-007 kept Torch optional and blocked default neural nets until tree and
econometric baselines existed. Those baselines now ship. Kronos (Shi et al.,
2025, arXiv:2508.02739, MIT) is an open K-line foundation model: a hierarchical
discrete tokenizer plus an autoregressive decoder-only transformer trained on
candlesticks. Dipcatcher needs that framework as a first-class forecast engine
without making GPU or Hub downloads a CI dependency, and without letting any
engine bypass fusion or the risk gate.

## Decision

1. Implement the Kronos two-stage contract in-repo as **robinhood+**
   (`robinhood_plus`). This is an internal Dipcatcher name. It is not
   affiliated with Robinhood Markets, Inc.
2. Default backend is NumPy: fixed-linear hierarchical binary-spherical
   quantization plus a lookback-conditioned hierarchical Markov decoder
   (s1 then s2 | s1). An optional tiny NumPy transformer decoder exists for
   architecture tests. Official Kronos weights load only via the optional
   `[nn]` extra and `backend: torch`, and never download from the Hub unless
   `allow_network: true`.
3. robinhood+ is a **core engine** and a **stamped challenger** until it
   beats **public-feature** ridge on a causal card (date-level IC of
   `rank_score` vs the ranking target; pinball/CRPS; Diebold–Mariano) **and**
   that card is not SYNTHETIC. Oracle columns (`planted_signal`) are not a
   fair champion. No Sharpe. Default `blend_weight` is 0: an engine that
   cannot beat the baseline must not size the book. Fusion still wraps any
   blended signal. The risk gate still applies. Name-level `vol_20` is not
   replaced. SYNTHETIC cannot take a champion alias.
4. Every `forecast_asof` stamps `robinhood_plus_n_ok` and
   `robinhood_plus_n_fallback`. A missing OHLC window is a fallback, never
   a fabricated Kronos path. Future `available_time` never enters lookback.
   Poison-the-next-bar tests gate leakage the same way the conformal suite
   does.
5. `backend: torch` is wired into `forecast_asof` for **local** Kronos-mini
   (or zoo) checkpoints with `allow_network: false`. Missing weights fail
   closed. They never silently fall back to numpy.
6. The family is **optional** in the research catalog (`robinhood_plus`).
   Existing required-family receipts stay valid. `dipcatcher robinhood-plus`
   `--compare` writes the champion/challenger card. No Sharpe, no live P&L
   claim.
7. ADR-007 remains in force for other neural architectures. This ADR opens
   only this K-line foundation engine, and only with Torch as an extra.

## Consequences

- `configs/base.yaml` enables robinhood+ with `backend: numpy` and
  `blend_weight: 0.0` after the SYNTHETIC ridge card.
- Forecast diagnostics stamp `core_engine=robinhood_plus` only when blend
  is applied and a name has a clean PIT K-line window; otherwise ridge /
  momentum remains the fallback and `n_fallback` is explicit.
- Jackknife+ `predict_interval` converts both LOO location and scores from
  standardized residual space via the test-time scale. Scaling only the
  half-width was a receipt-integrity bug (coverage ~0.07 vs the 1-2α floor),
  not a robinhood+ score.
- Pretrained Kronos-mini checkpoints are optional evidence, not the CI
  path. The NumPy decoder is a research-lab language model of K-line tokens,
  not a claim that it matches Hub weights.
- Fair G1 cards drop oracle columns and train ridge on `PUBLIC_FEATURES`.
  `configs/sota_protocol.yaml` freezes universe/horizons/embargo/sample_count
  /variant before scoring. Dual scoreboards: Dipcatcher IC/pinball/CRPS/DM
  and Kronos-paper path RankIC (DLinear in-repo; other TSFMs skipped without
  local weights). Calibration (Jackknife+, CQR/ACI Kupiec, PIT) is a gate.
  A file tape needs `event_time`, `available_time`, split-adjusted OHLC, and
  membership; Stooq EOD is session-close PIT, vendor-adjusted, not SIP.
