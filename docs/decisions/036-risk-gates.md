# ADR-036: Causal risk-controlled gates (Kelly / CRC / StepM size)

## Status

Accepted

## Date

2026-09-21

## Context

The 55-name public ridge book has IC ≈ 0. The user still wants Sharpe > 5,
max DD < 5%, and large net P&L. Vol targeting plus a 5% drawdown halt cannot
jointly produce that: Sharpe is invariant to *constant* leverage (rf = 0),
and a 2.5% vol target at Sharpe 5 is only ~12.5%/year. Risk engines cap
loss. They do not mint alpha. Lightspeed nautica already has a −20%/10d
crash flatten. CRC (ADR-012) and Romano–Wolf StepM (ADR-022) existed as
diagnostics, not as delay-1 size gates. Sign-flipping a losing book is
forbidden (ADR-032).

## Decision

1. Implement `quant_fund.risk.gates` as the causal gate library: Moreira–Muir
   vol targeting, remaining-budget / halt drawdown, Acerbi–Tasche ES cap,
   Thorp fractional Kelly (negative μ → 0), CRC loss-bound size, nautica
   crash analog, and expanding-window StepM allow/deny. Every scale applied
   to bar t uses information through t−1.
2. `BookRiskOverlay` additionally takes `min` with fractional Kelly and CRC.
   Kelly/CRC lookbacks default to 63 so existing short overlay unit tests
   stay on the vol/DD/ES path.
3. Catalog CS challengers `tsmom` (Moskowitz–Ooi–Pedersen 12–1 as a CS
   score), `vme` (AMP 2013 public proxy: 12–1 + George–Hwang 52w; no B/M),
   and `krauss` (Krauss–Do–Huck 2017 linear logistic on the same purged
   walk-forward as ridge). DNN stays behind ADR-007. Champion remains
   public ridge. `blend_weight` stays 0.
4. CLI: `dipcatcher ls hunt` (tape hunt + riskstack) and `dipcatcher ls race`
   (CS DM/RC/SPA/StepM + gated long-short). Paper-book only. No Alpaca.

## Consequences

- A DD-safe book can be *smaller*, not more skilled. StepM-gated size on
  this tape is expected to stay cash until some challenger actually
  rejects the no-skill null.
- Promotion is still pairwise DM of −IC vs ridge plus White RC / SPA /
  StepM on a non-SYNTHETIC public card. IS Sharpe does not promote.
