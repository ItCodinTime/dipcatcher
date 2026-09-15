# ADR-012: Conformal Risk Control for monotone tail loss

## Status

Accepted

## Date

2026-09-15

## Context

Split CQR / ACI (ADR-008, ADR-009) control *interval miscoverage*. Tail work needs a finite-sample bound on expected VaR-hit or drawdown-exceedance, a monotone 0-1 loss, not just \(P(Y \notin C(X))\). Neural nets remain blocked (ADR-007). The base quantile / historical bound stays the predictor; we only expand it.

## Options considered

- Coverage-only CQR / one-sided conformal — valid for miss probability, silent on other bounded risks.
- RCPS / Learn-then-Test (Bates, Angelopoulos, Lei, Malik, Jordan 2021) — distribution-free but more conservative concentration bounds.
- Conformal Risk Control (Angelopoulos, Bates, Malik, Jordan 2022) — for nested monotone losses, \(\mathbb{E}[L_{n+1}(\hat\lambda)] \le \alpha\) with an \(O(1/n)\) inflation.

## Decision

`ConformalRiskControl` wraps an existing loss bound. Hit loss is \(L = 1\{\text{loss} > \text{bound}(\lambda)\}\), \(B=1\). The threshold is the smallest \(\lambda\) with

\[
\frac{n\,\widehat L_n(\lambda)+B}{n+1}\le\alpha.
\]

If no candidate satisfies the inequality, use \(\lambda_{\max}\). Lab scores are CRC risk, nominal \(\alpha\), \(n\), and \(\hat\lambda\). No Sharpe. PIT / Kupiec of the *raw* bound stay specification diagnostics.

## Consequences

Expected calibration-style hit risk is controlled at \(\alpha\), which is stronger than reporting coverage of a two-sided return interval. CRC does not repair a misspecified Gaussian. Promotion still requires the raw model's PIT and Kupiec to remain visible.
