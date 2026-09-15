# ADR-010: Anytime-valid e-process for conformal coverage

## Status

Accepted

## Date

2026-09-15

## Context

Split CQR / ACI report a single holdout coverage number. Re-checking that
number after every date (or after every research peek) is a fixed-horizon
binomial / Kupiec test used as if it were sequential. We need evidence that
the empirical miss rate of prediction sets is consistent with nominal α that
remains valid at every prefix, with no peeking penalty.

## Options considered

- Fixed-horizon binomial or Kupiec on the full window — valid once; invalid
  if the lab inspects prefixes or re-runs after more dates arrive.
- Sequential p-values with α-spending / Bonferroni — valid but conservative
  and tied to a pre-declared look schedule.
- Anytime-valid e-process (Ramdas; Grünwald; Shafer–Vovk) — a nonnegative
  martingale under the null miss rate; Ville's inequality gives a threshold
  that may be monitored continuously.

## Decision

`quant_fund.metrics.evalues` monitors miss indicators \(X_t\in\{0,1\}\)
against \(H_0: \mathbb{E}[X_t]=\alpha\) with the Bernoulli betting /
likelihood-ratio e-value and a **fixed, predictable** alternative
\(\lambda=2\alpha\) (or \((\alpha+1)/2\) if \(2\alpha\ge 1\)):

\[
e_t
=
\lambda\frac{X_t}{\alpha}
+
(1-\lambda)\frac{1-X_t}{1-\alpha}
=
\Bigl(\frac{\lambda}{\alpha}\Bigr)^{X_t}
\Bigl(\frac{1-\lambda}{1-\alpha}\Bigr)^{1-X_t}.
\]

The e-process is the running product \(E_0=1\), \(E_t=\prod_{s=1}^t e_s\)
(one-step values clipped to \([0,10^{300}]\)). Under \(H_0\),
\(\mathbb{E}[e_t\mid\mathcal{F}_{t-1}]=1\), so \((E_t)\) is a nonnegative
martingale. Ville: \(\mathbb{P}(\sup_t E_t\ge 1/\alpha_{\mathrm{err}})
\le\alpha_{\mathrm{err}}\). The threshold is therefore \(1/\alpha_{\mathrm{err}}\)
(default \(\alpha_{\mathrm{err}}=0.05\Rightarrow 20\)). No look-schedule and
no multiplicity spend. \(\lambda>\alpha\) grows \(E_t\) on undercoverage
(miss streaks). `bench_e_coverage` reports coverage, final \(E_n\), whether
20 was ever crossed, and \(n\).

## Consequences

Coverage remains the lab point estimate. The e-process is sequential
evidence against nominal α, not a trading score and not a replacement for
PIT / pinball of the raw model (ADR-008). Growing \(E_t\) means too many
misses, not that the sets should be tightened by peeking at test residuals.
