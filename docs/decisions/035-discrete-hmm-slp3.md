# ADR-035: Discrete HMM (Jurafsky & Martin SLP3 Appendix A)

## Status

Accepted

## Date

2026-09-21

## Context

ADR-006 wraps `hmmlearn.hmm.GaussianHMM` for continuous regime
inference. That is a different model: Gaussian emissions, `hmmlearn`
solver, no textbook Eisner numbers. The user pointed at
https://web.stanford.edu/~jurafsky/slp3/A.pdf (Jurafsky & Martin
*Speech and Language Processing* 3ed, Appendix A). That appendix is
the discrete first-order HMM \(\lambda=(A,B,\pi)\) with Rabiner's
three problems: likelihood (forward), decoding (Viterbi), learning
(Baum–Welch). The running example is Eisner's ice-cream HMM
(COLD/HOT, observations \(\{1,2,3\}\)).

## Decision

1. Implement the appendix as `quant_fund.hmm` in linear space
   (matches the Eisner arithmetic). Log-space variants stay out of
   v1.
2. Do **not** replace `GaussianHMMRegime`. Discrete HMM and
   continuous Gaussian HMM stay separate.
3. CLI: `dipcatcher hmm eisner|train`. Research only. Not a CS
   ranker. `blend_weight` stays 0. Champion remains public ridge.
4. Unit tests lock Fig. A.5 / A.8: \(\alpha_1(H)=0.32\),
   \(\alpha_1(C)=0.02\), \(\alpha_2(H)=0.0404\), \(\alpha_2(C)=0.069\),
   \(P(3\,1\,3)\approx 0.028562\), Viterbi path HCH.

## Consequences

- Baum–Welch is unsupervised MLE on one sequence. It is not a
  promotion gate.
- Supervised MLE (`mle_supervised`) is the Appendix A.5 warm-up
  when states are visible.
- No live P&L claim.
