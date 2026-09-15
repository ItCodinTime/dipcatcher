# ADR-018: Conformal prediction sets for ranking / top-k

## Status

Accepted

## Date

2026-09-15

## Context

Ranking scores (`models.ranking`) order names within a date. We still need a
finite-sample answer to *which names belong in the top-k*. Marginal coverage of
a return interval (ADR-008) does not control errors on a shortlist. Cross-section
must stay per date: never stack raw names across timestamps. Neural rankers stay
blocked (ADR-007). Lab scores stay set size, FDR, and oracle top-k coverage — not
Sharpe.

## Options considered

### Option A: Naive score top-k

- Pros: one argsort per date.
- Cons: no finite-sample error control; a noisy ranker can fill the k slots with
  false names.

### Option B: Split conformal on the k-th threshold (set coverage)

Per calibration date, \(t_d=\min\{u_i:i\in\text{oracle top-}k\}\) where \(u_i\)
is the within-date score percentile. The set is \(\{i:u_i\ge\hat q\}\) with
\(\hat q\) the finite-sample lower \((1-\alpha)\) quantile of \(\{t_d\}\).
- Pros: \(P(\text{oracle top-}k\subseteq C)\ge 1-\alpha\) over exchangeable
  dates; one threshold; no BH.
- Cons: sets are often larger than \(k\); controls misses, not false discoveries.

### Option C: FDR conformalized selection (chosen default)

Jin & Candès (2023) cfBH / Bates et al. (2021) conformal p-values. Split dates
chronologically. H0 for name \(i\) on a date: *not* in that date's oracle top-k
(true label not among the \(k\) best). When \(k=\lfloor n/2\rfloor\) this is
H0: not better than the median. Null calibration scores are within-date
percentiles of names with true rank \(>k\). Test p-values are

\[
p_j=\frac{1+\#\{u\in U_{\mathrm{null}}:u\ge u_j\}}{n_{\mathrm{null}}+1}.
\]

BH runs **per test date**. Finite-sample FDR \(\le\alpha\) if those percentiles
are exchangeable between a test null and the calibration null pool.

- Pros: controls the fraction of selected names that are not truly top-k; wraps
  any ranking score; date-local multiplicity.
- Cons: does not guarantee that every true top-k name is selected (that is
  power / Option B).

### Option D: Neural ranker plus conformal

Forbidden until conformal baselines exist (ADR-007).

## Decision

`quant_fund.models.conformal_rank.conformal_topk` implements Option C by
default (`guarantee="fdr"`) and Option B as `guarantee="set_coverage"`. Ranking
models are not edited; they only supply scores. `bench_conformal_topk` reports
set size, FDR, oracle top-k coverage, and \(n_{\mathrm{dates}}\). No Sharpe.

## Consequences

Promotion can require FDR \(\le\alpha\) or date-level coverage \(\ge 1-\alpha\),
never a trading ratio. A well-ordered ranker yields small sets under FDR and
larger nesting sets under coverage. Date-level score drift is handled by
within-date percentiles; stacking raw scores would break that.
