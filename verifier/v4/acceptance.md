# Acceptance criteria v4 - research loop + flagship bench

Extends v3 (all prior criteria still hold). New:

1. Hypothesis traces: proper-score-only score keys (fail-closed), gate-verdict
   admissibility, receipt-bound SFT rendering.
2. Deterministic reward model: auditable weighted rules; honesty violation
   caps score at -10.
3. Curriculum builder: contracts -> interpretation -> research loop ->
   refusal ordering; deterministic from corpus hash + seed.
4. Red-team suite: roleplay/hypothetical/authority/evidence-laundering/
   encoding/synthetic-renaming attacks; laundering model fails, compliant
   model passes.
5. Dip Quality Score bench: causal detection with peak-regain recovery,
   unobservable horizons recorded as None, proper scores (Brier/log-loss/ECE),
   unconditional baseline, honesty gate on bench output keys, probability
   range enforcement.
6. Cited serving: provenance footer + honesty validation at inference.
7. CLI additions: redteam, dpo, curriculum, dipbench.
8. Honesty refinement: synthetic label rule fires on numeric presentation,
   not on commentary/refusals (regression-tested both ways).
9. Quality: full suite green; ruff clean.
