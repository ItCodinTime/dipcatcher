# Acceptance criteria v8 — corpus flywheel + host-independent gate

Extends v7 (all prior criteria hold). New:

1. **Host-independent test suite:** the fx1 gate passes on a host with no
   plugin root (`/app/.agents/plugins` absent) and no agent-gw credentials.
   Grammar/routing tests use stub scripts under a tmp `FX1_PLUGIN_ROOTS`
   with dummy credential *names*; production fail-closed preflight is
   unchanged.
2. **Property tests hold under adversarial dates:** dip-event property test
   uses unique, monotonically increasing date labels (hypothesis falsified
   the previous `i % 28` labels via `dates.index` ambiguity).
3. **Run-manifest eligibility:** the corpus loader recognizes the lab's
   research-run manifest schema — `claim: "research_only"` declares research
   scope; an explicit `live_pnl_claim` key always wins; without a claim the
   record fails closed (research_only=False, live_pnl_claim=True).
4. **Synthetic labeling:** manifests with `synthetic: true` become positive
   examples whose assistant text explicitly states the evidence is SYNTHETIC
   (simulated data, not market data). Evidence class is carried on
   `ReceiptRecord.evidence_class`.
5. **Flywheel measured:** `fx1 corpus build` over
   `data/metadata/research/runs` loads 88 manifests → 77 positive + 11
   negative (claim-less, refused). Latent display bug fixed: positive
   examples previously interpolated `record.schema` (a bound method)
   instead of `record.schema_name`.
6. **Triple gate still green:** 139 tests, ruff, mypy (47 files).
