# Acceptance criteria v2 - deepened turnover

Extends v1 (all v1 criteria still hold). New:

1. **Full harness registry:** `src/fx1/harness.py` covers all 23 public lab
   commands across four roles (data_engine, evaluation, verification,
   model_training); hidden `lab` and network `api` surfaces excluded.
2. **Multi-source corpus:** receipts + markdown notebooks + JSON ledgers, all
   provenance-hashed; nested `live_pnl_claim=true` anywhere in a ledger
   forces a negative example.
3. **Eval task bank:** built-in bank with >=5 honesty baits, domain tasks on
   proper scores/gates/microstructure, general regression tasks; bank catches
   a noncompliant model.
4. **Model cards:** versioned `fx-1.vX.Y` cards with eval deltas, ship gate
   (honesty native + domain up + general no-regress), research-scoped by
   construction.
5. **Inference backends:** hosted K3 (env-only credentials) and local fx-1
   (fail-closed without a passing model card); factory rejects unknown
   backends.
6. **CLI surface:** `fx1 corpus build|build-full`, `fx1 eval`,
   `fx1 modelcard`, `fx1 harness list|run`, `fx1 train manifest`.
7. **Makefile:** fx1-test / fx1-lint / fx1-corpus / fx1-corpus-full /
   fx1-eval targets.
8. **Quality:** full `tests/fx1` suite passes; ruff clean on src/fx1 and
   tests/fx1.
