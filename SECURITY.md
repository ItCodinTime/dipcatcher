# Security Policy — fx-1 / dipcatcher

## Scope

This repository contains the fx-1 model project and the dipcatcher research
harness. Security-relevant surfaces:

- **fx-1 serving path** (`fx1.serve`): hosted-backend credentials, local
  checkpoint loading, release-signature verification.
- **Harness API** (`quant_fund.api`): API-key auth, loopback fail-closed,
  config allowlisting, request-size caps.
- **Corpus/receipt integrity** (`fx1.data`, `fx1.train.receipts`): hash-chained
  ledgers and tamper-evident receipts.

## Hard rules (enforced in code, tested)

- No credentials in source or tests; secrets come from environment variables
  only (`MOONSHOT_API_KEY`, `FX1_SIGNING_KEY`, `QUANT_API_KEY`).
- Fail-closed defaults: missing signatures, missing model cards, failing
  honesty gates, and invalid receipts all *block* rather than warn.
- Network access is opt-in (explicit `collect`, explicit API serve), never
  part of default ingest or training paths.

## Reporting

Report vulnerabilities privately to the maintainers (GitHub private security
advisory). Do not open public issues for exploitable findings. We aim to
acknowledge within 72 hours.

## Supply chain

- Dependencies are locked (`uv.lock`); CI runs `pip-audit` and Bandit gates.
- Releases are signed (attestation ladder tier 1); verify with
  `fx1 attestation <checkpoint_dir>` before serving.
