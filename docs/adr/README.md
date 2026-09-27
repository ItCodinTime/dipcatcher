# Architecture decision records — atlas series

Records of *implicit* design decisions discovered while mapping the codebase
for `docs/ARCHITECTURE_ATLAS.md`. Each entry cites the code that implements
the decision, so the record can be re-verified.

Numbering is `NNNN-slug.md` to distinguish this series from the accepted
`docs/decisions/NNN-slug.md` series (ADR-001..036), which this series
cross-references rather than duplicates.

| ADR | Decision |
|---|---|
| [0001](0001-immutable-self-sealing-receipts.md) | Research receipts are immutable, self-sealing, atomically published evidence |
| [0002](0002-fx1-filesystem-boundary.md) | fx1 integrates with quant_fund through the receipt filesystem, never imports |
| [0003](0003-fx1-test-lane-separation.md) | `tests/fx1` is deliberately outside the default pytest testpaths |
| [0004](0004-fail-closed-defaults.md) | Degraded inputs fail closed instead of producing plausible-looking output |
| [0005](0005-paper-ledger-publish-order.md) | Ledger rows are durable before the resume cursor; resume is fingerprint-bound |
| [0006](0006-strict-config-forbid-extra.md) | `extra="forbid"` config validation rejects unknown keys at every level |
| [0007](0007-shadow-holds-no-capital.md) | Shadow/challenger slots record intent only and can never move cash |
| [0008](0008-forbidden-metric-key-scan.md) | Honesty is enforced by scanning artifact keys, not by convention |
