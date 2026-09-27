# Cost allocator numerical fixtures

These four `.npz` files contain only numeric allocation inputs. They were
reconstructed from the previously inspected US-wide validation tape to
regress solver conditioning, not to measure investment performance.

- Tracked bar snapshot SHA-256: `e22bf3eff634c742299b6495f8daf8f02adcc6cda47d9641e3094c0c191ae117`
- Benchmark config SHA-256: `2eb3a9e979121b6c749612278269386fb5a82cfc8a3fba3b839110361bb736b5`
- Cost-aware tournament config SHA-256: `4538d4a67d5dbca91b14d98b41e0450b7a9e7a307d2708d67f2da9b7e5741c52`
- Inputs: `alpha`, `covariance`, `uncertainty`, `previous`, `lower`, `upper`, `capacity`, `impact` in original NAV-fraction units.
- `original39` and `original114` are the first failures of the September 25 solver for momentum and reversal. `factor359` and `factor7` are later failures observed while trying an equivalent factored-risk formulation. Decision numbers start at one within validation.

The original sealed failure receipt and its evidence index are unchanged. These
fixtures do not create an untouched holdout or a forward shadow record.
