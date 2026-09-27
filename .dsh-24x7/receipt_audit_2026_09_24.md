# Receipt audit — 2026-09-24

## JSON validity

| Family | Files | Parse OK | Malformed |
|---|---|---|---|
| `.dsh-24x7/` | 306 | 306 | 0 |
| `receipts/` | 5 | 5 | 0 |
| `artifacts/` | 25 | 25 | 0 |
| **Total** | **336** | **336** | **0** |

## Reference integrity (PROOF.md, HANDOFF.md, README, all docs/*.md)

Backtick-quoted file refs extracted and resolved against repo root plus
`src/quant_fund/`, `scripts/`, `tests{,/unit,/regression,/end_to_end}`, `.dsh-24x7/`,
`receipts/`, `artifacts/`, `data/`, `configs/`, `docs/` prefixes.

**Genuinely broken references: 0.**

Categorized "missing" hits (all expected, none defects):

| Class | Examples | Verdict |
|---|---|---|
| Runtime products of runs (not committed) | `broker_state.json`, `equity.parquet`, `latest.json`, `promotion_dry_run.json`, `analytics_export.json`, `perf_bench.json`, `meta.json` | Expected — produced by runs |
| Remote-host receipts (disclosed remote) | `carry_1d.band15.json` lineage in PROOF.md §carry defects | Disclosed as remote `receipts/` |
| `/tmp` scratch refs in docs | `/tmp/expand_grid2.json`, `/tmp/dipcatcher-sp500-real-v1.json` | Session scratch, disclosed |
| Module/script names w/o dirs | `carry_engine.py`, `carry_research.py`, `sota_eval_kronos.py`, test filenames | All resolve under src/scripts/tests |
| Evidence receipts (path-less citations) | `MERGED_d1_native.json`, `MERGED_h4_native.json`, `d1_merged.json`, `h4_merged.json`, `merge_h4f.json`, `.losses.npz`, `.v2.npz` | All exist under `.dsh-24x7/{native,eval-full,mega-arena,lane-*}/` |

## Broken receipts

None. No fix PR needed — the audit is clean.
