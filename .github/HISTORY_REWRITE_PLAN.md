# History rewrite plan

This plan is not executed. Do not run it, do not force-push, and do not
delete the current remote until the owner has a read-only archive and has
accepted that every commit SHA will change.

Measured on 2026-09-27 from `git rev-list --objects --all` plus
`git cat-file --batch-check` (unique blobs, uncompressed) and
`git count-objects -vH` (pack):

| | |
|---|---|
| Pack | 383.16 MiB |
| Unique blobs | 4,322 |
| Uncompressed blob bytes | 518.4 MiB |

## What dominates history

| Path prefix | Unique blobs | Uncompressed |
|---|---:|---:|
| `data/file_us_wide/` | 26 | 244.8 MiB |
| `.dsh-24x7/` | 1,536 | 145.5 MiB |
| `artifacts/` (history only; `artifacts/repo_llm/checkpoint.pt` is not on `HEAD`) | 46 | 56.7 MiB |
| `data/` other than `file_us_wide` | 13 | 21.1 MiB |
| `third_party/` | 92 | 16.0 MiB |
| `uv.lock` (historical copies) | 9 | 7.2 MiB |
| `coverage.xml` (already gitignored) | 1 | 0.9 MiB |

Largest single blobs:

| Bytes | Object | Path |
|---:|---|---|
| 59,483,601 | `d155c76c49da` | `data/file_us_wide/silver/universe.parquet` |
| 57,986,711 | `1c1ab49b8e8b` | `artifacts/repo_llm/checkpoint.pt` |
| 41,072,342 | `988874d5ca76` | `data/file_us_wide/silver/bars.parquet` |
| 19,756,938 | `ed0983475f56` | `data/file_us_wide/bronze/bars.parquet` |
| ~8,020,320 each | several | `data/file_us_wide/metadata/oos_scores_*/**/*.npy` |

At the audit base, `data/file_us_wide/` and `.dsh-24x7/` were tracked.
As of `e749f1149`, `data/file_us_wide/` is no longer tracked, while
`.dsh-24x7/` remains tracked. Removing paths from the index without a
rewrite leaves their blobs in history.

Dropping `data/file_us_wide/`, `.dsh-24x7/`, and `artifacts/` removes about
447 MiB of the 518 MiB of uncompressed blobs. That is the rewrite worth doing.
Leave `uv.lock` history alone. `third_party/kronos/` is vendored source plus
one 5.8 MiB CSV; decide separately, it is not the bulk.

## Why this is not a secret-removal rewrite

A gitleaks 8.30.1 scan of all 239 commits on 2026-09-27 reported 188
`generic-api-key` hits. Every one is a content hash (sha256 or a 40-hex git
revision) or the synthetic canary in `test_probe_never_leaks_secret_values`.
No provider token prefix was found. Do not rotate credentials because of this
scan. The rewrite is for repository size.

## Honesty-contract constraint

Research receipts store `git_revision` and related hashes. A rewritten `main`
will not contain those commits. Old receipts stay reproducible only against
an archive of today's history. The rewritten repository must not be described
as the history those receipts were computed from.

## Steps, when the owner decides to do this

1. Push a mirror of the current repository to a new read-only remote
   (another GitHub repo, or a bundle). Confirm `git clone --mirror` of that
   archive still contains today's `HEAD` SHA. Do not delete the archive.
2. Clone a fresh mirror to a scratch machine. Work only on that clone.
3. Install `git-filter-repo` (not `git filter-branch`).
4. Dry-run the path list and confirm the expected blobs disappear:

   ```bash
   git filter-repo --dry-run \
     --invert-paths \
     --path data/file_us_wide \
     --path .dsh-24x7 \
     --path artifacts/repo_llm \
     --path coverage.xml
   ```

   Review the dry-run. Then run the same command without `--dry-run`.
5. On the rewritten tree, ignore those paths so they are not recommitted:

   ```
   data/file_us_wide/
   .dsh-24x7/
   artifacts/repo_llm/
   ```

   `coverage.xml` is already ignored. Regenerate research data outside git.
   The checkpoint under `artifacts/repo_llm/` is not a release artifact.
6. `git gc --prune=now` and compare `git count-objects -vH` to the table
   above. Expect the pack to fall well under 100 MiB if the three prefixes
   are gone.
7. Open a pull request from the rewritten clone only if the host allows a
   non-fast-forward update via a new branch. Replacing `main` cannot be a
   fast-forward. It needs the owner to:
   - temporarily allow a force-push on `main` (ruleset exception),
   - push with `--force-with-lease` after checking the lease SHA is still
     today's `HEAD`,
   - turn the force-push block back on,
   - rebase or close every open pull request (their SHAs will not apply).
8. Tell anyone with a clone to re-clone. Do not try to merge old branches
   back in; that restores the blobs.
9. Leave the archive remote in place so receipt `git_revision` values still
   resolve. Point `docs` at the archive when a receipt is cited from a
   pre-rewrite SHA.

Do not force-push `main` as part of adopting this document.
