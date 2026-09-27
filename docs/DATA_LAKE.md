# Data lake

Research storage for market data already on disk, including the Hugging Face
[OHLCV-1m](HF_OHLCV_1M.md) cache and local Parquet tapes such as
`data/file_us_wide`. This is not a broker, not a SIP vintage, and not a live
P&L path. Nothing here places orders.

## Storage

The lake root (default `data/lake/`, gitignored) has three layers:

| Path | What it is |
|---|---|
| `objects/sha256/<aa>/<digest>` | Exact file bytes, named by the SHA-256 of those bytes |
| `snapshots/<snapshot_id>.json` | Immutable manifest. The id is the SHA-256 of the canonical manifest |
| `partitions/source=<s>/symbol=<sym>/date=<YYYY-MM-DD>/<digest>.parquet` | Parts written by the lake, not by migration |

Hashes reuse `content_address` in `quant_fund.utils.reproducibility`, the same
chunked SHA-256 as worktree fingerprints. A Git LFS pointer is not hashed as
text: `content_sha256` is the pointer's `oid` (the real object) and
`stored_sha256` is the hash of the pointer bytes actually on disk. A smudged
payload is a `blob` whose two hashes match.

Snapshots are written once. A second commit of the same manifest is a no-op.
A different body under an existing id fails closed.

A research receipt may cite a snapshot without changing scorecard metrics.
`snapshot_receipt_fields` returns `provenance["data_snapshot_id"]`. The
default research provenance builder does not add it. `verify-research`
ignores a missing id and rejects a malformed one.

## Lineage

Every derived dataset records the input snapshot ids, a code hash, and the
parameters. The code hash is the SHA-256 of a source string, a callable's
source, or a file's real bytes. Records are append-only; the index points at
the latest record for a name.

```bash
uv run dipcatcher lineage show features/panel --root data/lake
uv run dipcatcher lineage verify --root data/lake
uv run dipcatcher lineage verify features/panel --root data/lake
```

`verify` re-hashes every object in the pinned snapshots and, when `code_path`
is set, the code file. Drift exits non-zero. The DAG printed by `show` follows
pinned input snapshot ids, not whatever the input dataset's latest record
happens to be.

## Queries

DuckDB reads the Parquet objects. `asof_bars` keeps a row only when both
`event_time` and `available_time` are at or before the timezone-aware
decision clock. A row whose available time precedes its event time is
lookahead and raises `LeakageError`. Files with neither column are refused,
so an event-time-only scan cannot stand in for a knowledge time.

Hugging Face month files stay in vendor form (`timestamp` is the minute
open). When the snapshot source is `hf_ohlcv_1m`, the query projects
`event_time` and `available_time` as `timestamp + 1 minute` (the close
convention in `DATA_CONTRACTS.md`) without rewriting the object.

`universe_asof` is survivorship-safe only when the snapshot can support it:

- A membership panel (`asof`, `security_id`, `available_time`) uses the latest
  panel date that was already knowable. Names that have dropped out of that
  date are absent. A later reconstitution is not applied backward.
- An interval table (`effective_from`, `effective_to`, `available_time`) keeps
  names whose window covers the decision and whose row was already available.
  A delisted name stays in until `effective_to`.
- Bars plus a security master can be passed to the existing `membership_asof`
  filter when a `UniverseConfig` is supplied.
- A bare symbol list is refused.

## Quality

`quality_report` scores gaps, duplicate keys, timestamps that move backwards
inside a symbol, OHLC envelope breaks (`low <= open, close <= high`, positive
finite prices), outlier log returns, and stale unchanged closes. The report
is JSON (`dipcatcher.lake.quality.v1`). Thresholds live on
`QualityThresholds`. `enforce=True` raises `QualityThresholdError` instead of
returning a failed report. Structural checks default to a zero tolerance.
`max_gap=None` reports the longest observed gap and does not gate it.

An LFS pointer fails `payload_present` because the real bytes are not in the
worktree. The checker does not fetch them.

```bash
uv run dipcatcher lake quality --path data/file_us_wide/bronze/bars.parquet \
    --max-gap-days 10
```

## Migration

`import_files` copies each source into the object store. It hashes the source
before and after the copy and refuses to continue if those hashes differ.
Multi-day files stay one object; they are not split into new Parquet parts,
because that would change bytes. A single symbol and a single date are
recorded as partition labels on the manifest entry only.

```bash
uv run dipcatcher lake import data/file_us_wide/bronze/bars.parquet \
    --dataset file_us_wide_bronze --root data/lake
```

`lagged_return_panel_bytes` is a deterministic research panel (prior close
within the symbol, canonical JSON). Reading it from the original path and
from the imported object yields the same bytes. That is the migration
guarantee: research output follows the bytes, and the bytes do not change.

Imported objects are not committed. `data/lake/` is gitignored along with the
other derived data directories.
