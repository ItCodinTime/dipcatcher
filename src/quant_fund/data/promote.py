"""Promote a collected source frame into a ``bars.parquet`` the parquet provider ingests.

``collect`` writes ``<root>/raw/sources/<source>.parquet`` in the full
source-PIT schema; ``data.source=parquet`` ingests ``<parquet_path>/bars.parquet``.
This is the bridge: it projects nothing away (the bar contract is a subset of
the source schema), enforces the same bar/PIT contract the provider enforces at
read time, refuses to clobber an existing bars file without ``force``, and writes
a provenance receipt that hashes the source parquet *and* its sidecar receipt so
the promotion step is itself auditable.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_fund.data.adapters.parquet import validate_bars_frame
from quant_fund.data.sources.base import SourceError
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

PROMOTE_RECEIPT_SCHEMA = "bar_promotion.v1"


def _dest_path(dest_dir: Path, filename: str) -> Path:
    relative = Path(filename)
    if relative.is_absolute() or relative.name != filename or filename in {"", ".", ".."}:
        raise SourceError("bars filename must be a plain relative name")
    if relative.suffix != ".parquet":
        raise SourceError("bars filename must use the .parquet suffix")
    return dest_dir.resolve() / filename


def _write_atomic(path: Path, write: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def promote_bars(
    source_file: str | Path,
    dest_dir: str | Path,
    *,
    filename: str = "bars.parquet",
    force: bool = False,
) -> dict[str, Any]:
    """Validate ``source_file`` against the bars contract and publish it as
    ``<dest_dir>/<filename>`` plus a provenance receipt."""
    src = Path(source_file).resolve()
    if not src.is_file():
        raise SourceError(f"source parquet not found: {src}")
    if src.suffix != ".parquet":
        raise SourceError("source file must be a .parquet")

    dest = _dest_path(Path(dest_dir), filename)
    if dest.exists() and not force:
        raise SourceError(f"{dest} already exists; pass force=True to replace it")

    frame = pl.read_parquet(src)
    if frame.is_empty():
        raise SourceError("source parquet has no rows")
    if "source" in frame.columns:
        labels = frame.get_column("source").cast(pl.String).unique().to_list()
        if len(labels) != 1:
            raise SourceError(
                f"promoted bars must carry exactly one source label, got {sorted(labels)}"
            )
    # The provider validates bars at read time; run the same contract here so a
    # bad frame fails at publish, not hours later inside ingest.
    validate_bars_frame(frame)

    source_receipt_path = src.with_suffix(".json")
    source_receipt_sha256: str | None = None
    source_label: str | None = None
    if source_receipt_path.is_file():
        source_receipt_sha256 = hashlib.sha256(source_receipt_path.read_bytes()).hexdigest()
        try:
            source_label = json.loads(source_receipt_path.read_text()).get("source")
        except (ValueError, AttributeError):
            source_receipt_sha256 = None
    dest.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(dest, lambda tmp: frame.write_parquet(tmp))
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()

    # Seal under the verifier's canonical_json convention (same bytes as
    # research.receipt_v2.seal_receipt) without a data→research edge.
    body = json.loads(
        canonical_json_bytes(
            {
                "schema": PROMOTE_RECEIPT_SCHEMA,
                "promoted_at": datetime.now(UTC).isoformat(),
                "source_parquet": str(src),
                "source_parquet_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                "source_receipt_sha256": source_receipt_sha256,
                "source_label": source_label,
                "dest": str(dest),
                "dest_sha256": digest,
                "rows": frame.height,
                "security_ids": frame.get_column("security_id").n_unique(),
                "event_time_range": {
                    "min": str(frame.get_column("event_time").min()),
                    "max": str(frame.get_column("event_time").max()),
                },
            }
        )
    )
    body.pop("receipt_sha256", None)
    receipt = {**body, "receipt_sha256": hash_bytes(canonical_json_bytes(body))}
    receipt_path = dest.with_suffix(".json")
    _write_atomic(
        receipt_path,
        lambda tmp: tmp.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        ),
    )
    return {"data": dest, "receipt": receipt_path, "rows": frame.height}
