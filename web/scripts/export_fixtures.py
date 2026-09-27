"""Export committed research artifacts into ``web/public/fixtures/``.

The research explorer web app is a static, read-only bundle. This script
materializes its data fixtures from the repo's real, committed evidence:

- ``receipts/*.json``           -> copied verbatim into ``fixtures/receipts/``
- ``artifacts/carry_*.json``    -> strategy stat fixtures (verbatim copies)
- ``artifacts/*equity*.parquet``-> equity-curve fixtures (compact JSON rows)
- ``fixtures/index.json``       -> manifest the app loads first, including a
                                   sha256 -> repo-path match table so the UI
                                   can show which embedded receipt hashes
                                   resolve to committed files in this checkout.

Sources are read-only; originals are never modified. Re-run after any
(sealed, intentional) change to ``receipts/`` or the committed artifacts::

    uv run --no-sync python web/scripts/export_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "web" / "public" / "fixtures"

# Directories scanned when resolving embedded sha256 fields to committed files.
# Bounded to keep the export fast; extend if new hash fields appear.
HASH_SCAN_ROOTS = ("src", "scripts", "configs", "examples", "receipts", "artifacts")
HASH_SCAN_MAX_BYTES = 20 * 1024 * 1024

EQUITY_SOURCES = (
    ("carry", "carry_equity_full.parquet", "carry_champion.json", "Carry champion"),
    (
        "carry_expanded",
        "carry_equity_expanded_full.parquet",
        "carry_champion_expanded.json",
        "Carry champion (expanded universe)",
    ),
)

ADAPTIVE_MIX_RECEIPT = "adaptive_mix_20asset_1d_20260922.json"


def _canonical_json_bytes(value: Any) -> bytes:
    """Mirror ``quant_fund.utils.hashing.canonical_json_bytes``.

    Kept dependency-free so the script runs under the shared env without
    importing the package under test.
    """

    def canon(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): canon(sub) for key, sub in item.items()}
        if isinstance(item, (list, tuple)):
            return [canon(sub) for sub in item]
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, bytes):
            return item.hex()
        scalar = getattr(item, "item", None)
        if callable(scalar):
            try:
                return canon(scalar())
            except (TypeError, ValueError):
                pass
        return item

    return json.dumps(
        canon(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _build_hash_index() -> dict[str, str]:
    """Map sha256 digests to repo paths.

    Matches both raw file bytes and the canonical-JSON re-encoding of ``.json``
    files (the repo's seal convention), which lets the UI distinguish "hash
    resolves to a committed file" from "hash refers to sealed bytes not in this
    checkout".
    """

    matches: dict[str, str] = {}
    for root in HASH_SCAN_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                if path.stat().st_size > HASH_SCAN_MAX_BYTES:
                    continue
                raw = path.read_bytes()
            except OSError:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            matches.setdefault(hashlib.sha256(raw).hexdigest(), rel)
            if path.suffix == ".json":
                try:
                    canonical = hashlib.sha256(_canonical_json_bytes(json.loads(raw))).hexdigest()
                except (ValueError, UnicodeDecodeError):
                    continue
                matches.setdefault(canonical, rel + " (canonical-json)")
    return matches


def _key_is_hashy(key: str) -> bool:
    lowered = key.lower()
    return "sha256" in lowered or "hash" in lowered or lowered.endswith("_sha")


def _collect_hashes(value: Any, into: set[str], parent_hashy: bool = False) -> None:
    """Gather every 64-hex string reachable from a receipt.

    A 64-hex string counts as a digest field when its own key is hash-ish
    (``script_sha256``) or when it sits directly inside a hash-ish table
    (``input_hashes.<file>``, ``bar_files_sha256.<file>``) — mirroring the
    web app's ``extractHashFields`` so index counts match the UI.
    """

    if isinstance(value, dict):
        for key, sub in value.items():
            hashy = _key_is_hashy(key)
            if isinstance(sub, str) and len(sub) == 64 and (hashy or parent_hashy):
                into.add(sub)
            _collect_hashes(sub, into, hashy)
    elif isinstance(value, list):
        for sub in value:
            _collect_hashes(sub, into, parent_hashy)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _export_equity(parquet_path: Path, out_path: Path) -> dict[str, Any]:
    """Convert an equity parquet to a compact JSON columnar fixture."""

    import pandas as pd  # shared env provides pandas/pyarrow

    frame = pd.read_parquet(parquet_path)
    required = {"event_time", "nav", "gross", "net", "turnover"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{parquet_path.name} missing columns: {sorted(missing)}")
    frame = frame.sort_values("event_time").reset_index(drop=True)
    points = [
        [
            row.event_time.date().isoformat(),
            round(float(row.nav), 6),
            round(float(row.gross), 8),
            round(float(row.net), 8),
            round(float(row.turnover), 8),
        ]
        for row in frame.itertuples(index=False)
    ]
    payload = {
        "fields": ["date", "nav", "gross", "net", "turnover"],
        "points": points,
        "rows": len(points),
        "source_file": parquet_path.relative_to(REPO_ROOT).as_posix(),
        "units": {"nav": "USD", "gross": "fraction", "net": "fraction", "turnover": "fraction"},
    }
    _write_json(out_path, payload)
    return {
        "rows": len(points),
        "first_date": points[0][0],
        "last_date": points[-1][0],
        "nav_final": points[-1][1],
    }


def _normalize_carry_stats(raw: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Carry champion artifacts store dev/holdout/full segment blobs."""

    out: dict[str, dict[str, float]] = {}
    for segment in ("dev", "holdout", "full"):
        blob = raw.get(segment)
        if isinstance(blob, dict):
            out[segment] = {k: v for k, v in blob.items() if isinstance(v, (int, float))}
    return out


def _strategy_entry(
    strategy_id: str,
    name: str,
    *,
    kind: str,
    data_source: str,
    provenance: dict[str, Any],
    segments: dict[str, dict[str, float]],
    extras: dict[str, Any] | None = None,
    stats_file: str | None = None,
    equity_file: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Emit a normalized strategy stats fixture and return its index entry."""

    stats_rel = stats_file or f"strategies/{strategy_id}.json"
    stats_payload = {
        "id": strategy_id,
        "name": name,
        "kind": kind,
        "data_source": data_source,
        "research_only": True,
        "live_pnl_claim": False,
        "provenance": provenance,
        "segments": segments,
        "extras": extras or {},
        "equity_file": equity_file,
    }
    _write_json(FIXTURES_DIR / stats_rel, stats_payload)
    entry = {
        "id": strategy_id,
        "name": name,
        "kind": kind,
        "data_source": data_source,
        "stats_file": stats_rel,
        "equity_file": equity_file,
        "has_equity": equity_file is not None,
        "segments": sorted(segments.keys()),
    }
    return entry, stats_rel


def main() -> None:
    receipts_dir = REPO_ROOT / "receipts"
    artifacts_dir = REPO_ROOT / "artifacts"
    if not receipts_dir.is_dir():
        raise SystemExit("receipts/ missing — run from the repo checkout")

    generated_at = datetime.now(UTC).isoformat()
    hash_index = _build_hash_index()

    # ---- receipts: verbatim copies + index rows ---------------------------
    receipt_entries: list[dict[str, Any]] = []
    receipt_hash_matches: dict[str, str] = {}
    for src in sorted(receipts_dir.glob("*.json")):
        rel = f"receipts/{src.name}"
        dst = FIXTURES_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        payload = json.loads(src.read_text())
        hashes: set[str] = set()
        _collect_hashes(payload, hashes)
        for digest in hashes:
            if digest in hash_index:
                receipt_hash_matches[digest] = hash_index[digest]
        receipt_entries.append(
            {
                "id": src.stem,
                "file": rel,
                "schema": payload.get("schema"),
                "evidence_level": payload.get("evidence_level"),
                "research_only": payload.get("research_only"),
                "live_pnl_claim": payload.get("live_pnl_claim"),
                "created_at": payload.get("created_at") or payload.get("generated_at"),
                "n_hashes": len(hashes),
                "n_hash_matches": sum(1 for h in hashes if h in hash_index),
            }
        )

    # ---- strategies --------------------------------------------------------
    strategies: list[dict[str, Any]] = []

    for strategy_id, parquet_name, champion_name, display in EQUITY_SOURCES:
        parquet_path = artifacts_dir / parquet_name
        champion_path = artifacts_dir / champion_name
        if not parquet_path.is_file() or not champion_path.is_file():
            raise SystemExit(f"missing artifacts for {strategy_id}")
        equity_rel = f"equity/{strategy_id}.json"
        equity_meta = _export_equity(parquet_path, FIXTURES_DIR / equity_rel)
        raw_stats = json.loads(champion_path.read_text())
        segments = _normalize_carry_stats(raw_stats)
        entry, _ = _strategy_entry(
            strategy_id,
            display,
            kind="equity_backed",
            data_source="real",
            provenance={
                "source_files": [
                    f"artifacts/{champion_name}",
                    f"artifacts/{parquet_name}",
                ],
                "note": "Committed research artifact; simulated NAV path, "
                "not live trading performance.",
                "equity": equity_meta,
            },
            segments=segments,
            equity_file=equity_rel,
        )
        strategies.append(entry)

    adaptive_path = receipts_dir / ADAPTIVE_MIX_RECEIPT
    if adaptive_path.is_file():
        receipt = json.loads(adaptive_path.read_text())
        receipt_rel = f"receipts/{ADAPTIVE_MIX_RECEIPT}"
        results = receipt.get("results", {})
        for allocator_id in ("adaptive", "equal"):
            blob = results.get(allocator_id)
            if not isinstance(blob, dict):
                continue
            segments = {
                name: {k: v for k, v in seg.items() if isinstance(v, (int, float))}
                for name, seg in (blob.get("segments") or {}).items()
                if isinstance(seg, dict)
            }
            extras = {
                key: blob[key] for key in blob if key.startswith("full_path") and key != "segments"
            }
            entry, _ = _strategy_entry(
                f"amix_{allocator_id}",
                f"Adaptive mix replay — {allocator_id} allocator",
                kind="receipt_stats",
                data_source=str(receipt.get("data_source", "unknown")),
                provenance={
                    "receipt": receipt_rel,
                    "pointer": f"results.{allocator_id}",
                    "note": "Segment stats recorded in sealed receipt "
                    f"{ADAPTIVE_MIX_RECEIPT}; no equity series exported.",
                },
                segments=segments,
                extras=extras,
            )
            strategies.append(entry)
        sleeves = receipt.get("paper_sleeves", {})
        for sleeve in sorted(sleeves):
            blob = sleeves[sleeve]
            if not isinstance(blob, dict):
                continue
            segments = {
                name: {k: v for k, v in seg.items() if isinstance(v, (int, float))}
                for name, seg in blob.items()
                if isinstance(seg, dict)
            }
            entry, _ = _strategy_entry(
                f"amix_sleeve_{sleeve}",
                f"Adaptive mix sleeve — {sleeve}",
                kind="receipt_stats",
                data_source=str(receipt.get("data_source", "unknown")),
                provenance={
                    "receipt": receipt_rel,
                    "pointer": f"paper_sleeves.{sleeve}",
                    "note": "Paper-sleeve segment stats recorded in sealed "
                    "receipt; diagnostic only, no equity series exported.",
                },
                segments=segments,
            )
            strategies.append(entry)

    index = {
        "generated_at": generated_at,
        "generator": "web/scripts/export_fixtures.py",
        "repo_revision": _git_revision(),
        "honesty": {
            "research_only": True,
            "live_pnl_claim": False,
            "note": "All fixtures are verbatim copies or mechanical exports of "
            "committed research evidence. The app renders stored fields; it "
            "makes no performance or live-trading claims.",
        },
        "strategies": strategies,
        "receipts": receipt_entries,
        "hash_matches": receipt_hash_matches,
    }
    _write_json(FIXTURES_DIR / "index.json", index)
    print(
        f"exported {len(receipt_entries)} receipts, "
        f"{len(strategies)} strategies, "
        f"{len(receipt_hash_matches)} hash matches -> {FIXTURES_DIR}"
    )


if __name__ == "__main__":
    main()
