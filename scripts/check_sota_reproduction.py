"""Compare a recomputed SOTA merge receipt against its committed reference.

Usage:
    python scripts/check_sota_reproduction.py \
        --recomputed /tmp/merged_d1.json \
        --reference .dsh-24x7/native/MERGED_d1_native.json

Reproduction means identical: every key, string, int, bool, and float must
match exactly (floats may pass under --float-rel-tol, reported as warnings —
a looser claim than exact reproduction). Volatile fields (timestamps,
implementation hashes, per-machine path spellings) are normalized before
comparison and reported separately so drift there is visible but not
conflated with a data failure. Exits 1 on any mismatch — fail closed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Fields that legitimately differ between runs without weakening the claim.
VOLATILE_EXACT_KEYS = {"created_at", "generated_at"}
# Fields reported as warnings only: implementation hashes change when the
# eval code changes (correct — the receipt is honest about it), and
# losses_file is the writer's chosen output name.
WARN_ONLY_KEYS = {"implementation_sha256", "losses_file", "losses_sha256", "timestamp_source"}
# Path-spelling normalization: absolute/relative/Windows paths all spell the
# same committed bytes; compare these maps by basename -> digest instead.
BASENAME_KEYED_MAPS = {"source_parts_sha256"}


def _norm(value: Any, key: str) -> Any:
    if key in BASENAME_KEYED_MAPS and isinstance(value, dict):
        return {Path(str(k).replace("\\", "/")).name: v for k, v in value.items()}
    if key == "parts" and isinstance(value, list):
        return sorted(Path(str(p).replace("\\", "/")).name for p in value)
    return value


def _compare(
    recomputed: Any,
    reference: Any,
    path: str,
    tol: float,
    mismatches: list[str],
    warnings: list[str],
) -> None:
    key = path.rsplit(".", 1)[-1]
    if key in VOLATILE_EXACT_KEYS:
        return
    recomputed, reference = _norm(recomputed, key), _norm(reference, key)
    if key in WARN_ONLY_KEYS:
        if recomputed != reference:
            warnings.append(f"{path}: {reference!r} -> {recomputed!r}")
        return
    if isinstance(recomputed, float) or isinstance(reference, float):
        try:
            rf, bf = float(recomputed), float(reference)
        except (TypeError, ValueError):
            mismatches.append(
                f"{path}: type {type(reference).__name__} -> {type(recomputed).__name__}"
            )
            return
        if math.isnan(rf) and math.isnan(bf):
            return
        if rf == bf:
            return
        if (
            tol > 0
            and math.isfinite(rf)
            and math.isfinite(bf)
            and math.isclose(rf, bf, rel_tol=tol)
        ):
            warnings.append(f"{path}: {bf!r} ~ {rf!r} (within rel_tol={tol})")
            return
        mismatches.append(f"{path}: {bf!r} -> {rf!r}")
        return
    if type(recomputed) is not type(reference):
        mismatches.append(f"{path}: type {type(reference).__name__} -> {type(recomputed).__name__}")
        return
    if isinstance(recomputed, dict):
        for k in sorted(set(recomputed) | set(reference)):
            if k not in recomputed:
                mismatches.append(f"{path}.{k}: missing in recomputed")
            elif k not in reference:
                mismatches.append(f"{path}.{k}: missing in reference")
            else:
                _compare(recomputed[k], reference[k], f"{path}.{k}", tol, mismatches, warnings)
        return
    if isinstance(recomputed, list):
        if len(recomputed) != len(reference):
            mismatches.append(f"{path}: len {len(reference)} -> {len(recomputed)}")
            return
        for i, (r, b) in enumerate(zip(recomputed, reference, strict=True)):
            _compare(r, b, f"{path}[{i}]", tol, mismatches, warnings)
        return
    if recomputed != reference:
        mismatches.append(f"{path}: {reference!r} -> {recomputed!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recomputed", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument(
        "--float-rel-tol",
        type=float,
        default=0.0,
        help="Relative tolerance for floats (default 0 = bit-exact reproduction)",
    )
    args = ap.parse_args()
    recomputed = json.loads(args.recomputed.read_text())
    reference = json.loads(args.reference.read_text())
    mismatches: list[str] = []
    warnings: list[str] = []
    _compare(recomputed, reference, "$", args.float_rel_tol, mismatches, warnings)
    for w in warnings:
        print(f"warning: {w}")
    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatch(es)")
        for m in mismatches[:50]:
            print(f"  {m}")
        return 1
    print(f"reproduction verified: {args.reference} == {args.recomputed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
