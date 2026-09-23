"""Splice contract-v2 TimesFM columns into matched evaluation shards.

Motivation: upstream ``TimesFM_2p5`` ``full_forecast`` channels are
``[q50, q10, q20, q30, q40, point, q60, q70, q80, q90]`` — channel 5 is the
point forecast. The legacy adapter scored ``q[:9]`` (drops q90, injects the
point as a pseudo-quantile), handicapping TimesFM in every remote-produced
shard. Corrected shards are timesfm-only reruns under the identical protocol,
bars, and seed.

All pairs are validated before any output is written: row count, asset
identity, bar hashes, and the full protocol tuple must match. Only the
``timesfm`` column is replaced; every other column is carried over untouched
and the splice records the source-file hashes of both inputs. Legacy shards
have no origin timestamps: positional alignment is checked, not presented as
timestamp proof.

Use ``--check-only`` to validate the fleet without writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

TARGET = "timesfm"
PROTOCOL_FIELDS = (
    "origins_per_asset",
    "samples_per_origin",
    "lookback",
    "window",
    "garch_window",
    "seed",
    "taus",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        data = {
            "crps": z["crps_matrix"],
            "pin": z["pinball_cube"],
            "aids": z["asset_ids"],
            "names": [str(x) for x in z["model_names"]],
            "meta": json.loads(str(z["meta_json"])),
        }
    crps, pin, names = data["crps"], data["pin"], data["names"]
    if crps.ndim != 2 or crps.shape[0] == 0 or crps.shape[1] != len(names):
        raise ValueError(f"{path}: invalid CRPS matrix shape")
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{path}: model names must be nonempty and unique")
    meta = data["meta"]
    if not isinstance(meta, dict) or not isinstance(meta.get("config"), dict):
        raise ValueError(f"{path}: missing evaluation metadata/config")
    taus = meta["config"].get("taus")
    if not isinstance(taus, list) or not taus or pin.shape != (*crps.shape, len(taus)):
        raise ValueError(f"{path}: pinball shape does not match CRPS and taus")
    if TARGET not in names:
        raise ValueError(f"{path}: no {TARGET} column")
    asset_names = meta.get("asset_names")
    if not isinstance(asset_names, list) or not asset_names:
        raise ValueError(f"{path}: missing asset names")
    return data


def splice_pair(original: Path, corrected: Path) -> dict[str, Any]:
    """Validate a pair and return a corrected copy without writing files."""
    a, b = load(original), load(corrected)
    am, bm = a["meta"], b["meta"]
    if a["crps"].shape[0] != b["crps"].shape[0]:
        raise ValueError(f"{original.name}: row count mismatch")
    # Merged h4 shards relabel assets as `X-4h`; the tfmfix shard is
    # pre-relabel. Strip the suffix before comparing (bars_sha256 still binds
    # the actual input file).
    norm = lambda xs: [str(x).rsplit("-", 1)[0] for x in xs]  # noqa: E731
    if norm(am["asset_names"]) != norm(bm["asset_names"]) or not np.array_equal(
        a["aids"], b["aids"]
    ):
        raise ValueError(f"{original.name}: asset identity mismatch")
    if am["bars_sha256"] != bm["bars_sha256"]:
        raise ValueError(f"{original.name}: bars provenance mismatch")
    for field in PROTOCOL_FIELDS:
        if am["config"].get(field) != bm["config"].get(field):
            raise ValueError(f"{original.name}: protocol mismatch: {field}")
    # The corrected shard is a timesfm-only rerun; its artifact hash binds the
    # same model weights (equal) — the *adapter* is what changed.
    if (
        am["artifact_sha256"].get(TARGET) != bm["artifact_sha256"].get(TARGET)
        and bm["artifact_sha256"].get(TARGET)
    ):
        raise ValueError(f"{original.name}: corrected model weights differ")

    # Pure-deterministic shared columns must be bit-identical — that is what
    # proves the rows are the same origins. Optimizer-dependent columns
    # (student_t/garch/fhs/lgbm/blend) may legitimately differ across fitter
    # versions; they are reported, not required.
    DETERMINISTIC = {"dip_gauss", "dip_ewma_t", "dip_empirical", "dip_empirical_long",
                     "dip_ewma_emp"}
    shared = [
        n for n in a["names"] if n in b["names"] and n != TARGET and n not in am.get("targets", [])
    ]
    diffs: dict[str, float] = {}
    for name in shared:
        ai0, bi0 = a["names"].index(name), b["names"].index(name)
        crps_eq = np.array_equal(a["crps"][:, ai0], b["crps"][:, bi0], equal_nan=True)
        pin_eq = np.array_equal(a["pin"][:, ai0], b["pin"][:, bi0], equal_nan=True)
        if name in DETERMINISTIC and not (crps_eq and pin_eq):
            raise ValueError(f"{original.name}: {name} differs — origins not aligned")
        if not (crps_eq and pin_eq):
            with np.errstate(invalid="ignore"):
                diffs[name] = float(
                    np.nanmax(np.abs(a["crps"][:, ai0] - b["crps"][:, bi0]))
                )

    ai, bi = a["names"].index(TARGET), b["names"].index(TARGET)
    if np.array_equal(a["crps"][:, ai], b["crps"][:, bi], equal_nan=True):
        raise ValueError(f"{original.name}: corrected column identical; nothing to fix")
    a["crps"][:, ai] = b["crps"][:, bi]
    a["pin"][:, ai] = b["pin"][:, bi]
    am.setdefault("transformations", []).append(
        {
            "operation": "replace_model_forecasts",
            "model": TARGET,
            "reason": (
                "contract v2: upstream full_forecast ch5 is the point forecast; "
                "legacy adapter included it and dropped a true decile"
            ),
            "inputs": {
                "original": {"path": str(original), "sha256": _sha256(original)},
                "corrected": {"path": str(corrected), "sha256": _sha256(corrected)},
            },
            "alignment": {
                "bars_asset_ids_protocol_equal": True,
                "shared_columns_bit_identical": [
                    n for n in shared if n not in diffs
                ],
                "optimizer_dependent_column_max_abs_diff": diffs,
                "origin_timestamps_verified": False,
                "note": "Legacy positional rows; source shards contain no origin timestamps.",
            },
        }
    )
    am["timesfm_corrected"] = "contract v2: np.delete(full_forecast, 5)"
    # Other target adapters were verified equivalent across checkouts
    # (kronos_samples identical; chronos2/bolt equivalent reshapes), so after
    # this splice the shard satisfies the v2 scoring contract for all targets.
    am["scoring_contract"] = "native_shapes_timesfm_point_first.v2+spliced"
    return a


def save(path: Path, data: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        crps_matrix=data["crps"],
        pinball_cube=data["pin"],
        asset_ids=data["aids"],
        model_names=np.asarray(data["names"]),
        meta_json=np.array(json.dumps(data["meta"])),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir", type=Path)
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        metavar="ORIG=TFMFIX",
        help="explicit original=corrected filename pairs (no implicit globbing)",
    )
    parser.add_argument("--suffix", default=".tfmv2.npz")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    prepared = []
    for pair in args.pairs:
        orig_s, _, corr_s = pair.partition("=")
        original, corrected = args.shard_dir / orig_s, args.shard_dir / corr_s
        if not original.exists() or not corrected.exists():
            raise ValueError(f"missing pair member: {pair}")
        fixed = splice_pair(original, corrected)
        for ext in (".losses.npz", ".fixed.npz", ".npz"):
            if original.name.endswith(ext):
                stem = original.name[: -len(ext)]
                break
        out = original.with_name(stem + args.suffix)
        prepared.append((out, fixed))
    if not args.check_only:
        for path, data in prepared:
            save(path, data)
    for path, data in prepared:
        print(
            f"{'CHECKED' if args.check_only else 'WROTE'} {path.name}: "
            f"rows={len(data['aids'])} models={len(data['names'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
