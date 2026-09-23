"""Splice canonical-tokenizer Kronos forecasts into matched evaluation shards.

All pairs are validated before any output is written. Challenger CRPS and
pinball arrays must match exactly, including missing-value locations; bars,
asset identities, and evaluation settings must also agree. Corrected model
and tokenizer hashes replace the original hashes, and source-file hashes bind
each transformation to its inputs. Legacy shards have no origin timestamps:
their positional alignment is checked, but is not presented as timestamp proof.

Four-hour ``.cc.npz`` outputs exclude ``dip_student_t`` uniformly and disclose
coverage and any subsequent complete-case row filtering. They retain the
correction's provenance.

Use ``--output-dir`` to preserve historical outputs, or ``--check-only`` to
validate the complete fleet without writing anything.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DIR = Path(r"D:\dipcatcher\.dsh-24x7\eval-full")
TARGET = "kronos_small"
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
    crps, pin, aids, names = data["crps"], data["pin"], data["aids"], data["names"]
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
    asset_names = meta.get("asset_names")
    if not isinstance(asset_names, list) or not asset_names:
        raise ValueError(f"{path}: missing asset names")
    if (
        aids.shape != (crps.shape[0],)
        or aids.dtype.kind not in "iu"
        or np.any(aids < 0)
        or np.any(aids >= len(asset_names))
    ):
        raise ValueError(f"{path}: invalid asset identities")
    targets = meta.get("targets")
    if (
        not isinstance(targets, list)
        or TARGET not in targets
        or len(set(targets)) != len(targets)
        or not set(targets).issubset(names)
    ):
        raise ValueError(f"{path}: invalid target declarations")
    for key in ("bars_sha256", "artifact_sha256"):
        values = meta.get(key)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"{path}: missing {key}")
        if any(
            not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in values.values()
        ):
            raise ValueError(f"{path}: invalid {key}")
    if not isinstance(meta.get("transformations", []), list):
        raise ValueError(f"{path}: invalid transformation history")
    return data


def splice_pair(original: Path, corrected: Path) -> dict[str, Any]:
    """Validate a pair and return a corrected copy without writing files."""
    a, b = load(original), load(corrected)
    am, bm = a["meta"], b["meta"]
    if a["crps"].shape[0] != b["crps"].shape[0]:
        raise ValueError(f"{original.name}: row count mismatch")
    if am["asset_names"] != bm["asset_names"] or not np.array_equal(a["aids"], b["aids"]):
        raise ValueError(f"{original.name}: asset identity mismatch")
    if am["bars_sha256"] != bm["bars_sha256"]:
        raise ValueError(f"{original.name}: bars provenance mismatch")
    for field in PROTOCOL_FIELDS:
        if field not in am["config"] or field not in bm["config"]:
            raise ValueError(f"{original.name}: missing protocol field {field}")
        if am["config"][field] != bm["config"][field]:
            raise ValueError(f"{original.name}: protocol mismatch: {field}")

    challengers = [name for name in a["names"] if name not in am["targets"]]
    corrected_challengers = [name for name in b["names"] if name not in bm["targets"]]
    if not challengers or set(challengers) != set(corrected_challengers):
        raise ValueError(f"{original.name}: challenger set mismatch")
    for name in challengers:
        ai, bi = a["names"].index(name), b["names"].index(name)
        for field in ("crps", "pin"):
            if not np.array_equal(a[field][:, ai], b[field][:, bi], equal_nan=True):
                raise ValueError(
                    f"{original.name}: {name} {field} differs (exact alignment required)"
                )

    artifact_keys = (TARGET, f"{TARGET}_tokenizer")
    for key in artifact_keys:
        if key not in am["artifact_sha256"] or key not in bm["artifact_sha256"]:
            raise ValueError(f"{original.name}: missing artifact identity: {key}")
    if am["artifact_sha256"][TARGET] != bm["artifact_sha256"][TARGET]:
        raise ValueError(f"{original.name}: corrected model weights differ")

    ai, bi = a["names"].index(TARGET), b["names"].index(TARGET)
    a["crps"][:, ai] = b["crps"][:, bi]
    a["pin"][:, ai] = b["pin"][:, bi]
    prior_hashes = {key: am["artifact_sha256"][key] for key in artifact_keys}
    corrected_hashes = {key: bm["artifact_sha256"][key] for key in artifact_keys}
    am["artifact_sha256"].update(corrected_hashes)
    am["kronos_small_corrected"] = "Kronos-Tokenizer-base pairing (canonical upstream)"
    am.setdefault("transformations", []).append(
        {
            "operation": "replace_model_forecasts",
            "model": TARGET,
            "inputs": {
                "original": {"path": str(original), "sha256": _sha256(original)},
                "corrected": {"path": str(corrected), "sha256": _sha256(corrected)},
            },
            "previous_artifact_sha256": prior_hashes,
            "replacement_artifact_sha256": corrected_hashes,
            "alignment": {
                "challengers": challengers,
                "loss_comparison": "exact CRPS and pinball equality, including NaN placement",
                "bars_asset_ids_protocol_equal": True,
                "origin_timestamps_verified": False,
                "note": "Legacy positional rows; source shards contain no origin timestamps.",
            },
        }
    )
    return a


def complete_case(data: dict[str, Any]) -> dict[str, Any]:
    """Exclude Student-t and retain the fixed shard's complete provenance."""
    out = copy.deepcopy(data)
    if "dip_student_t" not in out["names"]:
        raise ValueError("four-hour shard has no dip_student_t to exclude")
    coverage = {
        name: int(
            (np.isfinite(out["crps"][:, i]) & np.isfinite(out["pin"][:, i]).all(axis=1)).sum()
        )
        for i, name in enumerate(out["names"])
    }
    keep = [i for i, name in enumerate(out["names"]) if name != "dip_student_t"]
    crps, pin = out["crps"][:, keep], out["pin"][:, keep]
    finite = np.isfinite(crps).all(axis=1) & np.isfinite(pin).all(axis=(1, 2))
    if not finite.any():
        raise ValueError("four-hour subset has no complete observations")
    out["crps"], out["pin"], out["aids"] = crps[finite], pin[finite], out["aids"][finite]
    out["names"] = [out["names"][i] for i in keep]
    out["meta"]["complete_case_drop"] = ["dip_student_t"]
    out["meta"].setdefault("transformations", []).append(
        {
            "operation": "complete_case_model_subset",
            "excluded_models": ["dip_student_t"],
            "reason": "Uniform four-hour comparison excludes historical Student-t fit failures.",
            "n_rows_input": int(len(finite)),
            "n_rows_output": int(finite.sum()),
            "n_rows_dropped": int((~finite).sum()),
            "model_scored_counts_before_exclusion": coverage,
            "retained_input_row_indices": np.flatnonzero(finite).tolist(),
        }
    )
    return out


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
    parser.add_argument("shard_dir", nargs="?", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.shard_dir
    prepared = []
    for original in sorted(args.shard_dir.glob("*.losses.npz")):
        stem = original.name[: -len(".losses.npz")]
        if not stem.startswith(("d1_", "h4_")) or stem in ("d1_merged", "h4_merged"):
            continue
        prefix = "d1fix_" if stem.startswith("d1_") else "h4fix_"
        corrected = args.shard_dir / f"{prefix}{stem.split('_', 1)[1]}.losses.npz"
        if not corrected.exists():
            raise ValueError(f"{original.name}: missing corrected shard {corrected.name}")
        fixed = splice_pair(original, corrected)
        prepared.append((output_dir / f"{stem}.fixed.npz", fixed))
        if stem.startswith("h4_"):
            prepared.append((output_dir / f"{stem}.cc.npz", complete_case(fixed)))
    if not prepared:
        raise ValueError(f"{args.shard_dir}: no matched evaluation shards")
    if not args.check_only:
        output_dir.mkdir(parents=True, exist_ok=True)
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
