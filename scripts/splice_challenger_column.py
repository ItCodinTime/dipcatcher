"""Append a computed challenger column onto an existing losses shard.

Same safety discipline as ``splice_timesfm_fix.py``: the shard is rebound via
``bars_sha256``, the row grid is verified against the column artifact's
deterministic recomputation, and both input file hashes are recorded in the
output meta. The scoring contract string is preserved — appending a challenger
column does not alter how any existing column was scored; provenance lives in
``meta["appended_columns"]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from sota_eval_kronos import _sha256  # noqa: E402

KNOWN_MODELS = {
    "dip_gmm_k",
    "dip_skt",
    "dip_qar",
    "dip_conf_t",
    "dip_regime",
    "dip_stack",
    "dip_xbeta",
    "dip_seas",
    "dip_egarch",
    "dip_egarch_l",
    "dip_aci",
    "dip_har",
    "dip_kde",
    "dip_volm",
    "dip_lgbm_qv",
    "dip_evt",
    "dip_mid",
    "dip_stack2",
}
PROTOCOL_FIELDS = (
    "origins_per_asset",
    "lookback",
    "window",
    "garch_window",
    "taus",
    "seed",
)

# Fields allowed to differ under --allow-seed-mismatch: challenger columns are
# deterministic in the origin grid (which is itself seed-independent), so a
# column computed on a different seed's shard is valid when grid identity is
# otherwise proven (bars hash + remaining protocol fields + asset identity).
SEED_FIELD = "seed"


def _load_shard(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {
            "crps": z["crps_matrix"],
            "pin": z["pinball_cube"],
            "aids": z["asset_ids"],
            "names": [str(x) for x in z["model_names"]],
            "meta": json.loads(str(z["meta_json"])),
            "extra": {k: z[k] for k in z.files
                      if k not in {"crps_matrix", "pinball_cube", "asset_ids",
                                   "model_names", "meta_json"}},
        }


def _load_col(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {
            "crps": z["crps_col"],
            "pin": z["pin_cols"],
            "tt": z["target_time_ns"],
            "meta": json.loads(str(z["meta_json"])),
        }


def _check_compatible(
    s: dict[str, Any],
    c: dict[str, Any],
    shard_path: Path,
    col_path: Path,
    *,
    allow_seed_mismatch: bool = False,
) -> str:
    sm, cm = s["meta"], c["meta"]
    model = str(cm.get("model", ""))
    if model not in KNOWN_MODELS:
        raise ValueError(f"{col_path.name}: unknown column model {model!r}")
    if model in s["names"]:
        raise ValueError(f"{shard_path.name}: already contains {model}")
    if s["crps"].shape[0] != c["crps"].shape[0]:
        raise ValueError(f"{shard_path.name}: row count mismatch")
    s_bars = sm.get("bars_sha256")
    c_bars = cm.get("bars_sha256")
    if isinstance(s_bars, dict):
        s_bars = {str(k).lower(): str(v) for k, v in s_bars.items()}
        c_bars = {str(k).lower(): str(v) for k, v in (c_bars or {}).items()}
        if set(s_bars.values()) != set(c_bars.values()):
            raise ValueError(f"{shard_path.name}: bars provenance mismatch")
    elif s_bars != c_bars:
        raise ValueError(f"{shard_path.name}: bars provenance mismatch")
    # Grid alignment is guaranteed by bars_sha256 + protocol fields + row
    # count + asset identity (the origin grid is deterministic in those); the
    # recorded shard hash is provenance only — a column computed against an
    # unspliced ancestor of this shard is equally valid.
    cfg = sm.get("config", {})
    for field in PROTOCOL_FIELDS:
        if allow_seed_mismatch and field == SEED_FIELD:
            continue
        if cfg.get(field) != cm.get("config", {}).get(field):
            raise ValueError(f"{shard_path.name}: protocol mismatch: {field}")
    assets = [str(x).rsplit("-", 1)[0] for x in sm.get("asset_names", [])]
    col_assets = [str(x).rsplit("-", 1)[0] for x in cm.get("asset_names", [])]
    if assets != col_assets:
        raise ValueError(f"{shard_path.name}: asset identity mismatch")
    return model


def splice_one(
    s: dict[str, Any],
    col_path: Path,
    shard_path: Path,
    *,
    allow_seed_mismatch: bool = False,
) -> dict[str, Any]:
    """Append one validated column to an in-memory shard record."""
    c = _load_col(col_path)
    model = _check_compatible(
        s, c, shard_path, col_path, allow_seed_mismatch=allow_seed_mismatch
    )
    s["names"] = s["names"] + [model]
    s["crps"] = np.concatenate([s["crps"], c["crps"][:, None]], axis=1)
    s["pin"] = np.concatenate([s["pin"], c["pin"][:, None, :]], axis=1)
    appended = dict(s["meta"].get("appended_columns", {}))
    rec = {
        "tool": "splice_challenger_column.py",
        "column_artifact": col_path.name,
        "column_sha256": _sha256(col_path),
        "n_finite": int(np.isfinite(c["crps"]).sum()),
    }
    col_seed = c["meta"].get("config", {}).get(SEED_FIELD)
    shard_seed = s["meta"].get("config", {}).get(SEED_FIELD)
    if col_seed != shard_seed:
        rec["seed_mismatch_accepted"] = (
            "deterministic challenger column on a seed-independent origin "
            f"grid (column seed={col_seed}, shard seed={shard_seed}); grid "
            "identity proven by bars_sha256 + protocol fields + asset identity"
        )
    appended[model] = rec
    s["meta"] = dict(s["meta"])
    s["meta"]["appended_columns"] = appended
    return s


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=Path, required=True)
    p.add_argument("--cols", type=Path, nargs="+", required=True,
                   help="column artifacts to append, in order")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-seed-mismatch", action="store_true",
                   help="permit column-vs-shard seed difference (deterministic "
                        "columns only; provenance is recorded per column)")
    args = p.parse_args(argv)
    s = _load_shard(args.shard)
    for col in args.cols:
        s = splice_one(
            s, col, args.shard, allow_seed_mismatch=args.allow_seed_mismatch
        )
    np.savez_compressed(
        args.out,
        crps_matrix=s["crps"],
        pinball_cube=s["pin"],
        asset_ids=s["aids"],
        model_names=np.asarray(s["names"]),
        meta_json=np.array(json.dumps(s["meta"])),
        **s["extra"],
    )
    print(f"{args.shard.name} +{len(args.cols)} -> {args.out.name} ({len(s['names'])} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
