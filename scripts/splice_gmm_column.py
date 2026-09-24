"""Append a computed ``dip_gmm_k`` column onto an existing losses shard.

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

MODEL = "dip_gmm_k"
PROTOCOL_FIELDS = (
    "origins_per_asset",
    "lookback",
    "window",
    "garch_window",
    "taus",
    "seed",
)


def _load_shard(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {
            "crps": z["crps_matrix"],
            "pin": z["pinball_cube"],
            "aids": z["asset_ids"],
            "names": [str(x) for x in z["model_names"]],
            "meta": json.loads(str(z["meta_json"])),
            "extra": {
                k: z[k]
                for k in z.files
                if k not in {"crps_matrix", "pinball_cube", "asset_ids", "model_names", "meta_json"}
            },
        }


def _load_col(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {
            "crps": z["crps_col"],
            "pin": z["pin_cols"],
            "tt": z["target_time_ns"],
            "meta": json.loads(str(z["meta_json"])),
        }


def splice(shard_path: Path, col_path: Path) -> dict[str, Any]:
    s = _load_shard(shard_path)
    c = _load_col(col_path)
    sm, cm = s["meta"], c["meta"]
    if cm.get("model") != MODEL:
        raise ValueError(f"{col_path.name}: not a {MODEL} column artifact")
    if MODEL in s["names"]:
        raise ValueError(f"{shard_path.name}: already contains {MODEL}")
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
    if str(cm.get("shard_sha256")) != _sha256(shard_path):
        raise ValueError(f"{shard_path.name}: column was computed for a different shard")
    cfg = sm.get("config", {})
    for field in PROTOCOL_FIELDS:
        if cfg.get(field) != cm.get("config", {}).get(field):
            raise ValueError(f"{shard_path.name}: protocol mismatch: {field}")
    assets = [str(x).rsplit("-", 1)[0] for x in sm.get("asset_names", [])]
    col_assets = [str(x).rsplit("-", 1)[0] for x in cm.get("asset_names", [])]
    if assets != col_assets:
        raise ValueError(f"{shard_path.name}: asset identity mismatch")

    names = s["names"] + [MODEL]
    crps = np.concatenate([s["crps"], c["crps"][:, None]], axis=1)
    pin = np.concatenate([s["pin"], c["pin"][:, None, :]], axis=1)
    meta = dict(sm)
    appended = dict(meta.get("appended_columns", {}))
    appended[MODEL] = {
        "tool": "splice_gmm_column.py",
        "column_artifact": col_path.name,
        "column_sha256": _sha256(col_path),
        "shard_sha256_at_splice": _sha256(shard_path),
        "n_finite": int(np.isfinite(c["crps"]).sum()),
    }
    meta["appended_columns"] = appended
    return {
        "crps": crps,
        "pin": pin,
        "aids": s["aids"],
        "names": names,
        "meta": meta,
        "extra": s["extra"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pairs",
        type=Path,
        nargs="+",
        required=True,
        help="shard.npz col.gmmk.npz pairs (even count)",
    )
    p.add_argument("--suffix", default=".withgmm.npz")
    args = p.parse_args(argv)
    if len(args.pairs) % 2:
        p.error("--pairs expects shard/column file pairs")
    for i in range(0, len(args.pairs), 2):
        shard, col = args.pairs[i], args.pairs[i + 1]
        out_data = splice(shard, col)
        out = shard.with_suffix("")  # strip .npz
        out = (
            Path(str(out)[: -len(".losses")] + args.suffix)
            if str(out).endswith(".losses")
            else Path(str(shard) + args.suffix)
        )
        np.savez_compressed(
            out,
            crps_matrix=out_data["crps"],
            pinball_cube=out_data["pin"],
            asset_ids=out_data["aids"],
            model_names=np.asarray(out_data["names"]),
            meta_json=np.array(json.dumps(out_data["meta"])),
            **out_data["extra"],
        )
        print(f"{shard.name} + {col.name} -> {out.name} ({len(out_data['names'])} cols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
