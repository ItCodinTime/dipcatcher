"""Mega-arena: splice every lane column onto the canonical v2aug shards.

Discovers column artifacts under .dsh-24x7/lane-*/ by reading each npz's
meta_json — a file is a column artifact iff it carries ``crps_col`` and a
``meta.model`` in NEW_MODELS bound (via meta.shard) to one of the ten canonical
v2aug shards. Splices all discovered columns onto each shard in a fixed
model order, writes mega shards to .dsh-24x7/mega-arena/, then merges
daily + 4h via sota_eval_kronos.merge_parts semantics (invoked via CLI).

Usage: .venv/bin/python scripts/_mega_arena.py [--models m1,m2,...]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL_FULL = ROOT / ".dsh-24x7" / "eval-full"
OUT = ROOT / ".dsh-24x7" / "mega-arena"
SPLICE = ROOT / "scripts" / "splice_challenger_column.py"
EVAL = ROOT / "scripts" / "sota_eval_kronos.py"
PY = str(ROOT / ".venv" / "bin" / "python")

NEW_MODELS = [
    "dip_egarch",
    "dip_egarch_l",
    "dip_evt",
    "dip_seas",
    "dip_aci",
    "dip_har",
    "dip_kde",
    "dip_volm",
    "dip_xbeta",
    "dip_lgbm_qv",
    "dip_mid",
    "dip_stack2",
]

SHARD_KEYS = [
    "d1_bnbusdt_1d_deep",
    "d1_btcusdt_1d_deep",
    "d1_ethusdt_1d_deep",
    "d1_solusdt_1d_deep",
    "d1_xrpusdt_1d_deep",
    "h4f_bnbusdt_4h_deep",
    "h4f_btcusdt_4h_deep",
    "h4f_ethusdt_4h_deep",
    "h4f_solusdt_4h_deep",
    "h4f_xrpusdt_4h_deep",
]


def discover() -> dict[str, dict[str, Path]]:
    """{shard_key: {model: col_path}} for every lane column artifact."""
    found: dict[str, dict[str, Path]] = {k: {} for k in SHARD_KEYS}
    for npz in sorted(ROOT.glob(".dsh-24x7/lane-*/**/*.npz")):
        if "spliced" in npz.parts or "mega-arena" in npz.parts:
            continue
        try:
            with np.load(npz, allow_pickle=False) as z:
                if "crps_col" not in z.files or "meta_json" not in z.files:
                    continue
                meta = json.loads(str(z["meta_json"]))
        except Exception:
            continue
        model = str(meta.get("model", ""))
        shard_ref = str(meta.get("shard", ""))
        if model not in NEW_MODELS:
            continue
        for key in SHARD_KEYS:
            if key in shard_ref or key in npz.name:
                if model in found[key]:
                    print(f"WARN duplicate {model} for {key}: {npz} vs {found[key][model]}")
                else:
                    found[key][model] = npz
                break
    return found


def main() -> int:
    only = None
    if "--models" in sys.argv:
        i = sys.argv.index("--models")
        only = set(sys.argv[i + 1].split(","))
    found = discover()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spliced").mkdir(exist_ok=True)
    mega_shards = {"d1": [], "h4f": []}
    for key in SHARD_KEYS:
        src = EVAL_FULL / f"{key}.v2aug.npz"
        if not src.is_file():
            print(f"MISSING shard {src}")
            continue
        cols = [found[key][m] for m in NEW_MODELS if m in found[key] and (not only or m in only)]
        missing = [m for m in NEW_MODELS if m not in found[key]]
        out = OUT / "spliced" / f"{key}.mega.npz"
        if cols:
            cmd = [PY, str(SPLICE), "--shard", str(src), "--cols",
                   *[str(c) for c in cols], "--out", str(out)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                print(f"SPLICE FAIL {key}: {r.stdout}{r.stderr}")
                continue
            print(f"{key}: +{len(cols)} cols (missing: {missing or 'none'})")
        else:
            print(f"{key}: no cols found")
            continue
        mega_shards["d1" if key.startswith("d1") else "h4f"].append(str(out))
    for cell, parts in mega_shards.items():
        if not parts:
            continue
        receipt = OUT / f"merge_{cell}.json"
        cmd = [PY, str(EVAL), "--merge-parts", *parts, "--merge-out", str(receipt),
               "--bars-root", "data/raw/sources"]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if r.returncode:
            print(f"MERGE FAIL {cell}: {r.stdout[-800:]}{r.stderr[-800:]}")
        else:
            print(f"merged {cell} -> {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
