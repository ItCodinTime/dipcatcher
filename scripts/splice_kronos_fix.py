"""Splice corrected kronos_small (Tokenizer-base pairing) into eval shards.

The main d1_/h4_ runs scored kronos_small with Kronos-Tokenizer-2k, which is not
the upstream pairing for Kronos-small (README pairs it with Kronos-Tokenizer-base)
and degraded its forecasts ~3x. The d1fix_/h4fix_ runs re-scored kronos_small with
the canonical tokenizer under the same protocol/seed. This script:

 1. verifies challenger columns are identical between the two runs (determinism
    check — challengers use no run-level RNG);
 2. writes <shard>.fixed.npz = original matrix with the kronos_small column
    (CRPS + pinball) replaced by the corrected run's values;
 3. for every h4 shard, writes <shard>.cc.npz with dip_student_t removed and
    its NaN rows dropped — a uniform 13-model complete-case variant used for
    the headline 4h merge (dip_student_t fails on flat 4h windows on several
    assets, so it is excluded from the whole 4h comparison set).

Then merge the .fixed.npz (or .cc.npz where present) shards via the evaluator's
own --merge-parts path.

Usage: python splice_kronos_fix.py [shard_dir]   (default shown below)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(r"D:\dipcatcher\.dsh-24x7\eval-full")


def load(p: Path):
    z = np.load(p, allow_pickle=False)
    return {
        "crps": z["crps_matrix"],
        "pin": z["pinball_cube"],
        "aids": z["asset_ids"],
        "names": [str(x) for x in z["model_names"]],
        "meta": json.loads(str(z["meta_json"])),
    }


def save(out: Path, d: dict, names: list[str], meta_patch: dict | None = None):
    meta = dict(d["meta"])
    if meta_patch:
        meta.update(meta_patch)
    np.savez_compressed(
        out,
        crps_matrix=d["crps"],
        pinball_cube=d["pin"],
        asset_ids=d["aids"],
        model_names=np.asarray(names),
        meta_json=np.array(json.dumps(meta)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shard_dir", nargs="?", type=Path, default=DEFAULT_DIR)
    shard_dir = ap.parse_args().shard_dir
    report = []
    for main_npz in sorted(shard_dir.glob("*.losses.npz")):
        stem = main_npz.name[: -len(".losses.npz")]
        if not (stem.startswith("d1_") or stem.startswith("h4_")):
            continue
        tag = ("d1fix_" if stem.startswith("d1_") else "h4fix_") + stem.split("_", 1)[1]
        fix_npz = shard_dir / f"{tag}.losses.npz"
        if not fix_npz.exists():
            print(f"SKIP {stem}: no fix shard {fix_npz.name}")
            continue
        a, b = load(main_npz), load(fix_npz)
        assert a["crps"].shape[0] == b["crps"].shape[0], (stem, "row mismatch")

        ki = a["names"].index("kronos_small")
        kf = b["names"].index("kronos_small")
        # determinism check on challenger columns
        mism = []
        for name in a["names"]:
            if name in ("kronos_small",):
                continue
            if name not in b["names"]:
                mism.append(name + " (missing in fix)")
                continue
            j = b["names"].index(name)
            if not np.allclose(
                np.nan_to_num(a["crps"][:, a["names"].index(name)]),
                np.nan_to_num(b["crps"][:, j]),
                equal_nan=True,
            ):
                mism.append(name)
        if mism:
            print(f"WARN {stem}: challenger columns differ -> {mism} (splice anyway, flagging)")
        a["crps"][:, ki] = b["crps"][:, kf]
        a["pin"][:, ki, :] = b["pin"][:, kf, :]
        patch = {"kronos_small_corrected": "Kronos-Tokenizer-base pairing (canonical upstream)"}
        save(main_npz.with_name(stem + ".fixed.npz"), a, a["names"], patch)

        # complete-case variant: h4 shards always drop dip_student_t for a
        # uniform 13-model set (flat-window fit failures on several assets)
        nan_cols = [m for m in a["names"] if np.isnan(a["crps"][:, a["names"].index(m)]).any()]
        if stem.startswith("h4_") and "dip_student_t" in a["names"]:
            drop = set(nan_cols) - {"dip_student_t"}
            if drop:
                print(f"WARN {stem}: extra NaN columns {drop}; not producing cc variant")
            else:
                idx = [a["names"].index(m) for m in a["names"] if m != "dip_student_t"]
                cr = a["crps"][:, idx]
                pn = a["pin"][:, idx, :]
                ok = np.isfinite(cr).all(axis=1) & np.isfinite(pn).all(axis=(1, 2))
                d2 = {"crps": cr[ok], "pin": pn[ok], "aids": a["aids"][ok], "meta": a["meta"]}
                save(
                    main_npz.with_name(stem + ".cc.npz"),
                    d2,
                    [a["names"][i] for i in idx],
                    {"complete_case_drop": ["dip_student_t"],
                     "note": "dip_student_t fails on flat 4h windows; excluded from comparison set"},
                )
                report.append((stem, "cc", int(ok.sum()), len(ok), nan_cols))
        report.append((stem, "fixed", int(np.isfinite(a["crps"][:, ki]).sum()), a["crps"].shape[0], []))
    for r in report:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
