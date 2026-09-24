import glob
import json
import os

for f in sorted(glob.glob(r"D:\dipcatcher\.dsh-24x7\eval-full\*fix_*.json")):
    r = json.load(open(f))
    tag = os.path.basename(f).replace(".json", "")
    ks = r["mean_crps_pooled"].get("kronos_small")
    dips = {k: v for k, v in r["mean_crps_pooled"].items() if k.startswith("dip_")}
    worst_dip = max(dips.values())
    print(
        tag.ljust(28),
        "kronos_small=",
        round(ks, 5),
        "| worst dip=",
        round(worst_dip, 5),
        "| kronos beats worst dip?",
        ks < worst_dip,
    )
