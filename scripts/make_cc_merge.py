import json
from pathlib import Path

import numpy as np

DIR = Path(r"D:\dipcatcher\.dsh-24x7\eval-full")
# solusdt 4h: produce cc variant (drop dip_student_t for model-set uniformity)
z = np.load(DIR / "h4_solusdt_4h_deep.fixed.npz", allow_pickle=False)
names = [str(x) for x in z["model_names"]]
idx = [i for i, m in enumerate(names) if m != "dip_student_t"]
meta = json.loads(str(z["meta_json"]))
meta["complete_case_drop"] = ["dip_student_t"]
np.savez_compressed(
    DIR / "h4_solusdt_4h_deep.cc.npz",
    crps_matrix=z["crps_matrix"][:, idx],
    pinball_cube=z["pinball_cube"][:, idx, :],
    asset_ids=z["asset_ids"],
    model_names=np.asarray([names[i] for i in idx]),
    meta_json=np.array(json.dumps(meta)),
)
print("wrote solusdt cc:", [names[i] for i in idx])
