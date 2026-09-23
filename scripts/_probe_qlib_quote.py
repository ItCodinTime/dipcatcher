"""Probe qlib quote data around 2025-03-03 for SOLUSDT."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from incumbent_bench_qlib import write_qlib_provider  # noqa: E402
from incumbent_bench_vectorbt import load_panel  # noqa: E402


def main() -> None:
    frames = load_panel(
        [
            ROOT / "data/raw/sources/btcusdt_1d.parquet",
            ROOT / "data/raw/sources/ethusdt_1d.parquet",
            ROOT / "data/raw/sources/solusdt_1d.parquet",
        ]
    )
    tmp = tempfile.mkdtemp()
    prov = write_qlib_provider(Path(tmp) / "qlib_data", frames)

    import qlib  # noqa: E402
    from qlib.data import D  # noqa: E402

    qlib.init(provider_uri=str(prov), region="cn")

    out = open("probe_quote_out.txt", "w", encoding="utf-8")
    for sid in ("SOLUSDT", "BTCUSDT", "ETHUSDT"):
        df = D.features([sid], ["$open", "$close", "$volume"], start_time="2025-02-28", end_time="2025-03-05")
        out.write(f"=== {sid}\n{df.to_string()}\n\n")

    # also check the raw bin length vs calendar
    cal = (Path(tmp) / "qlib_data" / "calendars" / "day.txt").read_text().split()
    out.write(f"calendar n={len(cal)} first={cal[0]} last={cal[-1]}\n")
    import numpy as np  # noqa: E402

    b = np.fromfile(Path(tmp) / "qlib_data/features/solusdt/open.day.bin", dtype="<f4")
    out.write(f"sol open.bin len={len(b)} first_idx={b[0]} last3={b[-3:]}\n")
    idx = cal.index("2025-03-03")
    out.write(f"2025-03-03 cal index={idx} -> bin[{idx+1}]={b[idx+1]}\n")
    out.close()


if __name__ == "__main__":
    main()
