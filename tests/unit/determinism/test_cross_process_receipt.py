"""Cross-process determinism: a sealed receipt is identical under any hash seed.

In-process reruns share one PYTHONHASHSEED, so they cannot catch set-iteration
or dict-insertion-order leaks into sealed artifacts. Running the same lane in
two subprocesses with different hash seeds does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT = r"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from quant_fund.config.models import AppConfig
from quant_fund.paper.quantile_signals import QuantilePolicy
from quant_fund.paper.sim_live import StrategySlot, run_sim_live

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
rows = []
for k, sid in enumerate(["AAA", "BBB"]):
    px = 100.0 + k * 7.0
    for i in range(160):
        px *= 1.0 + 0.0003 * (((i * 7 + k * 13) % 11) - 5)
        rows.append(
            {
                "security_id": sid,
                "event_time": T0 + timedelta(hours=4 * i),
                "open": px * 0.999,
                "high": px * 1.003,
                "low": px * 0.997,
                "close": px,
                "volume": 1e6,
                "source": "binance",
            }
        )
bars = pl.DataFrame(rows)
root = Path(sys.argv[1])
bars_root = root / "bars"
bars_root.mkdir(parents=True)
for sid in ["AAA", "BBB"]:
    bars.filter(pl.col("security_id") == sid).write_parquet(
        bars_root / f"{sid.lower()}_1d.parquet"
    )
cfg = AppConfig.model_validate(
    {
        "data": {"root": str(root / "data"), "source": "synthetic"},
        "paper": {"ledger_subdir": "sim_live_test", "enable_shadow": False},
        "risk_gate": {"max_name": 0.5, "max_gross": 2.0, "max_net": 1.0},
    }
)
pol = QuantilePolicy(mode="long_flat", kappa=1.0, cost_gate=0.0, deadband=0.0)
res = run_sim_live(
    bars_root=bars_root,
    symbols=["AAA", "BBB"],
    interval="1d",
    config=cfg,
    champion=StrategySlot(name="empirical_long_flat", spec="empirical", policy=pol),
    challengers=[StrategySlot(name="ewma_emp_long_flat", spec="ewma_emp", policy=pol)],
    window=120,
    out_dir=root / "out",
    run_id="det-cross-proc",
)
sys.stdout.write(str(res.receipt_path))
"""


def _run_once(root: Path, hashseed: str) -> Path:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hashseed
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(root)],
        env=env,
        check=True,
        capture_output=True,
    )
    return Path(out.stdout.decode().strip())


def test_sim_live_receipt_is_byte_identical_across_hash_seeds(tmp_path: Path) -> None:
    from quant_fund.research.receipt_v2 import verify_receipt_file

    root_a, root_b = tmp_path / "a", tmp_path / "b"
    path_a = _run_once(root_a, "1")
    path_b = _run_once(root_b, "2")
    # Each receipt is sealed and verifies on its own.
    assert verify_receipt_file(path_a)["valid"] is True
    assert verify_receipt_file(path_b)["valid"] is True
    # Normalizing the machine-local absolute paths the seal legitimately
    # covers, the two runs must produce identical receipt content.
    text_a = path_a.read_text(encoding="utf-8").replace(str(root_a), "$ROOT")
    text_b = path_b.read_text(encoding="utf-8").replace(str(root_b), "$ROOT")
    obj_a, obj_b = json.loads(text_a), json.loads(text_b)
    assert obj_a.pop("receipt_sha256") != obj_b.pop("receipt_sha256")
    assert obj_a == obj_b, (
        "sim_live receipt content differs across PYTHONHASHSEED values "
        "(hash-ordering leak into a sealed artifact)"
    )
