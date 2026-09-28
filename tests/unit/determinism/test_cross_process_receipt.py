"""Cross-process determinism: receipt content is identical under any hash seed.

In-process reruns share one PYTHONHASHSEED, so they cannot catch set-iteration
or dict-insertion-order leaks into receipts and artifact digests. Running the
same lane in two subprocesses with different hash seeds does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl

_SCRIPT = r"""
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
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    path_a = _run_once(root_a, "1")
    path_b = _run_once(root_b, "2")
    # Normalizing the machine-local absolute paths (which legitimately differ
    # between run roots), the two runs must produce identical receipt content.
    text_a = path_a.read_text(encoding="utf-8").replace(str(root_a), "$ROOT")
    text_b = path_b.read_text(encoding="utf-8").replace(str(root_b), "$ROOT")
    obj_a, obj_b = json.loads(text_a), json.loads(text_b)
    for obj in (obj_a, obj_b):
        obj.pop("receipt_sha256", None)
    assert obj_a == obj_b, (
        "sim_live receipt content differs across PYTHONHASHSEED values "
        "(hash-ordering leak into a receipt artifact)"
    )


_GOLD_SCRIPT = r"""
import sys
from pathlib import Path

from quant_fund.config import load_config
from quant_fund.pipeline.dataset import build_gold

cfg = load_config("configs/research.yaml")
cfg.data.root = sys.argv[1]
cfg.data.synthetic_n_assets = 6
cfg.data.synthetic_n_days = 120
build_gold(cfg)
sys.stdout.write(str(Path(sys.argv[1]).resolve()))
"""


def test_gold_pipeline_frames_identical_across_hash_seeds(tmp_path: Path) -> None:
    """The synthetic data pipeline must be deterministic modulo ingest stamps.

    Set iteration or dict-order leaks into silver/gold parquet content would
    diverge across processes. ``ingested_time`` is intentionally wall-clock —
    it is audit metadata with a chain invariant (available_time <=
    ingested_time), never a feature/label input — so it is excluded.
    """

    def _run(root: Path, hashseed: str) -> Path:
        out = subprocess.run(
            [sys.executable, "-c", _GOLD_SCRIPT, str(root)],
            env={**os.environ, "PYTHONHASHSEED": hashseed},
            check=True,
            capture_output=True,
        )
        return Path(out.stdout.decode().strip())

    root_a = _run(tmp_path / "a", "1")
    root_b = _run(tmp_path / "b", "2")
    for rel in (
        "bronze/bars.parquet",
        "silver/bars.parquet",
        "silver/universe.parquet",
        "gold/features.parquet",
        "gold/labels.parquet",
    ):
        pa, pb = root_a / rel, root_b / rel
        if not (pa.is_file() and pb.is_file()):
            continue
        fa, fb = pl.read_parquet(pa), pl.read_parquet(pb)
        assert fa.shape == fb.shape, rel
        fa = fa.drop("ingested_time") if "ingested_time" in fa.columns else fa
        fb = fb.drop("ingested_time") if "ingested_time" in fb.columns else fb
        assert fa.equals(fb), f"{rel} content differs across PYTHONHASHSEED"
