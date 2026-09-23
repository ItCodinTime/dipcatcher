"""Compute the ``dip_gmm_k`` challenger column over an existing shard's exact
origin grid.

Replicates the origin loop of ``sota_eval_kronos.py`` (first_origin formula,
window/garch-window slicing, close-to-close target) but only evaluates the
deterministic Gaussian-mixture challenger — no target models, no other
challengers. Output binds to the shard via ``bars_sha256`` + config fields so a
splice tool can verify row alignment by construction (the grid is deterministic
given the same bars bytes and the same config).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sota_eval_kronos import TAUS, _fit_gmm, _sha256  # noqa: E402

from quant_fund.metrics.scoring import (  # noqa: E402
    crps_gaussian_mixture,
    gaussian_mixture_quantiles,
    pinball_loss,
)
from quant_fund.research.sota_evidence import validate_bars  # noqa: E402

MODEL = "dip_gmm_k"


def _find_bars(bars_root: Path, bars_sha256: dict[str, str]) -> Path:
    """Resolve the shard's ``{filename: sha256}`` binding to a local file."""
    if len(bars_sha256) != 1:
        raise ValueError(f"expected single-asset shard, got {sorted(bars_sha256)}")
    name, want = next(iter(bars_sha256.items()))
    cand = bars_root / name
    if not cand.is_file():
        found = [p for p in sorted(bars_root.glob("*.parquet")) if _sha256(p) == want]
        if len(found) != 1:
            raise ValueError(f"{name}: {len(found)} hash matches in {bars_root}")
        cand = found[0]
    if _sha256(cand) != want:
        raise ValueError(f"{cand.name}: sha256 mismatch vs shard record")
    return cand


def compute_column(bars_path: Path, cfg: dict) -> dict[str, np.ndarray]:
    """(crps[R], pin[R,T], target_time_ns[R]) over the shard's origin grid."""
    frame = pl.read_parquet(bars_path)
    event_times, interval = validate_bars(frame)
    closes = frame["close"].to_numpy().astype(float)
    n = closes.size
    origins = int(cfg["origins_per_asset"])
    lookback = int(cfg["lookback"])
    window = int(cfg["window"])
    garch_window = int(cfg["garch_window"])
    first_origin = max(lookback, window + 1, n - origins - 1)
    if first_origin >= n - 1:
        raise ValueError(f"{bars_path.name}: not enough bars ({n})")
    crps = np.full(n - 1 - first_origin, np.nan)
    pin = np.full((n - 1 - first_origin, len(TAUS)), np.nan)
    tt = np.empty(n - 1 - first_origin, dtype=np.int64)
    for row, i in enumerate(range(first_origin, n - 1)):
        closes_hist = closes[: i + 1]
        long_start = max(0, closes_hist.size - (garch_window + 1))
        rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        try:
            w, mu, sig = _fit_gmm(rets_long)
            crps[row] = float(
                crps_gaussian_mixture(np.array([y]), w, mu, sig)[0]
            )
            q = gaussian_mixture_quantiles(w, mu, sig, np.asarray(TAUS))
            for k, tau in enumerate(TAUS):
                if np.isfinite(q[k]):
                    pin[row, k] = float(
                        pinball_loss(np.array([y]), np.array([q[k]]), tau)[0]
                    )
        except Exception:  # noqa: BLE001 - honest NaN, row stays disclosed
            pass
    return {"crps_col": crps, "pin_cols": pin, "target_time_ns": tt,
            "bar_interval_ns": np.asarray(interval, dtype=np.int64)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=Path, required=True, help="existing losses shard to match")
    p.add_argument("--bars-root", type=Path, default=Path("data/raw/sources"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    with np.load(args.shard, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        n_rows = int(z["crps_matrix"].shape[0])
        shard_names = [str(x) for x in z["model_names"]]
    cfg = meta["config"]
    want_cols = int(cfg.get("origins_per_asset", 0)) >= 1
    if not want_cols or MODEL in shard_names:
        raise ValueError(f"{args.shard.name}: already has {MODEL} or bad config")

    bars = _find_bars(args.bars_root, meta["bars_sha256"])
    t0 = time.time()
    out = compute_column(bars, cfg)
    if out["crps_col"].shape[0] != n_rows:
        raise ValueError(
            f"{args.shard.name}: grid mismatch {out['crps_col'].shape[0]} != {n_rows}"
        )
    meta_out = {
        "tool": Path(__file__).name,
        "model": MODEL,
        "shard": args.shard.name,
        "shard_sha256": _sha256(args.shard),
        "bars": bars.name,
        "bars_sha256": meta["bars_sha256"],
        "bars_file_sha256": _sha256(bars),
        "asset_names": meta.get("asset_names"),
        "config": {k: cfg.get(k) for k in
                   ("origins_per_asset", "lookback", "window", "garch_window",
                    "taus", "seed")},
        "n_rows": n_rows,
        "n_finite_crps": int(np.isfinite(out["crps_col"]).sum()),
        "elapsed_s": round(time.time() - t0, 3),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        crps_col=out["crps_col"],
        pin_cols=out["pin_cols"],
        target_time_ns=out["target_time_ns"],
        bar_interval_ns=out["bar_interval_ns"],
        meta_json=np.array(json.dumps(meta_out)),
    )
    print(
        f"{args.shard.name}: {MODEL} rows={n_rows} "
        f"finite={meta_out['n_finite_crps']} mean_crps={np.nanmean(out['crps_col']):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
