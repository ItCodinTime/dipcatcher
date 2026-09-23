"""Compute the ``dip_seas`` challenger column over an existing shard's origin grid.

Replicates the origin loop of ``sota_eval_kronos.py`` (first_origin formula,
garch-window slicing, close-to-close target) exactly like
``_challenger_col.py``, evaluating a single deterministic challenger. Output
binds to the shard via ``bars_sha256`` + config fields so the splice tool can
verify row alignment by construction (the grid is deterministic given the
same bars bytes and the same config).

``dip_seas`` — calendar-seasonality-conditioned empirical distribution:

- Each bar is assigned a calendar slot from its UTC ``event_time``: daily
  shards (``bar_interval_ns == 86400e9``) bucket by day-of-week (7 slots,
  Monday=0, matching pandas ``dayofweek``); any other interval buckets by
  ``hour // 4`` (6 UTC slots), matching the 4h funding/session structure
  (funding prints at 00/08/16 UTC, US/EU/Asia session rotation).
- At origin ``i`` the forecast targets bar ``i + 1``, so the conditioning
  slot is ``slot(event_time[i+1])`` — the calendar bucket of the bar being
  predicted. Trailing returns are bucketed by their target bar's slot:
  ``rets_long[k] = close[k]/close[k-1] - 1`` joins slot ``s`` iff
  ``slot(event_time[bar k]) == s``. Only bars ``<= i`` enter the pool —
  strictly causal.
- Slot-conditional quantiles at ``LGBM_TAUS`` are shrunk toward the pooled
  empirical quantiles with pseudo-count ``SHRINK_K = 40``:
  ``q = (n_s * q_slot + K * q_global) / (n_s + K)``. Slots with fewer than
  ``MIN_SLOT_OBS = 15`` observations or non-finite slot quantiles fall back
  to the pooled quantiles; a non-finite pooled fit yields an honest NaN row.
- CRPS is scored by ``crps_from_quantiles(y, LGBM_TAUS, q)`` — the
  quantile-pinball trapezoid identity on the dense grid with the arena's
  disclosed exponential tail completion, the same scoring family used by
  ``dip_qar`` and ``dip_stack``. A ``crps_empirical`` cross-check on the
  deterministic 512-point quantile-grid sample of the same completed
  quantile function is recorded in meta (``crps_empirical_512_*``) but is
  not the scored value.

Fully deterministic: fixed grids, fixed constants, no RNG.
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

from sota_eval_kronos import (  # noqa: E402
    LGBM_TAUS,
    TAUS,
    _sha256,
    crps_from_quantiles,
    qf_at,
)

from quant_fund.metrics.scoring import crps_empirical, pinball_loss  # noqa: E402
from quant_fund.research.sota_evidence import validate_bars  # noqa: E402

MODEL = "dip_seas"
SHRINK_K = 40.0  # pseudo-count shrinkage of slot quantiles toward the pooled fit
MIN_SLOT_OBS = 15  # slot pools thinner than this fall back to the pooled fit
NS_PER_HOUR = 3_600_000_000_000
NS_PER_DAY = 86_400_000_000_000
# Deterministic 512-point quantile-grid sample used only for the meta-level
# crps_empirical cross-check of the scored crps_from_quantiles value.
CRPS_GRID = (np.arange(512) + 0.5) / 512.0


def _slot_ids(event_times: np.ndarray, interval_ns: int) -> tuple[np.ndarray, str]:
    """Calendar slot per bar index from UTC event timestamps (int64 ns).

    Daily bars (24h spacing) -> ``"dow"``: UTC day-of-week, Monday=0..Sunday=6
    (epoch day 1970-01-01 was a Thursday, hence the +3 offset). Other
    intervals -> ``"hour4"``: ``(utc_hour // 4)``, six slots aligned with the
    4h bar grid and funding/session boundaries.
    """
    if interval_ns == NS_PER_DAY:
        days = np.floor_divide(event_times, NS_PER_DAY)
        return ((days + 3) % 7).astype(np.int64), "dow"
    hours = np.floor_divide(event_times, NS_PER_HOUR) % 24
    return (hours // 4).astype(np.int64), "hour4"


def _seas_quantiles(
    rets_long: np.ndarray, train_slots: np.ndarray, slot: int
) -> tuple[np.ndarray, int, bool]:
    """Shrunk slot-conditional empirical quantiles at ``LGBM_TAUS``.

    ``rets_long[k]`` is the close-to-close return landing on the bar whose
    slot is ``train_slots[k]``. Returns ``(q[LGBM_TAUS], n_slot, fell_back)``;
    ``q`` is all-NaN only when even the pooled fit is non-finite.
    """
    q_global = np.quantile(rets_long, LGBM_TAUS)
    if not np.isfinite(q_global).all():
        return np.full(LGBM_TAUS.size, np.nan), 0, True
    pool = rets_long[train_slots == slot]
    n_s = int(pool.size)
    if n_s < MIN_SLOT_OBS:
        return q_global, n_s, True
    q_slot = np.quantile(pool, LGBM_TAUS)
    if not np.isfinite(q_slot).all():
        return q_global, n_s, True
    return (n_s * q_slot + SHRINK_K * q_global) / (n_s + SHRINK_K), n_s, False


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
    slots, scheme = _slot_ids(event_times, interval)
    n = closes.size
    origins = int(cfg["origins_per_asset"])
    lookback = int(cfg["lookback"])
    window = int(cfg["window"])
    garch_window = int(cfg["garch_window"])
    first_origin = max(lookback, window + 1, n - origins - 1)
    if first_origin >= n - 1:
        raise ValueError(f"{bars_path.name}: not enough bars ({n})")
    n_rows = n - 1 - first_origin
    crps = np.full(n_rows, np.nan)
    pin = np.full((n_rows, len(TAUS)), np.nan)
    tt = np.empty(n_rows, dtype=np.int64)
    # Meta-level diagnostics (not part of the spliced column payload).
    origin_slot = np.zeros(n_rows, dtype=np.int64)
    pool_n = np.zeros(n_rows, dtype=np.int64)
    fell_back = np.zeros(n_rows, dtype=bool)
    emp_diffs: list[float] = []
    for row, i in enumerate(range(first_origin, n - 1)):
        closes_hist = closes[: i + 1]
        long_start = max(0, closes_hist.size - (garch_window + 1))
        rets_long = np.diff(closes_hist[long_start:]) / closes_hist[long_start:-1]
        # rets_long[k] lands on bar index long_start+1+k; its slot is that bar's.
        train_slots = slots[long_start + 1 : i + 1]
        slot = int(slots[i + 1])  # slot of the bar being predicted
        y = float(closes[i + 1] / closes[i] - 1.0)
        tt[row] = int(event_times[i + 1])
        origin_slot[row] = slot
        try:
            q, n_s, fb = _seas_quantiles(rets_long, train_slots, slot)
            pool_n[row] = n_s
            fell_back[row] = fb
            if not np.isfinite(q).all():
                continue  # honest NaN row
            c = crps_from_quantiles(y, LGBM_TAUS, q)
            # Diagnostic only: empirical CRPS of the deterministic 512-point
            # quantile-grid sample of the same completed quantile function.
            samp = np.array([qf_at(LGBM_TAUS, q, t) for t in CRPS_GRID])
            emp = crps_empirical(y, samp)
            q_taus = np.array([qf_at(LGBM_TAUS, q, t) for t in TAUS])
            pins = np.array(
                [
                    float(pinball_loss(np.array([y]), np.array([qt]), tau)[0])
                    if np.isfinite(qt)
                    else np.nan
                    for qt, tau in zip(q_taus, TAUS)
                ]
            )
        except Exception:  # noqa: BLE001 - honest NaN, row stays disclosed
            continue
        crps[row] = c
        pin[row] = pins
        if np.isfinite(emp):
            emp_diffs.append(abs(emp - c))
    uniq = np.unique(origin_slot)
    diag = {
        "slot_scheme": scheme,
        "n_slots": int(slots.max()) + 1,
        "slot_origin_counts": {
            str(int(s)): int((origin_slot == s).sum()) for s in uniq
        },
        "slot_mean_pool_n": {
            str(int(s)): round(float(pool_n[origin_slot == s].mean()), 2)
            for s in uniq
        },
        "n_global_fallback_rows": int(fell_back.sum()),
        "shrinkage_k": SHRINK_K,
        "min_slot_obs": MIN_SLOT_OBS,
        "crps_method": (
            "crps_from_quantiles(y, LGBM_TAUS, q): 2*int pinball trapezoid on "
            "dense grid with exponential tail completion (same family as "
            "dip_qar/dip_stack); crps_empirical on a deterministic 512-point "
            "quantile-grid sample recorded as a cross-check only"
        ),
        "crps_empirical_512_mean_abs_diff": (
            float(np.mean(emp_diffs)) if emp_diffs else None
        ),
        "crps_empirical_512_max_abs_diff": (
            float(np.max(emp_diffs)) if emp_diffs else None
        ),
    }
    return {
        "crps_col": crps,
        "pin_cols": pin,
        "target_time_ns": tt,
        "bar_interval_ns": np.asarray(interval, dtype=np.int64),
        "diag": diag,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shard", type=Path, required=True, help="existing losses shard to match"
    )
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
    diag = out.pop("diag")
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
        "config": {
            k: cfg.get(k)
            for k in (
                "origins_per_asset",
                "lookback",
                "window",
                "garch_window",
                "taus",
                "seed",
            )
        },
        "n_rows": n_rows,
        "n_finite_crps": int(np.isfinite(out["crps_col"]).sum()),
        "bar_interval_ns": int(out["bar_interval_ns"]),
        **diag,
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
        f"finite={meta_out['n_finite_crps']} "
        f"mean_crps={np.nanmean(out['crps_col']):.6f} "
        f"scheme={diag['slot_scheme']} fallback_rows={diag['n_global_fallback_rows']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
