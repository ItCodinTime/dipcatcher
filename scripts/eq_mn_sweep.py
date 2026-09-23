"""Equity market-neutral sweep driver (screening only; confirm via run_backtest)."""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import numpy as np
from eq_mn_research import (
    demean_groups,
    eval_cfg,
    load_panel,
    rolling_sum,
    signals,
    zscore_xs,
)


def rank_avg(*arrs: np.ndarray) -> np.ndarray:
    """Equal-weight cross-sectional rank blend of signal panels (T,N)."""
    outs = []
    valid_any = np.zeros(arrs[0].shape, dtype=bool)
    for a in arrs:
        aa = np.where(np.isfinite(a), a, np.nan)
        valid_any |= np.isfinite(aa)
        order = np.argsort(np.where(np.isfinite(aa), aa, np.inf), axis=1)
        rk = np.full_like(aa, np.nan)
        rows = np.arange(aa.shape[0])[:, None]
        rk[rows, order] = np.tile(np.arange(aa.shape[1]), (aa.shape[0], 1))
        cnt = np.sum(np.isfinite(aa), axis=1, keepdims=True)
        rk = rk / np.where(cnt > 0, cnt, 1.0) - 0.5
        outs.append(np.where(np.isfinite(aa), rk, 0.0))
    s = sum(outs) / len(outs)
    return np.where(valid_any, s, np.nan)


def main() -> int:
    panel = load_panel()
    T, N = panel["tr"].shape
    sigs = signals(panel["tr"], panel["open"], panel["close"], panel["volu"], panel["sectors"])
    # continuation = +past return (flip of failed reversal family)
    r1 = np.full_like(panel["tr"], np.nan)
    r1[1:] = panel["tr"][1:] / panel["tr"][:-1] - 1
    lr1 = np.log1p(np.clip(r1, -0.99, 10))
    for k in (1, 2, 3, 5, 10, 21):
        r = rolling_sum(lr1, k)
        sigs[f"cont{k}"] = zscore_xs(r)
        sec_mean = demean_groups(np.nan_to_num(r), panel["sectors"])
        sigs[f"resid_cont{k}"] = zscore_xs(np.where(np.isfinite(r), r - sec_mean, np.nan))
        sigs[f"cont{k}_sec"] = demean_groups(zscore_xs(r), panel["sectors"])

    cut = next(
        i for i, d in enumerate(panel["dates"]) if d >= dt.datetime(2025, 1, 1, tzinfo=d.tzinfo)
    )
    print("split", cut, panel["dates"][cut])

    jobs: dict[str, dict] = {}
    # single-signal continuation & baseline grid
    for name in [
        "cont1",
        "cont2",
        "cont3",
        "cont5",
        "cont10",
        "cont21",
        "resid_cont1",
        "resid_cont3",
        "resid_cont5",
        "resid_cont10",
        "resid_cont21",
        "mom12_1",
        "mom12_1_sec",
        "cont1_sec",
        "cont3_sec",
        "cont5_sec",
        "cont10_sec",
        "cont21_sec",
    ]:
        for sec in (True, False):
            for vt in (0.10, 0.15):
                key = f"{name}{'_SN' if sec else ''}_vt{int(vt * 100)}"
                jobs[key] = {"sig": name, "sec": sec, "vt": vt}
    # combos (sector-neutral)
    combos = {
        "mix_c35": ["cont3", "cont5"],
        "mix_c3510": ["cont3", "cont5", "cont10"],
        "mix_c_mom": ["cont5", "mom12_1"],
        "mix_rc": ["resid_cont5", "resid_cont10"],
        "mix_all": ["cont3", "cont5", "cont10", "mom12_1", "zma20"],
    }
    for cname, members in combos.items():
        for vt in (0.10, 0.15):
            for sec in (True, False):
                jobs[f"{cname}{'_SN' if sec else ''}_vt{int(vt * 100)}"] = {
                    "combo": members,
                    "sec": sec,
                    "vt": vt,
                }

    out: dict[str, dict] = {}
    for key, j in jobs.items():
        if "combo" in j:
            sig = rank_avg(*[sigs[m] for m in j["combo"]])
        else:
            sig = sigs[j["sig"]]
        for sm in (None, 3.0):
            k2 = key + (f"_sm{sm:g}" if sm else "")
            try:
                m, w, r = eval_cfg(
                    sig,
                    panel,
                    gross=2.0,
                    cap=0.03,
                    sec=j["sec"],
                    t0=252,
                    t1=cut,
                    vt=j["vt"],
                    smooth=sm,
                )
                out[k2] = {
                    kk: m[kk] for kk in ("sharpe", "mdd", "cagr", "total", "mean_turn", "vol")
                }
                print(k2, json.dumps(out[k2], default=str), flush=True)
            except Exception as e:  # noqa: BLE001
                print(k2, "ERR", e, flush=True)
    pathlib.Path("artifacts").mkdir(exist_ok=True)
    pathlib.Path("artifacts/eqmn_sweep_cont.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print("wrote artifacts/eqmn_sweep_cont.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
