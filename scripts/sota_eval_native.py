"""Crossover eval under the Kronos paper's own protocol (arXiv:2508.02739, App. D).

This is the "beat them on their benchmark" lane, complementary to
``sota_eval_kronos.py`` (which scores next-bar return *distributions* by CRPS —
our native metric). Here the protocol is theirs:

- Test window: origins at or after 2024-07-01 (their pre-training cutoff is
  June 2024 — strict temporal separation per the paper).
- Look-back / horizon per frequency (paper Table 8):
    daily -> lookback 40, horizon 12 ;  4h -> lookback 90, horizon 18.
- Metrics (paper App. D):
    * Price-series IC / RankIC: per-sample correlation between predicted and
      true series per price channel, averaged. The lab's challengers model the
      close channel only, so this implementation scores the **close channel**
      for every model (disclosed deviation from the 4-channel average).
    * Return forecasting: r_hat = p_hat_{t+H}/p_t - 1; IC/RankIC computed
      across the pooled sample set per asset.
    * Realized-volatility forecasting: sigma2_hat = sum of squared log-returns
      of the predicted path; sigma2 = same on the actual path; MAE and R^2
      pooled per asset. For distributional challengers the honest analog of
      "predicted path vol" is the model-implied expected realized variance,
      E[sigma2] = (H-1) * sigma_hat^2 — scored identically for all models.

Target predicted paths:
    kronos   -> mean close path over S sampled OHLC trajectories (pred_len=H)
    chronos2 -> median (q50) path over horizon H
    bolt     -> median (q50) path over horizon H
    timesfm  -> point-forecast path (forecast channel 0) over horizon H

Challenger predicted paths (distributional -> honest point analog):
    mean path p_t * (1 + mu_hat)^h,  h = 1..H
    vol forecast (H-1) * sigma_hat^2
    H-step return (1 + mu_hat)^H - 1
Challengers are *distribution* models, not trajectory models: their mean path
is near-flat by construction, so the path-shape RankIC column is expected to
favor the generative targets. That is reported, not hidden — the claim of this
receipt is scoped to the metrics where distributional skill is the quantity
under test (return RankIC, vol MAE/R2), plus the honest loss column.

Per-origin predicted paths are saved to ``<out>.paths.npz`` so every metric is
recomputable without the models.

Usage (per asset, remote or local):
    .venv/bin/python scripts/sota_eval_native.py \
        --bars data/raw/sources/btcusdt_1d_deep.parquet \
        --kronos-repo third_party/kronos \
        --kronos kronos_small=data/models/Kronos-small,data/models/Kronos-Tokenizer-base \
        --chronos2 chronos2=data/models/chronos-2 \
        --bolt bolt_small=data/models/chronos-bolt-small \
        --timesfm timesfm=data/models/timesfm-2.5-200m-pytorch \
        --origins 300 --samples 8 --seed 7 \
        --out .dsh-24x7/native/btcusdt_1d_deep.json

Merge:
    .venv/bin/python scripts/sota_eval_native.py \
        --merge-parts .dsh-24x7/native/*.paths.npz --merge-out /tmp/native.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sota_eval_kronos import (  # noqa: E402
    _arch_fit,
    _conf_t_quantiles,
    _fit_gmm,
    _fit_skt,
    _fit_student_t,
    _lgbm_features,
    _qar_quantiles,
    _regime_quantiles,
    _sha256,
    _skt_pdf,
    ewma_next_sigma,
    lgbm_quantiles,
    parse_specs,
)

try:
    from quant_fund.research.sota_evidence import validate_bars  # noqa: E402
except ImportError:  # divergent checkouts: inline the same contract

    def validate_bars(frame: pl.DataFrame) -> tuple[np.ndarray, int]:
        required = {
            "security_id",
            "event_time",
            "available_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        if missing := required.difference(frame.columns):
            raise ValueError(f"missing columns {sorted(missing)}")
        if frame.height < 2 or frame["security_id"].n_unique() != 1:
            raise ValueError("each bar file must contain one security and at least two bars")
        if frame.select(pl.any_horizontal(pl.all().is_null()).any()).item():
            raise ValueError("bar data contains null values")
        times = frame["event_time"].dt.timestamp("ns").to_numpy()
        available = frame["available_time"].dt.timestamp("ns").to_numpy()
        steps = np.diff(times)
        if np.any(steps <= 0) or not np.all(steps == steps[0]):
            raise ValueError("bar event times must be strictly increasing with no gaps")
        if np.any(available < times) or np.any(available > times + steps[0]):
            raise ValueError("point-in-time violation: bar unavailable by next bar open")
        values = frame.select("open", "high", "low", "close", "volume").to_numpy()
        if not np.isfinite(values).all() or np.any(values[:, :4] <= 0):
            raise ValueError("prices must be positive and volume nonnegative, all finite")
        return times, int(steps[0])


TEST_START = pd.Timestamp("2024-07-01", tz="UTC")
# Paper Table 8: (lookback, horizon) by frequency.
FREQ_TABLE = {
    "1d": (40, 12),
    "4h": (90, 18),
    "1h": (80, 12),
}
CHALLENGER_NAMES = [
    "dip_gauss",
    "dip_student_t",
    "dip_ewma_t",
    "dip_empirical",
    "dip_empirical_long",
    "dip_garch_t",
    "dip_fhs",
    "dip_ewma_emp",
    "dip_lgbm_q",
    "dip_blend",
    "dip_gmm_k",
    "dip_skt",
    "dip_qar",
    "dip_conf_t",
    "dip_regime",
]


def challenger_params(rets: np.ndarray, rets_long: np.ndarray) -> dict[str, tuple[float, float]]:
    """(mu_hat, sigma_hat) per challenger — the distributional point analog."""
    out: dict[str, tuple[float, float]] = {}
    out["dip_gauss"] = (float(np.mean(rets)), float(np.std(rets, ddof=1)))
    nu, loc, sc = _fit_student_t(rets)
    if np.isfinite(nu) and np.isfinite(sc) and sc > 0 and nu > 2.0:
        # scale -> sd conversion for the Student-t
        out["dip_student_t"] = (float(loc), float(sc * np.sqrt(nu / (nu - 2.0))))
    else:
        out["dip_student_t"] = (np.nan, np.nan)
    out["dip_ewma_t"] = (0.0, ewma_next_sigma(rets))
    out["dip_empirical"] = (float(np.mean(rets)), float(np.std(rets, ddof=1)))
    out["dip_empirical_long"] = (
        float(np.mean(rets_long)),
        float(np.std(rets_long, ddof=1)),
    )
    # GARCH-t next-bar sigma
    try:
        fit = _arch_fit(rets_long * 100.0, vol="GARCH", dist="t", o=0)
        sig = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])) / 100.0
        out["dip_garch_t"] = (float(fit.params.get("mu", 0.0)) / 100.0, sig)
    except Exception:  # noqa: BLE001
        out["dip_garch_t"] = (np.nan, np.nan)
    # FHS: next-bar sigma from GARCH-normal fit
    try:
        fit = _arch_fit(rets_long * 100.0, vol="GARCH", dist="normal", o=1)
        sig = float(np.sqrt(fit.forecast(horizon=1).variance.iloc[-1, 0])) / 100.0
        out["dip_fhs"] = (float(fit.params.get("mu", 0.0)) / 100.0, sig)
    except Exception:  # noqa: BLE001
        out["dip_fhs"] = (np.nan, np.nan)
    # EWMA-weighted empirical: weighted mean/sd on rets_long
    try:
        w = 0.97 ** np.arange(rets_long.size - 1, -1, -1, dtype=float)
        w /= w.sum()
        mu_e = float(np.sum(w * rets_long))
        sd_e = float(np.sqrt(np.sum(w * (rets_long - mu_e) ** 2)))
        out["dip_ewma_emp"] = (mu_e, sd_e)
    except Exception:  # noqa: BLE001
        out["dip_ewma_emp"] = (np.nan, np.nan)
    # GMM: mixture mean/sd as the point analog
    try:
        gw, gmu, gsig = _fit_gmm(rets_long)
        mu_m = float(np.sum(gw * gmu))
        sd_m = float(np.sqrt(max(np.sum(gw * (gsig**2 + gmu**2)) - mu_m**2, 0.0)))
        out["dip_gmm_k"] = (mu_m, sd_m)
    except Exception:  # noqa: BLE001
        out["dip_gmm_k"] = (np.nan, np.nan)
    # skew-t: first two moments by quadrature on the fitted pdf
    try:
        from scipy.integrate import trapezoid

        xi, om, al, nu_s = _fit_skt(rets_long)
        zg = np.linspace(-12.0, 12.0, 8193)
        pdf = _skt_pdf(zg, al, nu_s)
        mass = trapezoid(pdf, zg)
        m1 = trapezoid(zg * pdf, zg) / mass
        m2 = trapezoid(zg * zg * pdf, zg) / mass
        out["dip_skt"] = (
            float(xi + om * m1),
            float(om * np.sqrt(max(m2 - m1 * m1, 0.0))),
        )
    except Exception:  # noqa: BLE001
        out["dip_skt"] = (np.nan, np.nan)
    # QAR: median as location, 5-95 spread mapped to sd (point analog only)
    try:
        q_qar = _qar_quantiles(rets_long)
        out["dip_qar"] = (
            float(q_qar[4]),
            float((q_qar[8] - q_qar[0]) / 3.2897072539),
        )
    except Exception:  # noqa: BLE001
        out["dip_qar"] = (np.nan, np.nan)
    # conf_t / regime: moments of the quantile-grid copy (point analog only)
    _grid = (np.arange(512) + 0.5) / 512.0
    for name, fn in (("dip_conf_t", _conf_t_quantiles), ("dip_regime", _regime_quantiles)):
        try:
            qg = fn(rets_long, _grid)
            out[name] = (float(np.mean(qg)), float(np.std(qg)))
        except Exception:  # noqa: BLE001
            out[name] = (np.nan, np.nan)
    return out


def challenger_paths(
    params: dict[str, tuple[float, float]],
    lgbm_q: np.ndarray | None,
    last_close: float,
    horizon: int,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, float]]:
    """Mean path / vol2 / H-step return for every challenger."""
    paths: dict[str, np.ndarray] = {}
    vol2: dict[str, float] = {}
    ret_hat: dict[str, float] = {}
    for name, (mu, sd) in params.items():
        if not (np.isfinite(mu) and np.isfinite(sd)):
            paths[name] = np.full(horizon, np.nan)
            vol2[name] = np.nan
            ret_hat[name] = np.nan
            continue
        growth = (1.0 + mu) ** np.arange(1, horizon + 1)
        paths[name] = last_close * growth
        vol2[name] = (horizon - 1) * sd * sd
        ret_hat[name] = float(growth[-1] - 1.0)
    # lgbm: 1-step q50 compounded (it has no multi-step head)
    if lgbm_q is not None and np.isfinite(lgbm_q).all():
        mid = lgbm_q[len(lgbm_q) // 2]
        growth = (1.0 + mid) ** np.arange(1, horizon + 1)
        paths["dip_lgbm_q"] = last_close * growth
        vol2["dip_lgbm_q"] = (horizon - 1) * float(np.var(lgbm_q))
        ret_hat["dip_lgbm_q"] = float(growth[-1] - 1.0)
    else:
        paths["dip_lgbm_q"] = np.full(horizon, np.nan)
        vol2["dip_lgbm_q"] = np.nan
        ret_hat["dip_lgbm_q"] = np.nan
    # blend: same (mu, sd) family — average of gauss and student params
    g = params["dip_gauss"]
    t = params["dip_student_t"]
    if np.isfinite(g[0]) and np.isfinite(t[0]):
        mu_b, sd_b = 0.5 * (g[0] + t[0]), 0.5 * (g[1] + t[1])
        growth = (1.0 + mu_b) ** np.arange(1, horizon + 1)
        paths["dip_blend"] = last_close * growth
        vol2["dip_blend"] = (horizon - 1) * sd_b * sd_b
        ret_hat["dip_blend"] = float(growth[-1] - 1.0)
    else:
        paths["dip_blend"] = np.full(horizon, np.nan)
        vol2["dip_blend"] = np.nan
        ret_hat["dip_blend"] = np.nan
    return paths, vol2, ret_hat


# --------------------------------------------------------------------------
# Target predictors -> close paths
# --------------------------------------------------------------------------


def kronos_path(
    predictor, history: pd.DataFrame, x_ts, y_ts, samples: int, horizon: int
) -> np.ndarray:
    """Mean close path over S sampled OHLC trajectories."""
    acc = np.zeros(horizon, dtype=float)
    for _ in range(samples):
        out = predictor.predict(
            history,
            x_ts,
            y_ts,
            horizon,
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
        acc += np.asarray(out["close"].iloc[:horizon], dtype=float)
    return acc / samples


def chronos2_path(pipe, closes_hist: np.ndarray, horizon: int) -> np.ndarray:
    import torch

    context = torch.from_numpy(np.asarray(closes_hist, dtype=np.float32))[None, None, :]
    quantiles, _mean = pipe.predict_quantiles(
        context, prediction_length=horizon, quantile_levels=[0.5]
    )
    q = np.asarray(quantiles[0], dtype=np.float64)
    return np.asarray(q[0, :, 0], dtype=float)  # (n_var, horizon, n_q)


def bolt_path(pipe, closes_hist: np.ndarray, horizon: int) -> np.ndarray:
    import torch

    context = torch.from_numpy(np.asarray(closes_hist, dtype=np.float32))[None, :]
    quantiles, _mean = pipe.predict_quantiles(
        context, prediction_length=horizon, quantile_levels=[0.5]
    )
    arr = np.asarray(quantiles, dtype=np.float64)
    return np.asarray(arr[0, :, 0], dtype=float)


def timesfm_path(model, closes_hist: np.ndarray, horizon: int) -> np.ndarray:
    _point, quant = model.forecast(
        horizon=horizon, inputs=[np.asarray(closes_hist, dtype=np.float64)]
    )
    q = np.asarray(quant, dtype=np.float64)
    return np.asarray(q[0, :, 5], dtype=float)  # channel 5 = point forecast


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def realized_vol2(closes: np.ndarray) -> float:
    lr = np.diff(np.log(closes))
    return float(np.sum(lr * lr))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    import scipy.stats as st

    if np.std(a) <= 0 or np.std(b) <= 0:
        return np.nan
    return float(st.spearmanr(a, b).statistic)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    import scipy.stats as st

    if np.std(a) <= 0 or np.std(b) <= 0:
        return np.nan
    return float(st.pearsonr(a, b).statistic)


# --------------------------------------------------------------------------
# Merge / summarize
# --------------------------------------------------------------------------


def summarize_parts(parts: list[dict]) -> dict:
    """Pool per-origin stats across assets; per-model scoreboard."""
    import scipy.stats as st

    model_names: list[str] = []
    for p in parts:
        for m in p["model_names"]:
            if m not in model_names:
                model_names.append(m)

    out: dict[str, Any] = {"schema": "native_eval.v1", "assets": {}}
    pooled_rankic: dict[str, list[float]] = {m: [] for m in model_names}
    pooled_volhat: dict[str, list[float]] = {m: [] for m in model_names}
    pooled_volact: dict[str, list[float]] = {m: [] for m in model_names}
    pooled_rethat: dict[str, list[float]] = {m: [] for m in model_names}
    pooled_retact: dict[str, list[float]] = {m: [] for m in model_names}

    def _vol_stats(vhat: np.ndarray, vact: np.ndarray) -> dict[str, float]:
        ok = np.isfinite(vhat) & np.isfinite(vact)
        if ok.sum() < 4:
            return {"vol_mae": np.nan, "vol_r2": np.nan}
        mae = float(np.mean(np.abs(vhat[ok] - vact[ok])))
        ss_res = float(np.sum((vact[ok] - vhat[ok]) ** 2))
        ss_tot = float(np.sum((vact[ok] - np.mean(vact[ok])) ** 2))
        return {"vol_mae": mae, "vol_r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan}

    for p in parts:
        asset = p["asset"]
        rank = p["path_rankic"]  # (origins, models)
        vhat = p["vol2_hat"]  # (origins, models)
        vact = p["vol2_actual"]  # (origins,)
        rethat = p["ret_hat"]  # (origins, models)
        ret = p["ret_actual"]  # (origins,)
        asset_row: dict[str, Any] = {}
        for j, m in enumerate(p["model_names"]):
            r = rank[:, j]
            rr = rethat[:, j]
            ok = np.isfinite(rr) & np.isfinite(ret)
            row = {
                "path_rankic_mean": float(np.nanmean(r)),
                "path_rankic_se": float(np.nanstd(r) / np.sqrt(np.isfinite(r).sum())),
                "ret_rankic": float(st.spearmanr(rr[ok], ret[ok]).statistic)
                if ok.sum() > 3
                else np.nan,
                "n_origins": int(np.isfinite(r).sum()),
            }
            row.update(_vol_stats(vhat[:, j], vact))
            asset_row[m] = row
            pooled_rankic[m].extend(r.tolist())
            pooled_volhat[m].extend(vhat[:, j].tolist())
            pooled_volact[m].extend(vact.tolist())
            pooled_rethat[m].extend(rr.tolist())
            pooled_retact[m].extend(ret.tolist())
        dupes = set(asset_row) & set(out["assets"].get(asset, {}))
        if dupes:
            print(
                f"WARNING: duplicate (asset,model) rows overwritten: {asset} {sorted(dupes)}",
                file=sys.stderr,
            )
        out["assets"].setdefault(asset, {}).update(asset_row)

    pooled: dict[str, Any] = {}
    for m in model_names:
        r = np.asarray(pooled_rankic[m])
        vh = np.asarray(pooled_volhat[m])
        va = np.asarray(pooled_volact[m])
        rr = np.asarray(pooled_rethat[m])
        ra = np.asarray(pooled_retact[m])
        ok = np.isfinite(rr) & np.isfinite(ra)
        row = {
            "path_rankic_mean": float(np.nanmean(r)),
            "ret_rankic": float(st.spearmanr(rr[ok], ra[ok]).statistic) if ok.sum() > 3 else np.nan,
            "n_origins": int(np.isfinite(r).sum()),
        }
        row.update(_vol_stats(vh, va))
        pooled[m] = row
    out["pooled"] = pooled
    return out


def merge_parts(args: argparse.Namespace) -> int:
    parts = []
    skipped: list[dict] = []
    for p in args.merge_parts:
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        if (
            (args.expect_freq and meta.get("freq") != args.expect_freq)
            or (args.expect_lookback and meta.get("lookback") != args.expect_lookback)
            or (args.expect_horizon and meta.get("horizon") != args.expect_horizon)
        ):
            skipped.append({"part": str(p), "meta": meta})
            continue
        parts.append(
            {
                "asset": meta["asset"],
                "part": str(p),
                "n_done": int(meta.get("n_origins_done", z["path_rankic"].shape[0])),
                "model_names": list(z["model_names"]),
                "path_rankic": z["path_rankic"],
                "vol2_hat": z["vol2_hat"],
                "vol2_actual": z["vol2_actual"],
                "ret_hat": z["ret_hat"],
                "ret_actual": z["ret_actual"],
            }
        )
    by_asset: dict[str, set[int]] = {}
    for p in parts:
        by_asset.setdefault(p["asset"], set()).add(p["n_done"])
    for asset, counts in by_asset.items():
        if len(counts) > 1:
            print(
                f"WARNING: {asset} parts disagree on origin coverage: {sorted(counts)}",
                file=sys.stderr,
            )
    if skipped:
        print(f"WARNING: skipped {len(skipped)} off-protocol part(s)", file=sys.stderr)
        for s in skipped:
            print(f"  {s['part']}: {s['meta']}", file=sys.stderr)
    if not parts:
        raise SystemExit("no parts survived protocol filters")
    receipt = summarize_parts(parts)
    receipt["parts"] = [str(p) for p in args.merge_parts]
    if skipped:
        receipt["skipped_off_protocol"] = skipped
    args.merge_out.parent.mkdir(parents=True, exist_ok=True)
    args.merge_out.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(receipt.get("pooled", {}), indent=2))
    return 0


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--kronos-repo", type=Path, default=None)
    ap.add_argument("--kronos", action="append", metavar="NAME=MODEL,TOK")
    ap.add_argument("--chronos2", action="append", metavar="NAME=DIR")
    ap.add_argument("--bolt", action="append", metavar="NAME=DIR")
    ap.add_argument("--timesfm", action="append", metavar="NAME=DIR")
    ap.add_argument("--target", action="append")
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--tokenizer-dir", type=Path, default=None)
    ap.add_argument("--chronos-model", type=Path, default=None)
    ap.add_argument("--lookback", type=int, default=None, help="default: paper Table 8 by freq")
    ap.add_argument("--horizon", type=int, default=None, help="default: paper Table 8 by freq")
    ap.add_argument("--window", type=int, default=250, help="challenger fit window")
    ap.add_argument("--garch-window", type=int, default=750)
    ap.add_argument("--origins", type=int, default=300)
    ap.add_argument("--samples", type=int, default=8, help="Kronos path draws per origin")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--test-start", type=str, default="2024-07-01")
    ap.add_argument("--no-targets", action="store_true", help="challengers only (smoke)")
    ap.add_argument(
        "--challengers",
        type=str,
        default=None,
        help="comma subset of CHALLENGER_NAMES to run (supplement shards)",
    )
    ap.add_argument("--no-lgbm", action="store_true", help="skip lgbm challenger (broken libomp)")
    ap.add_argument("--merge-parts", type=Path, nargs="+")
    ap.add_argument("--merge-out", type=Path)
    ap.add_argument("--expect-freq", choices=["1d", "4h", "1h"], default=None)
    ap.add_argument("--expect-lookback", type=int, default=None)
    ap.add_argument("--expect-horizon", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    args = ap.parse_args()

    if args.merge_parts:
        if not args.merge_out:
            ap.error("--merge-parts requires --merge-out")
        return merge_parts(args)
    if not args.bars or not args.out:
        ap.error("run mode requires --bars and --out")

    specs = [] if args.no_targets else parse_specs(args)
    challengers = args.challengers.split(",") if args.challengers else list(CHALLENGER_NAMES)
    unknown = [c for c in challengers if c not in CHALLENGER_NAMES]
    if unknown:
        raise SystemExit(f"unknown challengers: {unknown} (valid: {CHALLENGER_NAMES})")
    model_names = [s.name for s in specs] + challengers

    import torch

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    predictors: dict[str, Any] = {}
    if any(s.kind == "kronos" for s in specs):
        if not args.kronos_repo:
            ap.error("--kronos targets require --kronos-repo")
        sys.path.insert(0, str(args.kronos_repo.resolve()))
        from model import Kronos, KronosPredictor, KronosTokenizer

        for s in specs:
            if s.kind != "kronos":
                continue
            tokenizer = KronosTokenizer.from_pretrained(str(s.tokenizer_dir), local_files_only=True)
            kmodel = Kronos.from_pretrained(str(s.model_dir), local_files_only=True)
            kmodel.eval()
            tokenizer.eval()
            predictors[s.name] = KronosPredictor(
                kmodel, tokenizer, device="cpu", max_context=2048, clip=5.0
            )
    for s in specs:
        if s.kind == "chronos2":
            from chronos import Chronos2Pipeline

            predictors[s.name] = Chronos2Pipeline.from_pretrained(
                str(s.model_dir), device_map="cpu"
            )
        elif s.kind == "bolt":
            from chronos import ChronosBoltPipeline

            predictors[s.name] = ChronosBoltPipeline.from_pretrained(
                str(s.model_dir), device_map="cpu"
            )
        elif s.kind == "timesfm":
            import timesfm

            fm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                str(s.model_dir), torch_compile=False
            )
            fm.compile(
                timesfm.ForecastConfig(
                    max_context=max(args.lookback or 90, 512),
                    max_horizon=max(args.horizon or 18, 24),
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    infer_is_positive=False,
                    fix_quantile_crossing=True,
                )
            )
            predictors[s.name] = fm

    test_start = pd.Timestamp(args.test_start, tz="UTC")
    t0 = time.time()

    for bars_path in args.bars:
        frame = pl.read_parquet(bars_path).sort("event_time")
        event_times, interval = validate_bars(frame)
        interval_h = interval / 3.6e12  # ns -> hours
        if abs(interval_h - 24.0) < 1.0:
            freq = "1d"
        elif abs(interval_h - 4.0) < 0.5:
            freq = "4h"
        elif abs(interval_h - 1.0) < 0.25:
            freq = "1h"
        else:
            raise ValueError(f"{bars_path}: unsupported bar interval {interval_h}h")
        lookback = args.lookback or FREQ_TABLE[freq][0]
        horizon = args.horizon or FREQ_TABLE[freq][1]
        symbol = str(frame["security_id"][0])

        closes = frame["close"].cast(pl.Float64).to_numpy()
        n = frame.height
        rets_all = np.diff(closes) / closes[:-1]
        dow = pd.to_datetime(event_times[1:], unit="ns", utc=True).dayofweek.to_numpy(dtype=float)
        X = _lgbm_features(rets_all, dow)

        # eligible origins: i in test window, enough lookback, room for horizon
        ets = pd.to_datetime(event_times, unit="ns", utc=True)
        eligible = [
            i
            for i in range(max(lookback, args.window + 1), n - horizon - 1)
            if ets[i] >= test_start
        ]
        if len(eligible) > args.origins:
            step = len(eligible) / args.origins
            eligible = [eligible[int(k * step)] for k in range(args.origins)]
        if not eligible:
            raise ValueError(f"{symbol}: no eligible origins after {args.test_start}")

        path_rankic = np.full((len(eligible), len(model_names)), np.nan)
        path_ic = np.full_like(path_rankic, np.nan)
        vol2_hat = np.full_like(path_rankic, np.nan)
        vol2_actual = np.full(len(eligible), np.nan)
        ret_hat = np.full_like(path_rankic, np.nan)
        ret_actual = np.full(len(eligible), np.nan)

        for oi, i in enumerate(eligible):
            hist_close = closes[: i + 1]
            actual_path = closes[i + 1 : i + 1 + horizon]
            vol2_actual[oi] = realized_vol2(actual_path)
            ret_actual[oi] = float(closes[i + horizon] / closes[i] - 1.0)
            last_close = float(closes[i])

            for s in specs:
                col = model_names.index(s.name)
                try:
                    if s.kind == "kronos":
                        hist = frame.slice(0, i + 1)
                        pdf = (
                            hist.select(
                                ["open", "high", "low", "close", "volume"]
                                + (["amount"] if "amount" in hist.columns else [])
                            )
                            .tail(lookback)
                            .to_pandas()
                        )
                        x_ts = pd.Series(
                            pd.to_datetime(hist["event_time"].tail(lookback).to_numpy())
                        )
                        step = x_ts.iloc[-1] - x_ts.iloc[-2]
                        y_ts = pd.Series([x_ts.iloc[-1] + step * k for k in range(1, horizon + 1)])
                        path = kronos_path(
                            predictors[s.name], pdf, x_ts, y_ts, args.samples, horizon
                        )
                    elif s.kind == "chronos2":
                        path = chronos2_path(predictors[s.name], hist_close[-lookback:], horizon)
                    elif s.kind == "bolt":
                        path = bolt_path(predictors[s.name], hist_close[-lookback:], horizon)
                    elif s.kind == "timesfm":
                        path = timesfm_path(predictors[s.name], hist_close[-lookback:], horizon)
                    else:
                        continue
                    path_rankic[oi, col] = spearman(path, actual_path)
                    path_ic[oi, col] = pearson(path, actual_path)
                    vol2_hat[oi, col] = realized_vol2(path)
                    ret_hat[oi, col] = float(path[-1] / last_close - 1.0)
                except Exception:  # noqa: BLE001 - target failure -> NaN, disclosed
                    pass

            # challengers
            rets = np.diff(hist_close[-(args.window + 1) :]) / hist_close[-(args.window + 1) : -1]
            long_start = max(0, hist_close.size - (args.garch_window + 1))
            rets_long = np.diff(hist_close[long_start:]) / hist_close[long_start:-1]
            params = challenger_params(rets, rets_long)
            try:
                q_l = None if args.no_lgbm else lgbm_quantiles(X, rets_all, i, args.garch_window)
            except Exception:  # noqa: BLE001
                q_l = None
            paths, vol2s, reth = challenger_paths(params, q_l, last_close, horizon)
            for name in challengers:
                col = model_names.index(name)
                path_rankic[oi, col] = spearman(paths[name], actual_path)
                path_ic[oi, col] = pearson(paths[name], actual_path)
                vol2_hat[oi, col] = vol2s[name]
                ret_hat[oi, col] = reth[name]

            if (oi + 1) % args.checkpoint_every == 0 or oi == len(eligible) - 1:
                np.savez_compressed(
                    args.out.with_suffix(".paths.npz"),
                    path_rankic=path_rankic[: oi + 1],
                    path_ic=path_ic[: oi + 1],
                    vol2_hat=vol2_hat[: oi + 1],
                    vol2_actual=vol2_actual[: oi + 1],
                    ret_hat=ret_hat[: oi + 1],
                    ret_actual=ret_actual[: oi + 1],
                    model_names=np.asarray(model_names),
                    meta_json=np.array(
                        json.dumps(
                            {
                                "asset": symbol,
                                "freq": freq,
                                "lookback": lookback,
                                "horizon": horizon,
                                "test_start": str(test_start),
                                "n_origins_done": oi + 1,
                            }
                        )
                    ),
                )
                print(
                    f"{symbol} {oi + 1}/{len(eligible)} origins "
                    f"({(time.time() - t0) / 60:.1f} min)",
                    flush=True,
                )

        receipt = {
            "schema": "native_eval.v1",
            "protocol": "kronos-paper App.D (arXiv:2508.02739)",
            "asset": symbol,
            "freq": freq,
            "lookback": lookback,
            "horizon": horizon,
            "test_start": str(test_start),
            "n_origins": len(eligible),
            "bars_sha256": _sha256(bars_path),
            "script_sha256": _sha256(Path(__file__)),
            "seed": args.seed,
            "samples": args.samples,
            "metrics": summarize_parts(
                [
                    {
                        "asset": symbol,
                        "model_names": model_names,
                        "path_rankic": path_rankic,
                        "vol2_hat": vol2_hat,
                        "vol2_actual": vol2_actual,
                        "ret_hat": ret_hat,
                        "ret_actual": ret_actual,
                    }
                ]
            )["assets"][symbol],
            "runtime_s": time.time() - t0,
            "live_pnl_claim": False,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
        print(f"{symbol}: receipt -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
