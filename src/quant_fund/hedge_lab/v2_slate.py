"""dipcatcher.sota.v2 — lanes 2-5 of the frozen challenger slate.

Lane 1 (``ml_trees``) is registered by ``configs/sota_protocol_v2.yaml`` and
executed by ``data/file_us/metadata/_run_ml_lane.py``; it writes
``artifacts/hedge_lab/ml_lane.json`` citing ``protocol_sha256`` = sha256 of
the slate file bytes. This module covers the companion lanes registered in
``configs/sota_protocol_v2_lanes.yaml``:

- lane 2: pre-registered horizon change to ``future_idio_return_5`` / ``_20``
  with embargo = label horizon, same challenger trio vs public ridge;
- lane 3: causal Gaussian-HMM gate on the frozen top-5 12-1 book vs ungated
  + SPY under White's Reality Check (book overlay; never flips sign);
- lane 4: conservative offline contextual bandit over {cash, SPY, frozen
  top-5}, trained through 2024-12-31, frozen, scored once on the holdout;
- lane 5: wide-tape point-in-time universe, refit only the best tree from
  lane 1 (selected on selection-window IC before the holdout is scored).

Lanes 2 and 5 are ranker lanes: ``blend_weight`` may leave 0 only when a
challenger clears DM + White RC + Hansen SPA + Romano-Wolf StepM on the
frozen holdout AND the calibration gate passes. Lanes 3 and 4 are book-level
overlays; they never move ``blend_weight``. Not a live P&L claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from numpy.typing import NDArray

from quant_fund.config import load_config
from quant_fund.hedge_lab.directional import close_matrix, simple_returns, topk_long_returns
from quant_fund.hedge_lab.gated_race import slice_ic_window
from quant_fund.hedge_lab.lightspeed_book import _align_ic, _date_key, _ic_card
from quant_fund.hedge_lab.promotion import clears_book_overlay, clears_cs_promotion
from quant_fund.hedge_lab.resources import cap_blas_threads
from quant_fund.hedge_lab.scoreboard import book_economic_scoreboard
from quant_fund.lightspeed.specs import HOLDOUT_START, SELECTION_END
from quant_fund.metrics.inference import diebold_mariano, overlap_aware_hac_lags
from quant_fund.metrics.snooping import reality_check, spa_test, stepm
from quant_fund.models.ranking import PUBLIC_FEATURES, available_features, drop_oracle_columns
from quant_fund.pipeline.dataset import build_gold, design_matrix, panel
from quant_fund.pipeline.train import _label_horizon
from quant_fund.research.benches import oos_rank_scores

Array = NDArray[np.float64]

SLATE_PATH = "configs/sota_protocol_v2_lanes.yaml"
ALPHA = 0.05
_EPS = 1e-12

LANE1_TREES: tuple[str, ...] = ("gbrt", "lambdarank", "xgboost")


def _sha256_file(path: str | Path) -> str:
    """sha256 of raw file bytes — the convention _run_ml_lane.py uses."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_slate(path: str | Path = SLATE_PATH) -> tuple[dict[str, Any], str, str]:
    """Parse the lanes slate. Returns (slate, slate_sha256, parent_sha256).

    The parent file (``configs/sota_protocol_v2.yaml``) is hashed over its raw
    bytes, matching ``_run_ml_lane.py``; the lanes file is hashed over its raw
    bytes too so the citation survives whitespace-only accidents.
    """
    raw_path = Path(path)
    raw = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("lanes slate must be a mapping")
    for key in ("slate_id", "parent_slate", "evaluation", "lanes", "promotion"):
        if key not in raw:
            raise ValueError(f"lanes slate missing required key {key!r}")
    parent = Path(str(raw["parent_slate"]))
    if not parent.is_file():
        raise FileNotFoundError(f"parent slate missing: {parent}")
    return raw, _sha256_file(raw_path), _sha256_file(parent)


def _lane_spec(slate: dict[str, Any], lane: int) -> dict[str, Any]:
    for spec in slate["lanes"]:
        if int(spec["lane"]) == int(lane):
            return spec
    raise ValueError(f"lane {lane} is not in the frozen slate")


def _fingerprint(
    x: np.ndarray,
    y: np.ndarray,
    used: list[str],
    label: str,
    *,
    scheme: str,
    train_bars: int,
    val_bars: int,
    test_bars: int,
    embargo_bars: int,
) -> str:
    """Same cache key shape as ``_eval_paper_ic.py``."""
    digest = hashlib.sha256()
    digest.update(label.encode())
    digest.update(",".join(used).encode())
    digest.update(
        f"{scheme}|{train_bars}|{val_bars}|{test_bars}|{embargo_bars}".encode()
    )
    digest.update(np.ascontiguousarray(x[:: max(1, x.shape[0] // 4096)]).tobytes())
    digest.update(np.ascontiguousarray(y[:: max(1, y.shape[0] // 4096)]).tobytes())
    digest.update(str(x.shape).encode())
    return digest.hexdigest()[:16]


def _oos_scores(
    cfg: Any,
    label: str,
    engines: list[str],
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Purged walk-forward OOS scores, cached under a v2-private namespace.

    The shared ``oos_scores_<label>`` cache was written under the published
    race's eval knobs; v2 lanes run at the config's declared budget, so they
    use ``oos_scores_v2_<label>`` and never mix budgets silently.
    """
    gold = drop_oracle_columns(panel(cfg))
    gold = gold.filter(gold["security_id"] != cfg.data.benchmark_id)
    feats = available_features(gold.columns, PUBLIC_FEATURES)
    x, y, dates, used, ids = design_matrix(gold, label, feats)
    horizon = _label_horizon(label)
    fp = _fingerprint(
        x,
        y,
        used,
        label,
        scheme=cfg.validation.scheme,
        train_bars=int(cfg.validation.train_bars),
        val_bars=int(cfg.validation.val_bars),
        test_bars=int(cfg.validation.test_bars),
        embargo_bars=int(cfg.embargo_bars()),
    )
    cache_dir = Path(cfg.data.root) / "metadata" / f"oos_scores_v2_{label}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, Array] = {}
    for engine in engines:
        cache_path = cache_dir / f"{engine}_{fp}.npy"
        if use_cache and cache_path.is_file():
            cached = np.load(cache_path)
            if cached.shape[0] == x.shape[0]:
                scores[engine] = cached
                print(f"CACHED {engine} fp={fp}", flush=True)
                continue
        print(f"FIT {engine} label={label} fp={fp}", flush=True)
        scores[engine] = np.asarray(
            oos_rank_scores(
                engine, cfg, x, y, dates, ids, horizon_bars=horizon, feature_names=used
            ),
            dtype=float,
        )
        np.save(cache_path, scores[engine])
    n_dates = len({str(d) for d in dates.tolist()})
    return {
        "x": x,
        "y": y,
        "dates": dates,
        "ids": ids,
        "used": used,
        "horizon": int(horizon),
        "fingerprint": fp,
        "n_dates": int(n_dates),
        "scores": scores,
    }


def _lane_gates(
    aligned: dict[str, Array],
    *,
    benchmark: str,
    n_boot: int,
    lags: int | None,
) -> dict[str, Any]:
    """Sealed four-gate stack: DM of -IC vs benchmark + RC + SPA + StepM."""
    if benchmark not in aligned:
        raise ValueError("benchmark IC series missing")
    bench = aligned[benchmark]
    names = [n for n in aligned if n != benchmark]
    if not names or bench.size < 10:
        return {
            "dm": {},
            "reality_check_p": float("nan"),
            "spa_p_consistent": float("nan"),
            "stepm_rejected": [],
            "promote": False,
        }
    dm: dict[str, Any] = {}
    cols: list[Array] = []
    kept: list[str] = []
    for name in names:
        series = aligned[name]
        n = min(len(series), len(bench))
        if n < 10:
            continue
        result = diebold_mariano(
            -series[:n], -bench[:n], lags=lags, name_a=name, name_b=benchmark
        )
        dm[name] = {
            "statistic": result.statistic,
            "p_value": result.p_value,
            "preferred": result.preferred,
            "mean_loss_diff": result.mean_loss_diff,
            "lags": int(result.lags),
        }
        cols.append(series[:n] - bench[:n])
        kept.append(name)
    if not cols:
        return {
            "dm": dm,
            "reality_check_p": float("nan"),
            "spa_p_consistent": float("nan"),
            "stepm_rejected": [],
            "promote": False,
        }
    f = np.column_stack(cols)
    rc = reality_check(f, n_boot=n_boot, seed=7)
    spa = spa_test(f, n_boot=n_boot, seed=7)
    step = stepm(f, n_boot=n_boot, alpha=ALPHA, seed=7)
    rejected = [kept[i] for i, flag in enumerate(step.rejected) if flag]
    means = {
        name: float(np.mean(aligned[name][: min(len(aligned[name]), len(bench))]))
        for name in dm
    }
    cleared = [
        name
        for name, row in dm.items()
        if clears_cs_promotion(
            name=name,
            mean_ic=means[name],
            dm_preferred=str(row["preferred"]),
            dm_p=float(row["p_value"]),
            stepm_rejected=rejected,
            rc_p=float(rc.p_value),
            spa_p=float(spa.p_consistent),
            alpha=ALPHA,
        )
    ]
    promote = bool(cleared)
    return {
        "dm": dm,
        "reality_check_p": rc.p_value,
        "spa_p_consistent": spa.p_consistent,
        "spa_p_lower": spa.p_lower,
        "spa_p_upper": spa.p_upper,
        "stepm_rejected": rejected,
        "stepm_adjusted_p": dict(
            zip(names, [float(p) for p in step.adjusted_p], strict=False)
        ),
        "n_boot": int(n_boot),
        "hac_lags": lags,
        "promote": promote,
        "cleared": cleared,
        "note": (
            "promote is True only if the same challenger has mean IC > 0, "
            "wins pairwise DM of -IC vs ridge at 5%, and is inside the StepM "
            "rejection set, while White RC and SPA also reject no-skill. "
            "A less-negative IC does not clear. blend_weight stays 0 here."
        ),
    }


def _ic_windows(
    cards: list[dict[str, Any]],
    *,
    benchmark: str,
    n_boot: int,
    horizon: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    windows = {
        "full": (None, None),
        "selection": (None, SELECTION_END),
        "holdout": (HOLDOUT_START, None),
    }
    for name, (start, end) in windows.items():
        sliced = [slice_ic_window(c, start=start, end=end, horizon_bars=int(horizon)) for c in cards]
        aligned = _align_ic(sliced)
        n = int(aligned[benchmark].size) if benchmark in aligned else 0
        lags = overlap_aware_hac_lags(n, int(horizon)) if n else None
        slim = [
            {k: v for k, v in c.items() if k not in {"ic_series", "ic_dates"}}
            for c in sliced
        ]
        out[name] = {
            "n_dates": n,
            "hac_lags": lags,
            "cs_ic": slim,
            "gates": _lane_gates(aligned, benchmark=benchmark, n_boot=n_boot, lags=lags),
        }
    return out


def _calibration(cfg: Any) -> dict[str, Any]:
    """Jackknife+ floor / CQR-ACI Kupiec / PIT-KS. Fail-closed on error."""
    from quant_fund.research.sota_protocol import calibration_gate

    try:
        return calibration_gate(drop_oracle_columns(panel(cfg)), cfg)
    except Exception as exc:  # noqa: BLE001 - gate failure must fail closed
        return {
            "family": "calibration_gate",
            "pass": False,
            "recorded": False,
            "error": str(exc),
        }


def _write_receipt(receipt: dict[str, Any], artifact: str, root: Path) -> dict[str, Any]:
    payload = json.dumps(receipt, indent=2, default=str)
    public = Path(artifact)
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(payload, encoding="utf-8")
    meta = root / "metadata" / public.name
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(payload, encoding="utf-8")
    receipt["artifact_path"] = str(public)
    receipt["metadata_path"] = str(meta)
    return receipt


def _base_receipt(
    slate: dict[str, Any],
    slate_sha: str,
    parent_sha: str,
    lane: dict[str, Any],
    cfg: Any,
    n_boot: int,
) -> dict[str, Any]:
    source = "SYNTHETIC" if cfg.data.source == "synthetic" else str(cfg.data.source)
    return {
        "catalog": "hedge_lab_analytics",
        "protocol_id": slate["protocol_id"],
        "slate_id": slate["slate_id"],
        "protocol_sha256": parent_sha,
        "slate_sha256": slate_sha,
        "slate_file": SLATE_PATH,
        "parent_slate": slate["parent_slate"],
        "lane": int(lane["lane"]),
        "lane_name": lane["name"],
        "config": slate["evaluation"]["config"],
        "feature_set": "public",
        "benchmark": slate["evaluation"]["benchmark"],
        "selection_end": SELECTION_END,
        "holdout_start": HOLDOUT_START,
        "scheme": cfg.validation.scheme,
        "train_bars": int(cfg.validation.train_bars),
        "val_bars": int(cfg.validation.val_bars),
        "test_bars": int(cfg.validation.test_bars),
        "embargo_bars": int(cfg.embargo_bars()),
        "n_boot": int(n_boot),
        "alpha": ALPHA,
        "one_way_cost": float(slate["evaluation"]["one_way_cost"]),
        "data_source": source,
        "synthetic_not_promotable": source.upper() == "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "execution_claim": "paper_backtest",
        "champion": "ridge",
        "blend_weight": 0.0,
    }


def run_lane1_calibration(
    slate_path: str = SLATE_PATH,
) -> dict[str, Any]:
    """Attach the calibration gate to the lane-1 receipt (ml_lane.json).

    Lane 1's promotion list is the four statistical gates; the calibration
    gate is recorded alongside so a promote=true could not silently size the
    book without a green calibration check downstream.
    """
    slate, slate_sha, parent_sha = load_slate(slate_path)
    cfg = load_config(slate["evaluation"]["config"])
    receipt_path = Path(slate["evaluation"]["lane1_receipt"])
    if not receipt_path.is_file():
        raise FileNotFoundError("lane-1 receipt (ml_lane.json) not found")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    calibration = _calibration(cfg)
    receipt["calibration"] = calibration
    receipt["calibration_required_for_blend"] = True
    receipt["promote_statistical_gates"] = bool(receipt.get("promote"))
    receipt["promote"] = bool(
        receipt.get("promote")
        and calibration.get("pass")
        and str(receipt.get("data_source", "")).upper() != "SYNTHETIC"
    )
    payload = json.dumps(receipt, indent=2, default=str)
    receipt_path.write_text(payload, encoding="utf-8")
    meta = Path(cfg.data.root) / "metadata" / receipt_path.name
    meta.write_text(payload, encoding="utf-8")
    receipt["calibration_pass"] = bool(calibration.get("pass"))
    return receipt


def run_lane2(
    slate_path: str = SLATE_PATH,
    *,
    n_boot: int = 2000,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Pre-registered horizon change: future_idio_return_5 and _20."""
    slate, slate_sha, parent_sha = load_slate(slate_path)
    lane = _lane_spec(slate, 2)
    cfg = load_config(slate["evaluation"]["config"])
    engines = [slate["evaluation"]["benchmark"], *lane["challengers"]]
    per_label: dict[str, Any] = {}
    for label in lane["labels"]:
        horizon = _label_horizon(label)
        # Pre-registered: embargo equals the label horizon.
        cfg.validation.embargo_bars = int(horizon)
        blob = _oos_scores(cfg, label, engines, use_cache=use_cache)
        hac = overlap_aware_hac_lags(blob["n_dates"], horizon)
        cards = [
            _ic_card(name, blob["scores"][name], blob["y"], blob["dates"], hac)
            for name in engines
        ]
        windows = _ic_windows(
            cards, benchmark=slate["evaluation"]["benchmark"], n_boot=n_boot,
            horizon=horizon,
        )
        per_label[label] = {
            "horizon_bars": int(horizon),
            "embargo_bars": int(horizon),
            "fingerprint": blob["fingerprint"],
            "n_dates": int(blob["n_dates"]),
            "windows": windows,
        }
    receipt = _base_receipt(slate, slate_sha, parent_sha, lane, cfg, n_boot)
    receipt["embargo_bars"] = {
        label: int(_label_horizon(label)) for label in lane["labels"]
    }
    any_promote = any(
        set(per_label[label]["windows"]["selection"]["gates"].get("cleared") or [])
        & set(per_label[label]["windows"]["holdout"]["gates"].get("cleared") or [])
        for label in lane["labels"]
    )
    receipt.update(
        {
            "labels": list(lane["labels"]),
            "engines": engines,
            "results": per_label,
            "decision_window": "selection_and_holdout",
            "promote": bool(
                any_promote and str(cfg.data.source).upper() != "SYNTHETIC"
            ),
            "note": (
                "Pre-registered horizon slate. Embargo equals the label "
                "horizon. A horizon win still needs the calibration gate "
                "before blend_weight could move; the gate is shared across "
                "ranker lanes and recorded on the lane-1 receipt. Champion "
                "stays public ridge. Not live P&L."
            ),
        }
    )
    return _write_receipt(receipt, lane["receipt"], Path(cfg.data.root))


def _book_inputs(cfg: Any) -> tuple[list[Any], dict[str, Array]]:
    """Frozen top-5 12-1 inputs: pivot total-return closes from gold labels."""
    labels = pl.read_parquet(Path(cfg.data.root) / "gold" / "labels.parquet")
    price = "close_total_return" if "close_total_return" in labels.columns else "close"
    sub = labels.select(["event_time", "security_id", price]).drop_nulls()
    piv = sub.pivot(index="event_time", on="security_id", values=price).sort("event_time")
    dates = piv["event_time"].to_list()
    closes = {
        c: piv[c].to_numpy().astype(float) for c in piv.columns if c != "event_time"
    }
    n = min(len(v) for v in closes.values())
    for key in list(closes):
        closes[key] = np.asarray(closes[key][:n], dtype=float)
    return dates[:n], closes


def _window_mask(dates: list[Any], *, start: str | None, end: str | None) -> np.ndarray:
    keys = [_date_key(d) for d in dates]
    return np.asarray(
        [
            (start is None or k >= start) and (end is None or k <= end)
            for k in keys
        ],
        dtype=bool,
    )


def _drawdown(returns: Array) -> Array:
    wealth = np.cumprod(1.0 + np.where(np.isfinite(returns), returns, 0.0))
    peak = np.maximum.accumulate(wealth)
    return wealth / np.maximum(peak, _EPS) - 1.0


def _rolling_vol(returns: Array, window: int) -> Array:
    r = np.asarray(returns, dtype=float)
    out = np.full(r.size, np.nan)
    for i in range(window, r.size):
        out[i] = float(np.std(r[i - window : i], ddof=1))
    return out


def run_lane3(
    slate_path: str = SLATE_PATH,
    *,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Causal GaussianHMM gate on the frozen top-5 12-1 book."""
    from quant_fund.models.regime import GaussianHMMRegime

    slate, slate_sha, parent_sha = load_slate(slate_path)
    lane = _lane_spec(slate, 3)
    gate = lane["gate"]
    cfg = load_config(slate["evaluation"]["config"])
    dates, closes = _book_inputs(cfg)
    names = sorted(closes)
    _, px = close_matrix(closes, names)
    cost = float(slate["evaluation"]["one_way_cost"])
    book = topk_long_returns(px, top_k=5, one_way_cost=cost)
    if "SPY" not in closes:
        raise ValueError("SPY missing from the tape; cannot score the frozen book")
    spy = simple_returns(closes["SPY"])

    spy_r = np.asarray(spy, dtype=float)
    vol20 = _rolling_vol(spy_r, 20)
    feats = np.column_stack([spy_r, vol20])

    fit_mask = _window_mask(dates, start=None, end=str(gate["fit_through"]))
    fit_mask &= np.isfinite(feats).all(axis=1)
    regime = GaussianHMMRegime(n_states=int(gate["n_states"]), seed=int(gate["seed"]))
    regime.fit(feats[fit_mask])
    stress_state = next(
        (s for s, name in regime.labels.items() if name == "stress"), None
    )
    if stress_state is None:
        raise RuntimeError("HMM did not label a stress state")
    filtered = regime.predict_proba(feats)  # forward filter; causal at each t
    lag = int(gate["lag_bars"])
    mult = np.ones(len(dates), dtype=float)
    mult[lag:] = 1.0 - filtered[: len(dates) - lag, int(stress_state)]
    mult = np.clip(mult, 0.0, 1.0)
    switch = np.zeros(len(dates), dtype=float)
    switch[1:] = np.abs(mult[1:] - mult[:-1])
    gated = mult * book - float(gate["switch_cost"]) * switch

    out: dict[str, Any] = {}
    series = {
        "hmm_gated_topk5": gated,
        "topk5_12_1_ungated": book,
        "spy_buy_hold": spy,
    }
    for wname, (start, end) in {
        "selection": (None, SELECTION_END),
        "holdout": (HOLDOUT_START, None),
        "full": (None, None),
    }.items():
        mask = _window_mask(dates, start=start, end=end)
        mask[0] = False
        out[wname] = {
            "n_returns": int(mask.sum()),
            "cards": {
                k: book_economic_scoreboard(v[mask], data_source="file")
                for k, v in series.items()
            },
        }
    ho = _window_mask(dates, start=HOLDOUT_START, end=None)
    ho[0] = False
    diff = np.column_stack([gated[ho] - book[ho], gated[ho] - spy[ho]])
    excess_means = [float(np.mean(diff[:, 0])), float(np.mean(diff[:, 1]))]
    rc = reality_check(diff, n_boot=n_boot, seed=7)
    spa = spa_test(diff, n_boot=n_boot, seed=7)
    overlay_claim = clears_book_overlay(
        excess_means=excess_means,
        rc_p=float(rc.p_value),
        spa_p=float(spa.p_consistent),
    )
    receipt = _base_receipt(slate, slate_sha, parent_sha, lane, cfg, n_boot)
    receipt.update(
        {
            "book": lane["book"],
            "gate": dict(gate),
            "stress_state": int(stress_state),
            "hmm_labels": {str(k): v for k, v in regime.labels.items()},
            "hmm_means": np.asarray(regime.model.means_).tolist(),
            "multiplier_holdout_mean": float(np.mean(mult[ho])),
            "windows": out,
            "reality_check": {
                "statistic": rc.statistic,
                "p_value": rc.p_value,
                "best_index": rc.best_index,
                "differentials": ["gated_minus_ungated", "gated_minus_spy"],
            },
            "spa": {
                "p_consistent": spa.p_consistent,
                "p_lower": spa.p_lower,
                "p_upper": spa.p_upper,
            },
            "promote": False,
            "overlay_claim": overlay_claim,
            "affects_blend_weight": False,
            "note": (
                "Causal GaussianHMM gate on the frozen top-5 12-1 book. HMM "
                f"fit through {gate['fit_through']}; forward-filtered state "
                "only, multiplier in [0,1], never flips the sign. White RC "
                "over the three-model set on the holdout. Book-level claim; "
                "blend_weight stays 0. Not live P&L."
            ),
        }
    )
    return _write_receipt(receipt, lane["receipt"], Path(cfg.data.root))


def run_lane4(
    slate_path: str = SLATE_PATH,
    *,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Conservative offline contextual bandit over {cash, SPY, frozen top-5}."""
    from sklearn.linear_model import Ridge

    slate, slate_sha, parent_sha = load_slate(slate_path)
    lane = _lane_spec(slate, 4)
    cfg = load_config(slate["evaluation"]["config"])
    lane1 = Path(slate["evaluation"]["lane1_receipt"])
    if not lane1.is_file():
        raise FileNotFoundError("lane 4 requires the lane-1 receipt first")
    dates, closes = _book_inputs(cfg)
    names = sorted(closes)
    _, px = close_matrix(closes, names)
    cost = float(slate["evaluation"]["one_way_cost"])
    book = topk_long_returns(px, top_k=5, one_way_cost=cost)
    spy = simple_returns(closes["SPY"])
    n = len(dates)

    rewards = np.column_stack([np.zeros(n), spy, book])  # cash, SPY, topk5
    spy_dd = _drawdown(spy)
    book_dd = _drawdown(book)
    spy_vol = _rolling_vol(spy, 20)
    book_vol = _rolling_vol(book, 20)

    gold = drop_oracle_columns(panel(cfg))
    gold = gold.filter(gold["security_id"] != cfg.data.benchmark_id)
    cs = (
        gold.group_by("event_time")
        .agg(
            [
                pl.col("cs_z_mom_12_1").mean().alias("cs_mom_12_1_mean"),
                pl.col("cs_z_vol_20").mean().alias("cs_vol20_mean"),
                pl.col("cs_z_mom_12_1").std().alias("cs_mom_12_1_std"),
            ]
        )
        .sort("event_time")
    )
    cs_map = {
        _date_key(row[0]): (float(row[1]), float(row[2]), float(row[3]))
        for row in cs.iter_rows()
    }
    keys = [_date_key(d) for d in dates]
    lag = int(lane["state"]["lag_bars"])
    state = np.full((n, 8), np.nan)
    for t in range(lag, n):
        agg = cs_map.get(keys[t - lag])
        if agg is None:
            continue
        state[t] = [
            spy[t - lag],
            spy_vol[t - lag],
            book_vol[t - lag],
            book_dd[t - lag],
            spy_dd[t - lag],
            agg[0],
            agg[1],
            agg[2],
        ]

    train_mask = _window_mask(dates, start=None, end=str(lane["train_through"]))
    train_mask &= np.isfinite(state).all(axis=1)
    hold_mask = _window_mask(dates, start=HOLDOUT_START, end=None)
    hold_mask &= np.isfinite(state).all(axis=1)

    models: dict[int, Any] = {}
    bounds: dict[int, tuple[float, float]] = {}
    for a in range(3):
        model = Ridge(alpha=1.0)
        model.fit(state[train_mask], rewards[train_mask, a])
        models[a] = model
        bounds[a] = (
            float(np.min(rewards[train_mask, a])),
            float(np.max(rewards[train_mask, a])),
        )

    policy = np.zeros(n, dtype=int)
    for t in range(n):
        if not np.isfinite(state[t]).all():
            policy[t] = 0
            continue
        q = [
            float(np.clip(models[a].predict(state[t : t + 1])[0], *bounds[a]))
            for a in range(3)
        ]
        policy[t] = int(np.argmax(q))

    rl_ret = np.zeros(n, dtype=float)
    prev_action = 0
    for t in range(n):
        a = int(policy[t])
        rl_ret[t] = rewards[t, a] - (cost if a != prev_action else 0.0)
        prev_action = a

    out: dict[str, Any] = {}
    series = {
        "rl_policy": rl_ret,
        "spy_buy_hold": spy,
        "topk5_12_1_ungated": book,
    }
    for wname, (start, end) in {
        "selection": (None, SELECTION_END),
        "holdout": (HOLDOUT_START, None),
    }.items():
        mask = _window_mask(dates, start=start, end=end)
        mask &= np.isfinite(state).all(axis=1)
        mask[0] = False
        out[wname] = {
            "n_returns": int(mask.sum()),
            "cards": {
                k: book_economic_scoreboard(v[mask], data_source="file")
                for k, v in series.items()
            },
        }
    diff = np.column_stack(
        [
            rl_ret[hold_mask] - spy[hold_mask],
            rl_ret[hold_mask] - book[hold_mask],
        ]
    )
    rc = reality_check(diff, n_boot=n_boot, seed=7)
    spa = spa_test(diff, n_boot=n_boot, seed=7)
    overlay_claim = clears_book_overlay(
        excess_means=[float(np.mean(diff[:, 0])), float(np.mean(diff[:, 1]))],
        rc_p=float(rc.p_value),
        spa_p=float(spa.p_consistent),
    )
    action_counts = {
        list(lane["actions"])[a]: int((policy[hold_mask] == a).sum())
        for a in range(3)
    }
    receipt = _base_receipt(slate, slate_sha, parent_sha, lane, cfg, n_boot)
    receipt.update(
        {
            "actions": list(lane["actions"]),
            "method": lane["method"],
            "state_columns": list(lane["state"]["columns"]),
            "train_through": str(lane["train_through"]),
            "n_train_dates": int(train_mask.sum()),
            "holdout_action_counts": action_counts,
            "windows": out,
            "reality_check": {
                "statistic": rc.statistic,
                "p_value": rc.p_value,
                "best_index": rc.best_index,
                "differentials": ["rl_minus_spy", "rl_minus_ungated_topk5"],
            },
            "promote": False,
            "affects_blend_weight": False,
            "overlay_claim": overlay_claim,
            "overlay_replacement": overlay_claim,
            "note": (
                "Offline contextual bandit, actions {cash, SPY, frozen top-5}. "
                "Ridge Q per action clipped to the observed reward range; the "
                "policy cannot invent an action absent from the tape. Every "
                "action's reward is observable each day on the tape, so no "
                "propensity truncation is needed; the logged-action "
                "constraint is the closed three-action set. Trained through "
                f"{lane['train_through']}, frozen, scored once on the "
                "holdout. Reward is next-bar net return after a 10 bp switch "
                "cost, not Sharpe. Book-level overlay only; blend_weight "
                "stays 0. Not live P&L."
            ),
        }
    )
    return _write_receipt(receipt, lane["receipt"], Path(cfg.data.root))


def _best_tree_from_lane1(cfg: Any, slate: dict[str, Any]) -> dict[str, Any]:
    """Selection-window mean IC of the lane-1 trio on file_us, lane-1 budget.

    ``ml_lane.json`` stores only full-window ICs, so the trio is re-scored
    under the registered lane-1 settings (hedge_lab defaults, embargo 5) and
    the winner is chosen on the selection window — no holdout information.
    """
    label = "future_idio_return_1"
    blob = _oos_scores(cfg, label, list(LANE1_TREES), use_cache=True)
    hac = overlap_aware_hac_lags(blob["n_dates"], blob["horizon"])
    sel_ic: dict[str, float] = {}
    for name in LANE1_TREES:
        card = _ic_card(name, blob["scores"][name], blob["y"], blob["dates"], hac)
        sliced = slice_ic_window(card, end=SELECTION_END)
        sel_ic[name] = float(sliced["mean_ic"])
    finite = {k: v for k, v in sel_ic.items() if np.isfinite(v)}
    if not finite:
        raise RuntimeError("lane-1 trio ICs are all non-finite on the selection window")
    best = max(finite, key=lambda k: finite[k])
    return {"best_tree": best, "selection_mean_ic": sel_ic, "fingerprint": blob["fingerprint"]}


def run_lane5(
    slate_path: str = SLATE_PATH,
    *,
    n_boot: int = 2000,
    use_cache: bool = True,
) -> dict[str, Any]:
    """PIT universe rebuild on the wide tape; refit only lane 1's best tree."""
    slate, slate_sha, parent_sha = load_slate(slate_path)
    lane = _lane_spec(slate, 5)
    lane1_path = Path(slate["evaluation"]["lane1_receipt"])
    if not lane1_path.is_file():
        raise FileNotFoundError("lane 5 requires the lane-1 receipt first")

    cfg_narrow = load_config(slate["evaluation"]["config"])
    cfg_narrow.validation.embargo_bars = 5
    pick = _best_tree_from_lane1(cfg_narrow, slate)

    cfg = load_config(slate["evaluation"]["wide_config"])
    cfg.validation.embargo_bars = int(lane["embargo_bars"])
    engines = [slate["evaluation"]["benchmark"], pick["best_tree"]]

    rebuild_stats: dict[str, Any] = {}
    if lane.get("pit_rebuild") or lane["name"] == "pit_universe_rebuild":
        # Gold on the wide tape is stale-versioned; rebuild it from silver so
        # features/labels are materialized under the current PIT membership.
        # panel() then re-validates keys via require_panel_keys_in_membership.
        feats, labs = build_gold(cfg)
        rebuild_stats = {
            "gold_rebuilt": True,
            "n_feature_rows": int(feats.height),
            "n_label_rows": int(labs.height),
            "feature_set_version": feats.get_column("feature_set_version")
            .unique()
            .to_list()
            if "feature_set_version" in feats.columns
            else [],
        }

    blob = _oos_scores(cfg, lane["label"], engines, use_cache=use_cache)
    hac = overlap_aware_hac_lags(blob["n_dates"], blob["horizon"])
    cards = [
        _ic_card(name, blob["scores"][name], blob["y"], blob["dates"], hac)
        for name in engines
    ]
    windows = _ic_windows(
        cards, benchmark=slate["evaluation"]["benchmark"], n_boot=n_boot,
        horizon=blob["horizon"],
    )
    holdout = windows["holdout"]["gates"]
    receipt = _base_receipt(slate, slate_sha, parent_sha, lane, cfg, n_boot)
    receipt["config"] = slate["evaluation"]["wide_config"]
    receipt.update(
        {
            "label": lane["label"],
            "horizon_bars": int(lane["horizon_bars"]),
            "tape": lane["tape"],
            "universe": lane["universe"],
            "engines": engines,
            "best_tree": pick["best_tree"],
            "selection_mean_ic": pick["selection_mean_ic"],
            "lane1_fingerprint": pick["fingerprint"],
            "fingerprint": blob["fingerprint"],
            "features": blob["used"],
            "n_dates": int(blob["n_dates"]),
            "pit_rebuild": rebuild_stats,
            "windows": windows,
            "decision_window": "holdout",
            "promote": bool(
                holdout.get("promote")
                and str(cfg.data.source).upper() != "SYNTHETIC"
            ),
            "note": (
                "Wide-tape PIT membership (ADV-ranked, names enter and exit) "
                "is survivorship-honest within the vendor pool; the pool "
                "itself is still vendor-surviving. One tree, picked on "
                "selection-window IC before the holdout was scored. Champion "
                "stays public ridge; blend_weight stays 0 unless promote. "
                "Not live P&L."
            ),
        }
    )
    return _write_receipt(receipt, lane["receipt"], Path(cfg.data.root))


def main() -> None:
    parser = argparse.ArgumentParser(description="dipcatcher.sota.v2 lanes 2-5")
    parser.add_argument(
        "--lane",
        type=str,
        required=True,
        choices=["2", "3", "4", "5", "cal"],
    )
    parser.add_argument("--slate", default=SLATE_PATH)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    cap_blas_threads(0.6)
    if args.lane == "cal":
        receipt = run_lane1_calibration(args.slate)
        print(
            json.dumps(
                {
                    "promote": receipt["promote"],
                    "calibration_pass": receipt.get("calibration_pass"),
                }
            )
        )
        return
    runners = {2: run_lane2, 3: run_lane3, 4: run_lane4, 5: run_lane5}
    kwargs: dict[str, Any] = {"n_boot": args.n_boot}
    if args.lane in {"2", "5"}:
        kwargs["use_cache"] = not args.no_cache
    receipt = runners[int(args.lane)](args.slate, **kwargs)
    print(json.dumps({"promote": receipt["promote"], "artifact": receipt["artifact_path"]}))


if __name__ == "__main__":
    main()
