"""Public-feature CS race plus risk-controlled gates. Research only.

Walk-forward OOS scores on the 55-name file tape, pairwise DM of −IC vs
ridge, White RC / SPA / StepM. Then dollar-neutral CS long-short at 10 bp
and the causal gate stack (vol / DD / ES / Kelly / CRC / crash / StepM).

Does not move ``blend_weight``. Does not promote on IS Sharpe. SYNTHETIC
is not promotable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.hedge_lab.lightspeed_book import _align_ic, _date_key, _gates, _ic_card
from quant_fund.hedge_lab.mirror import long_short_path
from quant_fund.hedge_lab.promotion import names_clearing_both
from quant_fund.hedge_lab.resources import cap_blas_threads
from quant_fund.hedge_lab.scoreboard import book_economic_scoreboard
from quant_fund.lightspeed.specs import HOLDOUT_START, SELECTION_END
from quant_fund.metrics.inference import mean_tstat, overlap_aware_hac_lags
from quant_fund.risk.gates import GateSpec, apply_gate_stack

Array = NDArray[np.float64]

DEFAULT_ENGINES: tuple[str, ...] = (
    "ridge",
    "nautica",
    "classic",
    "classic_st",
    "reversal",
    "tsmom",
    "vme",
    "combo_ic",
    "krauss",
)

# Pre-declared confirmation set. Do not add lookalikes after seeing holdout.
CONFIRM_ENGINES: tuple[str, ...] = ("ridge", "tsmom", "nautica")

# Lane 1 of dipcatcher.sota.v2. Hashed before the fit. Do not append after the receipt.
ML_LANE_ENGINES: tuple[str, ...] = ("ridge", "gbrt", "lambdarank", "xgboost")


def slice_ic_window(
    card: dict[str, Any],
    *,
    start: str | None = None,
    end: str | None = None,
    horizon_bars: int = 1,
) -> dict[str, Any]:
    """Keep date ICs with ``start <= date <= end`` (inclusive, YYYY-MM-DD)."""
    pairs: list[tuple[str, float]] = []
    for stamp, value in zip(card["ic_dates"], card["ic_series"], strict=True):
        key = _date_key(stamp)
        if start is not None and key < start:
            continue
        if end is not None and key > end:
            continue
        pairs.append((key, float(value)))
    series = np.asarray([p[1] for p in pairs], dtype=float)
    n = int(series.size)
    hac = overlap_aware_hac_lags(n, int(horizon_bars)) if n else None
    _mean_t, t_stat, p_value = mean_tstat(series, hac)
    finite = series[np.isfinite(series)]
    mean = float(np.mean(finite)) if finite.size else float("nan")
    return {
        "name": card["name"],
        "mean_ic": mean,
        "t_ic": t_stat,
        "p_ic": p_value,
        "n_dates": n,
        "ic_series": [float(v) for v in series.tolist()],
        "ic_dates": [p[0] for p in pairs],
        "window_start": start,
        "window_end": end,
        "hac_lags": hac,
    }


def slice_path_window(
    dates: list[str],
    returns: Array,
    *,
    start: str | None = None,
    end: str | None = None,
) -> Array:
    """Keep book returns whose decision date is in ``[start, end]``."""
    r = np.asarray(returns, dtype=float)
    if r.size != len(dates):
        raise ValueError("returns must align with path dates")
    keep: list[float] = []
    for stamp, value in zip(dates, r.tolist(), strict=True):
        key = _date_key(stamp)
        if start is not None and key < start:
            continue
        if end is not None and key > end:
            continue
        keep.append(float(value))
    return np.asarray(keep, dtype=float)


def _gated_cards(returns: Array) -> dict[str, Any]:
    raw = book_economic_scoreboard(returns, data_source="file")
    stack = apply_gate_stack(returns, GateSpec())
    stepm = apply_gate_stack(returns, GateSpec(stepm_enable=True), stepm_excess=returns)
    return {
        "raw": raw,
        "riskstack": book_economic_scoreboard(stack.returns, data_source="file"),
        "riskstack_gate": stack.snapshot(),
        "stepm_stack": book_economic_scoreboard(stepm.returns, data_source="file"),
        "stepm_gate": stepm.snapshot(),
    }


def run_gated_race(
    config_path: str = "configs/hedge_lab.yaml",
    label: str = "future_idio_return_1",
    *,
    one_way_cost: float = 0.001,
    n_boot: int = 1000,
    k_frac: float = 0.2,
    engines: tuple[str, ...] | None = None,
    cpu_fraction: float = 0.6,
    artifact_name: str = "gated_race.json",
    protocol_id: str | None = None,
) -> dict[str, Any]:
    """CS challengers vs ridge, then DD-safe gates on the CS long-short."""
    from quant_fund.config import load_config
    from quant_fund.models.ranking import PUBLIC_FEATURES, available_features, drop_oracle_columns
    from quant_fund.pipeline.dataset import design_matrix, panel
    from quant_fund.pipeline.train import _label_horizon
    from quant_fund.research.benches import oos_rank_scores

    n_threads = cap_blas_threads(cpu_fraction)
    cfg = load_config(config_path)
    source = "SYNTHETIC" if cfg.data.source == "synthetic" else str(cfg.data.source)
    raw_gold = drop_oracle_columns(panel(cfg))
    gold = raw_gold.filter(raw_gold["security_id"] != cfg.data.benchmark_id)
    feats = available_features(gold.columns, PUBLIC_FEATURES)
    x, y, dates, used, ids = design_matrix(gold, label, feats)
    horizon = _label_horizon(label)
    n_dates = len({str(d) for d in dates.tolist()})
    hac = overlap_aware_hac_lags(int(n_dates), int(horizon)) if n_dates else None
    names = list(engines or DEFAULT_ENGINES)

    scores: dict[str, Array] = {}
    for name in names:
        scores[name] = oos_rank_scores(
            name, cfg, x, y, dates, ids, horizon_bars=horizon, feature_names=used
        )

    ic_cards = [_ic_card(name, scores[name], y, dates, hac) for name in names]
    aligned = _align_ic(ic_cards)
    gates = _gates(aligned, benchmark="ridge", n_boot=int(n_boot), lags=hac)
    selection_cards = [
        slice_ic_window(card, end=SELECTION_END, horizon_bars=horizon) for card in ic_cards
    ]
    holdout_cards = [
        slice_ic_window(card, start=HOLDOUT_START, horizon_bars=horizon) for card in ic_cards
    ]
    selection_aligned = _align_ic(selection_cards)
    holdout_aligned = _align_ic(holdout_cards)
    selection_n = int(selection_aligned["ridge"].size) if "ridge" in selection_aligned else 0
    holdout_n = int(holdout_aligned["ridge"].size) if "ridge" in holdout_aligned else 0
    selection_gates = _gates(
        selection_aligned,
        benchmark="ridge",
        n_boot=int(n_boot),
        lags=overlap_aware_hac_lags(selection_n, int(horizon)) if selection_n else None,
    )
    holdout_gates = _gates(
        holdout_aligned,
        benchmark="ridge",
        n_boot=int(n_boot),
        lags=overlap_aware_hac_lags(holdout_n, int(horizon)) if holdout_n else None,
    )
    cleared_both = names_clearing_both(
        list(selection_gates.get("cleared") or []),
        list(holdout_gates.get("cleared") or []),
    )

    ls_cards: dict[str, Any] = {}
    gated: dict[str, Any] = {}
    for name in names:
        path = long_short_path(
            scores[name], y, dates, ids, k_frac=k_frac, one_way_cost=float(one_way_cost)
        )
        rets = np.asarray(path["returns"], dtype=float)
        ls_cards[name] = path["economic"]
        gated[name] = _gated_cards(rets)

    best_dd_safe = None
    best_key = None
    for name, blob in gated.items():
        for key in ("riskstack", "stepm_stack"):
            card = blob[key]
            sharpe = float(card.get("sharpe") or float("nan"))
            dd = float(card.get("max_drawdown") or 0.0)
            if not np.isfinite(sharpe):
                continue
            if dd < -0.05:
                continue
            if best_dd_safe is None or sharpe > float(best_dd_safe.get("sharpe") or float("-inf")):
                best_dd_safe = {**card, "engine": name, "gate": key}
                best_key = f"{name}:{key}"

    promote = bool(cleared_both)
    if source.upper() == "SYNTHETIC":
        promote = False

    receipt: dict[str, Any] = {
        "catalog": "hedge_lab_analytics",
        "config": config_path,
        "label": label,
        "n_names": int(gold["security_id"].n_unique()),
        "n_rows": int(x.shape[0]),
        "n_dates": int(n_dates),
        "features": used,
        "scheme": cfg.validation.scheme,
        "train_bars": int(cfg.validation.train_bars),
        "one_way_cost": float(one_way_cost),
        "delay": 1,
        "cpu_fraction": float(cpu_fraction),
        "n_threads": int(n_threads),
        "engines": names,
        "cs_ic": [
            {k: v for k, v in c.items() if k not in {"ic_series", "ic_dates"}} for c in ic_cards
        ],
        "cs_ls": ls_cards,
        "gated_ls": {
            name: {
                "raw": blob["raw"],
                "riskstack": blob["riskstack"],
                "stepm_stack": blob["stepm_stack"],
                "riskstack_gate": blob["riskstack_gate"],
                "stepm_gate": blob["stepm_gate"],
            }
            for name, blob in gated.items()
        },
        "gates": gates,
        "selection_gates": selection_gates,
        "holdout_gates": holdout_gates,
        "cleared_both": cleared_both,
        "best_dd_safe": best_dd_safe,
        "best_dd_safe_key": best_key,
        "promote": promote,
        "champion": "ridge",
        "champion_alias": False,
        "blend_weight": 0.0,
        "protocol_id": protocol_id,
        "execution_claim": "paper_backtest",
        "live_pnl_claim": False,
        "research_only": True,
        "synthetic_not_promotable": source.upper() == "SYNTHETIC",
        "data_source": source,
        "note": (
            "Public PIT CS-z, future_idio_return_1, delay 1, 10 bp one-way. "
            "Promotion needs pairwise DM of −IC vs ridge plus White RC / SPA / "
            "StepM. Gates cap loss; they do not mint Sharpe 5 from IC≈0. "
            "Not a live P&L claim. blend_weight stays 0."
        ),
    }
    root = Path(cfg.data.root)
    out = root / "metadata" / artifact_name
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, default=str)
    out.write_text(payload, encoding="utf-8")
    public = Path("artifacts") / "hedge_lab" / artifact_name
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(payload, encoding="utf-8")
    receipt["receipt_path"] = str(out)
    receipt["artifact_path"] = str(public)
    return receipt


def _window_bundle(
    ic_cards: list[dict[str, Any]],
    ls_paths: dict[str, dict[str, Any]],
    *,
    start: str | None,
    end: str | None,
    n_boot: int,
    horizon_bars: int = 1,
) -> dict[str, Any]:
    sliced = [
        slice_ic_window(card, start=start, end=end, horizon_bars=horizon_bars) for card in ic_cards
    ]
    aligned = _align_ic(sliced)
    n_win = int(sliced[0]["n_dates"]) if sliced else 0
    lags = overlap_aware_hac_lags(n_win, int(horizon_bars)) if n_win else None
    gates = _gates(aligned, benchmark="ridge", n_boot=int(n_boot), lags=lags)
    ls_cards: dict[str, Any] = {}
    gated: dict[str, Any] = {}
    for name, path in ls_paths.items():
        rets = slice_path_window(path["dates"], path["returns"], start=start, end=end)
        ls_cards[name] = book_economic_scoreboard(rets, data_source="file")
        gated[name] = _gated_cards(rets)
    slim = [{k: v for k, v in c.items() if k not in {"ic_series", "ic_dates"}} for c in sliced]
    return {
        "selection_end": SELECTION_END,
        "holdout_start": HOLDOUT_START,
        "window_start": start,
        "window_end": end,
        "cs_ic": slim,
        "gates": gates,
        "cs_ls": ls_cards,
        "gated_ls": {
            name: {
                "raw": blob["raw"],
                "riskstack": blob["riskstack"],
                "stepm_stack": blob["stepm_stack"],
            }
            for name, blob in gated.items()
        },
    }


def run_holdout_confirm(
    config_path: str = "configs/hedge_lab.yaml",
    label: str = "future_idio_return_1",
    *,
    one_way_cost: float = 0.001,
    n_boot: int = 1000,
    k_frac: float = 0.2,
    cpu_fraction: float = 0.6,
) -> dict[str, Any]:
    """Pre-declared tsmom vs ridge on selection vs holdout IC windows.

    Walk-forward scores are still causal. Engine *choice* used the full
    OOS tape in Wave 156, so holdout here is a persistence check, not a
    license to retune or to flip the champion. Champion stays ridge.
    """
    from quant_fund.config import load_config
    from quant_fund.models.ranking import PUBLIC_FEATURES, available_features, drop_oracle_columns
    from quant_fund.pipeline.dataset import design_matrix, panel
    from quant_fund.pipeline.train import _label_horizon
    from quant_fund.research.benches import oos_rank_scores

    n_threads = cap_blas_threads(cpu_fraction)
    cfg = load_config(config_path)
    source = "SYNTHETIC" if cfg.data.source == "synthetic" else str(cfg.data.source)
    raw_gold = drop_oracle_columns(panel(cfg))
    gold = raw_gold.filter(raw_gold["security_id"] != cfg.data.benchmark_id)
    feats = available_features(gold.columns, PUBLIC_FEATURES)
    x, y, dates, used, ids = design_matrix(gold, label, feats)
    horizon = _label_horizon(label)
    n_dates = len({str(d) for d in dates.tolist()})
    hac = overlap_aware_hac_lags(int(n_dates), int(horizon)) if n_dates else None
    names = list(CONFIRM_ENGINES)

    scores: dict[str, Array] = {}
    for name in names:
        scores[name] = oos_rank_scores(
            name, cfg, x, y, dates, ids, horizon_bars=horizon, feature_names=used
        )

    ic_cards = [_ic_card(name, scores[name], y, dates, hac) for name in names]
    ls_paths: dict[str, dict[str, Any]] = {}
    for name in names:
        ls_paths[name] = long_short_path(
            scores[name], y, dates, ids, k_frac=k_frac, one_way_cost=float(one_way_cost)
        )

    selection = _window_bundle(
        ic_cards, ls_paths, start=None, end=SELECTION_END, n_boot=int(n_boot), horizon_bars=horizon
    )
    holdout = _window_bundle(
        ic_cards, ls_paths, start=HOLDOUT_START, end=None, n_boot=int(n_boot), horizon_bars=horizon
    )
    full = _window_bundle(
        ic_cards, ls_paths, start=None, end=None, n_boot=int(n_boot), horizon_bars=horizon
    )

    receipt: dict[str, Any] = {
        "catalog": "hedge_lab_analytics",
        "config": config_path,
        "label": label,
        "n_names": int(gold["security_id"].n_unique()),
        "n_rows": int(x.shape[0]),
        "n_dates": int(n_dates),
        "features": used,
        "scheme": cfg.validation.scheme,
        "train_bars": int(cfg.validation.train_bars),
        "one_way_cost": float(one_way_cost),
        "delay": 1,
        "cpu_fraction": float(cpu_fraction),
        "n_threads": int(n_threads),
        "engines": names,
        "selection_end": SELECTION_END,
        "holdout_start": HOLDOUT_START,
        "full": full,
        "selection": selection,
        "holdout": holdout,
        "promote": False,
        "champion": "ridge",
        "champion_alias": False,
        "blend_weight": 0.0,
        "execution_claim": "paper_backtest",
        "live_pnl_claim": False,
        "research_only": True,
        "synthetic_not_promotable": source.upper() == "SYNTHETIC",
        "data_source": source,
        "note": (
            "Pre-declared confirmation of tsmom vs ridge. Selection window "
            f"ends {SELECTION_END}; holdout starts {HOLDOUT_START}. "
            "Walk-forward scores are causal. Engine choice in Wave 156 used "
            "the full OOS IC, so holdout is a persistence check. Champion "
            "stays public ridge. blend_weight stays 0. Not a live P&L claim."
        ),
    }
    root = Path(cfg.data.root)
    out = root / "metadata" / "holdout_confirm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, default=str)
    out.write_text(payload, encoding="utf-8")
    public = Path("artifacts") / "hedge_lab" / "holdout_confirm.json"
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(payload, encoding="utf-8")
    receipt["receipt_path"] = str(out)
    receipt["artifact_path"] = str(public)
    return receipt
