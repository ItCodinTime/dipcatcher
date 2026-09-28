"""Contract checks for script-emitted receipt schemas.

The ``scripts/`` lanes write ``receipts/*.json`` evidence artifacts with their
own ``schema`` tags (``adaptive_mix_replay.v1``, ``basis_reversion_screen.v1``,
``incumbent_bench.v1``, ``basis_pair_candidate.v1``,
``adaptive_mix_band_search.v1``, ``fx1.dip_bench/v1``). These are research
receipts: every check below is a *re-derivation or bound* computed purely from
the sealed body — a headline scalar that disagrees with the embedded detail is
forged or mislabeled, and the verifier must say so.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any


def _as_finite_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        f = float(value)
        return f if math.isfinite(f) else None
    return None


def _finite_number(value: object) -> bool:
    return _as_finite_float(value) is not None


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _fraction(value: object) -> bool:
    f = _as_finite_float(value)
    return f is not None and 0.0 <= f <= 1.0


def _segment_metrics(metrics: object) -> list[str]:
    """Required segment statistics block: n + finite return/drawdown fields."""
    if not isinstance(metrics, Mapping):
        return ["segment_metrics_not_object"]
    errors: list[str] = []
    if not _positive_int(metrics.get("n")):
        errors.append("segment_n_not_positive_int")
    for key in ("net_return", "cagr", "sharpe", "max_drawdown"):
        if not _finite_number(metrics.get(key)):
            errors.append(f"segment_{key}_not_finite")
    return errors


def _allocation_simplex_errors(name: str, weights: object, sleeves: set[str]) -> list[str]:
    if not isinstance(weights, Mapping):
        return [f"{name}:allocation_not_object"]
    errors: list[str] = []
    total = 0.0
    for sleeve, w in weights.items():
        if sleeve not in sleeves:
            errors.append(f"{name}:unknown_sleeve:{sleeve}")
        if not _fraction(w):
            errors.append(f"{name}:weight_out_of_bounds:{sleeve}")
        else:
            total += float(w)
    if not errors and not (1.0 - 1e-6 <= total <= 1.0 + 1e-6):
        errors.append(f"{name}:allocation_not_simplex:{total}")
    return errors


def band_search_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``adaptive_mix_band_search.v1``: re-derive eligibility and the selected band.

    The embedded ``selection_rule`` is executable: a candidate is eligible iff
    ``cagr > 0``, ``sharpe > 0`` and ``max_drawdown > -0.05`` on development;
    ``selected_band`` must equal the eligible candidate with the highest
    development Sharpe, or ``null`` when none qualify.
    """
    errors: list[str] = []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ["candidates_missing"]
    eligible: list[tuple[float, object]] = []
    for i, cand in enumerate(candidates):
        if not isinstance(cand, Mapping):
            errors.append(f"candidate_{i}_not_object")
            continue
        if not _finite_number(cand.get("band")):
            errors.append(f"candidate_{i}_band_not_finite")
        errors.extend(f"candidate_{i}:{e}" for e in _segment_metrics(cand.get("development")))
        dev = cand.get("development") or {}
        if isinstance(dev, Mapping) and all(
            _finite_number(dev.get(k)) for k in ("cagr", "sharpe", "max_drawdown")
        ):
            cagr = _as_finite_float(dev.get("cagr"))
            sharpe = _as_finite_float(dev.get("sharpe"))
            drawdown = _as_finite_float(dev.get("max_drawdown"))
            if cagr is None or sharpe is None or drawdown is None:
                continue
            is_eligible = cagr > 0.0 and sharpe > 0.0 and drawdown > -0.05
            if cand.get("eligible") is not is_eligible:
                errors.append(f"candidate_{i}_eligible_mismatch")
            if is_eligible:
                eligible.append((sharpe, cand.get("band")))
    if not errors:
        expected = max(eligible, key=lambda t: t[0])[1] if eligible else None
        if payload.get("selected_band") != expected:
            errors.append("selected_band_not_argmax_dev_sharpe")
    return errors


def adaptive_mix_replay_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``adaptive_mix_replay.v1``: sleeve set agreement + simplex allocations."""
    errors: list[str] = []
    sleeves_obj = payload.get("paper_sleeves")
    if not isinstance(sleeves_obj, Mapping) or not sleeves_obj:
        return ["paper_sleeves_missing"]
    sleeves = set(sleeves_obj)
    for sleeve, seg in sleeves_obj.items():
        if not isinstance(seg, Mapping) or not seg:
            errors.append(f"paper_sleeve_{sleeve}_empty")
            continue
        for seg_name, metrics in seg.items():
            errors.extend(
                f"paper_sleeve_{sleeve}.{seg_name}:{e}" for e in _segment_metrics(metrics)
            )
    alloc = payload.get("mean_adaptive_allocation")
    if not isinstance(alloc, Mapping) or not alloc:
        errors.append("mean_adaptive_allocation_missing")
    else:
        for seg_name, weights in alloc.items():
            errors.extend(_allocation_simplex_errors(str(seg_name), weights, sleeves))
    if not _positive_int(payload.get("n_assets")):
        errors.append("n_assets_not_positive_int")
    if not _positive_int(payload.get("bar_count")):
        errors.append("bar_count_not_positive_int")
    return errors


def basis_reversion_screen_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``basis_reversion_screen.v1``: bounded fractions + four fold means."""
    errors: list[str] = []
    for key in ("positive_day_fraction", "hedged_pair_active_date_fraction"):
        if not _fraction(payload.get(key)):
            errors.append(f"{key}_out_of_bounds")
    for key in ("four_chronological_fold_means", "hedged_pair_four_fold_means"):
        folds = payload.get(key)
        if (
            not isinstance(folds, list)
            or len(folds) != 4
            or not all(_finite_number(v) for v in folds)
        ):
            errors.append(f"{key}_not_four_finite_means")
    for key in ("n_assets", "n_dates"):
        if not _positive_int(payload.get(key)):
            errors.append(f"{key}_not_positive_int")
    for key in (
        "mean_daily_gross_spread",
        "median_daily_gross_spread",
        "hedged_pair_mean_daily_gross_return",
    ):
        if not _finite_number(payload.get(key)):
            errors.append(f"{key}_not_finite")
    return errors


def basis_pair_candidate_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``basis_pair_candidate.v1``: development metrics + eligibility flag."""
    errors: list[str] = []
    if not isinstance(payload.get("development_eligible"), bool):
        errors.append("development_eligible_not_bool")
    dev = payload.get("development")
    if not isinstance(dev, Mapping):
        errors.append("development_missing")
    else:
        for key in ("total_return", "cagr", "sharpe", "max_drawdown"):
            if not _finite_number(dev.get(key)):
                errors.append(f"development_{key}_not_finite")
    if not _positive_int(payload.get("n_target_events")):
        errors.append("n_target_events_not_positive_int")
    if not _positive_int(payload.get("n_assets")):
        errors.append("n_assets_not_positive_int")
    return errors


def incumbent_bench_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``incumbent_bench.v1``: re-derive the reported medians from the raw runs."""
    errors: list[str] = []
    latency = payload.get("latency")
    if not isinstance(latency, Mapping):
        return ["latency_missing"]
    for side in ("dipcatcher", "qlib"):
        runs = latency.get(f"{side}_ms_all")
        median = latency.get(f"{side}_ms_median")
        if not isinstance(runs, list) or not runs or not all(_finite_number(v) for v in runs):
            errors.append(f"{side}_ms_all_not_finite_list")
            continue
        if not _finite_number(median):
            errors.append(f"{side}_ms_median_not_finite")
            continue
        expected = statistics.median(_as_finite_float(v) or 0.0 for v in runs)
        claimed = _as_finite_float(median)
        if claimed is None or abs(claimed - expected) > 0.51:
            errors.append(f"{side}_ms_median_rederive_mismatch")
    correctness = payload.get("correctness")
    if not isinstance(correctness, Mapping):
        errors.append("correctness_missing")
    else:
        if not isinstance(correctness.get("common_dates"), int) or correctness["common_dates"] < 0:
            errors.append("common_dates_not_nonneg_int")
        for key, value in correctness.items():
            if isinstance(value, float) and not _finite_number(value):
                errors.append(f"correctness_{key}_not_finite")
    for key in ("environment", "incumbent", "workload"):
        if not isinstance(payload.get(key), Mapping):
            errors.append(f"{key}_missing")
    return errors


def dip_bench_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``fx1.dip_bench/v1``: n_events must equal the sum of per-asset events."""
    errors: list[str] = []
    per_asset = payload.get("per_asset")
    if not isinstance(per_asset, Mapping) or not per_asset:
        return ["per_asset_missing"]
    total = 0
    for asset, info in per_asset.items():
        if not isinstance(info, Mapping):
            errors.append(f"per_asset_{asset}_not_object")
            continue
        bars, events = info.get("bars"), info.get("events")
        if not _positive_int(bars):
            errors.append(f"per_asset_{asset}_bars_invalid")
        if not isinstance(events, int) or isinstance(events, bool) or events < 0:
            errors.append(f"per_asset_{asset}_events_invalid")
            continue
        if isinstance(bars, int) and not isinstance(bars, bool) and events > bars:
            errors.append(f"per_asset_{asset}_events_exceed_bars")
        total += events
    n_events = payload.get("n_events")
    if not isinstance(n_events, int) or isinstance(n_events, bool) or n_events < 0:
        errors.append("n_events_invalid")
    elif not errors and n_events != total:
        errors.append("n_events_not_sum_of_per_asset")
    baseline = payload.get("baseline_recovery")
    if not isinstance(baseline, Mapping) or not all(_fraction(v) for v in baseline.values()):
        errors.append("baseline_recovery_not_fractions")
    params = payload.get("params")
    if isinstance(params, Mapping):
        horizons = params.get("horizons_bars")
        if not isinstance(horizons, Mapping) or not all(
            _positive_int(v) for v in horizons.values()
        ):
            errors.append("horizons_bars_not_positive_ints")
        if not _fraction(params.get("threshold")) or float(params.get("threshold", 1)) >= 1.0:
            errors.append("threshold_out_of_bounds")
    else:
        errors.append("params_missing")
    return errors


#: script-schema tag → contract-check function (dispatch lives in receipt_v2).
SCRIPT_RECEIPT_CONTRACTS: dict[str, Any] = {
    "adaptive_mix_band_search.v1": band_search_contract_errors,
    "adaptive_mix_replay.v1": adaptive_mix_replay_contract_errors,
    "basis_reversion_screen.v1": basis_reversion_screen_contract_errors,
    "basis_pair_candidate.v1": basis_pair_candidate_contract_errors,
    "incumbent_bench.v1": incumbent_bench_contract_errors,
    "fx1.dip_bench/v1": dip_bench_contract_errors,
}


def script_receipt_contract_errors(schema: object, payload: Mapping[str, Any]) -> list[str]:
    """Re-derive a script receipt's headline claims; ``[]`` when unknown schema."""
    check = SCRIPT_RECEIPT_CONTRACTS.get(str(schema))
    if check is None:
        return []
    try:
        result: list[str] = check(payload)
        return result
    except (TypeError, ValueError, KeyError, AttributeError, ZeroDivisionError):
        # A contract-check crash must never read as a verified receipt.
        return ["contract_check_error"]
