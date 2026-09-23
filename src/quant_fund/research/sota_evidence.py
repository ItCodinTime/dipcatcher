"""Input and observation contracts for external forecasting comparisons.

An asset is not an independent replication of a contemporaneous market shock.
Panel inference therefore operates on equal-weight losses by target timestamp,
using only timestamps with every declared asset and model present.
"""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
import polars as pl


def validate_losses(
    losses: np.ndarray,
    pinball: np.ndarray,
    asset_ids: np.ndarray,
    asset_names: list[str],
    model_names: list[str],
    n_taus: int,
) -> None:
    if not asset_names or len(set(asset_names)) != len(asset_names):
        raise ValueError("asset names must be nonempty and unique")
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("model names must be nonempty and unique")
    if losses.ndim != 2 or losses.shape[1] != len(model_names) or not len(losses):
        raise ValueError("loss matrix must have nonempty rows and one column per model")
    if pinball.shape != (*losses.shape, n_taus):
        raise ValueError("pinball cube must align with loss rows, models, and quantiles")
    if asset_ids.shape != (len(losses),) or not np.issubdtype(asset_ids.dtype, np.integer):
        raise ValueError("asset IDs must be an aligned integer vector")
    if np.any(asset_ids < 0) or np.any(asset_ids >= len(asset_names)):
        raise ValueError("asset ID is outside the declared asset names")
    if set(asset_ids.tolist()) != set(range(len(asset_names))):
        raise ValueError("every declared asset must have observations")
    if np.any(losses[np.isfinite(losses)] < 0) or np.any(pinball[np.isfinite(pinball)] < 0):
        raise ValueError("proper losses must be nonnegative")


def coverage_summary(
    losses: np.ndarray,
    pinball: np.ndarray,
    asset_ids: np.ndarray,
    asset_names: list[str],
    model_names: list[str],
    requested_per_asset: int | None = None,
) -> dict[str, Any]:
    """Report each model's missing observations before complete-case filtering."""
    validate_losses(losses, pinball, asset_ids, asset_names, model_names, pinball.shape[-1])
    if requested_per_asset is not None:
        if (
            isinstance(requested_per_asset, bool)
            or not isinstance(requested_per_asset, Integral)
            or requested_per_asset <= 0
        ):
            raise ValueError("requested_per_asset must be a positive integer")
        if any(np.sum(asset_ids == a) > requested_per_asset for a in range(len(asset_names))):
            raise ValueError("emitted origins exceed the requested_per_asset protocol")
    valid_crps = np.isfinite(losses)
    valid = valid_crps & np.isfinite(pinball).all(axis=2)

    def counts(mask: np.ndarray, column: int, requested: int) -> dict[str, Any]:
        emitted = int(mask.sum())
        scored = int(valid[mask, column].sum())
        return {
            "requested": requested,
            "emitted": emitted,
            "finite_crps": int(valid_crps[mask, column].sum()),
            "scored_all_metrics": scored,
            "missing_or_failed": requested - scored,
            "coverage_fraction": scored / requested if requested else None,
        }

    requested = {
        a: int(requested_per_asset)
        if requested_per_asset is not None
        else int(np.sum(asset_ids == a))
        for a in range(len(asset_names))
    }
    return {
        name: {
            **counts(np.ones(len(losses), dtype=bool), j, sum(requested.values())),
            "per_asset": {
                asset: counts(asset_ids == a, j, requested[a])
                for a, asset in enumerate(asset_names)
            },
        }
        for j, name in enumerate(model_names)
    }


def aligned_inference_losses(
    losses: np.ndarray,
    complete: np.ndarray,
    asset_ids: np.ndarray,
    asset_names: list[str],
    target_time_ns: np.ndarray | None,
    bar_interval_ns: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a balanced chronological panel, or an explicit unavailable state.

    Missing dates cannot silently become adjacent observations in a bootstrap.
    Inference is withheld for gaps; descriptive score/coverage tables remain.
    """
    if complete.shape != (len(losses),) or complete.dtype != np.dtype(bool):
        raise ValueError("complete mask must be an aligned boolean vector")
    if not np.isfinite(losses[complete]).all():
        raise ValueError("complete observations must contain finite losses")
    empty = np.empty((0, losses.shape[1]))
    info: dict[str, Any] = {
        "status": "unavailable",
        "method": "equal_weight_asset_loss_by_target_time",
        "n_observations": 0,
        "n_assets": len(asset_names),
    }
    if target_time_ns is None or bar_interval_ns is None:
        return empty, info | {"reason": "missing_target_timestamps_or_bar_interval"}
    times = np.asarray(target_time_ns)
    if times.shape != (len(losses),) or not np.issubdtype(times.dtype, np.integer):
        raise ValueError("target timestamps must be an aligned integer nanosecond vector")
    if (
        isinstance(bar_interval_ns, bool)
        or not isinstance(bar_interval_ns, Integral)
        or bar_interval_ns <= 0
        or np.any(times <= 0)
    ):
        raise ValueError("timestamps and bar interval must be positive")
    keys = np.rec.fromarrays([asset_ids, times])
    if len(np.unique(keys)) != len(keys):
        raise ValueError("duplicate asset/target timestamp observations")
    unique_times = np.unique(times)
    panel = []
    kept_times = []
    for timestamp in unique_times:
        mask = times == timestamp
        if int(mask.sum()) == len(asset_names) and complete[mask].all():
            panel.append(np.mean(losses[mask], axis=0))
            kept_times.append(timestamp)
    info.update(
        n_target_times=len(unique_times),
        n_balanced_times=len(kept_times),
        n_excluded_times=len(unique_times) - len(kept_times),
        bar_interval_ns=int(bar_interval_ns),
    )
    if len(panel) < 20:
        return empty, info | {"reason": "fewer_than_20_balanced_target_times"}
    if not np.all(np.diff(kept_times) == bar_interval_ns):
        return empty, info | {"reason": "gaps_in_balanced_target_times"}
    return np.asarray(panel), info | {
        "status": "computed",
        "n_observations": len(panel),
        "first_target_time_ns": int(kept_times[0]),
        "last_target_time_ns": int(kept_times[-1]),
        "scope": "conditional_on_balanced_complete_observations",
    }


def validate_bars(frame: pl.DataFrame) -> tuple[np.ndarray, int]:
    """Validate a single regularly spaced, causally available crypto bar series."""
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
    for name in ("event_time", "available_time"):
        if not isinstance(frame.schema[name], pl.Datetime):
            raise ValueError(f"{name} must be a datetime column")
    times = frame["event_time"].dt.timestamp("ns").to_numpy()
    available = frame["available_time"].dt.timestamp("ns").to_numpy()
    steps = np.diff(times)
    if np.any(steps <= 0) or not np.all(steps == steps[0]):
        raise ValueError("bar event times must be strictly increasing with no gaps")
    # Each close must be known before the next target bar starts. This also
    # ensures all earlier history is available at the corresponding decision.
    if np.any(available < times) or np.any(available > times + steps[0]):
        raise ValueError("point-in-time violation: bar unavailable by next bar open")
    values = frame.select("open", "high", "low", "close", "volume").to_numpy()
    if not np.isfinite(values).all() or np.any(values[:, :4] <= 0) or np.any(values[:, 4] < 0):
        raise ValueError("prices must be positive and volume nonnegative, all finite")
    if np.any(values[:, 1] < np.max(values[:, [0, 2, 3]], axis=1)) or np.any(
        values[:, 2] > np.min(values[:, [0, 1, 3]], axis=1)
    ):
        raise ValueError("inconsistent OHLC prices")
    return times, int(steps[0])


def paired_effect_intervals(
    losses: np.ndarray,
    model_names: list[str],
    targets: list[str],
    *,
    n_boot: int = 2000,
    seed: int = 7,
    block: float | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired CRPS effects on an already aligned chronological loss panel.

    One stationary-bootstrap draw resamples every comparison together. In
    addition to pointwise percentile intervals, the maximum standardized
    bootstrap deviation gives simultaneous intervals over the entire declared
    target/comparator family. Positive differences favor the comparator.
    These are approximate bootstrap intervals, not distribution-free bounds.
    """
    from quant_fund.metrics.inference import optimal_block_length, stationary_bootstrap_indices

    if (
        losses.ndim != 2
        or losses.shape[1] != len(model_names)
        or not np.isfinite(losses).all()
        or np.any(losses < 0)
    ):
        raise ValueError("effects require a finite nonnegative aligned loss panel")
    if len(set(model_names)) != len(model_names) or len(model_names) < 2:
        raise ValueError("effects require at least two unique models")
    if not targets or len(set(targets)) != len(targets) or not set(targets).issubset(model_names):
        raise ValueError("effect targets must be unique members of model_names")
    if isinstance(n_boot, bool) or not isinstance(n_boot, Integral) or n_boot < 2:
        raise ValueError("effect n_boot must be an integer >= 2")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("effect alpha must be in (0, 1)")
    n = len(losses)
    if n < 20:
        return {"status": "unavailable", "reason": "fewer_than_20_target_times"}
    pairs = [(target, name) for target in targets for name in model_names if name != target]
    differences = np.column_stack(
        [
            losses[:, model_names.index(target)] - losses[:, model_names.index(name)]
            for target, name in pairs
        ]
    )
    if block is None:
        candidates = [optimal_block_length(differences[:, j]) for j in range(len(pairs))]
        finite = [value for value in candidates if np.isfinite(value)]
        block = float(np.median(finite)) if finite else float(n ** (1 / 3))
        block = min(float(n), max(1.0, block))
    indices = stationary_bootstrap_indices(n, int(n_boot), block, np.random.default_rng(seed))
    # Draw counts avoid allocating a B x T x comparisons tensor.
    counts = np.asarray([np.bincount(draw, minlength=n) for draw in indices], dtype=float)
    boot = counts @ differences / n
    means = differences.mean(axis=0)
    lower, upper = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)
    se = boot.std(axis=0, ddof=1)
    tolerance = np.maximum(np.max(np.abs(differences), axis=0), 1e-12) * 1e-12
    estimable = np.isfinite(se) & (se > tolerance)
    critical = (
        float(
            np.quantile(
                np.max(np.abs((boot[:, estimable] - means[estimable]) / se[estimable]), axis=1),
                1 - alpha,
            )
        )
        if estimable.all()
        else None
    )
    comparisons: dict[str, dict[str, Any]] = {target: {} for target in targets}
    for j, (target, name) in enumerate(pairs):
        target_mean = float(losses[:, model_names.index(target)].mean())
        comparisons[target][name] = {
            "mean_target_minus_comparator": float(means[j]),
            "relative_crps_reduction": float(means[j] / target_mean) if target_mean > 0 else None,
            "pointwise_percentile_ci": [float(lower[j]), float(upper[j])],
            "simultaneous_ci": [
                float(means[j] - critical * se[j]),
                float(means[j] + critical * se[j]),
            ]
            if critical is not None
            else None,
        }
    return {
        "status": "computed",
        "method": "stationary_block_bootstrap_by_target_time",
        "sign": "positive_favors_comparator",
        "n_observations": n,
        "n_boot": int(n_boot),
        "seed": seed,
        "block_length": float(block),
        "confidence_level": 1 - alpha,
        "family_size": len(pairs),
        "simultaneous_method": "centered_max_absolute_standardized_bootstrap_deviation",
        "simultaneous_status": "computed"
        if critical is not None
        else "degenerate_comparison_variance",
        "comparisons": comparisons,
    }


def block_sensitivity(
    losses: np.ndarray,
    model_names: list[str],
    targets: list[str],
    *,
    n_boot: int,
    seed: int,
    blocks: tuple[float, ...] = (1.0, 5.0, 10.0, 20.0),
) -> list[dict[str, Any]]:
    """Expose MCS/SPA dependence on prespecified expected block lengths.

    Length one is the iid diagnostic; other lengths retain serial dependence.
    These sensitivity results are not searched to select a favorable p-value.
    """
    from quant_fund.metrics.snooping import model_confidence_set, spa_test

    results = []
    for block in sorted(set(blocks)):
        if not np.isfinite(block) or block < 1 or block > len(losses):
            raise ValueError("sensitivity block lengths must be in [1, n_observations]")
        mcs = model_confidence_set(-losses, n_boot=n_boot, seed=seed, block=block)
        mcs_valid = np.isfinite(mcs.p_values).all()
        spas = {}
        for target in targets:
            j = model_names.index(target)
            other = [k for k in range(len(model_names)) if k != j]
            spa = spa_test(losses[:, [j]] - losses[:, other], n_boot=n_boot, seed=seed, block=block)
            spas[target] = {
                "p_lower": spa.p_lower,
                "p_consistent": spa.p_consistent,
                "p_upper": spa.p_upper,
            }
        results.append(
            {
                "block_length": block,
                "n_observations": len(losses),
                "n_boot": n_boot,
                "seed": seed,
                "resampling": "iid" if block == 1 else "stationary_block",
                "mcs_status": "computed" if mcs_valid else "unavailable",
                "mcs_alpha": mcs.alpha,
                "mcs_included": {name: bool(mcs.included[j]) for j, name in enumerate(model_names)}
                if mcs_valid
                else {},
                "mcs_p_values": {name: float(mcs.p_values[j]) for j, name in enumerate(model_names)}
                if mcs_valid
                else {},
                "spa": spas,
            }
        )
    return results
