"""Expanding and rolling walk-forward splits on decision timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant_fund.config.models import ValidationConfig
from quant_fund.validation.purging import purge_mask


@dataclass(frozen=True)
class Fold:
    train_times: list[datetime]
    val_times: list[datetime]
    test_times: list[datetime]


def session_index(times: list[datetime]) -> dict[datetime, int]:
    uniq = sorted(set(times))
    return {t: i for i, t in enumerate(uniq)}


def walk_forward(
    times: list[datetime],
    config: ValidationConfig,
    *,
    horizon_bars: int,
    embargo_bars: int,
    scheme: str | None = None,
    label_end_times: list[datetime] | None = None,
) -> list[Fold]:
    if label_end_times is not None and len(label_end_times) != len(times):
        raise ValueError("label_end_times must align with times")
    uniq = sorted(set(times))
    n = len(uniq)
    scheme = scheme or config.scheme
    idx = session_index(uniq)
    # A date-level fold is unsafe if any security observed on that decision date
    # has a label reaching the holdout.  Use the latest endpoint conservatively.
    end_by_time: dict[datetime, datetime] | None = None
    if label_end_times is not None:
        end_by_time = {}
        for t, end in zip(times, label_end_times, strict=True):
            if end < t:
                raise ValueError("label_end_times must not precede times")
            previous = end_by_time.get(t)
            if previous is None or end > previous:
                end_by_time[t] = end
    folds: list[Fold] = []
    cursor = config.train_bars
    while True:
        val_end = cursor + config.val_bars
        test_end = val_end + config.test_bars
        if test_end > n:
            break
        if scheme == "rolling":
            tr_start = max(0, cursor - config.train_bars)
        else:
            tr_start = 0
        train_block = uniq[tr_start:cursor]
        val_block = uniq[cursor:val_end]
        test_block = uniq[val_end:test_end]
        keep = purge_mask(
            train_block,
            val_block[0],
            test_block[-1],
            horizon_bars,
            session_index=idx,
            label_end_times=(
                [end_by_time[t] for t in train_block] if end_by_time is not None else None
            ),
        )
        va0 = idx[val_block[0]]
        train_kept = [
            t for t, k in zip(train_block, keep, strict=True) if k and idx[t] + embargo_bars < va0
        ]
        folds.append(Fold(train_times=train_kept, val_times=val_block, test_times=test_block))
        cursor = test_end if scheme == "rolling" else test_end
        if scheme == "expanding":
            # next train includes previous test; cursor is the new train_end
            pass
    return folds


def assert_no_label_overlap(fold: Fold, horizon_bars: int, idx: dict[datetime, int]) -> None:
    """Fail if any train label window intersects a holdout session.

    Holdout may be non-contiguous (CPCV). Checks each contiguous holdout block
    separately so intervening train groups are not false-positives.
    """
    holdout = [t for t in fold.val_times + fold.test_times if t in idx]
    if not holdout:
        return
    ids = sorted({idx[t] for t in holdout})
    # contiguous blocks as inclusive [lo, hi]
    blocks: list[tuple[int, int]] = []
    lo = hi = ids[0]
    for j in ids[1:]:
        if j == hi + 1:
            hi = j
        else:
            blocks.append((lo, hi))
            lo = hi = j
    blocks.append((lo, hi))
    for t in fold.train_times:
        i = idx[t]
        label_end = i + horizon_bars
        if label_end <= i:
            continue
        for vmin, vmax in blocks:
            # (i, label_end] intersects [vmin, vmax]
            if i < vmax and label_end >= vmin:
                raise AssertionError(
                    f"label overlap at {t}: train window ({i}, {label_end}] "
                    f"vs holdout block [{vmin}, {vmax}]"
                )


def fold_ic_stability(
    fold_ics: list[float],
    *,
    min_ic: float = 0.0,
) -> dict[str, float | int]:
    """Fraction of folds with IC > ``min_ic`` plus mean/std of fold ICs.

    Used by promotion gates as multi-fold stability evidence.
    """
    import math

    vals = [float(x) for x in fold_ics if x is not None and math.isfinite(float(x))]
    n = len(vals)
    if n == 0:
        return {"n_folds": 0, "stability": 0.0, "mean_ic": float("nan"), "std_ic": float("nan")}
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
    stab = sum(1 for v in vals if v > min_ic) / n
    return {
        "n_folds": n,
        "stability": float(stab),
        "mean_ic": float(mean),
        "std_ic": float(var**0.5),
    }
