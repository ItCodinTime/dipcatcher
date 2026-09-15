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
) -> list[Fold]:
    uniq = sorted(set(times))
    n = len(uniq)
    scheme = scheme or config.scheme
    idx = session_index(uniq)
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
    holdout = [t for t in fold.val_times + fold.test_times if t in idx]
    if not holdout:
        return
    ids = [idx[t] for t in holdout]
    vmin, vmax = min(ids), max(ids)
    for t in fold.train_times:
        i = idx[t]
        label_end = i + horizon_bars
        if label_end > vmin and i < vmax:
            raise AssertionError(
                f"label overlap at {t}: train window ({i}, {label_end}] vs holdout [{vmin}, {vmax}]"
            )
