"""CSCV / PBO — combinatorially symmetric cross-validation (Bailey, Borwein,
López de Prado & Zhu 2014).

The panel of ``n_periods`` observations is divided into ``S`` contiguous
blocks; every one of the ``C(S, S/2)`` ways to pick half the blocks as
in-sample (the rest out-of-sample) yields one combination. Per combination
the IS-optimal trial is selected and its OOS rank percentile ``omega`` is
mapped through the logit; PBO is the fraction of combinations where the
IS-best trial underperforms the OOS median (``lambda < 0``).

``validation/cpcv.py`` is datetime/purge-oriented (its ``Fold`` carries
purge+embargo semantics that CSCV does not have), so the combinatorial
machinery here uses ``itertools.combinations`` directly over block indices —
the same enumeration pattern as ``combinatorial_purged_cv``.

Research diagnostics only — never a live P&L / promotion claim.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from quant_fund.proofcore.contracts import RealityFilterError

Array = NDArray[np.float64]

_OMEGA_CLIP = 1e-9
MIN_COMBINATIONS = 4
# WAVE2 §7.1: fail-closed cap on the number of IS/OOS combinations a CSCV
# computation may enumerate. C(S, S/2) grows super-exponentially (C(16, 8) is
# already 12_870), so an oversized s_blocks silently turns a diagnostic into
# an hours-long blowup. Exceeding the cap raises — never silently truncates.
COMBO_CAP = 10_000


def _check_combo_cap(n_combos: int, s_blocks: int, max_combos: int) -> None:
    """Fail-closed combination-count guard (WAVE2 §7.1)."""
    if max_combos < 1:
        raise RealityFilterError(f"max_combos must be >= 1, got {max_combos}")
    if n_combos > max_combos:
        raise RealityFilterError(
            f"cscv_combo_cap: C({s_blocks}, {s_blocks // 2}) = {n_combos} combinations "
            f"exceed the cap of {max_combos} — raise the cap explicitly via max_combos "
            "(no silent truncation)"
        )


def cscv_splits(
    n_periods: int,
    s_blocks: int = 16,
    *,
    max_combos: int | None = None,
) -> list[tuple[Array, Array]]:
    """All C(S, S/2) IS/OOS half-combinations over S contiguous blocks.

    Returns a list of ``(is_idx, oos_idx)`` period-index arrays. Blocks are
    contiguous and equal length; indices within each side are ascending.
    Fail-closed: ``s_blocks`` must be even and >= 2, and ``n_periods`` must be
    divisible by ``s_blocks`` (else block boundaries would be ambiguous) —
    violations raise ``RealityFilterError``.

    ``max_combos`` (WAVE2 §7.1, additive): when given, more than
    ``max_combos`` combinations raises ``RealityFilterError``
    (``cscv_combo_cap:...``) instead of enumerating. ``cscv_pbo`` passes its
    own default cap of ``COMBO_CAP``; the default None here keeps the pure
    enumeration helper's wave-1 behavior for direct callers.
    """
    n = int(n_periods)
    s = int(s_blocks)
    if s < 2 or s % 2 != 0:
        raise RealityFilterError(f"s_blocks must be even and >= 2, got {s_blocks}")
    if n < s or n % s != 0:
        raise RealityFilterError(
            f"n_periods ({n_periods}) must be divisible by s_blocks ({s_blocks})"
        )
    n_combos = comb(s, s // 2)
    if max_combos is not None:
        _check_combo_cap(n_combos, s, int(max_combos))
    block_len = n // s
    blocks = [np.arange(i * block_len, (i + 1) * block_len) for i in range(s)]
    splits: list[tuple[Array, Array]] = []
    for is_blocks in combinations(range(s), s // 2):
        is_set = frozenset(is_blocks)
        is_idx = np.concatenate([blocks[b] for b in sorted(is_set)])
        oos_idx = np.concatenate([blocks[b] for b in range(s) if b not in is_set])
        splits.append((is_idx, oos_idx))
    assert len(splits) == comb(s, s // 2)  # noqa: S101 — combinatorial invariant
    return splits


def pbo_from_performance(is_perf: Array, oos_perf: Array) -> dict[str, object]:
    """PBO from per-combination performance matrices, shape
    ``(n_combinations, n_trials)``, HIGHER = better.

    Per combination: rank trials IS (``scipy.stats.rankdata``, ties ->
    average rank), select the IS-best trial, take its OOS rank percentile
    ``omega = rank / (n_trials + 1)`` clipped to ``[1e-9, 1 - 1e-9]``, and
    ``lambda_c = log(omega / (1 - omega))``. ``PBO = P(lambda < 0)`` — the
    share of combinations where the IS-best trial ranks below the OOS median.

    Fail-closed honesty rules:
      - a combination row with any non-finite trial value is dropped and
        counted in ``'n_dropped'``;
      - fewer than 4 surviving combinations -> ``pbo`` NaN (matches
        ``metrics.overfitting.probability_of_backtest_overfitting``
        semantics: never a fabricated 0.0);
      - shape violations (non-2D, mismatched, <2 trials) -> pbo NaN with all
        rows counted as dropped.
    """
    ins = np.asarray(is_perf, dtype=float)
    oos = np.asarray(oos_perf, dtype=float)
    nan = float("nan")
    if ins.shape != oos.shape or ins.ndim != 2 or ins.shape[1] < 2:
        n_rows = int(ins.shape[0]) if ins.ndim >= 1 else 0
        return {"pbo": nan, "logits": [], "n_combinations": 0, "n_dropped": n_rows}
    n_trials = int(ins.shape[1])
    logits: list[float] = []
    n_dropped = 0
    for c in range(ins.shape[0]):
        is_row = ins[c]
        oos_row = oos[c]
        if not (np.isfinite(is_row).all() and np.isfinite(oos_row).all()):
            n_dropped += 1
            continue
        # IS-best by average-rank ordering (argmax of ranks == argmax of
        # values; rankdata keeps the tie convention explicit per spec).
        best = int(np.argmax(rankdata(is_row)))
        oos_rank = float(rankdata(oos_row)[best])
        omega = min(max(oos_rank / (n_trials + 1.0), _OMEGA_CLIP), 1.0 - _OMEGA_CLIP)
        logits.append(float(np.log(omega / (1.0 - omega))))
    n_comb = len(logits)
    pbo = float(np.mean([lam < 0.0 for lam in logits])) if n_comb >= MIN_COMBINATIONS else nan
    return {
        "pbo": pbo,
        "logits": logits,
        "n_combinations": n_comb,
        "n_dropped": n_dropped,
    }


def _periodic_sharpe(r: Array) -> float:
    """Per-period SR (NEVER annualized — A1 F1 convention); NaN if degenerate."""
    if r.size < 2 or not np.isfinite(r).all():
        return float("nan")
    sig = float(np.std(r, ddof=1))
    if sig == 0.0:
        return float("nan")
    return float(np.mean(r) / sig)


def cscv_pbo(
    returns: Array,
    s_blocks: int = 16,
    *,
    max_combos: int = COMBO_CAP,
) -> dict[str, object]:
    """End-to-end CSCV/PBO over a ``(n_periods, n_trials)`` return matrix.

    Per split, per trial the performance is the per-period Sharpe over the
    split's IS / OOS periods. Returns the ``pbo_from_performance`` payload
    plus ``'n_splits'``. Fail-closed: shape violations raise
    ``RealityFilterError``; trials with degenerate splits are dropped via the
    NaN-row rule inside ``pbo_from_performance``.

    ``max_combos`` (WAVE2 §7.1, additive, default ``COMBO_CAP`` = 10_000):
    more than ``max_combos`` combinations raises ``RealityFilterError``
    (``cscv_combo_cap:...``) fail-closed — never a silent truncation. Raise
    the cap explicitly (e.g. ``max_combos=20_000`` for ``s_blocks=16``) to opt
    into a larger enumeration.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2 or r.shape[0] < 2 * int(s_blocks) or r.shape[1] < 2:
        raise RealityFilterError(
            "returns must be (n_periods, n_trials) with n_periods >= 2 * s_blocks and n_trials >= 2"
        )
    splits = cscv_splits(int(r.shape[0]), int(s_blocks), max_combos=int(max_combos))
    n_trials = int(r.shape[1])
    is_perf = np.full((len(splits), n_trials), np.nan)
    oos_perf = np.full((len(splits), n_trials), np.nan)
    for c, (is_idx, oos_idx) in enumerate(splits):
        for k in range(n_trials):
            is_perf[c, k] = _periodic_sharpe(r[is_idx, k])
            oos_perf[c, k] = _periodic_sharpe(r[oos_idx, k])
    out = pbo_from_performance(is_perf, oos_perf)
    out["n_splits"] = len(splits)
    return out
