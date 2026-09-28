"""WAVE2 §7.1 CSCV combination-cap guard tests (W8).

Fail-closed: enumerating more than ``max_combos`` combinations raises
``RealityFilterError("cscv_combo_cap:...")`` — never a silent truncation.
At exactly the cap the computation proceeds unchanged; below the cap the
wave-1 behavior is untouched.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from quant_fund.proofcore.contracts import RealityFilterError
from quant_fund.reality.cscv import COMBO_CAP, cscv_pbo, cscv_splits


def _panel(n_periods: int, n_trials: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_periods, n_trials))


def test_default_cap_is_10_000() -> None:
    assert COMBO_CAP == 10_000


def test_pbo_default_cap_blocks_s16() -> None:
    """C(16, 8) = 12_870 > 10_000 -> fail-closed at the default cap."""
    assert comb(16, 8) == 12_870
    with pytest.raises(RealityFilterError, match="cscv_combo_cap"):
        cscv_pbo(_panel(32, 3), s_blocks=16)


def test_pbo_cap_error_message_mentions_explicit_override() -> None:
    with pytest.raises(RealityFilterError, match="raise the cap explicitly"):
        cscv_pbo(_panel(32, 3), s_blocks=16)


def test_pbo_explicit_override_above_cap_runs_full_enumeration() -> None:
    """Raising the cap explicitly opts in; no silent truncation — every one
    of the C(16, 8) combinations is actually computed."""
    out = cscv_pbo(_panel(32, 3), s_blocks=16, max_combos=20_000)
    assert out["n_splits"] == comb(16, 8)
    assert out["n_combinations"] + out["n_dropped"] == comb(16, 8)


def test_pbo_at_exactly_the_cap_passes() -> None:
    """== cap passes (boundary is inclusive)."""
    out = cscv_pbo(_panel(16, 2), s_blocks=8, max_combos=comb(8, 4))
    assert out["n_splits"] == comb(8, 4) == 70


def test_pbo_one_below_cap_raises() -> None:
    with pytest.raises(RealityFilterError, match="cscv_combo_cap"):
        cscv_pbo(_panel(16, 2), s_blocks=8, max_combos=comb(8, 4) - 1)


def test_pbo_below_default_cap_unchanged() -> None:
    """Wave-1 behavior below the cap is untouched (s=8 -> 70 combos)."""
    out = cscv_pbo(_panel(16, 2), s_blocks=8)
    assert out["n_splits"] == 70


def test_splits_opt_in_cap_fires() -> None:
    with pytest.raises(RealityFilterError, match="cscv_combo_cap"):
        cscv_splits(16, s_blocks=4, max_combos=comb(4, 2) - 1)


def test_splits_opt_in_cap_boundary_passes() -> None:
    splits = cscv_splits(16, s_blocks=4, max_combos=comb(4, 2))
    assert len(splits) == comb(4, 2) == 6


def test_splits_default_uncapped_for_direct_callers() -> None:
    """Additive-only contract: direct cscv_splits callers keep wave-1 behavior
    (the fail-closed default cap binds at the cscv_pbo computation entry)."""
    splits = cscv_splits(64, s_blocks=16)
    assert len(splits) == comb(16, 8)


def test_invalid_max_combos_rejected() -> None:
    with pytest.raises(RealityFilterError, match="max_combos must be >= 1"):
        cscv_splits(16, s_blocks=4, max_combos=0)
    with pytest.raises(RealityFilterError, match="max_combos must be >= 1"):
        cscv_pbo(_panel(16, 2), s_blocks=8, max_combos=0)
