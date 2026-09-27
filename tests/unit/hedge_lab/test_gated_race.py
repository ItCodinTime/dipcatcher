"""Gated-race window slicing + gate stack units (no panel needed)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.hedge_lab.gated_race import (
    _gated_cards,
    _window_bundle,
    slice_ic_window,
    slice_path_window,
)


def _card(name: str) -> dict:
    dates = [f"2024-1{d:01d}-{day:02d}" for d in (1, 2) for day in (1, 15)]
    dates = ["2024-11-01", "2024-12-15", "2025-01-10", "2025-02-01"]
    return {
        "name": name,
        "ic_dates": dates,
        "ic_series": [0.01, 0.02, 0.03, 0.04],
    }


def test_slice_ic_window_bounds_inclusive() -> None:
    c = _card("m")
    out = slice_ic_window(c, start="2024-12-15", end="2025-01-10")
    assert out["ic_dates"] == ["2024-12-15", "2025-01-10"]
    assert out["ic_series"] == [0.02, 0.03]
    assert out["n_dates"] == 2
    assert out["window_start"] == "2024-12-15"
    assert out["mean_ic"] == pytest.approx(0.025)


def test_slice_ic_window_empty_range() -> None:
    out = slice_ic_window(_card("m"), start="2030-01-01")
    assert out["n_dates"] == 0
    assert out["ic_series"] == []
    assert np.isnan(out["mean_ic"])


def test_slice_path_window_aligns_and_errors() -> None:
    dates = ["2024-12-30", "2025-01-03", "2025-01-10"]
    r = np.array([0.01, 0.02, 0.03])
    out = slice_path_window(dates, r, start="2025-01-01")
    assert out.tolist() == [0.02, 0.03]
    with pytest.raises(ValueError, match="align"):
        slice_path_window(dates, r[:2])


def _ic_card(name: str, dates: list[str], mean: float) -> dict:
    rng = np.random.default_rng(hash(name) % 2**32)
    return {
        "name": name,
        "ic_dates": dates,
        "ic_series": rng.normal(mean, 0.005, len(dates)).tolist(),
    }


def test_window_bundle_runs_gates_and_books() -> None:
    rng = np.random.default_rng(0)
    dates = [f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(40)]
    cards = [_ic_card("ridge", dates, 0.0), _ic_card("champ", dates, 0.08)]
    ls_paths = {
        "ridge": {"dates": dates, "returns": rng.normal(0.0, 0.01, len(dates))},
        "champ": {"dates": dates, "returns": rng.normal(0.001, 0.01, len(dates))},
    }
    out = _window_bundle(cards, ls_paths, start=None, end=None, n_boot=100)
    assert out["gates"]["promote"] is True
    assert set(out["cs_ls"]) == {"ridge", "champ"}
    assert set(out["gated_ls"]["champ"]) == {"raw", "riskstack", "stepm_stack"}
    assert len(out["cs_ic"]) == 2
    assert "ic_series" not in out["cs_ic"][0]  # slim cards


def test_gated_cards_produce_stacks() -> None:
    rng = np.random.default_rng(2)
    out = _gated_cards(rng.normal(0.0, 0.01, 300))
    for k in ("raw", "riskstack", "riskstack_gate", "stepm_stack", "stepm_gate"):
        assert k in out
    assert out["raw"]["research_only"] is True
