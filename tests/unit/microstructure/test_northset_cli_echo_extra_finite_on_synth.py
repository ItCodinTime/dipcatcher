"""NORTHSET_CLI_ECHO_EXTRA keys: present on synth; numeric not ±inf when finite."""

from __future__ import annotations

import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import NORTHSET_CLI_ECHO_EXTRA, bench_northset

# Non-numeric EXTRA stamps (strings / bools / optional None floors).
_NON_NUMERIC_EXTRA: frozenset[str] = frozenset(
    {
        "book_hypothesis_eligible",
        "claim",
        "concentration_top_finite_floor",
        "data_source",
        "depth_shape_finite_floor",
        "metrics_required_finite_ok",
        "session_book_hypothesis_eligible",
        "session_l2_identity_gate",
        "shape_columns_ensured",
        "sweep_primary_test_id",
    }
)


def _synth_receipt() -> dict:
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    return bench_northset(bars, cfg)


def test_extra_keys_present_on_synth_receipt() -> None:
    receipt = _synth_receipt()
    missing = sorted(k for k in NORTHSET_CLI_ECHO_EXTRA if k not in receipt)
    assert missing == [], missing


def test_extra_numeric_keys_not_inf_on_synth() -> None:
    """NaN allowed for sparse event diagnostics; ±inf fail-closed."""
    receipt = _synth_receipt()
    bad: list[str] = []
    for key in sorted(NORTHSET_CLI_ECHO_EXTRA - _NON_NUMERIC_EXTRA):
        val = receipt[key]
        try:
            f = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            bad.append(f"{key}:non_numeric={val!r}")
            continue
        if math.isinf(f):
            bad.append(f"{key}:{f!r}")
    assert bad == [], bad


def test_extra_non_numeric_keys_honest_types_on_synth() -> None:
    receipt = _synth_receipt()
    assert isinstance(receipt["claim"], str) and receipt["claim"].strip()
    assert isinstance(receipt["data_source"], str) and receipt["data_source"].strip()
    assert (
        isinstance(receipt["session_l2_identity_gate"], str)
        and receipt["session_l2_identity_gate"].strip()
    )
    assert receipt["sweep_primary_test_id"] == "H45_northset_follow_control"
    for key in (
        "book_hypothesis_eligible",
        "metrics_required_finite_ok",
        "session_book_hypothesis_eligible",
        "shape_columns_ensured",
    ):
        assert isinstance(receipt[key], bool), key
    # Optional floors may be None when unset
    for key in ("concentration_top_finite_floor", "depth_shape_finite_floor"):
        val = receipt[key]
        assert val is None or (isinstance(val, (int, float)) and math.isfinite(float(val))), key
