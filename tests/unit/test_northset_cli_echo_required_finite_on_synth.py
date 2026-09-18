"""NORTHSET_CLI_ECHO_REQUIRED keys are present + finite (or nonempty str) on synth."""

from __future__ import annotations

import math

from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import NORTHSET_CLI_ECHO_REQUIRED, bench_northset

# Receipt keys that are intentionally non-numeric strings when stamped.
_STRING_REQUIRED: frozenset[str] = frozenset({"book_source"})


def _synth_receipt() -> dict:
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    return bench_northset(bars, cfg)


def test_required_keys_present_on_synth_receipt() -> None:
    receipt = _synth_receipt()
    missing = sorted(k for k in NORTHSET_CLI_ECHO_REQUIRED if k not in receipt)
    assert missing == [], missing


def test_required_numeric_keys_finite_on_synth() -> None:
    receipt = _synth_receipt()
    bad: list[str] = []
    for key in sorted(NORTHSET_CLI_ECHO_REQUIRED - _STRING_REQUIRED):
        try:
            val = float(receipt[key])
        except (TypeError, ValueError):
            bad.append(f"{key}:non_numeric={receipt[key]!r}")
            continue
        if not math.isfinite(val):
            bad.append(f"{key}:{val!r}")
    assert bad == [], bad


def test_required_string_keys_nonempty_on_synth() -> None:
    receipt = _synth_receipt()
    for key in sorted(_STRING_REQUIRED & NORTHSET_CLI_ECHO_REQUIRED):
        val = receipt[key]
        assert isinstance(val, str) and val.strip(), f"{key}={val!r}"
