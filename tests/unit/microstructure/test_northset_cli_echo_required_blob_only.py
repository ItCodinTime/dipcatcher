"""Scoped northset CLI echo required-key frozenset + BLOB_ONLY allowlist.

Invariant: every stamped non-discovery / non-kyle receipt key is either
CLI-echoed or listed in NORTHSET_RECEIPT_BLOB_ONLY. REQUIRED keys must appear
in ``dipcatcher northset`` CLI source. REQUIRED ∩ BLOB_ONLY = ∅.
"""

from __future__ import annotations

import inspect

from quant_fund.cli import main as cli_main
from quant_fund.config.models import AppConfig
from quant_fund.data.adapters.synthetic import SyntheticMarketProvider
from quant_fund.northset.benches import (
    NORTHSET_CLI_ECHO_EXTRA,
    NORTHSET_CLI_ECHO_KEY_ALIASES,
    NORTHSET_CLI_ECHO_REQUIRED,
    NORTHSET_RECEIPT_BLOB_ONLY,
    NORTHSET_RECEIPT_CLASSIFIED,
    SESSION_RECEIPT_KEYS,
    bench_northset,
)


def _is_discovery_or_kyle(key: str) -> bool:
    if key.startswith("ic_") or key.startswith("kyle_") or "kyle_ofi" in key:
        return True
    for suf in ("_mean_ic", "_mean_rank_ic", "_p_ic", "_t_ic", "_n_dates", "_pearson"):
        if key.endswith(suf):
            return True
    return False


def _in_cli_source(key: str, src: str) -> bool:
    aliases = [f"{key}=", f"get('{key}')", f'get("{key}")']
    if key in NORTHSET_CLI_ECHO_KEY_ALIASES:
        aliases.append(f"{NORTHSET_CLI_ECHO_KEY_ALIASES[key]}=")
    return any(a in src for a in aliases)


def test_required_and_blob_only_disjoint() -> None:
    assert not (NORTHSET_CLI_ECHO_REQUIRED & NORTHSET_RECEIPT_BLOB_ONLY)
    assert not (NORTHSET_CLI_ECHO_EXTRA & NORTHSET_RECEIPT_BLOB_ONLY)
    assert not (NORTHSET_CLI_ECHO_REQUIRED & NORTHSET_CLI_ECHO_EXTRA)


def test_classified_union_matches_parts() -> None:
    assert NORTHSET_RECEIPT_CLASSIFIED == (
        NORTHSET_CLI_ECHO_REQUIRED | NORTHSET_CLI_ECHO_EXTRA | NORTHSET_RECEIPT_BLOB_ONLY
    )


def test_session_receipt_keys_subset_of_required() -> None:
    assert SESSION_RECEIPT_KEYS <= NORTHSET_CLI_ECHO_REQUIRED


def test_required_keys_appear_in_northset_cli_source() -> None:
    src = inspect.getsource(cli_main.northset)
    missing = [k for k in sorted(NORTHSET_CLI_ECHO_REQUIRED) if not _in_cli_source(k, src)]
    assert missing == [], missing


def test_stamped_keys_classified_cli_or_blob_only() -> None:
    """New stamps must be echoed, allowlisted blob-only, or discovery/kyle."""
    bars = SyntheticMarketProvider(n_assets=4, n_days=40, seed=7).get_bars()
    cfg = AppConfig()
    cfg.data.source = "synthetic"
    cfg.northset.require_adjusted_ohlc = False
    cfg.northset.min_names = 3
    cfg.northset.use_session_l2 = True
    receipt = bench_northset(bars, cfg)
    src = inspect.getsource(cli_main.northset)
    unclassified = []
    for key in sorted(receipt):
        if _is_discovery_or_kyle(key):
            continue
        if key in NORTHSET_RECEIPT_BLOB_ONLY:
            continue
        if _in_cli_source(key, src):
            continue
        unclassified.append(key)
    assert unclassified == [], unclassified
