"""BLOB_ONLY keys must not appear as dipcatcher northset CLI key= echoes.

Locks the blob-only allowlist: classified receipt keys that stay off CLI lines
must not silently leak into CLI source (would blur REQUIRED/EXTRA vs BLOB_ONLY).
"""

from __future__ import annotations

import inspect

from quant_fund.cli import main as cli_main
from quant_fund.northset.benches import (
    NORTHSET_CLI_ECHO_KEY_ALIASES,
    NORTHSET_CLI_ECHO_REQUIRED,
    NORTHSET_RECEIPT_BLOB_ONLY,
)


def _in_cli_source(key: str, src: str) -> bool:
    aliases = [f"{key}=", f"get('{key}')", f'get("{key}")']
    if key in NORTHSET_CLI_ECHO_KEY_ALIASES:
        aliases.append(f"{NORTHSET_CLI_ECHO_KEY_ALIASES[key]}=")
    return any(a in src for a in aliases)


def test_blob_only_disjoint_from_required() -> None:
    assert not (NORTHSET_RECEIPT_BLOB_ONLY & NORTHSET_CLI_ECHO_REQUIRED)


def test_blob_only_keys_absent_from_northset_cli_source() -> None:
    src = inspect.getsource(cli_main.northset)
    leaked = [k for k in sorted(NORTHSET_RECEIPT_BLOB_ONLY) if _in_cli_source(k, src)]
    assert leaked == [], leaked
