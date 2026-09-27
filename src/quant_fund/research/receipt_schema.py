"""Research-notebook schema 2 and the schema-1 migration.

Schema 2 adds a top-level ``backtest_overfitting`` block. Schema 1 receipts
stay valid without that block. Migration copies every existing field and, when
the block is absent, stamps an explicit uncomputed record. It does not
recompute metrics and does not alter sealed research numbers.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from quant_fund.research.catalog import (
    RESEARCH_RECEIPT_SCHEMA_VERSION,
    family_blob_forbidden_metrics_absent,
)

OVERFITTING_METRIC_KEYS = ("pbo", "dsr", "dsr_counted_trials", "psr", "min_trl")
OVERFITTING_REQUIRED_KEYS = (
    "claim",
    "research_only",
    "n_trials",
    "n_trials_effective",
    *OVERFITTING_METRIC_KEYS,
)
_UNIT_INTERVAL_KEYS = ("pbo", "dsr", "dsr_counted_trials", "psr")
_METRICS_STATUS = frozenset({"computed", "unavailable", "legacy_uncomputed"})


def unavailable_overfitting_block() -> dict[str, Any]:
    """Explicit empty diagnostic. Metrics are absent, not zero."""
    return {
        "claim": "research_diagnostic_only",
        "research_only": True,
        "metrics_status": "unavailable",
        "pbo": None,
        "dsr": None,
        "dsr_counted_trials": None,
        "psr": None,
        "min_trl": None,
        "n_trials": 0,
        "n_trials_effective": 0,
    }


def _legacy_trial_count(payload: dict[str, Any]) -> int:
    rankers = payload.get("rankers")
    if not isinstance(rankers, list):
        return 0
    count = 0
    for ranker in rankers:
        if not isinstance(ranker, dict):
            continue
        name = ranker.get("name")
        if isinstance(name, str) and name.strip() and not name.startswith("_"):
            count += 1
    return count


def migrate_research_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a schema-2 view of a research receipt.

    Schema 2 inputs are deep-copied unchanged. Schema 1 inputs are copied,
    their version is set to 2, and a missing overfitting block is filled with
    an uncomputed record whose trial count is the number of named rankers.
    Metric values already stored on a schema-1 block are kept. Any other
    schema raises ``ValueError``.
    """
    if not isinstance(payload, dict):
        raise TypeError("receipt must be an object")
    version = payload.get("schema_version")
    if isinstance(version, bool):
        raise ValueError("cannot migrate a boolean schema version")
    if version == RESEARCH_RECEIPT_SCHEMA_VERSION:
        return copy.deepcopy(payload)
    if version != 1:
        raise ValueError(f"cannot migrate research receipt schema {version!r}")
    migrated = copy.deepcopy(payload)
    migrated["schema_version"] = RESEARCH_RECEIPT_SCHEMA_VERSION
    existing = migrated.get("backtest_overfitting")
    if isinstance(existing, dict):
        return migrated
    block = unavailable_overfitting_block()
    trials = _legacy_trial_count(payload)
    block["n_trials"] = trials
    block["n_trials_effective"] = trials
    block["metrics_status"] = "legacy_uncomputed"
    block["migrated_from_schema"] = 1
    migrated["backtest_overfitting"] = block
    return migrated


def _unit_or_null(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and 0.0 <= number <= 1.0


def _min_trl_or_null(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and number >= 1.0


def _nonneg_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def overfitting_block_errors(notebook: dict[str, Any]) -> list[str]:
    """Fail-closed checks for the backtest-overfitting stamp.

    Schema 1 may omit the block. Schema 2 requires it. A present block, on
    either version, must carry the diagnostic keys, unit-interval probabilities,
    ``dsr <= psr`` and ``dsr_counted_trials <= psr`` when both ends are finite,
    and no forbidden headline key.
    """
    version = notebook.get("schema_version")
    block = notebook.get("backtest_overfitting")
    if version == 1 and block is None:
        return []
    errors: list[str] = []
    if not isinstance(block, dict):
        errors.append("backtest_overfitting_missing")
        return errors
    if not family_blob_forbidden_metrics_absent(block):
        errors.append("backtest_overfitting_forbidden_metrics")
    if block.get("claim") != "research_diagnostic_only":
        errors.append("backtest_overfitting_claim")
    if block.get("research_only") is not True:
        errors.append("backtest_overfitting_research_only")
    missing = [key for key in OVERFITTING_REQUIRED_KEYS if key not in block]
    if missing:
        errors.append("backtest_overfitting_fields_missing:" + ",".join(missing))
        return errors
    status = block.get("metrics_status")
    if status is not None and status not in _METRICS_STATUS:
        errors.append("backtest_overfitting_metrics_status")
    if not _nonneg_int(block.get("n_trials")):
        errors.append("backtest_overfitting_n_trials")
        return errors
    if not _nonneg_int(block.get("n_trials_effective")):
        errors.append("backtest_overfitting_n_trials_effective")
        return errors
    n_trials = int(block["n_trials"])
    n_effective = int(block["n_trials_effective"])
    if n_effective > n_trials:
        errors.append("backtest_overfitting_effective_exceeds_trials")
    if n_trials == 0 and n_effective != 0:
        errors.append("backtest_overfitting_effective_exceeds_trials")
    for key in _UNIT_INTERVAL_KEYS:
        if not _unit_or_null(block.get(key)):
            errors.append(f"backtest_overfitting_{key}")
    if not _min_trl_or_null(block.get("min_trl")):
        errors.append("backtest_overfitting_min_trl")
    if n_trials == 0:
        for key in OVERFITTING_METRIC_KEYS:
            if block.get(key) is not None:
                errors.append(f"backtest_overfitting_{key}_without_trials")
    psr = block.get("psr")
    if isinstance(psr, (int, float)) and not isinstance(psr, bool):
        for key in ("dsr", "dsr_counted_trials"):
            other = block.get(key)
            if (
                isinstance(other, (int, float))
                and not isinstance(other, bool)
                and float(other) > float(psr) + 1e-8
            ):
                errors.append(f"backtest_overfitting_{key}_above_psr")
    return errors
