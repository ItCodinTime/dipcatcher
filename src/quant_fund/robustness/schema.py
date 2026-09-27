"""Optional robustness extension on a research receipt.

Parent ``schema_version`` is left unchanged. Sealed schema-1 notebooks stay
valid without this block. The extension carries its own schema version.
Stamping and migration return copies; they do not mutate the input and they
refuse to write under a ``receipts`` directory.

This follows the additive pattern used for other optional receipt stamps:
old notebooks omit the block, a migrated *view* may attach an explicit
uncomputed block, and nothing recomputes or rewrites sealed numbers.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from quant_fund.research.catalog import family_blob_forbidden_metrics_absent
from quant_fund.robustness.threats import THREAT_NAMES

ROBUSTNESS_EXTENSION_SCHEMA_VERSION = 1

_METRICS_STATUS = frozenset({"computed", "unavailable", "legacy_uncomputed"})
_EVIDENCE = frozenset({"SYNTHETIC", "path_diagnostic"})
_RATIO_STATUS = frozenset(
    {
        "proven_tight",
        "proven_unbounded",
        "outer_bound",
        "vacuous_outer_bound",
        "undefined",
    }
)
_RADIUS_STATUS = frozenset(
    {
        "proven",
        "high_probability",
        "empirical",
        "exact_on_path",
        "unavailable",
        "not_found",
        "corollary",
    }
)


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _finite_number(value: object) -> bool:
    if _is_bool(value) or isinstance(value, int):
        return isinstance(value, int) and not _is_bool(value)
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _nonnegative_number(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, (int, float)) or _is_bool(value):
        return False
    number = float(value)
    return math.isfinite(number) and number >= 0.0


def unavailable_robustness_block() -> dict[str, Any]:
    """Explicit empty extension. Nothing was certified."""
    return {
        "schema_version": ROBUSTNESS_EXTENSION_SCHEMA_VERSION,
        "claim": "robustness_diagnostic_only",
        "research_only": True,
        "live_trading_claim": False,
        "metrics_status": "unavailable",
        "strategies": [],
    }


def _legacy_block() -> dict[str, Any]:
    block = unavailable_robustness_block()
    block["metrics_status"] = "legacy_uncomputed"
    block["migrated_from_absent"] = True
    return block


def migrate_robustness_view(notebook: dict[str, Any]) -> dict[str, Any]:
    """Return a copy that names the robustness extension without computing it.

    Schema-1 receipts and schema-2 receipts (the overfitting stamp uses
    parent schema 2) are accepted. An extension that is already present is
    kept verbatim. Parent ``schema_version`` is not changed. The input is
    not mutated, and no metric is recomputed.
    """
    if not isinstance(notebook, dict):
        raise TypeError("notebook must be an object")
    version = notebook.get("schema_version")
    if _is_bool(version) or version not in {1, 2}:
        raise ValueError(f"cannot migrate a receipt with schema_version {version!r}")
    migrated = copy.deepcopy(notebook)
    if "robustness" in migrated or "extensions_schema_version" in migrated:
        return migrated
    migrated["extensions_schema_version"] = ROBUSTNESS_EXTENSION_SCHEMA_VERSION
    migrated["robustness"] = _legacy_block()
    return migrated


def stamp_robustness(notebook: dict[str, Any], scorecards: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a copy of ``notebook`` with a computed robustness extension.

    Does not mutate ``notebook``. Does not write a file.
    """
    if not isinstance(notebook, dict):
        raise TypeError("notebook must be an object")
    if not isinstance(scorecards, list) or not all(isinstance(item, dict) for item in scorecards):
        raise TypeError("scorecards must be a list of objects")
    stamped = copy.deepcopy(notebook)
    stamped["extensions_schema_version"] = ROBUSTNESS_EXTENSION_SCHEMA_VERSION
    stamped["robustness"] = {
        "schema_version": ROBUSTNESS_EXTENSION_SCHEMA_VERSION,
        "claim": "robustness_diagnostic_only",
        "research_only": True,
        "live_trading_claim": False,
        "metrics_status": "computed",
        "strategies": copy.deepcopy(scorecards),
    }
    errors = robustness_extension_errors(stamped)
    if errors:
        raise ValueError("robustness scorecard is not stampable: " + "; ".join(errors))
    return stamped


def assert_stamp_target_allowed(path: Path) -> None:
    """Refuse paths inside a sealed ``receipts`` directory."""
    resolved = Path(path).resolve()
    if "receipts" in resolved.parts:
        raise ValueError(f"refusing to write a sealed receipts path: {resolved}")


def write_stamped_notebook(
    path: Path, notebook: dict[str, Any], scorecards: list[dict[str, Any]]
) -> dict[str, Any]:
    """Stamp and write a new notebook. Will not write under ``receipts/``."""
    assert_stamp_target_allowed(path)
    stamped = stamp_robustness(notebook, scorecards)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(stamped, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return stamped


def robustness_extension_errors(notebook: dict[str, Any]) -> list[str]:
    """Fail-closed checks for the optional robustness extension.

    Absence of both ``robustness`` and ``extensions_schema_version`` is valid.
    A present extension must carry schema version 1, the diagnostic claim,
    no forbidden headline keys, and well-typed radii.
    """
    if not isinstance(notebook, dict):
        return ["robustness_notebook_not_object"]
    has_block = "robustness" in notebook
    has_version = "extensions_schema_version" in notebook
    if not has_block and not has_version:
        return []
    errors: list[str] = []
    version = notebook.get("extensions_schema_version")
    if _is_bool(version) or version != ROBUSTNESS_EXTENSION_SCHEMA_VERSION:
        errors.append("robustness_extensions_schema_version")
    block = notebook.get("robustness")
    if not isinstance(block, dict):
        errors.append("robustness_missing")
        return errors
    if not family_blob_forbidden_metrics_absent(block):
        errors.append("robustness_forbidden_metrics")
    if block.get("schema_version") != ROBUSTNESS_EXTENSION_SCHEMA_VERSION or _is_bool(
        block.get("schema_version")
    ):
        errors.append("robustness_schema_version")
    if block.get("claim") != "robustness_diagnostic_only":
        errors.append("robustness_claim")
    if block.get("research_only") is not True:
        errors.append("robustness_research_only")
    if block.get("live_trading_claim") is not False:
        errors.append("robustness_live_trading_claim")
    status = block.get("metrics_status")
    if status not in _METRICS_STATUS:
        errors.append("robustness_metrics_status")
        return errors
    strategies = block.get("strategies")
    if not isinstance(strategies, list):
        errors.append("robustness_strategies")
        return errors
    if status in {"unavailable", "legacy_uncomputed"} and strategies:
        errors.append("robustness_strategies_present_when_uncomputed")
    if status == "legacy_uncomputed" and block.get("migrated_from_absent") is not True:
        errors.append("robustness_legacy_flag")
    if status == "computed":
        for index, scorecard in enumerate(strategies):
            errors.extend(_scorecard_errors(scorecard, index))
    return errors


def _scorecard_errors(scorecard: object, index: int) -> list[str]:
    prefix = f"robustness_strategy:{index}"
    if not isinstance(scorecard, dict):
        return [f"{prefix}:not_object"]
    errors: list[str] = []
    if not family_blob_forbidden_metrics_absent(scorecard):
        errors.append(f"{prefix}:forbidden_metrics")
    name = scorecard.get("strategy")
    if not isinstance(name, str) or not name.strip():
        errors.append(f"{prefix}:name")
    if scorecard.get("evidence_class") not in _EVIDENCE:
        errors.append(f"{prefix}:evidence_class")
    if scorecard.get("research_only") is not True:
        errors.append(f"{prefix}:research_only")
    if scorecard.get("live_trading_claim") is not False:
        errors.append(f"{prefix}:live_trading_claim")
    if (
        not isinstance(scorecard.get("limitations"), list)
        or not scorecard["limitations"]
        or not all(isinstance(item, str) and item.strip() for item in scorecard["limitations"])
    ):
        errors.append(f"{prefix}:limitations")
    errors.extend(_radius_errors(scorecard.get("certified_radius"), f"{prefix}:certified_radius"))
    errors.extend(
        _radius_errors(
            scorecard.get("empirical_attack_radius"), f"{prefix}:empirical_attack_radius"
        )
    )
    errors.extend(_distribution_errors(scorecard.get("distributional_robustness"), prefix))
    sensitivity = scorecard.get("sensitivity")
    if not isinstance(sensitivity, dict):
        errors.append(f"{prefix}:sensitivity")
    else:
        missing = [threat for threat in THREAT_NAMES if threat not in sensitivity]
        if missing:
            errors.append(f"{prefix}:sensitivity_missing:" + ",".join(missing))
        for threat, record in sensitivity.items():
            if threat not in THREAT_NAMES:
                errors.append(f"{prefix}:sensitivity_unknown:{threat}")
                continue
            errors.extend(_sensitivity_errors(record, f"{prefix}:sensitivity:{threat}"))
    comparable = scorecard.get("radii_comparable")
    if not isinstance(comparable, bool):
        errors.append(f"{prefix}:radii_comparable")
    elif comparable:
        certified = _radius_value(scorecard.get("certified_radius"))
        empirical = _radius_value(scorecard.get("empirical_attack_radius"))
        if certified is None or empirical is None or certified > empirical + 1e-6:
            errors.append(f"{prefix}:certificate_exceeds_attack")
    return errors


def _radius_value(block: object) -> float | None:
    if not isinstance(block, dict):
        return None
    value = block.get("value")
    if not isinstance(value, (int, float)) or _is_bool(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _radius_errors(block: object, prefix: str) -> list[str]:
    if not isinstance(block, dict):
        return [f"{prefix}:missing"]
    errors: list[str] = []
    if block.get("status") not in _RADIUS_STATUS:
        errors.append(f"{prefix}:status")
    if block.get("norm") not in {"l2", "linf"}:
        errors.append(f"{prefix}:norm")
    if not isinstance(block.get("proven"), bool):
        errors.append(f"{prefix}:proven")
    if not _nonnegative_number(block.get("value")):
        errors.append(f"{prefix}:value")
    return errors


def _sensitivity_errors(record: object, prefix: str) -> list[str]:
    if not isinstance(record, dict):
        return [f"{prefix}:missing"]
    errors: list[str] = []
    if record.get("status") not in _RADIUS_STATUS:
        errors.append(f"{prefix}:status")
    if not isinstance(record.get("unit"), str) or not record["unit"].strip():
        errors.append(f"{prefix}:unit")
    if not isinstance(record.get("method"), str) or not record["method"].strip():
        errors.append(f"{prefix}:method")
    flipped = record.get("flipped")
    if flipped is not None and not isinstance(flipped, bool):
        errors.append(f"{prefix}:flipped")
    if not _nonnegative_number(record.get("value")):
        errors.append(f"{prefix}:value")
    return errors


def _distribution_errors(block: object, prefix: str) -> list[str]:
    if not isinstance(block, dict):
        return [f"{prefix}:distribution"]
    errors: list[str] = []
    if block.get("reference") not in {"gaussian", "empirical"}:
        errors.append(f"{prefix}:reference")
    mean = block.get("worst_case_mean")
    ratio = block.get("worst_case_ratio")
    if (
        not isinstance(mean, dict)
        or mean.get("status") != "proven"
        or not _finite_number(mean.get("value"))
    ):
        errors.append(f"{prefix}:worst_case_mean")
    if not isinstance(ratio, dict) or ratio.get("status") not in _RATIO_STATUS:
        errors.append(f"{prefix}:worst_case_ratio")
    elif ratio["status"] in {"proven_tight", "outer_bound"}:
        value = ratio.get("value")
        if (
            _is_bool(value)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            errors.append(f"{prefix}:worst_case_ratio_value")
    elif ratio.get("value") is not None:
        errors.append(f"{prefix}:worst_case_ratio_value")
    return errors
