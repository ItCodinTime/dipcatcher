"""CI smoke for the robustness certifier.

Runs two SYNTHETIC toys, checks the proven linear radius against the
closed form, stamps a throwaway notebook, and validates the extension.
Does not read or write ``receipts/``.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from quant_fund.robustness.certify import certify
from quant_fund.robustness.schema import (
    robustness_extension_errors,
    stamp_robustness,
)
from quant_fund.robustness.strategies import FixedSchedule, LinearMargin


def run_smoke() -> dict[str, Any]:
    """Return a JSON-ready report. Raises if a proven check fails."""
    weights = np.array([0.5, 0.0, 0.0, 0.5])
    sample = np.array([1.0, -0.25, 0.1, 1.0])
    strategy = LinearMargin(weights, bias=0.0, name="smoke_linear")
    card = certify(
        strategy,
        sample,
        sigma=0.25,
        draws=64,
        alpha=0.01,
        seed=11,
        attack="cmaes",
        attack_steps=6,
        generations=5,
        gradient_steps=2,
        outcome_mean=0.4,
        outcome_scale=1.0,
        reference="gaussian",
        wasserstein_radius=0.1,
        evidence_class="SYNTHETIC",
    )
    analytic = card["analytic_radius"]["value"]
    certified = card["certified_radius"]["value"]
    if analytic is None or certified is None:
        raise RuntimeError("smoke linear toy did not produce finite radii")
    if not math.isclose(certified, analytic, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError("population certificate disagreed with the linear radius")
    if card["certified_radius"]["status"] != "proven":
        raise RuntimeError("population certificate was not marked proven")
    if card["distributional_robustness"]["worst_case_mean"]["status"] != "proven":
        raise RuntimeError("worst-case mean was not marked proven")
    mean_bound = card["distributional_robustness"]["worst_case_mean"]["value"]
    if not math.isclose(mean_bound, 0.3, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("worst-case mean disagreed with mean minus radius")
    schedule = FixedSchedule(np.array([0.0, 0.0, 1.0, 0.0]), name="smoke_schedule")
    scheduled = certify(
        schedule,
        np.array([0.0, 0.0, 1.0, 0.0]),
        sigma=0.25,
        draws=32,
        seed=3,
        run_attack=False,
        gradient="none",
        outcome_mean=1.0,
        outcome_scale=1.0,
        reference="gaussian",
        wasserstein_radius=0.2,
        evidence_class="SYNTHETIC",
    )
    jitter = scheduled["sensitivity"]["timing_jitter"]
    if jitter["value"] != 1.0 or jitter["status"] != "exact_on_path":
        raise RuntimeError("scheduled spike did not flip at a one-bar roll")
    notebook = {
        "schema_version": 1,
        "claim": "research_only",
        "note": "smoke notebook, not a sealed receipt",
    }
    stamped = stamp_robustness(notebook, [card, scheduled])
    errors = robustness_extension_errors(stamped)
    if errors:
        raise RuntimeError("stamped extension failed validation: " + "; ".join(errors))
    if notebook.get("robustness") is not None:
        raise RuntimeError("stamp mutated the input notebook")
    report = {
        "evidence_class": "SYNTHETIC",
        "research_only": True,
        "live_trading_claim": False,
        "linear": {
            "certified_radius": certified,
            "certified_status": card["certified_radius"]["status"],
            "empirical_attack_radius": card["empirical_attack_radius"]["value"],
            "empirical_status": card["empirical_attack_radius"]["status"],
            "gradient_attack_radius": card["gradient_attack"]["value"],
            "gradient_status": card["gradient_attack"]["status"],
            "worst_case_mean": mean_bound,
            "worst_case_ratio_status": card["distributional_robustness"]["worst_case_ratio"][
                "status"
            ],
            "radii_comparable": card["radii_comparable"],
        },
        "schedule_jitter_bars": jitter["value"],
        "extension_schema_version": stamped["extensions_schema_version"],
        "extension_errors": errors,
    }
    from quant_fund.leakage.patterns import find_forbidden_headline

    hits = find_forbidden_headline(json.dumps(report))
    if hits:
        raise RuntimeError(f"smoke report headlined forbidden tokens: {hits}")
    return report


def main() -> int:
    """Print the smoke report as JSON."""
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))
    return 0
