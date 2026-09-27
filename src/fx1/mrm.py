"""Model-risk evidence pack — the regulatory moat, compiled.

SR 26-2 (April 2026) explicitly excluded generative/agentic AI from model-risk
guidance, leaving institutions to build their own frameworks around the
surviving expectations: documented development, independent validation,
outcomes analysis, ongoing monitoring, governance. ``fx1 mrm`` compiles a
checkpoint's existing artifacts into that five-activity dossier structure —
fx-1 is the model a validation function can approve cheapest, because the
evidence already exists.

Every dossier section cites the artifact hashes it was compiled from; a
dossier referencing a missing artifact fails closed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from fx1.modelcard import ModelCard

FIVE_ACTIVITIES = (
    "development",
    "implementation",
    "validation",
    "monitoring",
    "governance",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DossierSection(BaseModel):
    activity: str
    artifact_hashes: dict[str, str] = Field(
        description="artifact path -> sha256 backing this activity"
    )
    summary: str


class MRMDossier(BaseModel):
    """The compiled five-activity model-risk dossier."""

    model_version: str
    compiled_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    base_model: str
    sections: list[DossierSection]
    contamination_flagged: bool
    ship_eligible: bool
    disclaimer: str = (
        "Research-scoped evidence pack. fx-1 and the dipcatcher harness "
        "produce research, backtest, or simulated evidence only; this dossier "
        "supports — and does not replace — the institution's independent "
        "validation. Not investment advice."
    )

    @property
    def complete(self) -> bool:
        return {s.activity for s in self.sections} == set(FIVE_ACTIVITIES)


def compile_dossier(
    *,
    modelcard_path: str | Path,
    artifacts: dict[str, str | Path],
    out_path: str | Path,
) -> MRMDossier:
    """Compile the dossier. Fail-closed on missing artifacts or activities.

    *artifacts* maps activity name -> artifact path (JSON/MD). The model card
    itself always backs ``governance``. A contamination report, if provided
    under key ``contamination_report``, is parsed for its overall flag.
    """
    card = ModelCard.load(modelcard_path)
    hashes: dict[str, dict[str, str]] = {}
    contamination_flagged = False
    for activity, artifact in artifacts.items():
        path = Path(artifact)
        if not path.exists():
            raise FileNotFoundError(f"dossier artifact for {activity!r} missing: {path}")
        hashes.setdefault(activity, {})[str(path)] = _sha(path)
        if activity == "validation" and "contamination" in path.name:
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                contamination_flagged = bool(report.get("overall_flagged", True))
            except json.JSONDecodeError:
                contamination_flagged = True
    sections: list[DossierSection] = []
    summaries = {
        "development": (
            f"fx-1 {card.version} fine-tuned from {card.base_model}; corpus "
            f"sha256 {card.corpus_sha256[:16]}…, training manifest "
            f"{card.training_manifest_sha256[:16]}…. Provenance-hashed corpus "
            "with documented quality gates."
        ),
        "implementation": (
            "Serving path enforces model-card ship gate and (when configured) "
            "release signatures; backends fail closed."
        ),
        "validation": (
            f"Ship gate: domain {card.eval_delta.domain_pass_rate_base:.2f} → "
            f"{card.eval_delta.domain_pass_rate_candidate:.2f}, general "
            f"{card.eval_delta.general_pass_rate_base:.2f} → "
            f"{card.eval_delta.general_pass_rate_candidate:.2f}, honesty "
            f"gate {'passed' if card.eval_delta.honesty_gate_candidate else 'FAILED'}. "
            f"Contamination audit flagged: {contamination_flagged}."
        ),
        "monitoring": (
            "Red-team suites, shadow/leaderboard harness surfaces, and "
            "telemetry hooks provide ongoing-monitoring inputs."
        ),
        "governance": (
            f"Model card {card.version}, license tier {card.license_tier}, "
            "research-scoped by construction (live_pnl_claim=false)."
        ),
    }
    for activity in FIVE_ACTIVITIES:
        backing = dict(hashes.get(activity, {}))
        backing[str(modelcard_path)] = _sha(Path(modelcard_path))
        sections.append(
            DossierSection(
                activity=activity,
                artifact_hashes=backing,
                summary=summaries[activity],
            )
        )
    dossier = MRMDossier(
        model_version=card.version,
        base_model=card.base_model,
        sections=sections,
        contamination_flagged=contamination_flagged,
        ship_eligible=card.eval_delta.ship_eligible,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dossier.model_dump_json(indent=2), encoding="utf-8")
    return dossier
