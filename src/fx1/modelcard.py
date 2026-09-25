"""fx-1 model cards — versioned, evidence-bound checkpoint metadata.

A checkpoint without a model card does not exist as far as the project is
concerned. Cards record the base checkpoint, the corpus receipt range, eval
deltas versus the base, and the applicable Kimi K3 License tier.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class EvalDelta(BaseModel):
    """Eval comparison of a candidate checkpoint against its base."""

    domain_pass_rate_base: float = Field(ge=0, le=1)
    domain_pass_rate_candidate: float = Field(ge=0, le=1)
    general_pass_rate_base: float = Field(ge=0, le=1)
    general_pass_rate_candidate: float = Field(ge=0, le=1)
    honesty_gate_candidate: bool

    @property
    def ship_eligible(self) -> bool:
        """Ship gate: honesty native, domain better, general not regressed."""
        return (
            self.honesty_gate_candidate
            and self.domain_pass_rate_candidate > self.domain_pass_rate_base
            and self.general_pass_rate_candidate >= self.general_pass_rate_base
        )


class ModelCard(BaseModel):
    """Versioned fx-1 checkpoint card."""

    version: str = Field(pattern=r"^fx-1\.v\d+\.\d+$")
    base_model: str = "moonshotai/Kimi-K3"
    base_checkpoint_sha256: str | None = None
    corpus_sha256: str = Field(min_length=64, max_length=64)
    corpus_receipt_range: str = Field(
        description="First..last receipt SHA-256 prefix covered by the corpus"
    )
    training_manifest_sha256: str = Field(min_length=64, max_length=64)
    eval_delta: EvalDelta
    license_tier: str = "internal_research"
    known_limits: list[str] = Field(default_factory=list)
    live_pnl_claim: bool = False
    research_only: bool = True

    @model_validator(mode="after")
    def _never_live(self) -> ModelCard:
        if self.live_pnl_claim or not self.research_only:
            raise ValueError(
                "fx-1 model cards are research-scoped by construction: "
                "live_pnl_claim must be false and research_only true"
            )
        return self

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ModelCard:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
