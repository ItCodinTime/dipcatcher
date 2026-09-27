"""Training configuration for fx-1.

fx-1 v0.x is LoRA/QLoRA only. Full fine-tuning of a 2.8T-parameter MoE is out
of scope without a dedicated compute-budget review. The ladder reflects
hardware reality: iterate on a proxy model, reserve full K3 runs for rental
multi-node clusters.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class LadderStage(StrEnum):
    """Development ladder stage for an fx-1 training run."""

    PROXY = "proxy"  # small open model / quantized build for iteration
    FINAL_K3 = "final_k3"  # full K3 LoRA on a rental multi-node cluster
    DISTILL = "distill"  # K3-LoRA teacher -> smaller servable fx-1 student


class LoRAConfig(BaseModel):
    rank: int = Field(default=16, ge=1, le=512)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class TrainConfig(BaseModel):
    """Validated, cost-disclosed fx-1 training run configuration."""

    run_name: str
    stage: LadderStage
    base_model: str = "moonshotai/Kimi-K3"
    corpus_jsonl: str
    eval_results_json: str  # eval harness MUST run before training
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    epochs: int = Field(default=1, ge=1, le=10)
    learning_rate: float = Field(default=1e-4, gt=0)
    # Hardware reality fields — must be estimated before any run launches.
    estimated_nodes: int = Field(ge=1)
    estimated_gpu_hours: float = Field(gt=0)
    estimated_cost_usd: float = Field(ge=0)
    license_tier: str = Field(
        default="internal_research",
        description="Kimi K3 License tier: internal_research | maas | commercial",
    )
    preserve_reasoning_content: bool = Field(
        default=True,
        description="K3 was trained in preserved-thinking-history mode; "
        "training data must keep reasoning_content + tool_calls intact.",
    )

    @model_validator(mode="after")
    def _k3_run_disclosure(self) -> TrainConfig:
        if self.stage == LadderStage.FINAL_K3:
            if self.base_model != "moonshotai/Kimi-K3":
                raise ValueError("FINAL_K3 stage must target moonshotai/Kimi-K3")
            if self.estimated_nodes < 2:
                raise ValueError(
                    "K3 is a 2.8T MoE (104B active); LoRA on the active "
                    "parameters is a multi-node job — estimated_nodes must be >= 2"
                )
            if not self.preserve_reasoning_content:
                raise ValueError(
                    "K3 requires preserved reasoning_content + tool_calls in training data"
                )
        return self
