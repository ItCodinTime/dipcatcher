"""Cluster configuration generation for K3-scale fx-1 training runs.

Hand-edited YAML is how training runs silently diverge from their receipts.
Cluster configs are *generated* from a validated :class:`ClusterSpec`, so the
artifact that goes to the scheduler is the same object the receipt hashes.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ClusterSpec(BaseModel):
    """Multi-node LoRA training specification for a 2.8T MoE base."""

    run_name: str
    nodes: int = Field(ge=2, description="K3 LoRA is multi-node by physics")
    gpus_per_node: int = Field(default=8, ge=1, le=8)
    zero_stage: int = Field(default=3, ge=2, le=3)
    precision: str = Field(default="bf16", pattern="^(bf16|fp8|mxfp4)$")
    gradient_checkpointing: bool = True
    sequence_length: int = Field(default=8192, ge=512, le=1_000_000)
    micro_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation: int = Field(default=16, ge=1)
    lora_rank: int = Field(default=16, ge=1, le=512)
    seed: int = Field(default=17)

    @model_validator(mode="after")
    def _sanity(self) -> ClusterSpec:
        if self.sequence_length >= 131_072 and self.nodes < 8:
            raise ValueError("long-context K3 training (>=128k) needs at least 8 nodes")
        if self.precision == "mxfp4" and self.zero_stage != 3:
            raise ValueError("mxfp4 QAT-aware runs require ZeRO stage 3")
        return self

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    def to_deepspeed_config(self) -> dict:
        return {
            "zero_optimization": {
                "stage": self.zero_stage,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
            "bf16": {"enabled": self.precision == "bf16"},
            "gradient_checkpointing": self.gradient_checkpointing,
            "train_micro_batch_size_per_gpu": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation,
            "seed": self.seed,
        }

    def save(self, out_dir: str | Path) -> dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        spec_path = out / "cluster_spec.json"
        ds_path = out / "deepspeed.json"
        spec_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        ds_path.write_text(json.dumps(self.to_deepspeed_config(), indent=2), encoding="utf-8")
        return {"spec": str(spec_path), "deepspeed": str(ds_path)}
