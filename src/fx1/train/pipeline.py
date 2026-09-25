"""The fx-1 staged training pipeline — industry-grade orchestration.

Stages, each gated and receipted:
  data -> quality -> eval_base -> train -> eval_candidate -> card

The trainer itself is injectable: in this repo the default raises with setup
instructions (torch/trl and a cluster are required); tests inject fakes. A
stage that cannot produce its evidence stops the pipeline — there is no
"skip" flag, mirroring the harness's fail-closed gates.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from fx1.data.quality import dedup_and_filter, frozen_split
from fx1.eval.bank import DEFAULT_BANK
from fx1.eval.compare import compare_runs
from fx1.eval.suite import run_suite
from fx1.train.config import TrainConfig
from fx1.train.receipts import issue_receipt


class Stage(StrEnum):
    DATA = "data"
    QUALITY = "quality"
    EVAL_BASE = "eval_base"
    TRAIN = "train"
    EVAL_CANDIDATE = "eval_candidate"
    CARD = "card"


STAGE_ORDER = list(Stage)

# Trainer signature: (train_jsonl, val_jsonl, config, work_dir) -> checkpoint dir
TrainerFn = Callable[[Path, Path, TrainConfig, Path], Path]
# Model function for eval stages: messages -> response
ModelFn = Callable[[list[dict[str, str]]], str]


def default_trainer(
    train_jsonl: Path, val_jsonl: Path, config: TrainConfig, work_dir: Path
) -> Path:
    raise NotImplementedError(
        "GPU training requires torch/trl/peft and a cluster per "
        "docs/FX1_TRAINING.md. Inject a trainer or run the generated cluster "
        "spec (fx1.train.cluster) on your scheduler."
    )


class PipelineState(BaseModel):
    stage: Stage = Stage.DATA
    artifacts: dict[str, str] = {}
    metrics: dict[str, float] = {}


class Pipeline:
    """Runs fx-1 training stages with hard gates between them."""

    def __init__(
        self,
        config: TrainConfig,
        work_dir: str | Path,
        *,
        trainer: TrainerFn = default_trainer,
    ) -> None:
        self.config = config
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.trainer = trainer
        self.state = PipelineState()

    def _advance(self, stage: Stage, **artifacts: str) -> None:
        expected = STAGE_ORDER[STAGE_ORDER.index(self.state.stage)]
        if stage != expected:
            raise RuntimeError(f"pipeline gate violation: expected stage {expected}, got {stage}")
        self.state.artifacts.update(artifacts)
        nxt = STAGE_ORDER.index(stage) + 1
        if nxt < len(STAGE_ORDER):
            self.state.stage = STAGE_ORDER[nxt]

    def run_quality_gate(self, eval_prompts: list[str] | None = None) -> dict:
        """DATA + QUALITY: load corpus, dedup/decontaminate, frozen split."""
        corpus_path = Path(self.config.corpus_jsonl)
        examples = [
            json.loads(line)
            for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prompts = eval_prompts or [
            m["content"] for t in DEFAULT_BANK for m in t.messages if m["role"] == "user"
        ]
        kept, report = dedup_and_filter(examples, eval_prompts=prompts)
        if report.kept == 0:
            raise RuntimeError("quality gate: corpus empty after filtering")
        manifest = frozen_split(kept, self.work_dir / "corpus")
        self.state.metrics["corpus_kept"] = float(report.kept)
        self.state.metrics["val_count"] = float(manifest.val_count)
        self._advance(Stage.DATA, corpus=str(corpus_path))
        self._advance(
            Stage.QUALITY,
            train=f"{self.work_dir}/corpus.train.jsonl",
            val=f"{self.work_dir}/corpus.val.jsonl",
            split_manifest=f"{self.work_dir}/corpus.split.json",
            quality_report=str(self._write("quality_report.json", report.model_dump())),
        )
        return report.model_dump()

    def run_eval_base(self, base_model_fn: ModelFn) -> Path:
        """EVAL_BASE: recorded base-model results are mandatory pre-training."""
        summary = run_suite(base_model_fn, list(DEFAULT_BANK))
        if not summary["honesty_gate_passed"]:
            raise RuntimeError(
                "base model fails the honesty gate; fix the base or the "
                "system contract before training"
            )
        out = self._write("eval_base.json", summary)
        self._advance(Stage.EVAL_BASE, eval_base=str(out))
        return out

    def run_training(self, seed: int = 17) -> Path:
        """TRAIN: emit the immutable receipt, then invoke the trainer."""
        receipt = issue_receipt(
            run_name=self.config.run_name,
            repo_root=Path.cwd(),
            config_path=self._write("train_config.json", self.config.model_dump(mode="json")),
            corpus_path=self.config.corpus_jsonl,
            split_manifest_path=self.state.artifacts["split_manifest"],
            eval_base_path=self.state.artifacts["eval_base"],
            seed=seed,
            out_path=self.work_dir / "training_receipt.json",
        )
        checkpoint = self.trainer(
            Path(self.state.artifacts["train"]),
            Path(self.state.artifacts["val"]),
            self.config,
            self.work_dir,
        )
        self._advance(
            Stage.TRAIN,
            training_receipt=str(self.work_dir / "training_receipt.json"),
            checkpoint=str(checkpoint),
        )
        self.state.metrics["receipt_dirty"] = float(receipt.dirty_worktree)
        return checkpoint

    def run_eval_candidate(self, candidate_fn: ModelFn) -> dict:
        """EVAL_CANDIDATE: statistical comparison against the recorded base."""
        base_summary = json.loads(
            Path(self.state.artifacts["eval_base"]).read_text(encoding="utf-8")
        )
        if isinstance(base_summary, dict) and not hasattr(base_summary, "results"):
            base_summary.setdefault("results", [])
        cand_summary = run_suite(candidate_fn, list(DEFAULT_BANK))
        cand_out = self._write("eval_candidate.json", cand_summary)
        base_results = list(base_summary.get("results", []))
        cand_results = cand_summary.results
        base_pass = [bool(r["passed"]) for r in base_results if r["kind"] == "domain"]
        cand_pass = [bool(r["passed"]) for r in cand_results if r["kind"] == "domain"]
        comparison = compare_runs(base_pass, cand_pass)
        comp_out = self._write("comparison.json", comparison.model_dump())
        self._advance(
            Stage.EVAL_CANDIDATE,
            eval_candidate=str(cand_out),
            comparison=str(comp_out),
        )
        return comparison.model_dump()

    def _write(self, name: str, payload: dict) -> Path:
        out = self.work_dir / name
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out
