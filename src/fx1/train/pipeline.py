"""The fx-1 staged training pipeline — industry-grade orchestration.

Stages, each gated and receipted:
  data -> quality -> eval_base -> train -> eval_candidate -> card

The trainer itself is injectable: in this repo the default raises with setup
instructions (torch/trl and a cluster are required); tests inject fakes. A
stage that cannot produce its evidence stops the pipeline — there is no
"skip" flag, mirroring the harness's fail-closed gates.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from fx1.data.quality import dedup_and_filter, frozen_split
from fx1.eval.bank import DEFAULT_BANK
from fx1.eval.compare import compare_runs
from fx1.eval.suite import run_suite
from fx1.modelcard import EvalDelta, ModelCard
from fx1.train.config import TrainConfig
from fx1.train.receipts import issue_receipt
from fx1.train.run import build_training_manifest


class Stage(StrEnum):
    DATA = "data"
    QUALITY = "quality"
    EVAL_BASE = "eval_base"
    TRAIN = "train"
    EVAL_CANDIDATE = "eval_candidate"
    CARD = "card"
    COMPLETE = "complete"


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
        train_config_path = self._write("train_config.json", self.config.model_dump(mode="json"))
        manifest_config = self.config.model_copy(
            update={"eval_results_json": self.state.artifacts["eval_base"]}
        )
        manifest_path = self.work_dir / "training_manifest.json"
        build_training_manifest(manifest_config, manifest_path)
        receipt = issue_receipt(
            run_name=self.config.run_name,
            repo_root=Path.cwd(),
            config_path=train_config_path,
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
            training_manifest=str(manifest_path),
            training_receipt=str(self.work_dir / "training_receipt.json"),
            checkpoint=str(checkpoint),
        )
        self.state.metrics["receipt_dirty"] = float(receipt.dirty_worktree)
        return checkpoint

    def run_eval_candidate(self, candidate_fn: ModelFn) -> dict:
        """EVAL_CANDIDATE: enforce the complete statistical ship gate."""
        base_summary = json.loads(
            Path(self.state.artifacts["eval_base"]).read_text(encoding="utf-8")
        )
        if isinstance(base_summary, dict) and not hasattr(base_summary, "results"):
            base_summary.setdefault("results", [])
        cand_summary = run_suite(candidate_fn, list(DEFAULT_BANK))
        cand_out = self._write("eval_candidate.json", cand_summary)
        base_results = list(base_summary.get("results", []))
        cand_results = cand_summary.results
        base_tasks = {str(r["task"]): r for r in base_results}
        cand_tasks = {str(r["task"]): r for r in cand_results}
        if base_tasks.keys() != cand_tasks.keys():
            raise RuntimeError("candidate ship gate: eval task sets do not match")
        domain_names = [str(r["task"]) for r in base_results if r["kind"] == "domain"]
        general_names = [str(r["task"]) for r in base_results if r["kind"] == "general"]
        if not domain_names or not general_names:
            raise RuntimeError("candidate ship gate: domain and general tasks are required")
        base_pass = [bool(base_tasks[name]["passed"]) for name in domain_names]
        cand_pass = [bool(cand_tasks[name]["passed"]) for name in domain_names]
        comparison = compare_runs(base_pass, cand_pass)
        comp_out = self._write("comparison.json", comparison.model_dump())

        def _pass_rate(results: dict[str, dict], names: list[str]) -> float:
            return sum(bool(results[name]["passed"]) for name in names) / len(names)

        delta = EvalDelta(
            domain_pass_rate_base=comparison.base_pass_rate,
            domain_pass_rate_candidate=comparison.candidate_pass_rate,
            general_pass_rate_base=_pass_rate(base_tasks, general_names),
            general_pass_rate_candidate=_pass_rate(cand_tasks, general_names),
            honesty_gate_candidate=bool(cand_summary["honesty_gate_passed"]),
            domain_significant_improvement=comparison.significant_improvement,
        )
        gate = {
            "ship_eligible": delta.ship_eligible,
            "eval_delta": delta.model_dump(),
            "comparison": comparison.model_dump(),
        }
        gate_out = self._write("ship_gate.json", gate)
        if not delta.ship_eligible:
            failed = [
                name
                for name, passed in {
                    "domain_significant_improvement": delta.domain_significant_improvement,
                    "domain_pass_rate_improved": (
                        delta.domain_pass_rate_candidate > delta.domain_pass_rate_base
                    ),
                    "general_no_regression": (
                        delta.general_pass_rate_candidate >= delta.general_pass_rate_base
                    ),
                    "honesty_gate": delta.honesty_gate_candidate,
                }.items()
                if not passed
            ]
            raise RuntimeError(f"candidate ship gate failed: {', '.join(failed)}")
        self._advance(
            Stage.EVAL_CANDIDATE,
            eval_candidate=str(cand_out),
            comparison=str(comp_out),
            ship_gate=str(gate_out),
        )
        return comparison.model_dump()

    def run_card(self, *, known_limits: list[str] | None = None) -> Path:
        """CARD: bind a ship-eligible checkpoint to its immutable evidence."""
        gate = json.loads(Path(self.state.artifacts["ship_gate"]).read_text())
        if gate.get("ship_eligible") is not True:
            raise RuntimeError("model card blocked: candidate ship gate did not pass")
        corpus_path = Path(self.config.corpus_jsonl)
        receipt_shas: list[str] = []
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line).get("receipt_sha256")
                if not isinstance(value, str) or len(value) != 64:
                    raise RuntimeError("model card blocked: corpus provenance is invalid")
                receipt_shas.append(value)
        if not receipt_shas:
            raise RuntimeError("model card blocked: corpus has no receipt provenance")
        receipt_shas.sort()
        manifest_path = Path(self.state.artifacts["training_manifest"])
        checkpoint = Path(self.state.artifacts["checkpoint"])
        checkpoint.mkdir(parents=True, exist_ok=True)
        card = ModelCard(
            version=self.config.run_name,
            base_model=self.config.base_model,
            corpus_sha256=self._sha256(corpus_path),
            corpus_receipt_range=(f"{receipt_shas[0][:16]}..{receipt_shas[-1][:16]}"),
            training_manifest_sha256=self._sha256(manifest_path),
            eval_delta=EvalDelta.model_validate(gate["eval_delta"]),
            license_tier=self.config.license_tier,
            known_limits=known_limits or [],
        )
        out = checkpoint / "modelcard.json"
        card.save(out)
        self._advance(Stage.CARD, modelcard=str(out))
        return out

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write(self, name: str, payload: dict) -> Path:
        out = self.work_dir / name
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out
