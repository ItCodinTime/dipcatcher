"""fx-1 training pipeline (LoRA/QLoRA/DPO on Kimi K3 open weights)."""

from fx1.train.cluster import ClusterSpec
from fx1.train.config import LadderStage, TrainConfig
from fx1.train.dpo import PreferencePair, build_preference_pairs
from fx1.train.pipeline import Pipeline, Stage
from fx1.train.receipts import TrainingReceipt, issue_receipt, verify_training_receipt
from fx1.train.run import build_training_manifest
from fx1.train.tracking import Tracker

__all__ = [
    "ClusterSpec",
    "LadderStage",
    "Pipeline",
    "PreferencePair",
    "Stage",
    "Tracker",
    "TrainConfig",
    "TrainingReceipt",
    "build_preference_pairs",
    "build_training_manifest",
    "issue_receipt",
    "verify_training_receipt",
]
