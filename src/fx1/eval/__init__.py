"""fx-1 evaluation harness — built before any training runs."""

from fx1.eval.bank import DEFAULT_BANK, DOMAIN_TASKS, GENERAL_TASKS, HONESTY_BAITS
from fx1.eval.compare import ComparisonResult, compare_runs
from fx1.eval.contamination import ContaminationReport, run_contamination_audit
from fx1.eval.masking import MemoryGapReport, mask_task, masked_twins, memory_gap_report
from fx1.eval.redteam import REDTEAM_TASKS
from fx1.eval.suite import EvalResult, EvalTask, run_suite
from fx1.eval.timepart import TimePartition, partition_tasks, post_cutoff_pass_rate

__all__ = [
    "DEFAULT_BANK", "DOMAIN_TASKS", "GENERAL_TASKS", "HONESTY_BAITS",
    "REDTEAM_TASKS", "ComparisonResult", "ContaminationReport", "EvalResult",
    "EvalTask", "MemoryGapReport", "TimePartition", "compare_runs", "mask_task",
    "masked_twins", "memory_gap_report", "partition_tasks",
    "post_cutoff_pass_rate", "run_contamination_audit", "run_suite",
]
