"""fx-1 evaluation harness — built before any training runs.

PEP 562 lazy facade: submodule symbols resolve on first attribute access
so light consumers (``fx1 doctor``, ``--help``) never pay the
capability/calibration import chain. ``fx1.eval.capability`` and friends
import each other directly — that path stays eager by design.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fx1.eval.bank import (
        DEFAULT_BANK as DEFAULT_BANK,
    )
    from fx1.eval.bank import (
        DOMAIN_TASKS as DOMAIN_TASKS,
    )
    from fx1.eval.bank import (
        GENERAL_TASKS as GENERAL_TASKS,
    )
    from fx1.eval.bank import (
        HONESTY_BAITS as HONESTY_BAITS,
    )
    from fx1.eval.capability import (
        CapabilityEvalReport as CapabilityEvalReport,
    )
    from fx1.eval.capability import (
        run_capability_eval as run_capability_eval,
    )
    from fx1.eval.compare import (
        ComparisonResult as ComparisonResult,
    )
    from fx1.eval.compare import (
        compare_runs as compare_runs,
    )
    from fx1.eval.contamination import (
        ContaminationReport as ContaminationReport,
    )
    from fx1.eval.contamination import (
        run_contamination_audit as run_contamination_audit,
    )
    from fx1.eval.masking import (
        MemoryGapReport as MemoryGapReport,
    )
    from fx1.eval.masking import (
        mask_task as mask_task,
    )
    from fx1.eval.masking import (
        masked_twins as masked_twins,
    )
    from fx1.eval.masking import (
        memory_gap_report as memory_gap_report,
    )
    from fx1.eval.redteam import REDTEAM_TASKS as REDTEAM_TASKS
    from fx1.eval.rephrased import (
        rephrased_twins as rephrased_twins,
    )
    from fx1.eval.rephrased import (
        run_rephrased_gap as run_rephrased_gap,
    )
    from fx1.eval.suite import (
        EvalResult as EvalResult,
    )
    from fx1.eval.suite import (
        EvalTask as EvalTask,
    )
    from fx1.eval.suite import (
        run_suite as run_suite,
    )
    from fx1.eval.timepart import (
        TimePartition as TimePartition,
    )
    from fx1.eval.timepart import (
        partition_tasks as partition_tasks,
    )
    from fx1.eval.timepart import (
        post_cutoff_pass_rate as post_cutoff_pass_rate,
    )

_EXPORTS = {
    "bank": ("DEFAULT_BANK", "DOMAIN_TASKS", "GENERAL_TASKS", "HONESTY_BAITS"),
    "capability": ("CapabilityEvalReport", "run_capability_eval"),
    "compare": ("ComparisonResult", "compare_runs"),
    "contamination": ("ContaminationReport", "run_contamination_audit"),
    "masking": ("MemoryGapReport", "mask_task", "masked_twins", "memory_gap_report"),
    "redteam": ("REDTEAM_TASKS",),
    "rephrased": ("rephrased_twins", "run_rephrased_gap"),
    "suite": ("EvalResult", "EvalTask", "run_suite"),
    "timepart": ("TimePartition", "partition_tasks", "post_cutoff_pass_rate"),
}
_ATTR_TO_MODULE = {attr: mod for mod, attrs in _EXPORTS.items() for attr in attrs}

__all__ = sorted(_ATTR_TO_MODULE)


def __getattr__(name: str) -> Any:
    module = _ATTR_TO_MODULE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"fx1.eval.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
