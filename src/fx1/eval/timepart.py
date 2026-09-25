"""Post-cutoff time-partitioned evaluation (LiveCodeBench-style).

Any eval task whose source artifact was created *after* the base model's
knowledge cutoff cannot be memorized from pretraining. This module partitions
the task set by creation timestamp and reports the post-cutoff slice
separately — the ship gate treats post-cutoff domain performance as the
least-contaminated competence estimate.
"""

from __future__ import annotations

from pydantic import BaseModel

from fx1.eval.suite import EvalTask


class TimePartition(BaseModel):
    cutoff: str
    pre_cutoff: list[str] = []  # task names
    post_cutoff: list[str] = []
    undated: list[str] = []


def partition_tasks(tasks: list[EvalTask], created: dict[str, str], cutoff: str) -> TimePartition:
    """Split tasks by creation date against the base model's cutoff.

    *created* maps task name -> ISO creation date of its source artifact.
    Tasks without provenance dates are listed as undated and excluded from
    both headline slices — undated evidence is not time-evidence.
    """
    pre: list[str] = []
    post: list[str] = []
    undated: list[str] = []
    for task in tasks:
        stamp = created.get(task.name)
        if stamp is None:
            undated.append(task.name)
        elif stamp > cutoff:
            post.append(task.name)
        else:
            pre.append(task.name)
    return TimePartition(cutoff=cutoff, pre_cutoff=pre, post_cutoff=post, undated=undated)


def post_cutoff_pass_rate(partition: TimePartition, results: dict[str, bool]) -> float | None:
    """Pass rate over the post-cutoff slice; None if the slice is empty."""
    if not partition.post_cutoff:
        return None
    hits = [results.get(name, False) for name in partition.post_cutoff]
    return sum(hits) / len(hits)
