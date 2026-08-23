"""
Relation applicability rules — which relations apply to which task specs.

Used by the verification engine to avoid running irrelevant relations
and to track relation coverage statistics.
"""

from __future__ import annotations

from morphda.relations.base import MetamorphicRelation
from morphda.relations import ALL_RELATIONS
from morphda.tasks.schema import TaskSpec


def get_applicable_relations(
    task_spec: TaskSpec,
    relations: list[MetamorphicRelation] | None = None,
) -> list[MetamorphicRelation]:
    """Return the subset of relations applicable to this task."""
    if relations is None:
        relations = ALL_RELATIONS
    return [r for r in relations if r.is_applicable(task_spec)]


def applicability_report(
    tasks: list[TaskSpec],
    relations: list[MetamorphicRelation] | None = None,
) -> dict:
    """
    Compute relation applicability statistics across a task set.

    Returns a dict with:
      - per_relation: {relation_id: coverage_fraction}
      - per_task: {task_id: [applicable_relation_ids]}
    """
    if relations is None:
        relations = ALL_RELATIONS

    per_relation: dict[str, int] = {r.relation_id: 0 for r in relations}
    per_task: dict[str, list[str]] = {}

    for task in tasks:
        applicable = [r for r in relations if r.is_applicable(task)]
        per_task[task.task_id] = [r.relation_id for r in applicable]
        for r in applicable:
            per_relation[r.relation_id] += 1

    n = len(tasks)
    return {
        "per_relation": {rid: count / n for rid, count in per_relation.items()},
        "per_task": per_task,
        "mean_relations_per_task": sum(len(v) for v in per_task.values()) / n if n else 0,
    }
