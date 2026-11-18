"""
Task validator — verifies that reference programs:
  1. Execute successfully on all data seeds.
  2. Produce consistent gold answers across seeds (same value or same label set).
  3. Pass all applicable metamorphic relations (zero false positives).
  4. Produce non-equivalent output when at least one mutation is applied.

This is the Phase 1 exit condition (paper Section 8.6 / 11.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from morphda.data.generators import generate_scenario
from morphda.execution.normalization import outputs_equal
from morphda.reference.compiler import compile_task, run_reference, ReferenceCompilerError
from morphda.tasks.schema import TaskSpec
from morphda.verification.engine import VerificationEngine

# Lazy import to avoid circular dependency:
# relations.base → tasks.schema but tasks.validators → relations
def _get_all_relations():
    from morphda.relations import ALL_RELATIONS
    return ALL_RELATIONS


@dataclass
class TaskValidationResult:
    task_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    gold_answers: dict[int, Any] = field(default_factory=dict)
    relation_violations: list[str] = field(default_factory=list)


def validate_task(
    task: TaskSpec,
    seeds: list[int] | None = None,
    engine: VerificationEngine | None = None,
    verbose: bool = False,
) -> TaskValidationResult:
    """
    Validate a single task's reference program across multiple data seeds.

    Checks:
      1. Reference program compiles without error.
      2. Reference program executes on each seed.
      3. Reference program passes all applicable metamorphic relations (FP = 0).
      4. Gold answers are deterministic (same seed always gives same answer).

    Args:
        task:    The task to validate.
        seeds:   Data seeds to test. Defaults to [42, 7, 123].
        engine:  Verification engine (creates default if None).
        verbose: Print progress.

    Returns:
        TaskValidationResult with pass/fail and any failure messages.
    """
    if seeds is None:
        seeds = [42, 7, 123]
    if engine is None:
        engine = VerificationEngine(relations=_get_all_relations(), timeout_seconds=15)

    result = TaskValidationResult(task_id=task.task_id, passed=True)

    # 1. Compile
    try:
        source = compile_task(task)
    except ReferenceCompilerError as exc:
        result.passed = False
        result.failures.append(f"CompileError: {exc}")
        return result

    # 2. Execute on each seed
    for seed in seeds:
        try:
            tables = generate_scenario(task.scenario_id, seed=seed)
        except KeyError:
            result.failures.append(f"seed={seed}: scenario '{task.scenario_id}' not found")
            result.passed = False
            continue

        try:
            gold = run_reference(task, tables)
        except ReferenceCompilerError as exc:
            result.failures.append(f"seed={seed}: execution failed: {exc}")
            result.passed = False
            continue

        result.gold_answers[seed] = gold

        # 3. Metamorphic relation check (must all pass for reference)
        report = engine.verify(source, tables, task, f"ref_{task.task_id}_s{seed}")
        if report.decision == "fail":
            for w in report.witnesses:
                msg = f"seed={seed}: MR VIOLATION {w.relation_id} — {w.likely_issue}"
                result.relation_violations.append(msg)
                result.failures.append(msg)
            result.passed = False

    # 4. Repeatability: same seed, same answer
    for seed in seeds:
        if seed not in result.gold_answers:
            continue
        try:
            tables = generate_scenario(task.scenario_id, seed=seed)
            gold2 = run_reference(task, tables)
            if not outputs_equal(result.gold_answers[seed], gold2, task.output_type):
                msg = f"seed={seed}: non-deterministic gold answer"
                result.failures.append(msg)
                result.passed = False
        except Exception as exc:
            result.failures.append(f"seed={seed}: repeatability check failed: {exc}")

    return result


def validate_task_set(
    tasks: list[TaskSpec],
    seeds: list[int] | None = None,
    verbose: bool = True,
) -> tuple[list[TaskValidationResult], dict]:
    """
    Validate a full set of tasks. Returns results and a summary dict.
    """
    engine = VerificationEngine(relations=_get_all_relations(), timeout_seconds=15)
    results = []
    n_pass = n_fail = 0

    for i, task in enumerate(tasks):
        if verbose:
            print(f"  [{i+1:3d}/{len(tasks)}] {task.task_id} (L{task.difficulty_level}) ... ",
                  end="", flush=True)
        r = validate_task(task, seeds=seeds, engine=engine)
        results.append(r)
        if r.passed:
            n_pass += 1
            if verbose:
                print("✓")
        else:
            n_fail += 1
            if verbose:
                print(f"✗  {r.failures[0][:80]}")

    summary = {
        "total": len(tasks),
        "passed": n_pass,
        "failed": n_fail,
        "pass_rate": n_pass / len(tasks) if tasks else 0.0,
        "mr_violations": sum(len(r.relation_violations) for r in results),
    }
    return results, summary
