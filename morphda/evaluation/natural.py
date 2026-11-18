"""
Natural-agent evaluation (paper Sections 6-7).

Takes the programs an agent wrote on the source data seed and answers two questions:

  1. Which of them are *accidental corrects*?  Each program is re-executed
     UNCHANGED on held-out data seeds; wrong on any of them => wrong.
  2. How well does MORPH-DA flag the wrong ones?  Conditions:
        A  naive labels   (correct on the source seed)       , MORPH-DA with 1 transformation seed
        B  cross-seed labels (correct on source + held-out)  , MORPH-DA with 1 transformation seed
        C  cross-seed labels                                  , MORPH-DA with 2 transformation seeds

Programs whose original execution failed are excluded from every denominator.
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from morphda.data.generators import generate_scenario
from morphda.evaluation.metrics import compute_verification_metrics
from morphda.execution.normalization import outputs_equal
from morphda.execution.sandbox import execute_program
from morphda.reference.compiler import ReferenceCompilerError, run_reference
from morphda.relations import ALL_RELATIONS
from morphda.tasks.schema import TaskSpec
from morphda.verification.engine import VerificationEngine

SOURCE_SEED = 42
HELD_OUT_SEEDS = (7, 123)
VERIFIER_SEEDS = (42, 7)  # Condition B uses the first, Condition C uses both
DISCRIMINABILITY_SEEDS = (7, 42, 123)


# ── Task selection ───────────────────────────────────────────────────────────

def scope_is_discriminating(task: TaskSpec, seeds: Iterable[int] = DISCRIMINABILITY_SEEDS) -> bool:
    """
    True if dropping the task's whole scope (row filters and date window) changes
    the gold answer on at least one seed. Otherwise a program that ignores the
    scope passes on every seed regardless of implementation (paper Section 5).
    """
    if not task.filters and task.date is None:
        return True
    unscoped = task.model_copy(update={"filters": [], "date": None})
    for seed in seeds:
        tables = generate_scenario(task.scenario_id, seed=seed)
        try:
            unscoped_gold = run_reference(unscoped, tables)
        except ReferenceCompilerError:
            return True  # scope is structural (e.g. period comparison): cannot be dropped
        if not outputs_equal(run_reference(task, tables), unscoped_gold, task.output_type):
            return True
    return False


# ── Program loading ──────────────────────────────────────────────────────────

def load_source_programs(path: str | Path, seed: int = SOURCE_SEED) -> dict[str, dict]:
    """
    Read programs.jsonl and return {task_id: record} for the source seed.
    If a program_id appears more than once (an interrupted run was resumed),
    the last record wins.
    """
    latest: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            rec = json.loads(line)
            if int(rec.get("data_seed", -1)) == seed:
                latest[rec["task_id"]] = rec
    return latest


# ── Per-program evaluation ───────────────────────────────────────────────────

def _matches_gold(source: str, task: TaskSpec, seed: int, timeout: int) -> tuple[bool, bool, Any]:
    """Run `source` on (scenario, seed). Returns (executed, correct, tables)."""
    tables = generate_scenario(task.scenario_id, seed=seed)
    result = execute_program(source, tables, timeout)
    if not result.success:
        return False, False, tables
    gold = run_reference(task, tables)
    return True, outputs_equal(result.output, gold, task.output_type), tables


def evaluate_program(args: tuple) -> dict:
    """Worker: evaluate one program. Picklable for ProcessPoolExecutor."""
    task, source, held_out, verifier_seeds, timeout = args

    executed, correct_source, tables = _matches_gold(source, task, SOURCE_SEED, timeout)
    row: dict[str, Any] = {
        "task_id": task.task_id,
        "executed": executed,
        "correct_source": correct_source,
        "correct_held_out": {},
        "flags": {},
    }
    if not executed:
        return row

    for seed in held_out:
        _, ok, _ = _matches_gold(source, task, seed, timeout)
        row["correct_held_out"][str(seed)] = ok

    for rng_seed in verifier_seeds:
        engine = VerificationEngine(ALL_RELATIONS, timeout_seconds=timeout, rng_seed=rng_seed)
        report = engine.verify(source, tables, task, f"{task.task_id}_rng{rng_seed}")
        row["flags"][str(rng_seed)] = report.decision == "fail"
    return row


def evaluate_programs(
    programs: dict[str, dict],
    tasks: Iterable[TaskSpec],
    held_out: tuple[int, ...] = HELD_OUT_SEEDS,
    verifier_seeds: tuple[int, ...] = VERIFIER_SEEDS,
    timeout: int = 20,
    workers: int = 1,
) -> list[dict]:
    """Evaluate every program that belongs to one of `tasks`."""
    work = [
        (t, programs[t.task_id]["generated_program"], held_out, verifier_seeds, timeout)
        for t in tasks if t.task_id in programs
    ]
    if workers <= 1:
        return [evaluate_program(w) for w in work]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(evaluate_program, work, chunksize=1))


# ── Labels, conditions, metrics ──────────────────────────────────────────────

def is_accidental_correct(row: dict) -> bool:
    """Correct on the source seed, wrong on at least one held-out seed."""
    return bool(
        row["executed"] and row["correct_source"]
        and not all(row["correct_held_out"].values())
    )


def _wrong_naive(row: dict) -> bool:
    return not row["correct_source"]


def _wrong_cross_seed(row: dict) -> bool:
    return not (row["correct_source"] and all(row["correct_held_out"].values()))


def _flag_one(row: dict) -> bool:
    return row["flags"][str(VERIFIER_SEEDS[0])]


def _flag_either(row: dict) -> bool:
    return any(row["flags"][str(s)] for s in VERIFIER_SEEDS)


CONDITIONS = {
    "A_naive":      (_wrong_naive, _flag_one),
    "B_cross_seed": (_wrong_cross_seed, _flag_one),
    "C_two_seeds":  (_wrong_cross_seed, _flag_either),
}


def _confusion(labels: list[bool], flags: list[bool]) -> dict[str, int]:
    return {
        "tp": sum(l and f for l, f in zip(labels, flags)),
        "fp": sum((not l) and f for l, f in zip(labels, flags)),
        "tn": sum((not l) and (not f) for l, f in zip(labels, flags)),
        "fn": sum(l and (not f) for l, f in zip(labels, flags)),
    }


def _metrics(labels: list[bool], flags: list[bool]) -> dict[str, float]:
    m = compute_verification_metrics(labels, flags)
    return {"precision": m.precision, "recall": m.recall, "fpr": m.false_positive_rate,
            "f1": m.f1, "aar": m.accepted_answer_risk}


def bootstrap_ci(labels: list[bool], flags: list[bool], n_boot: int = 3000,
                 seed: int = 0, level: float = 0.95) -> dict[str, list[float]]:
    """
    Percentile CIs for precision / recall / FPR, resampling tasks.
    Each task contributes exactly one program here, so the task-clustered
    bootstrap is a resample of programs.
    """
    rng = np.random.default_rng(seed)
    labels_a, flags_a = np.array(labels), np.array(flags)
    n = len(labels_a)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = _metrics(labels_a[idx].tolist(), flags_a[idx].tolist())
        for k in ("precision", "recall", "fpr"):
            draws[k].append(m[k])
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in draws.items()}


def summarize(rows: list[dict], n_boot: int = 3000) -> dict:
    """Accidental-correct counts plus Conditions A/B/C for one model."""
    usable = [r for r in rows if r["executed"]]
    seed_correct = [r for r in usable if r["correct_source"]]
    accidental = [r for r in usable if is_accidental_correct(r)]

    out: dict[str, Any] = {
        "n_tasks": len(rows),
        "n_exec_failures": len(rows) - len(usable),
        "n_correct_source_seed": len(seed_correct),
        "n_accidental_correct": len(accidental),
        "n_cross_seed_correct": len(seed_correct) - len(accidental),
        "accidental_correct_flagged": sum(_flag_one(r) for r in accidental),
        "conditions": {},
    }
    for name, (wrong_fn, flag_fn) in CONDITIONS.items():
        labels = [wrong_fn(r) for r in usable]
        flags = [flag_fn(r) for r in usable]
        cond = {**_confusion(labels, flags), "n": len(usable), **_metrics(labels, flags)}
        if name == "B_cross_seed":
            cond["ci95"] = bootstrap_ci(labels, flags, n_boot=n_boot)
        out["conditions"][name] = cond
    return out
