#!/usr/bin/env python3
"""
Pilot mutation study — Day 3 deliverable.

Runs the full mutation → filter → MORPH-DA verification pipeline on a
small set of tasks with trusted reference programs.

Usage:
    python scripts/pilot_mutation_study.py

Output:
    runs/pilot_mutation_study/results.jsonl
    runs/pilot_mutation_study/summary.txt
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make the project importable from the scripts directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from morphda.data.generators import generate_scenario
from morphda.evaluation.metrics import compute_mutation_score, compute_verification_metrics
from morphda.mutations import ALL_OPERATORS
from morphda.reference.compiler import compile_task, run_reference
from morphda.relations import ALL_RELATIONS
from morphda.tasks.schema import FilterSpec, MetricSpec, RankingSpec, TaskSpec
from morphda.verification.engine import VerificationEngine

# ─── Pilot task definitions ───────────────────────────────────────────────────

PILOT_TASKS = [
    TaskSpec(
        task_id="pilot_001",
        scenario_id="retail01",
        question_family="grouped_sum_rank",
        difficulty_level=2,
        inputs=["orders"],
        filters=[FilterSpec(column="order_status", operator="not_equal", value="cancelled")],
        metric=MetricSpec(name="total_revenue", operation="sum", column="revenue"),
        group_by=["category"],
        ranking=RankingSpec(direction="descending", k=1),
        output_type="label",
        canonical_question=(
            "Which category had the highest total revenue among non-cancelled orders?"
        ),
    ),
    TaskSpec(
        task_id="pilot_002",
        scenario_id="retail01",
        question_family="scalar_mean",
        difficulty_level=1,
        inputs=["orders"],
        filters=[FilterSpec(column="order_status", operator="not_equal", value="cancelled")],
        metric=MetricSpec(name="avg_revenue", operation="mean", column="revenue"),
        output_type="scalar",
        canonical_question=(
            "What is the mean revenue of non-cancelled orders?"
        ),
    ),
    TaskSpec(
        task_id="pilot_003",
        scenario_id="retail01",
        question_family="scalar_distinct_count",
        difficulty_level=1,
        inputs=["orders"],
        metric=MetricSpec(name="distinct_customers", operation="count_distinct", column="customer_id"),
        output_type="scalar",
        canonical_question="How many distinct customers placed orders?",
    ),
]

DATA_SEEDS = [42, 7, 123]


def run_pilot() -> None:
    out_dir = Path("runs/pilot_mutation_study")
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = VerificationEngine(relations=ALL_RELATIONS, timeout_seconds=15)
    results = []

    print(f"{'='*60}")
    print("MORPH-DA Pilot Mutation Study")
    print(f"Tasks: {len(PILOT_TASKS)}  |  Seeds: {len(DATA_SEEDS)}  |  Operators: {len(ALL_OPERATORS)}")
    print(f"{'='*60}\n")

    for task in PILOT_TASKS:
        print(f"▶ Task: {task.task_id} — {task.canonical_question[:60]}...")

        for seed in DATA_SEEDS:
            tables = generate_scenario(task.scenario_id, seed=seed)

            # ── Reference program ──────────────────────────────────────────
            reference_source = compile_task(task)
            try:
                gold_answer = run_reference(task, tables)
            except Exception as exc:
                print(f"  ✗ Reference failed on seed={seed}: {exc}")
                continue

            # Verify reference passes all relations (FP check)
            ref_report = engine.verify(reference_source, tables, task, f"ref_{task.task_id}_s{seed}")
            if ref_report.decision == "fail":
                print(f"  ⚠ Reference FAILED relation checks on seed={seed}: {ref_report.violated_relations} violations")
                for w in ref_report.witnesses[:2]:
                    print(f"    - {w.relation_id}: {w.likely_issue}")

            # ── Mutant generation and detection ────────────────────────────
            n_valid = 0
            n_killed = 0
            mutant_rows = []

            for operator in ALL_OPERATORS:
                record = operator.generate(reference_source, task, task.task_id)
                if record is None or not record.syntax_valid:
                    continue

                # Quick equivalence check: run on first seed
                from morphda.execution.sandbox import execute_program
                ref_result = execute_program(reference_source, tables, timeout_seconds=10)
                mut_result = execute_program(record.mutated_program, tables, timeout_seconds=10)

                if not mut_result.success:
                    continue  # runtime-invalid mutant

                # Check non-equivalence on this seed
                from morphda.execution.normalization import outputs_equal
                if outputs_equal(ref_result.output, mut_result.output, task.output_type):
                    continue  # provisionally equivalent on this seed

                record.execution_valid = True
                record.contract_valid = True
                record.non_equivalent_seeds = [seed]
                n_valid += 1

                # Run MORPH-DA verifier on mutant
                mut_report = engine.verify(
                    record.mutated_program, tables, task,
                    f"mut_{task.task_id}_s{seed}_{operator.operator_id}"
                )
                killed = mut_report.decision == "fail"
                if killed:
                    n_killed += 1

                mutant_rows.append({
                    "task_id": task.task_id,
                    "seed": seed,
                    "operator_id": operator.operator_id,
                    "mutation_family": operator.mutation_family,
                    "valid": True,
                    "killed": killed,
                    "violated_relations": mut_report.violated_relations,
                    "witness_relation": mut_report.witnesses[0].relation_id if mut_report.witnesses else None,
                })

            ms = n_killed / n_valid if n_valid > 0 else 0.0
            print(f"  seed={seed}: gold={gold_answer!r:20}  "
                  f"valid_mutants={n_valid:3d}  killed={n_killed:3d}  "
                  f"mutation_score={ms:.1%}")
            results.extend(mutant_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        with open(out_dir / "results.jsonl", "w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

        all_killed = [r["killed"] for r in results]
        all_families = [r["mutation_family"] for r in results]
        task_ids_list = [r["task_id"] for r in results]

        metrics = compute_mutation_score(all_killed, all_families)

        summary = [
            "\n" + "="*60,
            "SUMMARY",
            "="*60,
            f"Total valid mutants:   {metrics.n_valid_mutants}",
            f"Total killed:          {metrics.n_killed}",
            f"Micro mutation score:  {metrics.micro_mutation_score:.1%}",
            f"Macro mutation score:  {metrics.macro_mutation_score:.1%}",
            "",
            "Per-family kill rate:",
        ]
        for fam, rate in sorted(metrics.per_family_kill_rate.items()):
            summary.append(f"  {fam:<20} {rate:.1%}")

        summary_text = "\n".join(summary)
        print(summary_text)

        with open(out_dir / "summary.txt", "w") as f:
            f.write(summary_text + "\n")

        print(f"\nResults written to {out_dir}/")
    else:
        print("No results generated.")


if __name__ == "__main__":
    run_pilot()
