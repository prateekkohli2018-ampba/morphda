#!/usr/bin/env python3
"""
Generate the deterministic (rule-based) mutant corpus.

For each task × operator combination:
  1. Compile trusted reference program.
  2. Apply mutation operator.
  3. Validate: syntax, execution, non-equivalence on 5 validation seeds.
  4. Write valid mutants to frozen corpus.

Usage:
    python scripts/generate_rule_mutants.py [--tasks retail01 web01] [--seeds 5]

Output:
    benchmark/frozen_mutants/rulemut_corpus.jsonl
    benchmark/frozen_mutants/rulemut_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphda.data.generators import generate_scenario
from morphda.execution.normalization import outputs_equal
from morphda.execution.sandbox import execute_program
from morphda.logging.schemas import record_to_dict
from morphda.logging.writer import LogWriter
from morphda.mutations import ALL_OPERATORS
from morphda.reference.compiler import compile_task, run_reference, ReferenceCompilerError
from morphda.tasks.factory import generate_task_set


def validate_mutant(
    reference_source: str,
    mutated_source: str,
    task,
    scenario_id: str,
    validation_seeds: list[int],
) -> tuple[bool, bool, bool, list[int]]:
    """
    Returns: (execution_valid, contract_valid, is_non_equivalent, non_equivalent_seeds)
    """
    non_eq_seeds = []
    for seed in validation_seeds:
        try:
            tables = generate_scenario(scenario_id, seed=seed)
        except KeyError:
            continue

        ref_result = execute_program(reference_source, tables, timeout_seconds=10)
        if not ref_result.success:
            continue

        mut_result = execute_program(mutated_source, tables, timeout_seconds=10)
        if not mut_result.success:
            return False, False, False, []

        # Contract: same output type
        if type(mut_result.output) != type(ref_result.output):
            # allow int/float equivalence
            if not (isinstance(mut_result.output, (int, float)) and
                    isinstance(ref_result.output, (int, float))):
                return True, False, False, []

        if not outputs_equal(ref_result.output, mut_result.output, task.output_type):
            non_eq_seeds.append(seed)

    is_non_equiv = len(non_eq_seeds) > 0
    return True, True, is_non_equiv, non_eq_seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--validation-seeds", nargs="+", type=int,
                        default=[42, 7, 123, 99, 17])
    parser.add_argument("--max-per-operator", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path("benchmark/frozen_mutants")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = generate_task_set(args.scenarios)
    print(f"Tasks: {len(tasks)}  |  Operators: {len(ALL_OPERATORS)}")
    print(f"Validation seeds: {args.validation_seeds}\n")

    stats = Counter()
    corpus_path = out_dir / "rulemut_corpus.jsonl"

    with LogWriter(corpus_path) as w:
        for task_idx, task in enumerate(tasks):
            try:
                ref_source = compile_task(task)
                # Quick sanity: reference executes on seed 42
                tables = generate_scenario(task.scenario_id, seed=42)
                run_reference(task, tables)
            except Exception as exc:
                print(f"  SKIP {task.task_id}: reference failed: {exc}")
                stats["task_skipped"] += 1
                continue

            task_valid = 0
            for operator in ALL_OPERATORS:
                if args.max_per_operator and stats[f"op_{operator.operator_id}"] >= args.max_per_operator:
                    continue

                record = operator.generate(ref_source, task, task.task_id)
                stats["generated"] += 1
                if record is None:
                    stats["not_applicable"] += 1
                    continue
                if not record.syntax_valid:
                    stats["syntax_invalid"] += 1
                    continue

                exec_valid, contract_valid, non_equiv, ne_seeds = validate_mutant(
                    ref_source, record.mutated_program, task,
                    task.scenario_id, args.validation_seeds,
                )

                if not exec_valid:
                    stats["exec_invalid"] += 1
                    continue
                if not contract_valid:
                    stats["contract_invalid"] += 1
                    continue
                if not non_equiv:
                    stats["equivalent"] += 1
                    continue

                record.execution_valid = True
                record.contract_valid = True
                record.non_equivalent_seeds = ne_seeds
                stats["valid"] += 1
                stats[f"op_{operator.operator_id}"] += 1
                task_valid += 1
                w.write(record_to_dict(record))

            print(f"  [{task_idx+1:3d}/{len(tasks)}] {task.task_id} (L{task.difficulty_level}): "
                  f"{task_valid} valid mutants")

    # Summary
    print(f"\n{'='*60}")
    print("MUTANT CORPUS SUMMARY")
    print(f"{'='*60}")
    print(f"Generated:         {stats['generated']}")
    print(f"Not applicable:    {stats['not_applicable']}")
    print(f"Syntax invalid:    {stats['syntax_invalid']}")
    print(f"Exec invalid:      {stats['exec_invalid']}")
    print(f"Contract invalid:  {stats['contract_invalid']}")
    print(f"Equivalent:        {stats['equivalent']}")
    print(f"Valid non-equiv:   {stats['valid']}")
    print(f"\nCorpus → {corpus_path}")

    with open(out_dir / "rulemut_stats.json", "w") as f:
        json.dump(dict(stats), f, indent=2)


if __name__ == "__main__":
    main()
