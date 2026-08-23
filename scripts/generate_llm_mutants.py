#!/usr/bin/env python3
"""
Generate LLM-hidden-fault mutants (LLMMut track, Section 11.6).

Uses a different model from the primary analysis agent.
The mutator model does NOT see MORPH-DA relation descriptions.

Generated mutants are oracle-filtered: non-equivalent on validation seeds.
Targets 300-500 valid non-equivalent mutants (paper Section 33).

Usage:
    python scripts/generate_llm_mutants.py --model claude-sonnet-4-6 --candidates-per-task 3

Output:
    benchmark/frozen_mutants/llmmut_corpus.jsonl
    benchmark/frozen_mutants/llmmut_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphda.agents.llm_gateway import build_llm_client
from morphda.data.generators import generate_scenario
from morphda.execution.normalization import outputs_equal
from morphda.execution.sandbox import execute_program
from morphda.logging.writer import LogWriter
from morphda.mutations.llm_mutator import generate_llm_mutants, LLM_MUTATOR_SYSTEM
from morphda.reference.compiler import compile_task, run_reference, ReferenceCompilerError
from morphda.tasks.factory import generate_task_set


VALIDATION_SEEDS = [42, 7, 123, 99, 17]


def validate_llm_mutant(
    reference_source: str,
    mutated_source: str,
    task,
    seeds: list[int],
) -> tuple[bool, bool, list[int]]:
    """
    Validate a LLM-generated mutant.

    Returns: (exec_valid, non_equivalent, non_equiv_seeds)
    """
    non_equiv_seeds = []
    for seed in seeds:
        try:
            tables = generate_scenario(task.scenario_id, seed=seed)
        except Exception:
            continue

        ref_result = execute_program(reference_source, tables, timeout_seconds=10)
        if not ref_result.success:
            continue

        mut_result = execute_program(mutated_source, tables, timeout_seconds=10)
        if not mut_result.success:
            return False, False, []

        if not outputs_equal(ref_result.output, mut_result.output, task.output_type):
            non_equiv_seeds.append(seed)

    is_non_equiv = len(non_equiv_seeds) > 0
    return True, is_non_equiv, non_equiv_seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Mutator model (should differ from analysis agent)")
    parser.add_argument("--candidates-per-task", type=int, default=3)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path("benchmark/frozen_mutants")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = generate_task_set()
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    # Use a different model from the primary agent when possible
    print(f"LLM mutator model: {args.model}")
    print(f"Tasks: {len(tasks)} | Candidates per task: {args.candidates_per_task}")
    print(f"Validation seeds: {VALIDATION_SEEDS}\n")

    llm = build_llm_client(model=args.model)

    stats = Counter()
    corpus_path = out_dir / "llmmut_corpus.jsonl"

    with LogWriter(corpus_path) as w:
        for task_idx, task in enumerate(tasks):
            # Compile reference
            try:
                ref_source = compile_task(task)
                tables_test = generate_scenario(task.scenario_id, seed=42)
                run_reference(task, tables_test)
            except Exception as exc:
                print(f"  SKIP {task.task_id}: ref failed: {exc}")
                stats["task_skipped"] += 1
                continue

            # Generate LLM candidates
            from morphda.tasks.schema import TaskSpec

            class _TaskProxy:
                canonical_question = task.canonical_question
                task_id = task.task_id
                scenario_id = task.scenario_id
                output_type = task.output_type

            candidates = generate_llm_mutants(
                task_spec=_TaskProxy(),
                reference_source=ref_source,
                llm_fn=llm,
                n_candidates=args.candidates_per_task,
            )
            stats["generated"] += len(candidates)

            task_valid = 0
            for candidate in candidates:
                if not candidate.syntax_valid:
                    stats["syntax_invalid"] += 1
                    continue

                exec_valid, non_equiv, ne_seeds = validate_llm_mutant(
                    ref_source, candidate.mutated_program, task, VALIDATION_SEEDS
                )

                if not exec_valid:
                    stats["exec_invalid"] += 1
                    continue
                if not non_equiv:
                    stats["equivalent"] += 1
                    continue

                stats["valid"] += 1
                task_valid += 1

                record = {
                    "mutant_id": f"llmmut_{task.task_id}_{hashlib.sha256(candidate.mutated_program.encode()).hexdigest()[:8]}",
                    "task_id": task.task_id,
                    "reference_hash": candidate.reference_hash,
                    "generation_source": "llmmut",
                    "mutation_family": "llm_hidden",
                    "mutation_operator": f"llmmut_{args.model}",
                    "mutated_program": candidate.mutated_program,
                    "syntax_valid": True,
                    "execution_valid": True,
                    "contract_valid": True,
                    "non_equivalent_seeds": ne_seeds,
                    "held_out_operator": False,
                    "candidate_index": candidate.candidate_index,
                }
                w.write(record)

            print(f"  [{task_idx+1:3d}/{len(tasks)}] {task.task_id} (L{task.difficulty_level}): "
                  f"{task_valid} valid LLM mutants", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print("LLM MUTANT CORPUS SUMMARY")
    print(f"{'='*60}")
    print(f"Generated:         {stats['generated']}")
    print(f"Syntax invalid:    {stats['syntax_invalid']}")
    print(f"Exec invalid:      {stats['exec_invalid']}")
    print(f"Equivalent:        {stats['equivalent']}")
    print(f"Valid non-equiv:   {stats['valid']}")
    print(f"\nCorpus → {corpus_path}")

    with open(out_dir / "llmmut_stats.json", "w") as f:
        json.dump(dict(stats), f, indent=2)


if __name__ == "__main__":
    main()
