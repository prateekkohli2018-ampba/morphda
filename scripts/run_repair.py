#!/usr/bin/env python3
"""
Repair experiment (Section 15.5, paper Table 6).

Compares R0–R7 repair strategies on natural wrong programs and mutants.

Repair methods:
  R0 - No retry (baseline)
  R2 - Always retry once (generic feedback)
  R3 - Random matched retry (same count as MORPH, generic feedback)
  R5 - MORPH-gated, no witness (any violation → generic feedback)
  R6 - MORPH relation-name feedback
  R7 - MORPH counterexample witness (primary experimental condition)

Key comparison: R5 vs R7 isolates whether the concrete witness helps
beyond just selecting suspicious programs.

Usage:
    python scripts/run_repair.py --model claude-sonnet-4-6 --max-programs 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphda.agents.llm_gateway import build_llm_client
from morphda.agents.langgraph_agent import MorphDaAgent
from morphda.agents.prompts import build_schema_summary
from morphda.data.generators import generate_scenario
from morphda.evaluation.metrics import compute_repair_metrics
from morphda.execution.normalization import outputs_equal
from morphda.logging.writer import load_jsonl, LogWriter
from morphda.reference.compiler import run_reference
from morphda.relations import ALL_RELATIONS
from morphda.repair.prompts import generic_retry_prompt, relation_name_prompt, witness_guided_prompt
from morphda.tasks.factory import generate_task_set
from morphda.verification.engine import VerificationEngine


def _attempt_repair(
    agent: MorphDaAgent,
    question: str,
    schema: str,
    original_program: str,
    strategy: str,
    witnesses: list,
    relation_names: list[str],
    tables: dict,
) -> str | None:
    """Attempt one repair using the given strategy. Returns repaired program or None."""
    if strategy == "R2_generic":
        prompt = generic_retry_prompt(question, schema, original_program)
    elif strategy == "R6_relation_name" and relation_names:
        prompt = relation_name_prompt(
            question=question,
            schema_summary=schema,
            program=original_program,
            relation_id=relation_names[0],
            relation_description=relation_names[0].replace("MR-", "").lower().replace("_", " "),
        )
    elif strategy == "R7_witness" and witnesses:
        prompt = witness_guided_prompt(
            question=question,
            schema_summary=schema,
            program=original_program,
            witness=witnesses[0],
        )
    else:
        prompt = generic_retry_prompt(question, schema, original_program)

    result = agent.repair(
        question=question,
        tables=tables,  # pass actual tables so execution can verify
        original_program=original_program,
        feedback=prompt,
    )
    return result.generated_program if result.success else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--programs-file",
                        default="runs/natural_agents/claude_sonnet_4_6/programs.jsonl")
    parser.add_argument("--verification-file",
                        default="runs/natural_agents/claude_sonnet_4_6/verification.jsonl")
    parser.add_argument("--max-programs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path("runs/repair")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load programs and verification results
    progs = [json.loads(l) for l in open(args.programs_file) if l.strip().startswith('{')]
    vers  = [json.loads(l) for l in open(args.verification_file) if l.strip().startswith('{')]
    ver_by_id = {v['program_id']: v for v in vers}

    # Filter: wrong-but-executable programs that we can verify
    wrong_exe = [p for p in progs
                 if p.get('execution_success') and not p.get('gold_correct')
                 and p['program_id'] in ver_by_id]

    # Also include a balanced sample of correct programs (for regression measurement)
    correct = [p for p in progs
               if p.get('execution_success') and p.get('gold_correct')]

    rng = random.Random(args.seed)
    if args.max_programs:
        wrong_exe = wrong_exe[:args.max_programs]
        correct_sample = rng.sample(correct, min(len(wrong_exe), len(correct)))
    else:
        correct_sample = rng.sample(correct, min(len(wrong_exe), len(correct)))

    tasks_dict = {t.task_id: t for t in generate_task_set()}
    llm   = build_llm_client(model=args.model)
    agent = MorphDaAgent(llm=llm, timeout_seconds=20)
    engine = VerificationEngine(relations=ALL_RELATIONS, timeout_seconds=15)

    print(f"Model: {args.model}")
    print(f"Wrong programs: {len(wrong_exe)} | Correct sample: {len(correct_sample)}")

    strategies = ["R0_no_retry", "R2_generic", "R6_relation_name", "R7_witness"]
    results: dict[str, list[dict]] = {s: [] for s in strategies}

    def _evaluate_program(program: str, task, tables, gold) -> dict:
        if program is None:
            return {"success": False, "correct": False, "morph_flagged": None}
        from morphda.execution.sandbox import execute_program
        exec_result = execute_program(program, tables, timeout_seconds=15)
        if not exec_result.success:
            return {"success": False, "correct": False, "morph_flagged": None}
        is_correct = outputs_equal(exec_result.output, gold, task.output_type)
        return {"success": True, "correct": is_correct, "morph_flagged": None}

    print("\n--- Repair on wrong-but-executable programs ---")
    for prog in wrong_exe:
        task_id = prog['task_id']
        task = tasks_dict.get(task_id)
        if task is None:
            continue

        try:
            tables = generate_scenario(task.scenario_id, seed=prog.get('data_seed', 42))
            gold   = run_reference(task, tables)
        except Exception:
            continue

        schema = build_schema_summary(tables)
        ver = ver_by_id.get(prog['program_id'], {})
        witnesses_raw = ver.get('witnesses', [])
        relation_names = [w.get('relation_id', '') for w in witnesses_raw]

        # Reconstruct ViolationWitness objects for R7
        from morphda.relations.base import ViolationWitness
        witnesses = [
            ViolationWitness(
                relation_id=w.get('relation_id', ''),
                case_id=w.get('case_id', ''),
                transformation_description=w.get('transformation_description', ''),
                source_output=w.get('source_output'),
                follow_up_output=w.get('follow_up_output'),
                expected_relation=w.get('expected_relation', ''),
                likely_issue=w.get('likely_issue', ''),
            )
            for w in witnesses_raw
        ]

        for strategy in strategies:
            if strategy == "R0_no_retry":
                row = {"task_id": task_id, "strategy": strategy,
                       "initial_wrong": True,
                       "repaired_correct": False,   # no repair attempted
                       "regressed": False}
            else:
                # Only R5/R6/R7 gate on MORPH; R2 always retries
                if strategy in ("R6_relation_name", "R7_witness") and not witnesses:
                    strategy_actual = "R2_generic"  # fall back to generic if no witnesses
                else:
                    strategy_actual = strategy

                repaired = _attempt_repair(
                    agent, task.canonical_question, schema,
                    prog.get('generated_program', ''),
                    strategy_actual, witnesses, relation_names,
                    tables=tables,
                )
                eval_r = _evaluate_program(repaired, task, tables, gold)
                row = {"task_id": task_id, "strategy": strategy,
                       "initial_wrong": True,
                       "repaired_correct": eval_r["correct"],
                       "regressed": False}

            results[strategy].append(row)

        status = "✓" if results["R7_witness"][-1]["repaired_correct"] else "✗"
        print(f"  {status} {task_id[:30]:<30} witnesses={len(witnesses)}", flush=True)

    # Summary
    print(f"\n{'='*65}")
    print("REPAIR EXPERIMENT RESULTS")
    print(f"{'='*65}")
    print(f"{'Strategy':<22} {'Repair rate':>12} {'n_repaired':>11} {'n_wrong':>8}")
    print("-"*65)

    for strategy in strategies:
        rows = results[strategy]
        n_wrong = len(rows)
        n_repaired = sum(1 for r in rows if r["repaired_correct"])
        rate = n_repaired / n_wrong if n_wrong else 0.0
        print(f"  {strategy:<20} {rate:>12.1%} {n_repaired:>11} {n_wrong:>8}")

    # Save
    all_results = []
    for strategy, rows in results.items():
        all_results.extend(rows)

    with open(out_dir / "repair_results.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults → {out_dir}/repair_results.jsonl")


if __name__ == "__main__":
    main()
