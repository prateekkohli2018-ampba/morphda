#!/usr/bin/env python3
"""
End-to-end verification experiment (RuleMut track).

Usage:
    python scripts/run_verification.py [--workers 4] [--eval-seed 999]

Output:
    runs/verification/results.jsonl
    runs/verification/summary.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphda.data.generators import generate_scenario
from morphda.evaluation.metrics import compute_mutation_score
from morphda.logging.writer import load_jsonl, LogWriter
from morphda.relations import ALL_RELATIONS, UNIVERSAL_RELATIONS, FILTER_RELATIONS, AGGREGATION_RELATIONS, GROUPING_RELATIONS
from morphda.tasks.factory import generate_task_set
from morphda.verification.engine import VerificationEngine


def _verify_one(args: tuple) -> dict:
    """Worker function: verify one mutant with all methods."""
    mutant, task_dict, eval_seed = args

    task_id = mutant["task_id"]
    program = mutant["mutated_program"]

    # Rebuild task from dict (can't pickle TaskSpec easily, so use factory)
    task = task_dict.get(task_id)
    if task is None:
        return {}

    try:
        tables = generate_scenario(task.scenario_id, seed=eval_seed)
    except Exception:
        return {}

    prog_hash = hashlib.sha256(program.encode()).hexdigest()[:8]
    mut_id = f"mut_{task_id}_{mutant['mutation_operator']}_{prog_hash}"

    methods = {
        "universal_only": VerificationEngine(relations=UNIVERSAL_RELATIONS, timeout_seconds=8),
        "filter_agg":     VerificationEngine(relations=UNIVERSAL_RELATIONS + FILTER_RELATIONS + AGGREGATION_RELATIONS, timeout_seconds=8),
        "full_morph_da":  VerificationEngine(relations=ALL_RELATIONS, timeout_seconds=8),
    }

    row = {
        "mutant_id": mut_id,
        "task_id": task_id,
        "mutation_family": mutant["mutation_family"],
        "mutation_operator": mutant["mutation_operator"],
        "killed_by": {},
    }

    for method_name, engine in methods.items():
        try:
            report = engine.verify(program, tables, task, f"{method_name}_{mut_id}")
            row["killed_by"][method_name] = report.decision == "fail"
        except Exception:
            row["killed_by"][method_name] = False

    row["killed_any"] = any(row["killed_by"].values())
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="benchmark/frozen_mutants/rulemut_corpus.jsonl")
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path("runs/verification")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = {t.task_id: t for t in generate_task_set()}
    mutants = load_jsonl(args.corpus)
    print(f"Loaded {len(mutants)} mutants | workers={args.workers}")

    work_items = [(m, tasks, args.eval_seed) for m in mutants]
    results = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_verify_one, item): i for i, item in enumerate(work_items)}
        done = 0
        for future in as_completed(futures):
            row = future.result()
            if row:
                results.append(row)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(mutants)} done", flush=True)

    # Write results
    with LogWriter(out_dir / "results.jsonl") as w:
        # Overwrite: open fresh
        pass
    with open(out_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    # Summary
    families = [r["mutation_family"] for r in results]
    task_ids_list = [r["task_id"] for r in results]

    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total mutants evaluated: {len(results)}\n")

    method_names = ["universal_only", "filter_agg", "full_morph_da"]
    for method_name in method_names:
        killed = [r["killed_by"].get(method_name, False) for r in results]
        ms = compute_mutation_score(killed, families)
        print(f"  {method_name:<22} micro={ms.micro_mutation_score:.1%}  "
              f"macro={ms.macro_mutation_score:.1%}  killed={ms.n_killed}/{ms.n_valid_mutants}")

    print()
    full_killed = [r["killed_by"].get("full_morph_da", False) for r in results]
    full_ms = compute_mutation_score(full_killed, families)
    print("Per-family kill rate (Full MORPH-DA):")
    for fam, rate in sorted(full_ms.per_family_kill_rate.items()):
        n_fam = sum(1 for f in families if f == fam)
        k_fam = sum(1 for k, f in zip(full_killed, families) if f == fam and k)
        print(f"  {fam:<22} {rate:.1%}  ({k_fam}/{n_fam})")

    with open(out_dir / "summary.txt", "w") as f:
        f.write(f"Total mutants: {len(results)}\n")
        for method_name in method_names:
            killed = [r["killed_by"].get(method_name, False) for r in results]
            ms = compute_mutation_score(killed, families)
            f.write(f"{method_name}: micro={ms.micro_mutation_score:.1%} macro={ms.macro_mutation_score:.1%}\n")

    print(f"\nResults → {out_dir}/")


if __name__ == "__main__":
    main()
