#!/usr/bin/env python3
"""
Parallel natural-agent track — runs tasks concurrently via thread pool.

Usage:
    python scripts/run_natural_agents_parallel.py --model claude-sonnet-4-6 --seeds 42 7 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphda.agents.llm_gateway import build_llm_client
from morphda.agents.langgraph_agent import MorphDaAgent
from morphda.agents.prompts import build_schema_summary
from morphda.data.generators import generate_scenario
from morphda.evaluation.metrics import compute_verification_metrics
from morphda.execution.normalization import outputs_equal
from morphda.logging.schemas import ProgramRecord, VerificationRecord, record_to_dict
from morphda.reference.compiler import run_reference
from morphda.relations import ALL_RELATIONS
from morphda.tasks.factory import generate_task_set
from morphda.verification.engine import VerificationEngine


_lock = threading.Lock()
_results: list[dict] = []


def _run_one(args: tuple) -> dict | None:
    task, seed, model, timeout = args
    try:
        llm = build_llm_client(model=model)
        agent = MorphDaAgent(llm=llm, timeout_seconds=timeout)
        engine = VerificationEngine(relations=ALL_RELATIONS, timeout_seconds=timeout)

        tables = generate_scenario(task.scenario_id, seed=seed)
        gold = run_reference(task, tables)

        result = agent.run(
            question=task.canonical_question,
            tables=tables,
            task_id=f"{task.task_id}_s{seed}",
        )

        is_correct = False
        if result.success and result.source_output is not None:
            is_correct = outputs_equal(result.source_output, gold, task.output_type)

        morph_flagged = False
        morph_report = None
        if result.success and result.generated_program:
            morph_report = engine.verify(result.generated_program, tables, task,
                                          f"{task.task_id}_s{seed}")
            morph_flagged = morph_report.decision == "fail"

        status = "correct" if is_correct else ("wrong_exe" if result.success else "exe_fail")
        return {
            "task_id": task.task_id,
            "difficulty": task.difficulty_level,
            "scenario": task.scenario_id,
            "seed": seed,
            "model": model,
            "status": status,
            "is_correct": is_correct,
            "exe_success": result.success,
            "morph_flagged": morph_flagged,
            "morph_violations": morph_report.violated_relations if morph_report else 0,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "model_latency_ms": result.model_latency_ms,
        }
    except Exception as exc:
        return {
            "task_id": task.task_id, "seed": seed, "model": model,
            "status": "error", "is_correct": False, "exe_success": False,
            "morph_flagged": False, "morph_violations": 0, "error": str(exc)[:200],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--max-tasks", type=int, default=None)
    args = parser.parse_args()

    model_slug = args.model.replace("/", "_").replace("-", "_")
    out_dir = Path(f"runs/natural_agents/{model_slug}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = generate_task_set()
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    work_items = [(t, s, args.model, args.timeout) for t in tasks for s in args.seeds]
    print(f"Tasks={len(tasks)} × Seeds={len(args.seeds)} = {len(work_items)} runs | workers={args.workers}")

    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, item): item for item in work_items}
        for future in as_completed(futures):
            row = future.result()
            if row:
                results.append(row)
            done += 1
            if done % 20 == 0:
                correct_so_far = sum(1 for r in results if r.get("is_correct"))
                print(f"  {done}/{len(work_items)} done | acc so far: {correct_so_far}/{len(results)}", flush=True)

    # Write results
    with open(out_dir / "natural_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    # Summary
    n_total  = len(results)
    n_correct = sum(1 for r in results if r.get("is_correct"))
    n_exe_ok  = sum(1 for r in results if r.get("exe_success"))
    n_wrong_exe = sum(1 for r in results if r.get("exe_success") and not r.get("is_correct"))

    acc = n_correct / n_total if n_total else 0.0
    esr = n_exe_ok  / n_total if n_total else 0.0
    wer = n_wrong_exe / n_exe_ok if n_exe_ok else 0.0

    # MORPH-DA metrics
    labels = [not r.get("is_correct") for r in results]
    preds  = [r.get("morph_flagged", False) for r in results]
    vm = compute_verification_metrics(labels, preds) if sum(labels) > 0 else None

    # Per-difficulty breakdown
    by_level: dict[int, dict] = {}
    for r in results:
        lvl = r.get("difficulty", 0)
        d = by_level.setdefault(lvl, {"total": 0, "correct": 0, "wrong_exe": 0})
        d["total"] += 1
        if r.get("is_correct"):
            d["correct"] += 1
        elif r.get("exe_success"):
            d["wrong_exe"] += 1

    print(f"\n{'='*60}")
    print("NATURAL AGENT RESULTS")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Total runs:              {n_total}")
    print(f"Task accuracy:           {acc:.1%}")
    print(f"Execution success rate:  {esr:.1%}")
    print(f"Wrong-but-executable:    {wer:.1%}")
    if vm:
        print(f"MORPH-DA precision:      {vm.precision:.1%}")
        print(f"MORPH-DA recall:         {vm.recall:.1%}")
        print(f"MORPH-DA FPR:            {vm.false_positive_rate:.1%}")

    print("\nPer-difficulty level:")
    for lvl in sorted(by_level.keys()):
        d = by_level[lvl]
        acc_l = d["correct"] / d["total"] if d["total"] else 0.0
        wer_l = d["wrong_exe"] / d["total"] if d["total"] else 0.0
        print(f"  L{lvl}: acc={acc_l:.0%}  wrong_exe={wer_l:.0%}  n={d['total']}")

    summary = {
        "model": args.model, "seeds": args.seeds,
        "n_total": n_total, "n_correct": n_correct,
        "task_accuracy": acc, "execution_success_rate": esr,
        "wrong_but_executable_rate": wer,
        "morph_precision": vm.precision if vm else None,
        "morph_recall":    vm.recall    if vm else None,
        "morph_fpr":       vm.false_positive_rate if vm else None,
        "by_difficulty": {str(k): v for k, v in by_level.items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults → {out_dir}/")


if __name__ == "__main__":
    main()
