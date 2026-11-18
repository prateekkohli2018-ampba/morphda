#!/usr/bin/env python3
"""
Natural-agent evaluation: accidental corrects and Conditions A/B/C (paper Table 2).

Reads the seed-42 programs an agent produced, re-executes them unchanged on
held-out data seeds, runs MORPH-DA with two transformation seeds, and reports
precision / recall / FPR / F1 / AAR per condition. No LLM calls are made.

Usage:
    python scripts/evaluate_natural_agents.py runs/natural_agents/claude_haiku_4_5 [more run dirs]
        [--workers 4] [--all-tasks] [--n-boot 3000]

Output (per run dir):
    evaluation_rows.jsonl      one row per program
    evaluation_summary.json    counts + Conditions A/B/C
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphda.evaluation.natural import (
    evaluate_programs,
    load_source_programs,
    scope_is_discriminating,
    summarize,
)
from morphda.tasks.factory import generate_task_set


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}"


def print_summary(name: str, s: dict) -> None:
    print(f"\n{name}: {s['n_tasks']} tasks, {s['n_exec_failures']} original execution failures excluded")
    print(f"  correct on source seed: {s['n_correct_source_seed']}  |  "
          f"cross-seed correct: {s['n_cross_seed_correct']}  |  "
          f"accidental corrects: {s['n_accidental_correct']} "
          f"(MORPH-DA flags {s['accidental_correct_flagged']})")
    print(f"  {'condition':<14}{'prec':>7}{'recall':>8}{'FPR':>7}{'F1':>7}{'AAR':>7}   TP/FP/TN/FN")
    for cond, c in s["conditions"].items():
        print(f"  {cond:<14}{_pct(c['precision']):>7}{_pct(c['recall']):>8}{_pct(c['fpr']):>7}"
              f"{_pct(c['f1']):>7}{_pct(c['aar']):>7}   {c['tp']}/{c['fp']}/{c['tn']}/{c['fn']}")
        if "ci95" in c:
            ci = c["ci95"]
            print(f"  {'':<14}95% CI  precision [{_pct(ci['precision'][0]).strip()}, {_pct(ci['precision'][1]).strip()}]"
                  f"  recall [{_pct(ci['recall'][0]).strip()}, {_pct(ci['recall'][1]).strip()}]"
                  f"  FPR [{_pct(ci['fpr'][0]).strip()}, {_pct(ci['fpr'][1]).strip()}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="directories containing programs.jsonl")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--all-tasks", action="store_true",
                        help="skip the scope-discriminability filter (paper uses the filtered set)")
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    tasks = generate_task_set()
    if not args.all_tasks:
        kept = [t for t in tasks if scope_is_discriminating(t)]
        print(f"Scope-discriminating tasks: {len(kept)}/{len(tasks)} "
              f"(excluded: {sorted({t.task_id for t in tasks} - {t.task_id for t in kept})})")
        tasks = kept

    for run_dir in map(Path, args.run_dirs):
        programs = load_source_programs(run_dir / "programs.jsonl")
        rows = evaluate_programs(programs, tasks, timeout=args.timeout, workers=args.workers)
        summary = summarize(rows, n_boot=args.n_boot)

        with open(run_dir / "evaluation_rows.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        (run_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
        print_summary(run_dir.name, summary)


if __name__ == "__main__":
    main()
