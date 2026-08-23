#!/usr/bin/env python3
"""
Phase 1 exit condition: validate all reference programs.

Checks:
  1. Every task compiles without error.
  2. Every reference program executes on all data seeds.
  3. Zero metamorphic-relation violations on reference programs (FPR = 0).
  4. Gold answers are deterministic.

Usage:
    python scripts/validate_references.py [--seeds 42 7 123] [--verbose]

Output:
    runs/validation/reference_validation.jsonl
    runs/validation/summary.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphda.tasks.factory import generate_task_set, task_set_summary
from morphda.tasks.validators import validate_task_set
from morphda.logging.writer import LogWriter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path("runs/validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating task set...")
    tasks = generate_task_set()
    print(task_set_summary(tasks))
    print()

    print(f"Validating {len(tasks)} tasks on seeds {args.seeds} ...\n")
    results, summary = validate_task_set(tasks, seeds=args.seeds, verbose=args.verbose)

    # Write JSONL
    with LogWriter(out_dir / "reference_validation.jsonl") as w:
        for r in results:
            w.write({
                "task_id": r.task_id,
                "passed": r.passed,
                "failures": r.failures,
                "gold_answers": {str(k): str(v) for k, v in r.gold_answers.items()},
                "mr_violations": r.relation_violations,
            })

    # Summary
    lines = [
        "",
        "=" * 60,
        "REFERENCE VALIDATION SUMMARY",
        "=" * 60,
        f"Total tasks:       {summary['total']}",
        f"Passed:            {summary['passed']}",
        f"Failed:            {summary['failed']}",
        f"Pass rate:         {summary['pass_rate']:.1%}",
        f"MR violations:     {summary['mr_violations']} (must be 0)",
        "",
    ]

    if summary['failed'] > 0:
        lines.append("FAILED TASKS:")
        for r in results:
            if not r.passed:
                lines.append(f"  {r.task_id}: {r.failures[0][:80]}")
        lines.append("")

    summary_text = "\n".join(lines)
    print(summary_text)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary_text)

    exit_code = 0 if summary['failed'] == 0 and summary['mr_violations'] == 0 else 1
    print(f"\nPhase 1 exit condition: {'PASS ✓' if exit_code == 0 else 'FAIL ✗'}")
    print(f"Results → {out_dir}/")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
