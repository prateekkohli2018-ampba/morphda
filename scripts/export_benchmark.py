#!/usr/bin/env python3
"""
Export the benchmark to plain files so it can be used without this package.

Writes, for the 101 tasks and the chosen data seeds:
    <out>/task_specs/<task_id>.yaml            structured task specification (verifier-only)
    <out>/questions.jsonl                      what the agent sees: task_id, question, scenario, tables
    <out>/gold/seed<N>.jsonl                   reference answers per task
    <out>/data/<scenario>/seed<N>/<table>.csv  the tables themselves

Usage:
    python scripts/export_benchmark.py                       # seeds 42 7 123 -> benchmark/export
    python scripts/export_benchmark.py --seeds 42 --out /tmp/morphda_bench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphda.data.generators import generate_scenario
from morphda.reference.compiler import run_reference
from morphda.tasks.factory import generate_task_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--out", default="benchmark/export")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "task_specs").mkdir(parents=True, exist_ok=True)
    tasks = generate_task_set()

    with open(out / "questions.jsonl", "w", encoding="utf-8") as fh:
        for t in tasks:
            (out / "task_specs" / f"{t.task_id}.yaml").write_text(
                yaml.safe_dump(t.model_dump(mode="json"), sort_keys=False)
            )
            fh.write(json.dumps({
                "task_id": t.task_id,
                "scenario_id": t.scenario_id,
                "tables": t.inputs,
                "question": t.canonical_question,
            }) + "\n")

    scenarios = sorted({t.scenario_id for t in tasks})
    for seed in args.seeds:
        for scenario in scenarios:
            tables = generate_scenario(scenario, seed=seed)
            d = out / "data" / scenario / f"seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            for name, df in tables.items():
                df.to_csv(d / f"{name}.csv", index=False)

        (out / "gold").mkdir(exist_ok=True)
        with open(out / "gold" / f"seed{seed}.jsonl", "w", encoding="utf-8") as fh:
            for t in tasks:
                gold = run_reference(t, generate_scenario(t.scenario_id, seed=seed))
                fh.write(json.dumps({"task_id": t.task_id, "gold": gold}, default=str) + "\n")

    print(f"Exported {len(tasks)} tasks, {len(scenarios)} scenarios, seeds {args.seeds} -> {out}/")


if __name__ == "__main__":
    main()
