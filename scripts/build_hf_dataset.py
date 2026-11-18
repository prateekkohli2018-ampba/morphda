#!/usr/bin/env python3
"""
Package MORPH-DA as a Hugging Face dataset repository (JSONL + CSV, no extra dependencies).

Configs written:
    tasks            101 tasks: question, task specification, reference answers per seed
    <table name>     one config per data table, one split per data seed (seed42, seed7, seed123)
    mutants          seeded-fault programs with fault family and operator
    agent_programs   programs written by three hosted models, with per-seed correctness

The paper and poster are copied under paper/. The dataset is licensed CC BY 4.0.
Nothing is uploaded by this script.

Usage:
    python scripts/build_hf_dataset.py --repo-id YOUR_NAME/morph-da --out ../morphda-hf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from morphda.data.generators import generate_scenario  # noqa: E402
from morphda.evaluation.natural import scope_is_discriminating  # noqa: E402
from morphda.reference.compiler import run_reference  # noqa: E402
from morphda.tasks.factory import generate_task_set  # noqa: E402

SEEDS = (42, 7, 123)
MODELS = ("claude_haiku_4_5", "claude_sonnet_4_6", "claude_opus_4_5")
CARD_TEMPLATE = Path(__file__).with_name("hf_dataset_card.md")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def build_tasks(out: Path) -> tuple[list, dict[str, str]]:
    tasks = generate_task_set()
    gold = {s: {} for s in SEEDS}
    for s in SEEDS:
        for t in tasks:
            gold[s][t.task_id] = run_reference(t, generate_scenario(t.scenario_id, seed=s))
    rows = []
    for t in tasks:
        spec = t.model_dump(mode="json")
        rows.append({
            "task_id": t.task_id,
            "scenario_id": t.scenario_id,
            "tables": list(t.inputs),
            "question": t.canonical_question,
            "question_family": spec["question_family"],
            "difficulty_level": spec["difficulty_level"],
            "output_type": spec["output_type"],
            "scope_discriminating": scope_is_discriminating(t),
            **{f"gold_seed{s}": json.dumps(gold[s][t.task_id], default=str) for s in SEEDS},
            "task_spec": json.dumps(spec, default=str),
        })
    _write_jsonl(out / "data" / "tasks.jsonl", rows)
    return tasks, {t.task_id: t.scenario_id for t in tasks}


def build_tables(out: Path, tasks: list) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """One CSV per (table, seed). Returns ({table: [scenario ids]}, {table: schema info})."""
    owners: dict[str, set[str]] = {}
    schemas: dict[str, dict] = {}
    for scenario in sorted({t.scenario_id for t in tasks}):
        for seed in SEEDS:
            for name, df in generate_scenario(scenario, seed=seed).items():
                owners.setdefault(name, set()).add(scenario)
                info = schemas.setdefault(name, {"columns": {c: str(t) for c, t in df.dtypes.items()}, "rows": {}})
                info["rows"][seed] = len(df)
                path = out / "data" / "tables" / name / f"seed{seed}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(path, index=False)
    clashes = {n: s for n, s in owners.items() if len(s) > 1}
    if clashes:
        raise SystemExit(f"table names shared by several scenarios: {clashes}")
    return {n: sorted(s) for n, s in owners.items()}, schemas


def build_mutants(out: Path) -> int:
    rows = []
    for m in _read_jsonl(ROOT / "benchmark" / "frozen_mutants" / "rulemut_corpus.jsonl"):
        digest = hashlib.sha256(m["mutated_program"].encode()).hexdigest()[:8]
        rows.append({
            "mutant_id": f"mut_{m['task_id']}_{m['mutation_operator']}_{digest}",
            "task_id": m["task_id"],
            "mutation_family": m["mutation_family"],
            "mutation_operator": m["mutation_operator"],
            "mutated_program": m["mutated_program"],
            "non_equivalent_seeds": m.get("non_equivalent_seeds", []),
        })
    _write_jsonl(out / "data" / "mutants.jsonl", rows)
    return len(rows)


def build_agent_programs(out: Path, scenarios: dict[str, str]) -> int:
    rows = []
    for model in MODELS:
        base = ROOT / "runs" / "natural_agents" / model
        evals = {r["task_id"]: r for r in _read_jsonl(base / "evaluation_rows.jsonl")}
        latest: dict[str, dict] = {}
        for p in _read_jsonl(base / "programs.jsonl"):
            if p.get("data_seed") == 42:
                latest[p["task_id"]] = p
        for task_id, p in sorted(latest.items()):
            e = evals.get(task_id)
            held = (e or {}).get("correct_held_out") or {}
            rows.append({
                "model": p.get("model_id", model),
                "task_id": task_id,
                "scenario_id": scenarios.get(task_id),
                "generated_program": p["generated_program"],
                "executed_on_seed42": bool(e["executed"] if e else p.get("execution_success")),
                "correct_seed42": e["correct_source"] if e else p.get("gold_correct"),
                "correct_seed7": held.get("7"),
                "correct_seed123": held.get("123"),
            })
    _write_jsonl(out / "data" / "agent_programs.jsonl", rows)
    return len(rows)


def _schema_markdown(tables: dict[str, list[str]], schemas: dict[str, dict]) -> str:
    blocks = []
    for name in tables:
        rows = schemas[name]["rows"]
        cols = ", ".join(f"`{c}` ({t})" for c, t in schemas[name]["columns"].items())
        sizes = ", ".join(f"seed{s}: {rows[s]:,}" for s in SEEDS)
        blocks.append(f"- **`{name}`** — scenario `{tables[name][0]}`; rows {sizes}.  \n  Columns: {cols}")
    return "\n".join(blocks)


def write_card(out: Path, repo_id: str, tables: dict[str, list[str]], schemas: dict[str, dict], counts: dict) -> None:
    configs = [
        "- config_name: tasks\n  data_files:\n  - split: train\n    path: data/tasks.jsonl",
        "- config_name: mutants\n  data_files:\n  - split: train\n    path: data/mutants.jsonl",
        "- config_name: agent_programs\n  data_files:\n  - split: train\n    path: data/agent_programs.jsonl",
    ]
    table_lines = []
    for name, scen in tables.items():
        splits = "\n".join(f"  - split: seed{s}\n    path: data/tables/{name}/seed{s}.csv" for s in SEEDS)
        configs.append(f"- config_name: {name}\n  data_files:\n{splits}")
        table_lines.append(f"| `{name}` | {', '.join(scen)} |")
    card = CARD_TEMPLATE.read_text(encoding="utf-8")
    for key, value in {
        "{{REPO_ID}}": repo_id,
        "{{CONFIGS_YAML}}": "\n".join(configs),
        "{{TABLE_ROWS}}": "\n".join(table_lines),
        "{{TABLE_SCHEMAS}}": _schema_markdown(tables, schemas),
        "{{N_TASKS}}": str(counts["tasks"]),
        "{{N_MUTANT_ROWS}}": str(counts["mutants"]),
        "{{N_PROGRAMS}}": str(counts["agent_programs"]),
    }.items():
        card = card.replace(key, value)
    (out / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. your-name/morph-da")
    ap.add_argument("--out", default="../morphda-hf")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    tasks, scenarios = build_tasks(out)
    tables, schemas = build_tables(out, tasks)
    counts = {
        "tasks": len(tasks),
        "mutants": build_mutants(out),
        "agent_programs": build_agent_programs(out, scenarios),
    }

    (out / "paper").mkdir()
    for name in ("morphda_paper.pdf", "morphda_poster.pdf", "morphda_poster.png"):
        shutil.copy(ROOT / "paper" / name, out / "paper" / name)
    shutil.copy(Path(__file__).with_name("hf_dataset_LICENSE.txt"), out / "LICENSE")
    write_card(out, args.repo_id, tables, schemas, counts)

    print(f"Built {out}: {counts}, tables={sorted(tables)}")


if __name__ == "__main__":
    main()
