#!/usr/bin/env python3
"""
Natural-agent track experiment (Section 15.1).

Runs LLM agents on all tasks × seeds, verifies with MORPH-DA,
and evaluates gold correctness.

Prerequisites:
  - ANTHROPIC_API_KEY environment variable set
  - Install: pip install anthropic langchain-anthropic

Usage:
    python scripts/run_natural_agents.py --model claude-opus-4-5 --seeds 42 7 123

Output:
    runs/natural_agents/{model_id}/programs.jsonl
    runs/natural_agents/{model_id}/verification.jsonl
    runs/natural_agents/{model_id}/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphda.data.generators import generate_scenario
from morphda.evaluation.metrics import (
    compute_verification_metrics,
    compute_mutation_score,
)
from morphda.logging.schemas import ProgramRecord, VerificationRecord, record_to_dict
from morphda.logging.writer import LogWriter
from morphda.reference.compiler import run_reference
from morphda.tasks.factory import generate_task_set
from morphda.verification.engine import VerificationEngine
from morphda.relations import ALL_RELATIONS
from morphda.agents.langgraph_agent import MorphDaAgent
from morphda.agents.prompts import build_schema_summary
from morphda.execution.normalization import outputs_equal


def _build_llm(model_id: str):
    """
    Build an LLM client.

    Tries in order:
      1. morphda.agents.llm_gateway (Anthropic-compatible gateway)
      2. LangChain ChatAnthropic (requires ANTHROPIC_API_KEY)
      3. LangChain ChatOpenAI (requires OPENAI_API_KEY)
    """
    # 1. Generic gateway client (uses ANTHROPIC_API_KEY or MORPH_DA_API_KEY)
    try:
        from morphda.agents.llm_gateway import build_llm_client
        llm = build_llm_client(model=model_id)
        llm.invoke([{"role": "user", "content": "ping"}])
        print(f"Using gateway: {model_id}")
        return llm
    except Exception as e:
        print(f"Gateway unavailable ({e}), trying LangChain...")

    # 2. Anthropic API via LangChain
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_id, temperature=0.0)
    except ImportError:
        pass

    # 3. OpenAI-compatible API via LangChain
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, temperature=0.0)
    except ImportError:
        pass

    raise ImportError("No LLM backend available. Try: pip install langchain-anthropic")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-4-5")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit tasks for pilot")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    model_slug = args.model.replace("/", "_").replace("-", "_")
    out_dir = Path(f"runs/natural_agents/{model_slug}")
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = generate_task_set()
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]

    engine  = VerificationEngine(relations=ALL_RELATIONS, timeout_seconds=args.timeout)
    llm     = _build_llm(args.model)
    agent   = MorphDaAgent(llm=lambda msgs: llm.invoke(msgs), timeout_seconds=args.timeout)

    print(f"Model: {args.model} | Tasks: {len(tasks)} | Seeds: {args.seeds}")

    n_correct = n_total = n_wrong_exe = n_exe_fail = 0
    morph_labels: list[bool] = []    # True = incorrect program
    morph_preds:  list[bool] = []    # True = MORPH-DA flagged

    with (
        LogWriter(out_dir / "programs.jsonl") as prog_w,
        LogWriter(out_dir / "verification.jsonl") as ver_w,
    ):
        for task in tasks:
            for seed in args.seeds:
                try:
                    tables = generate_scenario(task.scenario_id, seed=seed)
                except Exception:
                    continue

                # Gold answer
                try:
                    gold = run_reference(task, tables)
                except Exception:
                    continue

                schema = build_schema_summary(tables)

                # Agent run
                result = agent.run(
                    question=task.canonical_question,
                    tables=tables,
                    task_id=f"{task.task_id}_s{seed}",
                )

                n_total += 1
                prog_id = f"{model_slug}_{task.task_id}_s{seed}"

                is_correct = False
                if result.success:
                    try:
                        is_correct = outputs_equal(
                            result.source_output, gold, task.output_type
                        )
                    except Exception:
                        pass
                    if not is_correct:
                        n_wrong_exe += 1
                    else:
                        n_correct += 1
                else:
                    n_exe_fail += 1

                # MORPH-DA verification
                morph_flagged = False
                if result.generated_program and result.success:
                    report = engine.verify(
                        result.generated_program, tables, task, prog_id
                    )
                    morph_flagged = report.decision == "fail"

                    ver_w.write(record_to_dict(VerificationRecord(
                        program_id=prog_id,
                        task_id=task.task_id,
                        data_seed=seed,
                        source="natural",
                        decision=report.decision,
                        applicable_relations=report.applicable_relations,
                        violated_relations=report.violated_relations,
                        total_python_runs=report.total_python_runs,
                        total_latency_ms=report.total_latency_ms,
                        witnesses=[w.to_dict() for w in report.witnesses],
                    )))

                morph_labels.append(not is_correct)
                morph_preds.append(morph_flagged)

                prog_w.write(record_to_dict(ProgramRecord(
                    program_id=prog_id,
                    task_id=task.task_id,
                    scenario_id=task.scenario_id,
                    data_seed=seed,
                    source="natural",
                    model_id=args.model,
                    agent_harness="morphda_agent_v1",
                    generated_program=result.generated_program or "",
                    execution_success=result.success,
                    source_output=result.source_output,
                    gold_correct=is_correct,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    model_latency_ms=result.model_latency_ms,
                )))

                status = "✓" if is_correct else ("✗" if result.success else "E")
                morph_status = "⚠" if morph_flagged else " "
                print(f"  {status}{morph_status} {task.task_id} s={seed}", flush=True)

    # Summary
    acc = n_correct / n_total if n_total else 0.0
    wer = n_wrong_exe / (n_total - n_exe_fail) if (n_total - n_exe_fail) > 0 else 0.0
    esr = (n_total - n_exe_fail) / n_total if n_total else 0.0

    summary = {
        "model": args.model,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_exe_fail": n_exe_fail,
        "n_wrong_executable": n_wrong_exe,
        "task_accuracy": acc,
        "execution_success_rate": esr,
        "wrong_but_executable_rate": wer,
    }

    if sum(morph_labels) > 0:
        vm = compute_verification_metrics(morph_labels, morph_preds)
        summary["morph_precision"] = vm.precision
        summary["morph_recall"]    = vm.recall
        summary["morph_fpr"]       = vm.false_positive_rate

    print(f"\n{'='*55}")
    print(f"Task accuracy:              {acc:.1%}")
    print(f"Execution success rate:     {esr:.1%}")
    print(f"Wrong-but-executable rate:  {wer:.1%}")
    if "morph_precision" in summary:
        print(f"MORPH-DA precision:         {summary['morph_precision']:.1%}")
        print(f"MORPH-DA recall:            {summary['morph_recall']:.1%}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults → {out_dir}/")


if __name__ == "__main__":
    main()
