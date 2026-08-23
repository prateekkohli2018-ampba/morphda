#!/usr/bin/env python3
"""
Generate paper result tables (Section 21) from frozen experimental data.

Tables generated:
  Table 1: Benchmark composition
  Table 2: Mutant corpus quality
  Table 3: Verification on controlled mutants
  Table 4: Detection by fault family

Usage:
    python scripts/make_result_tables.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from morphda.data.generators import BUILTIN_SCENARIOS
from morphda.evaluation.metrics import compute_mutation_score, compute_verification_metrics
from morphda.logging.writer import load_jsonl
from morphda.tasks.factory import generate_task_set, task_set_summary


def table1_benchmark_composition() -> str:
    """Table 1: Benchmark composition by split."""
    tasks = generate_task_set()
    by_level = Counter(t.difficulty_level for t in tasks)
    by_scenario = Counter(t.scenario_id for t in tasks)
    n_single = sum(1 for t in tasks if len(t.joins) == 0)
    n_multi  = len(tasks) - n_single
    n_easy   = sum(1 for t in tasks if t.difficulty_level <= 2)
    n_hard   = sum(1 for t in tasks if t.difficulty_level >= 3)

    lines = [
        "\nTable 1: Benchmark Composition",
        "=" * 70,
        f"{'Component':<35} {'Count':>8}",
        "-" * 70,
        f"  {'Controlled task specifications':<33} {len(tasks):>8}",
        f"  {'Dataset scenarios':<33} {len(BUILTIN_SCENARIOS):>8}",
        f"  {'Single-table tasks':<33} {n_single:>8}",
        f"  {'Multi-table tasks':<33} {n_multi:>8}",
        f"  {'Difficulty L1-L2 (easy)':<33} {n_easy:>8}",
        f"  {'Difficulty L3-L5 (hard)':<33} {n_hard:>8}",
        "",
        f"  {'By difficulty level:':<33}",
    ]
    for lvl, cnt in sorted(by_level.items()):
        lines.append(f"    {'L' + str(lvl):<31} {cnt:>8}")

    lines += [
        "",
        f"  {'By scenario:':<33}",
    ]
    for scen, cnt in sorted(by_scenario.items()):
        lines.append(f"    {scen:<31} {cnt:>8}")

    lines.append("=" * 70)
    return "\n".join(lines)


def table2_mutant_corpus_quality() -> str:
    """Table 2: Mutant corpus quality."""
    corpus_path = Path("benchmark/frozen_mutants/rulemut_corpus.jsonl")
    stats_path  = Path("benchmark/frozen_mutants/rulemut_stats.json")

    if not corpus_path.exists():
        return "\nTable 2: Not available (run generate_rule_mutants.py first)"

    mutants = load_jsonl(str(corpus_path))
    by_family: dict[str, dict] = defaultdict(lambda: {"valid": 0})
    for m in mutants:
        fam = m.get("mutation_family", "unknown")
        by_family[fam]["valid"] += 1

    import json
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

    lines = [
        "\nTable 2: Mutant Corpus Quality",
        "=" * 75,
        f"{'Component':<35} {'Count':>8}",
        "-" * 75,
        f"  {'Generated candidates':<33} {stats.get('generated', '?'):>8}",
        f"  {'Not applicable':<33} {stats.get('not_applicable', '?'):>8}",
        f"  {'Syntax invalid':<33} {stats.get('syntax_invalid', 0):>8}",
        f"  {'Execution invalid':<33} {stats.get('exec_invalid', '?'):>8}",
        f"  {'Contract invalid':<33} {stats.get('contract_invalid', '?'):>8}",
        f"  {'Provisionally equivalent':<33} {stats.get('equivalent', '?'):>8}",
        f"  {'Valid non-equivalent (corpus)':<33} {stats.get('valid', len(mutants)):>8}",
        "",
        f"  {'By fault family:':<33}",
    ]
    for fam, d in sorted(by_family.items()):
        lines.append(f"    {fam:<31} {d['valid']:>8}")

    lines.append("=" * 75)
    return "\n".join(lines)


def table3_verification_controlled() -> str:
    """Table 3: Verification performance on controlled mutants."""
    results_path = Path("runs/verification/results.jsonl")
    if not results_path.exists():
        return "\nTable 3: Not available (run run_verification.py first)"

    results = load_jsonl(str(results_path))
    families = [r.get("mutation_family", "unknown") for r in results]

    lines = [
        "\nTable 3: Verification on Controlled Mutants",
        "=" * 85,
        f"{'Method':<25} {'Micro MS':>9} {'Macro MS':>9} {'Killed':>8} {'Total':>7}",
        "-" * 85,
    ]

    method_labels = {
        "no_verifier":    "Execution only (B0)",
        "universal_only": "Universal MRs only (B7)",
        "filter_agg":     "Universal+Filter+Agg MRs",
        "full_morph_da":  "Full MORPH-DA (B8)",
    }

    for method_key, label in method_labels.items():
        if method_key == "no_verifier":
            lines.append(f"  {label:<23}     0.0%      0.0%       0 {len(results):>7}")
            continue
        killed = [r.get("killed_by", {}).get(method_key, False) for r in results]
        ms = compute_mutation_score(killed, families)
        lines.append(
            f"  {label:<23} {ms.micro_mutation_score:>8.1%} {ms.macro_mutation_score:>9.1%} "
            f"{ms.n_killed:>8} {ms.n_valid_mutants:>7}"
        )

    lines.append("=" * 85)
    return "\n".join(lines)


def table4_detection_by_fault_family() -> str:
    """Table 4: Detection rate by fault family for key methods."""
    results_path = Path("runs/verification/results.jsonl")
    if not results_path.exists():
        return "\nTable 4: Not available"

    results = load_jsonl(str(results_path))
    families = sorted({r.get("mutation_family", "unknown") for r in results})
    methods  = ["universal_only", "filter_agg", "full_morph_da"]

    lines = [
        "\nTable 4: Detection Rate by Fault Family",
        "=" * 80,
        f"{'Fault Family':<22} {'Universal':>10} {'Filter+Agg':>11} {'Full MORPH':>11} {'N':>6}",
        "-" * 80,
    ]

    for fam in families:
        fam_results = [r for r in results if r.get("mutation_family") == fam]
        n = len(fam_results)
        row_parts = [f"  {fam:<20}"]
        for method in methods:
            killed_fam = sum(1 for r in fam_results if r.get("killed_by", {}).get(method))
            rate = killed_fam / n if n > 0 else 0.0
            row_parts.append(f"{rate:>10.1%}")
        row_parts.append(f"{n:>7}")
        lines.append("".join(row_parts))

    # Overall row
    all_n = len(results)
    lines.append("-" * 80)
    overall = ["  OVERALL               "]
    for method in methods:
        killed_all = sum(1 for r in results if r.get("killed_by", {}).get(method))
        rate = killed_all / all_n if all_n > 0 else 0.0
        overall.append(f"{rate:>10.1%}")
    overall.append(f"{all_n:>7}")
    lines.append("".join(overall))
    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    out_dir = Path("paper")
    out_dir.mkdir(exist_ok=True)

    tables = [
        table1_benchmark_composition(),
        table2_mutant_corpus_quality(),
        table3_verification_controlled(),
        table4_detection_by_fault_family(),
    ]

    full_text = "\n\n".join(tables)
    print(full_text)

    with open(out_dir / "result_tables.txt", "w") as f:
        f.write(full_text + "\n")
    print(f"\n→ paper/result_tables.txt")


if __name__ == "__main__":
    main()
